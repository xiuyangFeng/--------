"""Golden regression: re-run stage B for finished jobs and diff against their stored results.

Usage:
    PYTHONPATH=. python -m wss_deploy.regress --jobs-root outputs/wss_deploy_jobs --out /tmp/regress \
        [--job ID ...] [--device cuda|cpu] [--atol 1e-5] [--release-root DIR]

Every selected job is copied (input, clean STL, centreline, stage_a.json, job.json) to ``--out/<id>``,
stage B is executed with the job's own release/mapping/compute settings, and the new ``summary.json`` and
``field.npz`` are compared to the originals.  The exit code is non-zero if any job exceeds the tolerance.
This is the acceptance gate for refactors of the feature pipeline, the exporters and the run record.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

COPY_FILES = ("input.stl", "input_clean_mm.stl", "stage_a.json", "job.json")
SUMMARY_KEYS = ("peak", "wss_field_pa", "per_branch", "geometry", "surface_statistics", "quality",
                "volume_statistics", "murray_shares", "caps", "cloud", "mapping", "branch_names")
SKIP_NUMERIC = {"timing_s", "seconds_per_model", "created_at", "elapsed"}


def _walk(a, b, path, atol, diffs):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key in SKIP_NUMERIC:
                continue
            if key not in a or key not in b:
                diffs.append((f"{path}.{key}", "missing", None))
                continue
            _walk(a[key], b[key], f"{path}.{key}", atol, diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((path, "length", (len(a), len(b))))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, f"{path}[{i}]", atol, diffs)
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        if a != b:
            diffs.append((path, "value", (a, b)))
    else:
        try:
            x, y = float(a), float(b)
        except (TypeError, ValueError):
            if a != b:
                diffs.append((path, "value", (a, b)))
            return
        if not (np.isfinite(x) and np.isfinite(y)):
            if not (np.isnan(x) and np.isnan(y)):
                diffs.append((path, "nonfinite", (x, y)))
        elif abs(x - y) > atol * max(1.0, abs(x), abs(y)):
            diffs.append((path, "numeric", (x, y)))


def compare_arrays(old: Path, new: Path, atol: float) -> tuple[dict, list]:
    a, b = np.load(old), np.load(new)
    report, failures = {}, []
    for key in sorted(set(a.files) | set(b.files)):
        if key not in a.files or key not in b.files:
            report[key] = "missing"
            failures.append((key, "missing", None))
            continue
        x, y = a[key], b[key]
        if x.shape != y.shape:
            report[key] = f"shape {x.shape} vs {y.shape}"
            failures.append((key, "shape", (x.shape, y.shape)))
            continue
        if x.dtype.kind in "fc":
            finite = np.isfinite(x) & np.isfinite(y)
            if not np.array_equal(np.isfinite(x), np.isfinite(y)):
                failures.append((key, "nan-pattern", None))
            diff = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else 0.0
            scale = float(max(1.0, np.max(np.abs(x[finite])))) if finite.any() else 1.0
            report[key] = diff
            if diff > atol * scale:
                failures.append((key, "numeric", diff))
        else:
            equal = bool(np.array_equal(x, y))
            report[key] = "equal" if equal else "differs"
            if not equal:
                failures.append((key, "value", None))
    return report, failures


def run_job(src: Path, dst: Path, *, device: str, atol: float, release_root: str | None,
            reference: Path | None = None) -> dict:
    """Re-run stage B from the inputs in ``src`` and compare with ``reference`` (default: ``src``)."""
    from .registry import ReleaseRegistry
    from .pipeline import stage_b
    reference = Path(reference) if reference is not None else src
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for name in COPY_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
    if (src / "centerline").is_dir():
        shutil.copytree(src / "centerline", dst / "centerline")
    job = json.loads((src / "job.json").read_text(encoding="utf-8"))
    stage_a = json.loads((dst / "stage_a.json").read_text(encoding="utf-8"))
    # Historical records store an absolute clean-STL path; point it at the copy so the
    # regression never depends on the source directory staying in place.
    stage_a.setdefault("input_check", {})["clean_stl"] = str(dst / "input_clean_mm.stl")
    (dst / "stage_a.json").write_text(json.dumps(stage_a, ensure_ascii=False, indent=1), encoding="utf-8")
    compute = job.get("compute") or {}
    release_id = (job.get("model_release") or {}).get("id")
    registry = ReleaseRegistry(release_root, device=device)
    release = registry.load(release_id, device=device, seed_count=compute.get("seed_count"))
    started = time.perf_counter()
    meta = stage_b(dst, job["mapping"], release, confirmed=True, case_id=job.get("case_id"),
                   device=device, seed_count=compute.get("seed_count"), threads=compute.get("threads"))
    elapsed = time.perf_counter() - started
    old_summary = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    new_summary = json.loads((dst / "summary.json").read_text(encoding="utf-8"))
    diffs = []
    for key in SUMMARY_KEYS:
        if key in old_summary or key in new_summary:
            _walk(old_summary.get(key), new_summary.get(key), key, atol, diffs)
    arrays, array_failures = compare_arrays(reference / "field.npz", dst / "field.npz", atol)
    return {"job": src.name, "release": release_id, "device": device, "seconds": round(elapsed, 2),
            "reference": str(reference),
            "summary_diffs": [{"path": p, "kind": k, "values": v} for p, k, v in diffs[:50]],
            "summary_diff_count": len(diffs), "arrays": arrays,
            "array_failures": [{"key": k, "kind": kind, "value": v} for k, kind, v in array_failures],
            "passed": not diffs and not array_failures}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs-root", default="outputs/wss_deploy_jobs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--job", action="append", default=None, help="job id (repeatable); default: all done jobs with a bound release")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--release-root", default=None)
    parser.add_argument("--reference-root", default=None,
                        help="directory holding <id>/summary.json + field.npz to compare against "
                             "(default: the jobs root; use a previous --out to compare code versions)")
    parser.add_argument("--report", default=None, help="write JSON report here (default: <out>/regress_report.json)")
    args = parser.parse_args(argv)
    root, out = Path(args.jobs_root), Path(args.out)
    reference_root = Path(args.reference_root) if args.reference_root else None
    out.mkdir(parents=True, exist_ok=True)
    ids = args.job
    if not ids:
        ids = []
        for path in sorted(root.glob("*/job.json")):
            job = json.loads(path.read_text(encoding="utf-8"))
            if job.get("status") == "done" and (job.get("model_release") or {}).get("id") and job.get("mapping") \
                    and (path.parent / "summary.json").is_file() and (path.parent / "field.npz").is_file():
                ids.append(path.parent.name)
    results = []
    for job_id in ids:
        print(f"[regress] {job_id} ...", file=sys.stderr, flush=True)
        try:
            result = run_job(root / job_id, out / job_id, device=args.device, atol=args.atol, release_root=args.release_root,
                             reference=(reference_root / job_id) if reference_root else None)
        except Exception as exc:  # keep going; the report records the failure
            result = {"job": job_id, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        extra = result.get("error") or f"summary diffs {result.get('summary_diff_count')}, array failures {len(result.get('array_failures', []))}, {result.get('seconds')} s"
        print(f"[regress] {job_id}: {status} ({extra})", file=sys.stderr, flush=True)
    report = {"jobs_root": str(root), "reference_root": str(reference_root or root), "device": args.device, "atol": args.atol,
              "passed": all(r["passed"] for r in results), "results": results}
    report_path = Path(args.report) if args.report else out / "regress_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "n": len(results),
                      "failed": [r["job"] for r in results if not r["passed"]], "report": str(report_path)}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
