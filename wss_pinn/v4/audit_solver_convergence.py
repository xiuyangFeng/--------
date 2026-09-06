"""Per-time-step solver convergence audit from the Fluent transcripts of all 173 cases.

For every case the largest ``Fluent_*.out`` transcript (the run that produced
the current ``ascii_in`` frames) is parsed into per-time-step records:

* number of inner iterations used (journal cap: ``dual-time-iterate 1280 20``);
* whether Fluent printed ``solution is converged`` for the step;
* the last scaled residuals (continuity, x/y/z velocity) of the step.

The 81 exported steps (``1120..1280`` stride 2) are then summarised so that a
solver-quality whitelist can be written per case (see the 2026-09-03 rebuild
plan, P0-4 §5.2).  Nothing is written outside the anatomy-prep audit root.

Frozen 2026-09-05: the per-case ``solver_quality`` tier is defined on the
**residual value only**.  The Fluent ``solution is converged`` flag is a
case-setup artefact: 77/173 cases carry ``convergence-criterion-type 3`` (no
convergence check), so the flag can never fire there regardless of how well the
step converged; the other 96 cases use type 0 (absolute, continuity 1e-3).
Both the flag count and the criterion type are still recorded as provenance.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from wss_pinn.utils import ROOT, sha256_file, utc_now

SOURCE_MANIFEST = ROOT / "data_wss_pinn/volume_uvwp_bc_rcr_v4_train138_test35/manifest.json"
PREP_ROOT = ROOT / "outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903"
AUDIT_ROOT = PREP_ROOT / "audits/solver_convergence"
EXPORT_STEPS = list(range(1120, 1281, 2))
ITERATION_RE = re.compile(
    r"^\s*!?\s*(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+\d+:\d+:\d+\s+(\d+)"
)
STEP_RE = re.compile(r"Flow time = ([0-9.eE+-]+)s, time step = (\d+)")
JOURNAL_RE = re.compile(r"dual-time-iterate\s+(\d+)\s+(\d+)")
READ_CASE_RE = re.compile(r"read-case\s+(\S+)")
CRITERION_RE = re.compile(rb"convergence-criterion-type (\d)")
# Frozen 2026-09-05 (residual-only; see module docstring).
TIER_DEFINITION = (
    "max over the 81 export steps of the last continuity residual of the step: "
    "A <= 1e-3; B < 2e-3; C < 5e-3; D >= 5e-3; unknown = no transcript"
)
CRITERION_TYPE_NAMES = {0: "absolute", 1: "relative", 2: "relative-or-absolute", 3: "none"}


def solver_quality_tier(report: dict[str, Any]) -> str:
    value = (report.get("export_steps") or {}).get("continuity_last_max")
    if value is None:
        return "unknown"
    if value <= 1.0e-3:
        return "A"
    if value < 2.0e-3:
        return "B"
    if value < 5.0e-3:
        return "C"
    return "D"


def _case_file(raw_dir: Path, journal_text: str) -> Path | None:
    """The ``.cas.gz`` the journal reads (by basename; journal paths may be stale)."""

    match = READ_CASE_RE.search(journal_text)
    if match:
        candidate = raw_dir / Path(match.group(1)).name
        if candidate.is_file():
            return candidate
    cases = sorted(raw_dir.glob("*.cas.gz"))
    return cases[0] if len(cases) == 1 else None


def convergence_criterion_type(case_file: Path | None) -> int | None:
    if case_file is None:
        return None
    with gzip.open(case_file, "rb") as handle:
        for index, line in enumerate(handle):
            match = CRITERION_RE.search(line)
            if match:
                return int(match.group(1))
            if index > 200_000:
                break
    return None


def annotate_solver_quality(report: dict[str, Any]) -> dict[str, Any]:
    """Attach the frozen residual-only tier and the case convergence-check setting."""

    raw_dir = ROOT / "data_new" / report["canonical_id"]
    journal_path = (report.get("journal") or {}).get("path")
    journal_text = Path(journal_path).read_text(encoding="utf-8", errors="ignore") if journal_path and Path(journal_path).is_file() else ""
    case_file = _case_file(raw_dir, journal_text)
    criterion = convergence_criterion_type(case_file)
    export = report.get("export_steps") or {}
    report["solver_quality"] = {
        "tier": solver_quality_tier(report),
        "definition": TIER_DEFINITION,
        "continuity_last_max": export.get("continuity_last_max"),
        "continuity_last_p95": export.get("continuity_last_p95"),
        "flag_converged_export_steps": export.get("converged"),
        "convergence_criterion_type": criterion,
        "convergence_criterion_name": CRITERION_TYPE_NAMES.get(criterion) if criterion is not None else None,
        "case_file": str(case_file) if case_file else None,
    }
    return report


def _case_ids() -> list[dict[str, Any]]:
    payload = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    return [{"canonical_id": row["canonical_id"], "role": row["role"]} for row in payload["cases"]]


def _transcripts(raw_dir: Path) -> list[Path]:
    """Newest transcript first: the run that produced the current ``ascii_in`` frames.

    Size is not a valid selector (a superseded January run can be marginally
    larger than the September re-run of the same case).
    """

    candidates = list((raw_dir / "Global_conditions").glob("Fluent_*.out")) + list(raw_dir.glob("Fluent_*.out"))
    return sorted(candidates, key=lambda path: path.stat().st_mtime_ns, reverse=True)


def parse_transcript(path: Path) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "Updating solution at time level" in line:
                if current is not None:
                    steps.append(current)
                current = {"step": None, "iterations": 0, "converged": False, "last_residuals": None}
                continue
            if current is None:
                continue
            match = ITERATION_RE.match(line)
            if match:
                current["iterations"] += 1
                current["last_residuals"] = [float(match.group(k)) for k in range(2, 6)]
                continue
            if "solution is converged" in line:
                current["converged"] = True
                continue
            step = STEP_RE.search(line)
            if step:
                current["step"] = int(step.group(2))
                current["flow_time_s"] = float(step.group(1))
    if current is not None:
        steps.append(current)
    return steps


def audit_case(entry: dict[str, Any]) -> dict[str, Any]:
    canonical_id = entry["canonical_id"]
    output = AUDIT_ROOT / "cases" / f"{canonical_id.replace('/', '__')}.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    raw_dir = ROOT / "data_new" / canonical_id
    transcripts = _transcripts(raw_dir)
    journal = next(iter(sorted(raw_dir.glob("*.jou"))), None)
    journal_text = journal.read_text(encoding="utf-8", errors="ignore") if journal else ""
    journal_match = JOURNAL_RE.search(journal_text)
    report: dict[str, Any] = {
        "schema_version": 1,
        "canonical_id": canonical_id,
        "role": entry["role"],
        "created_at": utc_now(),
        "journal": {
            "path": str(journal) if journal else None,
            "time_steps": int(journal_match.group(1)) if journal_match else None,
            "max_iterations_per_step": int(journal_match.group(2)) if journal_match else None,
        },
        "transcripts": [
            {"path": str(path), "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in transcripts
        ],
    }
    frames = sorted((raw_dir / "ascii_in").glob("*-[0-9]*"))
    frame_mtime = max((path.stat().st_mtime_ns for path in frames), default=0)
    if not transcripts:
        report.update({"status": "no_transcript", "gate_pass": False})
    else:
        chosen = transcripts[0]
        report["transcript_newer_than_frames"] = bool(chosen.stat().st_mtime_ns >= frame_mtime - 24 * 3600 * 10**9)
        steps = parse_transcript(chosen)
        by_step = {row["step"]: row for row in steps if row["step"] is not None}
        export = [by_step[s] for s in EXPORT_STEPS if s in by_step]
        cap = report["journal"]["max_iterations_per_step"]
        residuals = np.asarray([row["last_residuals"] for row in export if row["last_residuals"]], dtype=np.float64)
        continuity = residuals[:, 0] if len(residuals) else np.zeros(0)
        report.update(
            {
                "status": "parsed",
                "transcript": {"path": str(chosen), "sha256": sha256_file(chosen), "size_bytes": chosen.stat().st_size},
                "parsed_steps": len(steps),
                "steps_with_index": len(by_step),
                "first_step": min(by_step) if by_step else None,
                "last_step": max(by_step) if by_step else None,
                "all_steps": {
                    "converged": int(sum(1 for row in by_step.values() if row["converged"])),
                    "count": len(by_step),
                },
                "export_steps": {
                    "found": len(export),
                    "converged": int(sum(1 for row in export if row["converged"])),
                    "iteration_cap_hits": int(sum(1 for row in export if cap and row["iterations"] >= cap)),
                    "iterations_p50": float(np.median([row["iterations"] for row in export])) if export else None,
                    "continuity_last_p50": float(np.percentile(continuity, 50)) if len(continuity) else None,
                    "continuity_last_p95": float(np.percentile(continuity, 95)) if len(continuity) else None,
                    "continuity_last_max": float(np.max(continuity)) if len(continuity) else None,
                    "velocity_last_max": float(np.max(residuals[:, 1:])) if len(residuals) else None,
                    "unconverged_steps": [row["step"] for row in export if not row["converged"]],
                },
            }
        )
        report["gate_pass"] = bool(len(export) == 81 and report["export_steps"]["converged"] == 81)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, output)
    return report


def _safe(entry: dict[str, Any]) -> dict[str, Any]:
    try:
        return audit_case(entry)
    except Exception as error:  # noqa: BLE001
        return {"canonical_id": entry["canonical_id"], "role": entry["role"], "error": repr(error), "traceback": traceback.format_exc(), "gate_pass": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    entries = _case_ids()
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    reports = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for report in pool.map(_safe, entries):
            ex = report.get("export_steps", {})
            print(
                f"{report['canonical_id']}: {report.get('status', 'error')} converged {ex.get('converged')}/{ex.get('found')} "
                f"cap-hits {ex.get('iteration_cap_hits')} cont p95 {ex.get('continuity_last_p95')}",
                flush=True,
            )
            reports.append(report)
    for report in reports:
        if "error" in report:
            continue
        annotate_solver_quality(report)
        output = AUDIT_ROOT / "cases" / f"{report['canonical_id'].replace('/', '__')}.json"
        output.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    parsed = [r for r in reports if r.get("status") == "parsed"]
    annotated = [r for r in reports if "solver_quality" in r]
    tier_counts: dict[str, dict[str, int]] = {}
    for r in annotated:
        tier = r["solver_quality"]["tier"]
        bucket = tier_counts.setdefault(tier, {"all": 0, "train": 0, "test": 0})
        bucket["all"] += 1
        bucket[r["role"]] = bucket.get(r["role"], 0) + 1
    criterion_counts: dict[str, int] = {}
    for r in annotated:
        key = str(r["solver_quality"]["convergence_criterion_type"])
        criterion_counts[key] = criterion_counts.get(key, 0) + 1
    summary = {
        "schema_version": 2,
        "created_at": utc_now(),
        "solver_quality": {
            "definition": TIER_DEFINITION,
            "frozen": "2026-09-05",
            "tier_counts": tier_counts,
            "convergence_criterion_type_counts": criterion_counts,
            "by_case": sorted(
                (
                    {
                        "canonical_id": r["canonical_id"],
                        "role": r["role"],
                        "tier": r["solver_quality"]["tier"],
                        "continuity_last_max": r["solver_quality"]["continuity_last_max"],
                        "flag_converged_export_steps": r["solver_quality"]["flag_converged_export_steps"],
                        "convergence_criterion_type": r["solver_quality"]["convergence_criterion_type"],
                    }
                    for r in annotated
                ),
                key=lambda row: (row["tier"], row["canonical_id"]),
            ),
        },
        "cases": len(reports),
        "parsed": len(parsed),
        "no_transcript": sorted(r["canonical_id"] for r in reports if r.get("status") == "no_transcript"),
        "errors": [{"canonical_id": r["canonical_id"], "error": r["error"]} for r in reports if "error" in r],
        "export_all_81_converged": sorted(r["canonical_id"] for r in parsed if r["gate_pass"]),
        "not_all_converged": sorted(
            (
                {
                    "canonical_id": r["canonical_id"],
                    "role": r["role"],
                    "converged": r["export_steps"]["converged"],
                    "found": r["export_steps"]["found"],
                    "iteration_cap_hits": r["export_steps"]["iteration_cap_hits"],
                    "continuity_last_p95": r["export_steps"]["continuity_last_p95"],
                    "continuity_last_max": r["export_steps"]["continuity_last_max"],
                }
                for r in parsed
                if not r["gate_pass"]
            ),
            key=lambda row: row["converged"],
        ),
        "journal_iteration_caps": sorted({r["journal"]["max_iterations_per_step"] for r in reports if r.get("journal", {}).get("max_iterations_per_step") is not None}),
    }
    (AUDIT_ROOT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"all-81-converged {len(summary['export_all_81_converged'])}/{len(parsed)}; no transcript {len(summary['no_transcript'])}")
    print("solver_quality tiers:", json.dumps(tier_counts, sort_keys=True), "| criterion types:", json.dumps(criterion_counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
