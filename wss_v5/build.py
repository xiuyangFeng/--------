"""CLI: build V5 case bundles (pilot / all / selected) in parallel, aggregate gates, refresh the snapshot manifest.

Examples
--------
    python -m wss_v5.build --pilot --workers 12 --run-name pilot_20260906
    python -m wss_v5.build --all --workers 24 --resume --run-name full_20260906
    python -m wss_v5.build --cases AG/fast/CHEN_SHI_MING --no-hash --domain-subsample 2000
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from wss_pinn.utils import git_state, sha256_file, utc_now

from . import contract as C
from .sources import Registry

PILOT_FIXED = [
    "AG/fast/CHEN_SHI_MING",          # normal AG, comma-delimited
    "AG/fast/HAN_JIAN_JUN",           # whitespace-delimited wall export
    "AAA/ruputer/LI_LAO_PING",        # wide aneurysm sac (largest atlas curvature)
    "ILO/WANG_JIN_MING-0/before",     # ILO stenosis, worst residual (tier C)
    "AAA/unruputer/HAN_JIAN_FU",      # first-frame node reorder + wall pressure gauge offset
    "AG/slow/LIU_FENG",               # one anatomy wall node missing from the export (mask)
    "AAA/ruputer/ZHOU_KE_XUN",        # polyhedral mesh, 09-02 recompute, STL re-extracted from mesh wall
    "AAA/ruputer/GUO_AI_JUN",         # protocol exception (R_total)
    "AAA/unruputer/CHEN_SHU_LIN",     # inlet area ratio 0.888
    "ILO/WANG_SHU_SHENG-0/before",    # formerly node-value volume export (fixed 09-04)
]


def _topology_rows(canonical_id: str) -> int:
    path = C.TOPOLOGY_AUDIT_DIR / f"{canonical_id.replace('/', '__')}.json"
    return int(json.loads(path.read_text(encoding="utf-8"))["anatomy_rows"])


def pilot_cases(registry: Registry) -> list[str]:
    train = [c for c in registry.all_cases() if registry.role(c) == "train" and c not in PILOT_FIXED]
    sizes = {c: _topology_rows(c) for c in train}
    largest = max(sizes, key=sizes.get)
    smallest = min(sizes, key=sizes.get)
    return PILOT_FIXED + [largest, smallest]


def _worker(canonical_id: str, root: str, hash_files: bool, domain_subsample: int, n_generate: int, seed: int) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from .build_case import build_case  # imported in the worker so numpy threading env applies

    registry = Registry.load()
    try:
        return build_case(canonical_id, registry, Path(root), hash_files=hash_files, seed=seed,
                          domain_subsample=domain_subsample, n_generate=n_generate)
    except Exception as exc:  # noqa: BLE001 - defensive: build_case already catches, this guards the pool
        return {"canonical_id": canonical_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(), "gates": {"pass": False, "checks": {}, "waivers": {}}}


def _already_built(canonical_id: str, root: Path) -> bool:
    manifest = C.case_dir(canonical_id, root) / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("schema_version") == C.SCHEMA_VERSION and payload.get("gates", {}).get("pass", False) \
        and Path(payload["bundle"]["path"]).is_file()


def _fmt(value: Any, spec: str = ".3f") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "✓" if value else "✗"
    if isinstance(value, (int,)):
        return str(value)
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def summarize(reports: list[dict[str, Any]], root: Path, run_name: str) -> dict[str, Any]:
    audit_dir = root / "audits" / run_name
    audit_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in reports:
        m = r.get("metrics", {})
        pc = m.get("pointcloud", {})
        it = pc.get("inside_test", {})
        gates = r.get("gates", {})
        failed = [k for k, v in gates.get("checks", {}).items() if not v and k not in gates.get("waivers", {})]
        rows.append({
            "case": r["canonical_id"], "role": r.get("role"), "status": r["status"], "gate_pass": gates.get("pass", False),
            "n_volume": m.get("counts", {}).get("n_volume"), "n_wall": m.get("counts", {}).get("n_wall"), "wall_missing": m.get("counts", {}).get("n_wall_missing"),
            "align_gap": m.get("atlas_alignment", {}).get("median_abs_rel_gap"),
            "normal_med_deg": pc.get("normals_pca_vs_mesh", {}).get("angle_median_deg"), "normal_p95_deg": pc.get("normals_pca_vs_mesh", {}).get("angle_p95_deg"),
            "recall": it.get("anatomy_cells", {}).get("inside_fraction_hybrid"), "ext_false": it.get("extension_cells_near_interface", {}).get("inside_fraction_hybrid"),
            "shell_false": it.get("shell_1_3mm_outside", {}).get("inside_fraction_hybrid"), "gen_leak": pc.get("generated_queries", {}).get("leak_fraction"),
            "wall_dp_pa": m.get("labels", {}).get("wall_pressure_minus_adjacent_cell_median_pa"),
            "mass_bal": m.get("mass_balance_peak_rel_inlet"),
            "protocol": m.get("conditions", {}).get("protocol_id"), "ratio": m.get("conditions", {}).get("inlet_area_ratio"),
            "size_mb": (r.get("bundle", {}).get("bytes") or 0) / 1e6, "seconds": r.get("timings_s", {}).get("total"),
            "failed_checks": failed, "waivers": list(gates.get("waivers", {}).keys()), "error": r.get("error"),
        })
    n_pass = sum(1 for row in rows if row["gate_pass"])
    summary = {"run_name": run_name, "created_at": utc_now(), "schema_version": C.SCHEMA_VERSION, "cases": len(rows), "gate_pass": n_pass,
               "gate_result": "pass" if n_pass == len(rows) and rows else "fail", "gate_thresholds": C.GATES, "rows": rows, "code": git_state()}
    (audit_dir / "gate_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# V5 build gate report — {run_name}", "", f"schema {C.SCHEMA_VERSION} · {utc_now()} · {n_pass}/{len(rows)} cases pass · overall **{summary['gate_result']}**", "",
             "| case | role | status | gate | N_v | N_w | miss | align gap | normal med/p95 ° | recall | ext false | shell false | gen leak | wall Δp Pa | mass bal | protocol | ratio | MB | s | failed / waived |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |"]
    for row in rows:
        lines.append("| " + " | ".join([
            row["case"], str(row["role"]), row["status"], _fmt(row["gate_pass"]), _fmt(row["n_volume"]), _fmt(row["n_wall"]), _fmt(row["wall_missing"]),
            _fmt(row["align_gap"]), f"{_fmt(row['normal_med_deg'], '.1f')}/{_fmt(row['normal_p95_deg'], '.1f')}", _fmt(row["recall"], ".4f"), _fmt(row["ext_false"], ".4f"),
            _fmt(row["shell_false"], ".4f"), _fmt(row["gen_leak"], ".4f"), _fmt(row["wall_dp_pa"], ".2f"), _fmt(row["mass_bal"], ".3f"), str(row["protocol"]),
            _fmt(row["ratio"]), _fmt(row["size_mb"], ".0f"), _fmt(row["seconds"], ".0f"),
            ("; ".join(row["failed_checks"]) + (" / waived: " + ", ".join(row["waivers"]) if row["waivers"] else "")) or (row["error"] or ""),
        ]) + " |")
    (audit_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def refresh_snapshot_manifest(root: Path, registry: Registry) -> dict[str, Any]:
    cases = []
    for manifest in sorted((root / "cases").glob("*/manifest.json")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        cases.append({"canonical_id": payload["canonical_id"], "role": payload["role"], "cohort": payload["cohort"], "patient_group": payload["patient_group"],
                      "validation_fold": payload["validation_fold"], "bundle": payload["bundle"], "counts": payload["counts"], "gate_pass": payload["gates"]["pass"],
                      "created_at": payload["created_at"]})
    snapshot = {
        "schema_version": C.SCHEMA_VERSION, "snapshot": root.name, "updated_at": utc_now(),
        "split": {"path": str(C.SPLIT_PATH), "sha256": sha256_file(C.SPLIT_PATH), "content_sha256": registry.split.get("content_sha256")},
        "atlas_dir": str(C.ATLAS_DIR), "topology_audit_dir": str(C.TOPOLOGY_AUDIT_DIR),
        "expected_cases": len(registry.all_cases()), "built_cases": len(cases), "built_pass": sum(1 for c in cases if c["gate_pass"]),
        "total_bytes": sum(c["bundle"]["bytes"] for c in cases), "cases": cases, "code": git_state(),
        "contract": {"length_units": "mm (Fluent m x 1000)", "frames": len(C.EXPECTED_STEPS), "steps": [C.EXPECTED_STEPS[0], C.EXPECTED_STEPS[-1], 2],
                     "step_dt_s": C.STEP_DT_S, "period_s": C.PERIOD_S, "hdf5_compression": C.HDF5_COMPRESSION, "chunk_points": C.HDF5_CHUNK_POINTS,
                     "dependency_tags": [C.TAG_MODEL_FEATURE, C.TAG_QUADRATURE, C.TAG_AUDIT, C.TAG_OFFLINE_REF, C.TAG_LABEL]},
    }
    (root / "manifest.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")
    return snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--all", action="store_true")
    group.add_argument("--cases", nargs="+")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--root", type=Path, default=C.SNAPSHOT_ROOT)
    parser.add_argument("--run-name", default=time.strftime("run_%Y%m%d_%H%M%S"))
    parser.add_argument("--resume", action="store_true", help="skip cases whose manifest already passes the gate")
    parser.add_argument("--no-hash", action="store_true", help="skip per-frame sha256 (smoke tests only)")
    parser.add_argument("--domain-subsample", type=int, default=C.WINDING_SUBSAMPLE)
    parser.add_argument("--n-generate", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    registry = Registry.load()
    if args.pilot:
        cases = pilot_cases(registry)
    elif args.all:
        cases = registry.all_cases()
    else:
        cases = list(args.cases)
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cases if not (args.resume and _already_built(c, root))]
    print(f"[v5-build] run={args.run_name} root={root} cases={len(cases)} todo={len(todo)} workers={args.workers}", flush=True)
    reports: list[dict[str, Any]] = []
    for c in cases:
        if c not in todo:
            reports.append(json.loads((C.case_dir(c, root) / "report.json").read_text(encoding="utf-8")))
    started = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, c, str(root), not args.no_hash, args.domain_subsample, args.n_generate, args.seed): c for c in todo}
            for future in as_completed(futures):
                report = future.result()
                reports.append(report)
                gate = report.get("gates", {}).get("pass")
                print(f"[v5-build] {report['canonical_id']}: {report['status']} gate={gate} "
                      f"t={report.get('timings_s', {}).get('total', 0):.0f}s {('ERROR ' + report['error']) if report.get('error') else ''}", flush=True)
    order = {c: i for i, c in enumerate(cases)}
    reports.sort(key=lambda r: order.get(r["canonical_id"], 1 << 30))
    summary = summarize(reports, root, args.run_name)
    refresh_snapshot_manifest(root, registry)
    print(f"[v5-build] done in {time.time() - started:.0f}s: {summary['gate_pass']}/{summary['cases']} pass -> {summary['gate_result']}", flush=True)
    return 0 if summary["gate_result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
