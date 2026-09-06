"""Build the unique-segment centerline feature atlas for all 173 cases.

Outputs (never touching ``data_new`` or historical data roots):

``outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903/atlas/cases/<cid>/atlas.npz``
``outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903/atlas/cases/<cid>/summary.json``
``outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903/atlas/atlas_summary.json``

The atlas is expressed in *true millimetres* of the Fluent frame, i.e. the
legacy V2 "mm" products are rescaled by ``1000 / fluent_to_mm_unit_factor``
(see the unit audit: the authoritative STLs are exactly the Fluent wall x1000
for 130/173 cases and within 0.5% for another 40; the frozen per-case factor
810-1088 is a legacy bbox inference and is not a physical unit).
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from wss_pinn.utils import ROOT, sha256_file, utc_now
from wss_pinn.v4.centerline_atlas import (
    ATLAS_SAMPLE_COLUMNS,
    ENDPOINT_HALF_WIDTH_MM,
    JUNCTION_HALF_WIDTH_MM,
    build_feature_atlas,
    detect_kinks,
    load_v2_graph,
    save_atlas,
)

SOURCE_MANIFEST = ROOT / "data_wss_pinn/volume_uvwp_bc_rcr_v4_train138_test35/manifest.json"
CENTERLINE_ROOT = ROOT / "outputs/centerline_v2_full_173_20260828/cases"
PREP_ROOT = ROOT / "outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903"
ATLAS_ROOT = PREP_ROOT / "atlas"
CURVATURE_P99_TARGET = 0.5
CURVATURE_HARD_MAX = 10.0
WHITELIST_REVIEW_THRESHOLD = 0.5


def _case_ids() -> list[dict[str, Any]]:
    payload = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    return [{"canonical_id": row["canonical_id"], "role": row["role"]} for row in payload["cases"]]


def build_case(entry: dict[str, Any]) -> dict[str, Any]:
    canonical_id = entry["canonical_id"]
    output_dir = ATLAS_ROOT / "cases" / canonical_id
    if "source_root" in entry:
        source_root = Path(entry["source_root"])
    else:
        # keep a case on its previously registered source (e.g. the 3 mesh-wall
        # re-extractions) when the cohort is rebuilt without --source-root
        source_root = Path(CENTERLINE_ROOT)
        prior_summary = output_dir / "summary.json"
        if prior_summary.is_file():
            prior_root = json.loads(prior_summary.read_text(encoding="utf-8")).get("source_root")
            if prior_root:
                source_root = Path(prior_root)
    case_dir = source_root / canonical_id
    selection = json.loads((case_dir / "surface_selection.json").read_text(encoding="utf-8"))
    factor = float(selection["fluent_to_mm_unit_factor"])
    # legacy V2 products live in Fluent m x factor; mesh-wall products are already true mm
    graph = load_v2_graph(case_dir, unit_rescale=1000.0 / factor)
    atlas = build_feature_atlas(canonical_id, graph)
    summary = save_atlas(atlas, output_dir / "atlas.npz")
    kinks = detect_kinks(atlas)
    rows = atlas.unique_rows()
    curvature = rows[:, ATLAS_SAMPLE_COLUMNS.index("curvature_per_mm")]
    d_end = rows[:, ATLAS_SAMPLE_COLUMNS.index("dist_to_endpoint_mm")]
    d_junction = rows[:, ATLAS_SAMPLE_COLUMNS.index("dist_to_junction_mm")]
    peak = int(np.argmax(curvature))
    location = (
        "endpoint"
        if d_end[peak] <= ENDPOINT_HALF_WIDTH_MM
        else "junction" if d_junction[peak] <= JUNCTION_HALF_WIDTH_MM else "interior"
    )
    summary.update(
        {
            "role": entry["role"],
            "created_at": utc_now(),
            "source_root": str(source_root),
            "surface_source": selection.get("selection_source", ""),
            "inputs": {
                "graph_vtp": {"path": str(case_dir / "centerline_graph_mm.vtp"), "sha256": sha256_file(case_dir / "centerline_graph_mm.vtp")},
                "result": {"path": str(case_dir / "result.json"), "sha256": sha256_file(case_dir / "result.json")},
                "surface_selection": {"path": str(case_dir / "surface_selection.json"), "sha256": sha256_file(case_dir / "surface_selection.json")},
            },
            "outputs": {"atlas_npz": {"path": str(output_dir / "atlas.npz"), "sha256": sha256_file(output_dir / "atlas.npz")}},
            "peak_location_class": location,
            "peak_dist_to_endpoint_mm": float(d_end[peak]),
            "peak_dist_to_junction_mm": float(d_junction[peak]),
            "curvature_gt_0p5_samples": int(np.sum(curvature > WHITELIST_REVIEW_THRESHOLD)),
            "needs_whitelist_review": bool(np.max(curvature) > WHITELIST_REVIEW_THRESHOLD),
            "kinks": kinks,
            "gates": {
                "finite": bool(np.isfinite(rows).all()),
                "seven_segments": len(atlas.segments) == 7,
                "curvature_hard_max_lt_10": float(np.max(curvature)) < CURVATURE_HARD_MAX,
                "radius_positive": bool(np.all(rows[:, ATLAS_SAMPLE_COLUMNS.index("radius_mm")] > 0.0)),
                "no_interior_kinks": kinks["interior"] == 0,
            },
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


def _safe(entry: dict[str, Any]) -> dict[str, Any]:
    try:
        return build_case(entry)
    except Exception as error:  # noqa: BLE001
        return {"canonical_id": entry["canonical_id"], "role": entry["role"], "error": repr(error), "traceback": traceback.format_exc()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Centerline V2-layout root to read the selected cases from (default: frozen 2026-08-28 root)",
    )
    parser.add_argument("--summary-only", action="store_true", help="only rebuild the cohort summary from per-case summaries on disk")
    args = parser.parse_args()
    entries = _case_ids()
    if args.cases:
        wanted = set(args.cases)
        entries = [row for row in entries if row["canonical_id"] in wanted]
    if args.source_root is not None:
        for row in entries:
            row["source_root"] = str(args.source_root / "cases")
    ATLAS_ROOT.mkdir(parents=True, exist_ok=True)
    reports = []
    if not args.summary_only:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for report in pool.map(_safe, entries):
                if "error" in report:
                    print(f"{report['canonical_id']}: ERROR {report['error']}", flush=True)
                else:
                    c = report["curvature_per_mm"]
                    kk = report["kinks"]
                    print(
                        f"{report['canonical_id']}: samples={report['unique_samples']} "
                        f"k p99={c['p99']:.3f} max={c['max']:.3f} ({report['peak_location_class']}) "
                        f"kR max={report['curvature_times_radius']['max']:.2f} "
                        f"win={report['sg_window_samples']['min']}..{report['sg_window_samples']['max']} "
                        f"kinks int/end={kk['interior']}/{kk['endpoint']} "
                        f"res max={report['fit_residual_mm']['max']:.3f}",
                        flush=True,
                    )
                reports.append(report)
    # the cohort summary always reflects every per-case summary on disk (mixed sources allowed)
    all_ids = [row["canonical_id"] for row in _case_ids()]
    role_of = {row["canonical_id"]: row["role"] for row in _case_ids()}
    disk_reports = []
    for canonical_id in all_ids:
        path = ATLAS_ROOT / "cases" / canonical_id / "summary.json"
        if path.is_file():
            disk_reports.append(json.loads(path.read_text(encoding="utf-8")))
    errors = [r for r in reports if "error" in r]
    reports = disk_reports + errors
    good = [r for r in reports if "error" not in r]
    all_curv = []
    all_k_r = []
    all_windows = []
    for report in good:
        table = np.load(report["outputs"]["atlas_npz"]["path"])["table"]
        dup = table[:, ATLAS_SAMPLE_COLUMNS.index("is_child_duplicate")] < 0.5
        all_curv.append(table[dup, ATLAS_SAMPLE_COLUMNS.index("curvature_per_mm")])
        all_k_r.append(table[dup, ATLAS_SAMPLE_COLUMNS.index("curvature_times_radius")])
        all_windows.append(table[dup, ATLAS_SAMPLE_COLUMNS.index("sg_window_samples")])
    cohort = np.concatenate(all_curv) if all_curv else np.zeros(0)
    cohort_k_r = np.concatenate(all_k_r) if all_k_r else np.zeros(0)
    cohort_windows = np.concatenate(all_windows) if all_windows else np.zeros(0)
    train_curv = np.concatenate(
        [
            np.load(r["outputs"]["atlas_npz"]["path"])["table"][:, ATLAS_SAMPLE_COLUMNS.index("curvature_per_mm")]
            for r in good
            if r["role"] == "train"
        ]
    ) if good else np.zeros(0)
    summary = {
        "schema_version": 2,
        "created_at": utc_now(),
        "cases": len(all_ids),
        "built": len(good),
        "sg_window_policy": sorted({r["provenance"].get("sg_window_policy", "fixed SG11") for r in good}),
        "source_roots": sorted({r.get("source_root", "") for r in good}),
        "mesh_wall_cases": sorted(r["canonical_id"] for r in good if r.get("surface_source") == "fluent_case_anatomy_wall"),
        "errors": [{"canonical_id": r["canonical_id"], "error": r["error"]} for r in reports if "error" in r],
        "unit_contract": "true mm = Fluent m x 1000; legacy V2 frame rescaled by 1000/fluent_to_mm_unit_factor",
        "cohort_curvature_per_mm": {
            "samples": int(len(cohort)),
            "p50": float(np.percentile(cohort, 50)) if len(cohort) else None,
            "p99": float(np.percentile(cohort, 99)) if len(cohort) else None,
            "max": float(np.max(cohort)) if len(cohort) else None,
            "p99_target_lt_0p5": bool(np.percentile(cohort, 99) < CURVATURE_P99_TARGET) if len(cohort) else None,
            "hard_max_lt_10": bool(np.max(cohort) < CURVATURE_HARD_MAX) if len(cohort) else None,
            "train138_p99": float(np.percentile(train_curv, 99)) if len(train_curv) else None,
            "train138_max": float(np.max(train_curv)) if len(train_curv) else None,
        },
        "cohort_curvature_times_radius": {
            "p99": float(np.percentile(cohort_k_r, 99)) if len(cohort_k_r) else None,
            "max": float(np.max(cohort_k_r)) if len(cohort_k_r) else None,
            "fraction_gt_1": float(np.mean(cohort_k_r > 1.0)) if len(cohort_k_r) else None,
        },
        "sg_window_samples": {
            "min": int(np.min(cohort_windows)) if len(cohort_windows) else None,
            "p50": float(np.median(cohort_windows)) if len(cohort_windows) else None,
            "p99": float(np.percentile(cohort_windows, 99)) if len(cohort_windows) else None,
            "max": int(np.max(cohort_windows)) if len(cohort_windows) else None,
            "fraction_gt_11": float(np.mean(cohort_windows > 11)) if len(cohort_windows) else None,
        },
        "kink_gate": {
            "rule": good[0]["kinks"]["rule"] if good else None,
            "interior_total": int(sum(r["kinks"]["interior"] for r in good)),
            "junction_total": int(sum(r["kinks"]["junction"] for r in good)),
            "endpoint_total": int(sum(r["kinks"]["endpoint"] for r in good)),
            "cases_with_interior_kinks": sorted(r["canonical_id"] for r in good if r["kinks"]["interior"]),
            "cases_with_endpoint_kinks": [
                {"canonical_id": r["canonical_id"], "samples": r["kinks"]["endpoint"], "curvature_max_per_mm": r["curvature_per_mm"]["max"]}
                for r in sorted(good, key=lambda row: row["canonical_id"])
                if r["kinks"]["endpoint"]
            ],
        },
        "peak_location_classes": {
            cls: sorted(r["canonical_id"] for r in good if r["peak_location_class"] == cls)
            for cls in ("endpoint", "junction", "interior")
        },
        "whitelist_review_queue": sorted(
            (
                {
                    "canonical_id": r["canonical_id"],
                    "role": r["role"],
                    "curvature_max_per_mm": r["curvature_per_mm"]["max"],
                    "peak_location_class": r["peak_location_class"],
                    "peak_dist_to_endpoint_mm": r["peak_dist_to_endpoint_mm"],
                    "peak_dist_to_junction_mm": r["peak_dist_to_junction_mm"],
                    "samples_gt_0p5": r["curvature_gt_0p5_samples"],
                    "fit_residual_max_mm": r["fit_residual_mm"]["max"],
                }
                for r in good
                if r["needs_whitelist_review"]
            ),
            key=lambda row: -row["curvature_max_per_mm"],
        ),
        "gate_failures": [
            {"canonical_id": r["canonical_id"], "failed": [k for k, v in r["gates"].items() if not v]}
            for r in good
            if not all(r["gates"].values())
        ],
        "cases_report": {
            r["canonical_id"]: {
                "curvature_p99": r["curvature_per_mm"]["p99"],
                "curvature_max": r["curvature_per_mm"]["max"],
                "curvature_times_radius_max": r["curvature_times_radius"]["max"],
                "sg_window_max": r["sg_window_samples"]["max"],
                "kinks_interior": r["kinks"]["interior"],
                "kinks_endpoint": r["kinks"]["endpoint"],
                "peak_location_class": r["peak_location_class"],
                "fit_residual_max_mm": r["fit_residual_mm"]["max"],
                "unique_samples": r["unique_samples"],
                "max_s_from_root_mm": r["max_s_from_root_mm"],
                "unit_rescale": r["provenance"]["unit_rescale"],
                "surface_source": r.get("surface_source", ""),
                "isolated_graph_points_ignored": r["provenance"].get("isolated_graph_points_ignored", 0),
            }
            for r in good
        },
    }
    (ATLAS_ROOT / "atlas_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"built {len(good)}/{len(reports)}; cohort curvature p99={summary['cohort_curvature_per_mm']['p99']} max={summary['cohort_curvature_per_mm']['max']}")
    return 0 if not errors and len(good) == len(all_ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())
