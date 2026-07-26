#!/usr/bin/env python3
"""Audit and summarize the three strict xyz-only controls."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from training_wss_min.tools.analyze_d2_k64_ilo_structure_results import AGG_KEYS, sha256
from training_wss_min.tools.analyze_s3_rootcause_results import build_record, comparison


REPO = Path(__file__).resolve().parents[2]
PREFLIGHT = REPO / "training_wss_min/preflight"
MANIFEST = PREFLIGHT / "xyz_input_ablation_20260724_prepared.json"
OUT_JSON = PREFLIGHT / "xyz_input_ablation_20260724_results_analysis.json"
OUT_SUMMARY = PREFLIGHT / "xyz_input_ablation_20260724_results_summary.csv"
OUT_PAIRS = PREFLIGHT / "xyz_input_ablation_20260724_paired_case_stats.csv"
PARENTS = {
    "d2_c125_k64_xyz": (
        "d2_c125_k64_xyzgeom",
        REPO / "training_wss_min/runs/pointnetpp_sa1_scale_single_seed/outputs/d2_rand5000_c125_k64",
    ),
    "m1_d2k64_mixed138_test36_xyz": (
        "m1_d2k64_mixed138_test36_xyzgeom",
        REPO / "training_wss_min/runs/pointnetpp_d2_k64_ilo_structure/outputs/m1_d2k64_mixed138_test36",
    ),
    "s2_d2k64_pnxr_mixed_xyz": (
        "s2_d2k64_pnxr_mixed_xyzgeom",
        REPO / "training_wss_min/runs/pointnetpp_d2_k64_ilo_structure/outputs/s2_d2k64_pnxr_mixed",
    ),
}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = manifest.get("configs", [])
    if manifest.get("status") != "prepared" or len(rows) != 3:
        raise RuntimeError("expected the exact three-arm xyz-only manifest")
    records: dict[str, dict] = {}
    comparisons = []
    for index, row in enumerate(rows):
        experiment_id = row["experiment_id"]
        parent_id, parent_run = PARENTS[experiment_id]
        records[parent_id] = build_record(parent_id, parent_run, 1234, "xyzgeom_parent")
        records[experiment_id] = build_record(
            experiment_id, Path(row["run_dir"]), 1234, "xyz_only",
            Path(row["config"]), row["sha256"],
        )
        comparisons.append(comparison(
            f"{experiment_id}_vs_xyzgeom_parent", records[parent_id],
            records[experiment_id], 20260725 + index * 100,
        ))
    if any(record["integrity"] != "passed" for record in records.values()):
        failed = {key: value["integrity_errors"] for key, value in records.items() if value["integrity"] != "passed"}
        raise RuntimeError(f"integrity failures: {failed}")
    mixed_xyz = comparison(
        "s2_pnxr_xyz_vs_m1_xyz", records["m1_d2k64_mixed138_test36_xyz"],
        records["s2_d2k64_pnxr_mixed_xyz"], 20261025,
    )
    payload = {
        "schema_version": 1,
        "analysis_date": "2026-07-25",
        "scope": "three frozen-parent xyz-only controls; historical test27/test36 engineering screens",
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha256(MANIFEST),
        "records": records,
        "paired_comparisons": comparisons,
        "secondary_structure_comparison": mixed_xyz,
        "interpretation": (
            "All primary contrasts are xyz-only minus their exact xyz+geom parent; "
            "the S2_xyz vs M1_xyz contrast isolates the residual core under xyz-only input."
        ),
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        fields = ["experiment_id", "variant", "best_epoch", *AGG_KEYS, "AG_r2", "AAA_r2", "ILO_r2", "integrity"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records.values():
            writer.writerow({
                "experiment_id": record["experiment_id"], "variant": record["variant"],
                "best_epoch": record["history"].get("best_epoch"),
                **{key: record["best"][key] for key in AGG_KEYS},
                "AG_r2": record["domains"]["AG"], "AAA_r2": record["domains"]["AAA"],
                "ILO_r2": record["domains"]["ILO"], "integrity": record["integrity"],
            })
    with OUT_PAIRS.open("w", newline="", encoding="utf-8") as handle:
        fields = ["comparison", "control", "treatment", "delta_r2cb", "delta_mae", "delta_rmse", "delta_high_wss_r2", "delta_top10_iou", "delta_p99_ratio", "case_r2_mean_delta", "case_r2_ci95_low", "case_r2_ci95_high", "case_r2_wins", "case_r2_losses"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in [*comparisons, mixed_xyz]:
            delta, cases = result["aggregate_treatment_minus_control"], result["paired_case_stats"]["case_r2"]
            writer.writerow({
                "comparison": result["comparison"], "control": result["control"], "treatment": result["treatment"],
                "delta_r2cb": delta["physical_r2_casebalanced"], "delta_mae": delta["physical_mae"],
                "delta_rmse": delta["physical_rmse"], "delta_high_wss_r2": delta["high_wss_r2"],
                "delta_top10_iou": delta["physical_top10_iou"], "delta_p99_ratio": delta["physical_p99_amplitude_ratio"],
                "case_r2_mean_delta": cases["mean_delta"], "case_r2_ci95_low": cases["ci95_low"],
                "case_r2_ci95_high": cases["ci95_high"], "case_r2_wins": cases["wins"], "case_r2_losses": cases["losses"],
            })
    print(json.dumps({"status": "passed", "records": len(records), "json": str(OUT_JSON)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
