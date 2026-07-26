#!/usr/bin/env python3
"""Audit the single-seed S3-GEOPE + SA3 coarse-attention screen."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from training_wss_min.tools.analyze_d2_k64_ilo_structure_results import AGG_KEYS, sha256
from training_wss_min.tools.analyze_s3_rootcause_results import PARENT, build_record, comparison


REPO = Path(__file__).resolve().parents[2]
PREFLIGHT = REPO / "training_wss_min/preflight"
MANIFEST = PREFLIGHT / "s3_sa3_coarse_attention_single_prepared.json"
OUT_JSON = PREFLIGHT / "s3_sa3_coarse_attention_single_results_analysis.json"
OUT_SUMMARY = PREFLIGHT / "s3_sa3_coarse_attention_single_results_summary.csv"
OUT_PAIRS = PREFLIGHT / "s3_sa3_coarse_attention_single_paired_case_stats.csv"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = manifest.get("configs", [])
    if manifest.get("status") != "prepared" or len(rows) != 1:
        raise RuntimeError("expected the exact single-attention manifest")
    row = rows[0]
    parent = build_record("s3_geope_s1234", PARENT[1234], 1234, "s3_geope_parent")
    child = build_record(row["experiment_id"], Path(row["run_dir"]), 1234, row["variant"], Path(row["config"]), row["sha256"])
    if parent["integrity"] != "passed" or child["integrity"] != "passed":
        raise RuntimeError(f"integrity failures: parent={parent['integrity_errors']} child={child['integrity_errors']}")
    paired = comparison("sa3_coarse_attention_vs_s3_geope_s1234", parent, child, 20260725)
    delta = paired["aggregate_treatment_minus_control"]
    decision = "Weak-Go signal" if delta["physical_r2_casebalanced"] >= 0.005 and delta["high_wss_r2"] >= 0 else "No-Go"
    payload = {
        "schema_version": 1, "analysis_date": "2026-07-25",
        "scope": "single seed=1234 S3-GEOPE + SA3 32-center, 4-head coarse-attention screen; historical mixed test36",
        "manifest": str(MANIFEST.resolve()), "manifest_sha256": sha256(MANIFEST),
        "records": {parent["experiment_id"]: parent, child["experiment_id"]: child},
        "paired_comparison": paired, "decision": decision,
        "interpretation_limits": ["Single seed only; not a multi-seed or independent confirmation.", "ckpt_best remains selected only by train loss."],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        fields = ["experiment_id", "variant", "best_epoch", *AGG_KEYS, "AG_r2", "AAA_r2", "ILO_r2", "integrity"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in (parent, child):
            writer.writerow({"experiment_id": record["experiment_id"], "variant": record["variant"], "best_epoch": record["history"].get("best_epoch"), **{key: record["best"][key] for key in AGG_KEYS}, "AG_r2": record["domains"]["AG"], "AAA_r2": record["domains"]["AAA"], "ILO_r2": record["domains"]["ILO"], "integrity": record["integrity"]})
    with OUT_PAIRS.open("w", newline="", encoding="utf-8") as handle:
        cases = paired["paired_case_stats"]["case_r2"]
        writer = csv.DictWriter(handle, fieldnames=["comparison", "delta_r2cb", "delta_mae", "delta_rmse", "delta_high_wss_r2", "delta_top10_iou", "case_r2_mean_delta", "case_r2_ci95_low", "case_r2_ci95_high", "case_r2_wins", "case_r2_losses"])
        writer.writeheader()
        writer.writerow({"comparison": paired["comparison"], "delta_r2cb": delta["physical_r2_casebalanced"], "delta_mae": delta["physical_mae"], "delta_rmse": delta["physical_rmse"], "delta_high_wss_r2": delta["high_wss_r2"], "delta_top10_iou": delta["physical_top10_iou"], "case_r2_mean_delta": cases["mean_delta"], "case_r2_ci95_low": cases["ci95_low"], "case_r2_ci95_high": cases["ci95_high"], "case_r2_wins": cases["wins"], "case_r2_losses": cases["losses"]})
    print(json.dumps({"status": "passed", "decision": decision, "json": str(OUT_JSON)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
