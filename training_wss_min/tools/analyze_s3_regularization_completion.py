#!/usr/bin/env python3
"""Analyze all seed1234 S3 regularization rates and pairwise crosses."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from training_wss_min.tools.analyze_d2_k64_ilo_structure_results import AGG_KEYS, sha256
from training_wss_min.tools.analyze_s3_rootcause_results import PARENT, build_record, comparison

ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = ROOT / "training_wss_min/preflight"
MANIFEST = PREFLIGHT / "s3_regularization_completion_20260725_prepared.json"
OUT_JSON = PREFLIGHT / "s3_regularization_completion_20260725_results_analysis.json"
OUT_SUMMARY = PREFLIGHT / "s3_regularization_completion_20260725_results_summary.csv"
OUT_PAIRS = PREFLIGHT / "s3_regularization_completion_20260725_paired_case_stats.csv"
OLD = {
    "head_dropout010_s1234": ROOT / "training_wss_min/runs/pointnetpp_s3_regularization/outputs/head_dropout01_s1234",
    "droppath005_s1234": ROOT / "training_wss_min/runs/pointnetpp_s3_regularization/outputs/droppath005_s1234",
    "droppath010_s1234": ROOT / "training_wss_min/runs/pointnetpp_s3_regularization/outputs/droppath010_s1234",
    "neighbordrop005_s1234": ROOT / "training_wss_min/runs/pointnetpp_s3_regularization/outputs/neighbordrop005_s1234",
}
CURVES = {
    "head_dropout": ("head_dropout005_s1234", "head_dropout010_s1234", "head_dropout015_s1234", "head_dropout020_s1234"),
    "drop_path": ("droppath005_s1234", "droppath010_s1234", "droppath015_s1234", "droppath020_s1234"),
    "neighbor_drop": ("neighbordrop005_s1234", "neighbordrop010_s1234", "neighbordrop015_s1234", "neighbordrop020_s1234"),
}

def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = manifest["configs"]
    if manifest.get("status") != "prepared" or len(rows) != 20:
        raise RuntimeError("expected exact 20-arm completion manifest")
    parent = build_record("s3_geope_s1234", PARENT[1234], 1234, "s3_parent")
    records = {parent["experiment_id"]: parent}
    for eid, run in OLD.items():
        records[eid] = build_record(eid, run, 1234, eid.rsplit("_s", 1)[0])
    for row in rows:
        records[row["experiment_id"]] = build_record(row["experiment_id"], Path(row["run_dir"]), 1234, row["variant"], Path(row["config"]), row["sha256"])
    failed = {eid: r["integrity_errors"] for eid, r in records.items() if r["integrity"] != "passed"}
    if failed: raise RuntimeError(f"integrity failures: {failed}")
    comparisons = []
    for eid, record in records.items():
        if eid != parent["experiment_id"]:
            comparisons.append(comparison(f"{eid}_vs_s3", parent, record, 20260725 + len(comparisons) * 100))
    by_treatment = {item["treatment"]: item for item in comparisons}
    curve_rows = {}
    for family, eids in CURVES.items():
        curve_rows[family] = []
        for eid in eids:
            delta = by_treatment[eid]["aggregate_treatment_minus_control"]
            rate = int(eid.split("_")[-2][-3:]) / 100
            curve_rows[family].append({"experiment_id": eid, "rate": rate, "r2cb": records[eid]["best"]["physical_r2_casebalanced"], "delta_r2cb": delta["physical_r2_casebalanced"], "delta_mae": delta["physical_mae"], "delta_high_wss_r2": delta["high_wss_r2"]})
    pairs = []
    for row in rows:
        fields = row["regularizers"]
        if len(fields) != 2: continue
        eid = row["experiment_id"]
        pair_delta = by_treatment[eid]["aggregate_treatment_minus_control"]
        terms = []
        for field, rate in fields.items():
            key = {"model.dropout": "head_dropout", "model.drop_path_rate": "droppath", "model.neighbor_drop_rate": "neighbordrop"}[field]
            rate_tag = f"{round(float(rate) * 100):03d}"
            terms.append(f"{key}{rate_tag}_s1234")
        additive = sum(by_treatment[term]["aggregate_treatment_minus_control"]["physical_r2_casebalanced"] for term in terms)
        pairs.append({"experiment_id": eid, "regularizers": fields, "delta_r2cb": pair_delta["physical_r2_casebalanced"], "delta_mae": pair_delta["physical_mae"], "delta_high_wss_r2": pair_delta["high_wss_r2"], "additive_single_delta_r2cb": additive, "interaction_r2cb": pair_delta["physical_r2_casebalanced"] - additive})
    ranking = sorted((r for eid, r in records.items() if eid != parent["experiment_id"]), key=lambda r: r["best"]["physical_r2_casebalanced"], reverse=True)
    payload = {"schema_version": 1, "analysis_date": "2026-07-25", "scope": "S3-GEOPE seed1234 regularization completion: full single-rate curves and pairwise 0.05/0.10 crosses; historical mixed test36", "manifest": str(MANIFEST.resolve()), "manifest_sha256": sha256(MANIFEST), "records": records, "paired_comparisons": comparisons, "single_rate_curves": curve_rows, "pairwise_interactions": pairs, "rank_by_r2cb": [{"rank": i, "experiment_id": r["experiment_id"], "r2cb": r["best"]["physical_r2_casebalanced"], "delta_r2cb": by_treatment[r["experiment_id"]]["aggregate_treatment_minus_control"]["physical_r2_casebalanced"]} for i, r in enumerate(ranking, 1)], "limits": ["All comparisons are seed1234 only.", "test36 is a historical engineering screening set; rankings are not independent confirmation."]}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as h:
        fields=["experiment_id","variant","best_epoch",*AGG_KEYS,"AG_r2","AAA_r2","ILO_r2","integrity"]
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
        for r in records.values(): w.writerow({"experiment_id":r["experiment_id"],"variant":r["variant"],"best_epoch":r["history"].get("best_epoch"),**{k:r["best"][k] for k in AGG_KEYS},"AG_r2":r["domains"]["AG"],"AAA_r2":r["domains"]["AAA"],"ILO_r2":r["domains"]["ILO"],"integrity":r["integrity"]})
    with OUT_PAIRS.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=["comparison","delta_r2cb","delta_mae","delta_rmse","delta_high_wss_r2","delta_top10_iou","case_r2_mean_delta","case_r2_ci95_low","case_r2_ci95_high","case_r2_wins","case_r2_losses"]);w.writeheader()
        for r in comparisons:
            d,c=r["aggregate_treatment_minus_control"],r["paired_case_stats"]["case_r2"]
            w.writerow({"comparison":r["comparison"],"delta_r2cb":d["physical_r2_casebalanced"],"delta_mae":d["physical_mae"],"delta_rmse":d["physical_rmse"],"delta_high_wss_r2":d["high_wss_r2"],"delta_top10_iou":d["physical_top10_iou"],"case_r2_mean_delta":c["mean_delta"],"case_r2_ci95_low":c["ci95_low"],"case_r2_ci95_high":c["ci95_high"],"case_r2_wins":c["wins"],"case_r2_losses":c["losses"]})
    print(json.dumps({"status":"passed","records":len(records),"best":ranking[0]["experiment_id"],"json":str(OUT_JSON)},ensure_ascii=False))
if __name__=="__main__": main()
