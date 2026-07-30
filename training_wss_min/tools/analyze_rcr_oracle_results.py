#!/usr/bin/env python3
"""Audit and summarize the completed RCR Oracle O0/O1a/O2 matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from training_wss_min.tools.analyze_d2_k64_ilo_structure_results import (
    AGG_KEYS,
    sha256,
)
from training_wss_min.tools.analyze_s3_rootcause_results import (
    build_record,
    comparison,
)


ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = ROOT / "training_wss_min/preflight"
MANIFEST = PREFLIGHT / "rcr_oracle_matrix_20260728_prepared.json"
SUBMISSION = PREFLIGHT / "rcr_oracle_matrix_20260728_submission.json"
OUT_JSON = PREFLIGHT / "rcr_oracle_matrix_20260728_results_analysis.json"
OUT_SUMMARY = PREFLIGHT / "rcr_oracle_matrix_20260728_results_summary.csv"
OUT_PAIRS = PREFLIGHT / "rcr_oracle_matrix_20260728_paired_case_stats.csv"

ORDER = (
    "o0_geometry_s1234",
    "o1a_true_rcr_s1234",
    "o2_shuffled_rcr_s1234",
)
LABELS = {
    "o0_geometry_s1234": "O0 geometry",
    "o1a_true_rcr_s1234": "O1a true RCR",
    "o2_shuffled_rcr_s1234": "O2 shuffled RCR",
}
DOMAINS = ("AG", "AAA", "ILO")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extra_metrics(record: dict) -> dict[str, float]:
    raw = read_json(
        Path(record["run_dir"]) / "eval/ckpt_best/metrics.json"
    )["test"]
    high = raw["regional_field"]["high_wss"]
    return {
        "high_wss_nrmse_range": high["nrmse_range"],
        "high_wss_mae": high["mae"],
        "physical_top10_ratio": raw["calibration"]["top10_pred_true_ratio"],
        "physical_p99_ratio": raw["calibration"]["p99_pred_true_ratio"],
    }


def main() -> None:
    manifest = read_json(MANIFEST)
    submission = read_json(SUBMISSION)
    rows = manifest["configs"]
    if (
        manifest.get("status") != "prepared"
        or [row["experiment_id"] for row in rows] != list(ORDER)
        or manifest["protocol"]
        != "S3 PointNeXt-R + LocalGeoPE; mixed 138/0/36; seed1234; legacy_vertex"
    ):
        raise RuntimeError("expected exact prepared RCR Oracle matrix")

    records: dict[str, dict] = {}
    for row in rows:
        experiment_id = row["experiment_id"]
        record = build_record(
            experiment_id,
            Path(row["run_dir"]),
            1234,
            experiment_id.split("_s1234")[0],
            Path(row["config"]),
            row["config_sha256"],
        )
        record["label"] = LABELS[experiment_id]
        record["input_dim"] = row["input_dim"]
        record["extra"] = extra_metrics(record)
        records[experiment_id] = record

    failed = {
        experiment_id: record["integrity_errors"]
        for experiment_id, record in records.items()
        if record["integrity"] != "passed"
    }
    if failed:
        raise RuntimeError(f"integrity failures: {failed}")

    o0 = records["o0_geometry_s1234"]
    comparisons = [
        comparison(
            "o1a_true_rcr_vs_o0_geometry",
            o0,
            records["o1a_true_rcr_s1234"],
            20260728,
        ),
        comparison(
            "o2_shuffled_rcr_vs_o0_geometry",
            o0,
            records["o2_shuffled_rcr_s1234"],
            20260828,
        ),
        comparison(
            "o1a_true_rcr_vs_o2_shuffled_rcr",
            records["o2_shuffled_rcr_s1234"],
            records["o1a_true_rcr_s1234"],
            20260928,
        ),
    ]
    for item in comparisons:
        control = records[item["control"]]
        treatment = records[item["treatment"]]
        item["extra_treatment_minus_control"] = {
            key: treatment["extra"][key] - control["extra"][key]
            for key in treatment["extra"]
        }

    o1a_vs_o0 = comparisons[0]
    o2_vs_o0 = comparisons[1]
    o1a_delta = o1a_vs_o0["aggregate_treatment_minus_control"]
    o2_delta = o2_vs_o0["aggregate_treatment_minus_control"]
    o1a_case = o1a_vs_o0["paired_case_stats"]["case_r2"]
    decision_pass = (
        o1a_delta["physical_r2_casebalanced"] > 0.03
        and o1a_delta["physical_mae"] < 0
        and o1a_delta["physical_top10_iou"] > 0
        and o1a_case["ci95_low"] > 0
        and abs(o2_delta["physical_r2_casebalanced"]) < 0.01
    )
    decision = (
        "true_rcr_signal_confirmed_for_followup"
        if decision_pass
        else "rcr_signal_not_confirmed"
    )

    payload = {
        "schema_version": 1,
        "analysis_date": "2026-07-28",
        "scope": (
            "S3 PointNeXt-R + LocalGeoPE RCR information-limit Oracle; "
            "O0 geometry, O1a true RCR, O2 within-split/stratum shuffled RCR; "
            "historical mixed test36"
        ),
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha256(MANIFEST),
        "submission": submission,
        "records": records,
        "paired_comparisons": comparisons,
        "decision": decision,
        "decision_basis": [
            "O1a improves physical R2_cb by more than 0.03.",
            "O1a lowers physical MAE and improves top10 IoU.",
            "The paired case-level R2 mean-difference bootstrap interval excludes zero.",
            "O2 shuffled RCR remains within 0.01 R2_cb of O0.",
        ],
        "interpretation_limits": [
            "All arms use seed1234 only; this is not a multi-seed confirmation.",
            "test36 is a repeatedly reused engineering development set, not an independent final test set.",
            "ckpt_best is selected by train loss; ckpt_last is sensitivity-only.",
            "The result supports RCR information content, but not immediate deployment because true RCR is not a routinely available model input.",
            "O1b outlet flow fractions remain deferred because direct source coverage is only 58/174 and the feature is solver-output leakage.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "experiment_id",
            "label",
            "input_dim",
            "best_epoch",
            *AGG_KEYS,
            "high_wss_nrmse_range",
            "physical_p99_ratio",
            "AG_r2",
            "AAA_r2",
            "ILO_r2",
            "last_minus_best_r2cb",
            "integrity",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for experiment_id in ORDER:
            record = records[experiment_id]
            writer.writerow(
                {
                    "experiment_id": experiment_id,
                    "label": record["label"],
                    "input_dim": record["input_dim"],
                    "best_epoch": record["history"].get("best_epoch"),
                    **{key: record["best"][key] for key in AGG_KEYS},
                    "high_wss_nrmse_range": record["extra"][
                        "high_wss_nrmse_range"
                    ],
                    "physical_p99_ratio": record["extra"][
                        "physical_p99_ratio"
                    ],
                    "AG_r2": record["domains"]["AG"],
                    "AAA_r2": record["domains"]["AAA"],
                    "ILO_r2": record["domains"]["ILO"],
                    "last_minus_best_r2cb": record["last_minus_best"][
                        "physical_r2_casebalanced"
                    ],
                    "integrity": record["integrity"],
                }
            )

    with OUT_PAIRS.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "comparison",
            "control",
            "treatment",
            "delta_r2cb",
            "delta_mae",
            "delta_rmse",
            "delta_high_wss_nrmse_range",
            "delta_high_wss_r2",
            "delta_top10_iou",
            "delta_p99_ratio",
            "delta_AG",
            "delta_AAA",
            "delta_ILO",
            "case_r2_mean_delta",
            "case_r2_ci95_low",
            "case_r2_ci95_high",
            "case_r2_wins",
            "case_r2_losses",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in comparisons:
            delta = item["aggregate_treatment_minus_control"]
            extra = item["extra_treatment_minus_control"]
            domains = item["domain_r2_delta"]
            cases = item["paired_case_stats"]["case_r2"]
            writer.writerow(
                {
                    "comparison": item["comparison"],
                    "control": item["control"],
                    "treatment": item["treatment"],
                    "delta_r2cb": delta["physical_r2_casebalanced"],
                    "delta_mae": delta["physical_mae"],
                    "delta_rmse": delta["physical_rmse"],
                    "delta_high_wss_nrmse_range": extra[
                        "high_wss_nrmse_range"
                    ],
                    "delta_high_wss_r2": delta["high_wss_r2"],
                    "delta_top10_iou": delta["physical_top10_iou"],
                    "delta_p99_ratio": extra["physical_p99_ratio"],
                    "delta_AG": domains["AG"],
                    "delta_AAA": domains["AAA"],
                    "delta_ILO": domains["ILO"],
                    "case_r2_mean_delta": cases["mean_delta"],
                    "case_r2_ci95_low": cases["ci95_low"],
                    "case_r2_ci95_high": cases["ci95_high"],
                    "case_r2_wins": cases["wins"],
                    "case_r2_losses": cases["losses"],
                }
            )

    print(
        json.dumps(
            {
                "status": "passed",
                "records": len(records),
                "decision": decision,
                "o1a_delta_r2cb": o1a_delta["physical_r2_casebalanced"],
                "o2_delta_r2cb": o2_delta["physical_r2_casebalanced"],
                "json": str(OUT_JSON),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
