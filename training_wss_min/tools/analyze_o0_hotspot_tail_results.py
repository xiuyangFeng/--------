#!/usr/bin/env python3
"""Audit and summarize the completed O0 hotspot/tail objective matrix."""

from __future__ import annotations

import csv
import json
import subprocess
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
MANIFEST = PREFLIGHT / "o0_hotspot_tail_matrix_20260728_prepared.json"
SUBMISSION = PREFLIGHT / "o0_hotspot_tail_matrix_20260728_submission.json"
OUT_JSON = PREFLIGHT / "o0_hotspot_tail_matrix_20260728_results_analysis.json"
OUT_SUMMARY = PREFLIGHT / "o0_hotspot_tail_matrix_20260728_results_summary.csv"
OUT_PAIRS = PREFLIGHT / "o0_hotspot_tail_matrix_20260728_paired_case_stats.csv"

O0_ID = "o0_geometry_s1234"
O0_RUN = (
    ROOT
    / "training_wss_min/runs/pointnetpp_rcr_oracle/outputs/o0_geometry_s1234"
)
ORDER = (
    "h1_hotspot_bce_q90_lam020_s1234",
    "h2_pinball_q90_lam020_s1234",
)
LABELS = {
    O0_ID: "O0 geometry",
    "h1_hotspot_bce_q90_lam020_s1234": "H1 hotspot BCE q90 λ0.20",
    "h2_pinball_q90_lam020_s1234": "H2 pinball q90 λ0.20",
}
DOMAINS = ("AG", "AAA", "ILO")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extra_metrics(record: dict, checkpoint: str = "ckpt_best") -> dict[str, float]:
    raw = read_json(
        Path(record["run_dir"]) / f"eval/{checkpoint}/metrics.json"
    )["test"]
    high = raw["regional_field"]["high_wss"]
    return {
        "high_wss_nrmse_range": high["nrmse_range"],
        "high_wss_mae": high["mae"],
        "physical_top10_ratio": raw["calibration"]["top10_pred_true_ratio"],
        "physical_p99_ratio": raw["calibration"]["p99_pred_true_ratio"],
    }


def slurm_states() -> list[dict[str, str]]:
    command = [
        "/public/slurm/bin/sacct",
        "-n",
        "-P",
        "-j",
        "11012,11013",
        "--format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS",
    ]
    output = subprocess.check_output(command, text=True)
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        job_id, job_name, state, exit_code, elapsed, max_rss = line.split("|")
        if "." in job_id:
            continue
        rows.append(
            {
                "job_id": job_id,
                "job_name": job_name,
                "state": state,
                "exit_code": exit_code,
                "elapsed": elapsed,
                "max_rss": max_rss,
            }
        )
    return rows


def main() -> None:
    manifest = read_json(MANIFEST)
    submission = read_json(SUBMISSION)
    rows = manifest["configs"]
    if (
        manifest.get("status") != "prepared"
        or [row["experiment_id"] for row in rows] != list(ORDER)
        or "mixed 138/0/36" not in manifest["protocol"]
    ):
        raise RuntimeError("expected exact prepared O0 hotspot/tail matrix")

    o0 = build_record(O0_ID, O0_RUN, 1234, "geometry")
    o0["label"] = LABELS[O0_ID]
    o0["extra"] = extra_metrics(o0)
    o0["extra_last"] = extra_metrics(o0, "ckpt_last")

    records = {O0_ID: o0}
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
        record["out_dim"] = row["out_dim"]
        record["extra"] = extra_metrics(record)
        record["extra_last"] = extra_metrics(record, "ckpt_last")
        records[experiment_id] = record

    failed = {
        experiment_id: record["integrity_errors"]
        for experiment_id, record in records.items()
        if record["integrity"] != "passed"
    }
    if failed:
        raise RuntimeError(f"integrity failures: {failed}")

    states = slurm_states()
    if not states or any(
        row["state"] != "COMPLETED" or row["exit_code"] != "0:0"
        for row in states
    ):
        raise RuntimeError(f"Slurm jobs not cleanly completed: {states}")

    comparisons = []
    for offset, experiment_id in enumerate(ORDER):
        item = comparison(
            f"{experiment_id}_vs_o0_geometry",
            o0,
            records[experiment_id],
            20260728 + offset * 100,
        )
        item["extra_treatment_minus_control"] = {
            key: records[experiment_id]["extra"][key] - o0["extra"][key]
            for key in records[experiment_id]["extra"]
        }
        comparisons.append(item)

    by_treatment = {item["treatment"]: item for item in comparisons}
    decisions = {}
    for experiment_id in ORDER:
        item = by_treatment[experiment_id]
        delta = item["aggregate_treatment_minus_control"]
        extra = item["extra_treatment_minus_control"]
        primary_name = (
            "delta_top10_iou"
            if experiment_id.startswith("h1_")
            else "delta_high_wss_nrmse_range"
        )
        primary_value = (
            delta["physical_top10_iou"]
            if experiment_id.startswith("h1_")
            else extra["high_wss_nrmse_range"]
        )
        primary_threshold = 0.02 if experiment_id.startswith("h1_") else -0.002
        primary_pass = (
            primary_value >= primary_threshold
            if experiment_id.startswith("h1_")
            else primary_value <= primary_threshold
        )
        guards = {
            "normalized_r2_casebalanced": {
                "delta": delta["normalized_r2_casebalanced"],
                "threshold": -0.01,
                "pass": delta["normalized_r2_casebalanced"] >= -0.01,
            },
            "physical_mae_casebalanced": {
                "delta": delta["physical_mae"],
                "threshold": 0.05,
                "pass": delta["physical_mae"] <= 0.05,
            },
        }
        decisions[experiment_id] = {
            "primary": {
                "metric": primary_name,
                "delta": primary_value,
                "threshold": primary_threshold,
                "pass": primary_pass,
            },
            "guards": guards,
            "decision": (
                "go"
                if primary_pass and all(guard["pass"] for guard in guards.values())
                else "no_go"
            ),
        }

    payload = {
        "schema_version": 1,
        "analysis_date": "2026-07-28",
        "scope": (
            "O0 geometry-only S3 PointNeXt-R + LocalGeoPE; H1 hotspot BCE "
            "and H2 q90 pinball; historical mixed test36"
        ),
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha256(MANIFEST),
        "submission": submission,
        "slurm_states": states,
        "records": records,
        "paired_comparisons": comparisons,
        "registered_gate_decisions": decisions,
        "combination_decision": "do_not_combine",
        "interpretation_limits": [
            "All arms use seed1234 only; this is a single-seed engineering screen.",
            "test36 is a repeatedly reused engineering development set, not an independent final test set.",
            "ckpt_best is selected by train loss because the frozen split has no validation set; ckpt_last is sensitivity-only.",
            "The registered primary deltas are aggregate point/case metrics; paired case bootstrap intervals are supporting evidence, not substitutes for the registered thresholds.",
            "No area-weighted metric is used because some cases do not meet the area-definition requirements.",
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
            "best_epoch",
            *AGG_KEYS,
            "high_wss_nrmse_range",
            "physical_p99_ratio",
            "AG_r2",
            "AAA_r2",
            "ILO_r2",
            "last_minus_best_r2cb",
            "last_minus_best_high_wss_nrmse_range",
            "integrity",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for experiment_id in (O0_ID, *ORDER):
            record = records[experiment_id]
            writer.writerow(
                {
                    "experiment_id": experiment_id,
                    "label": record["label"],
                    "best_epoch": record["history"].get("best_epoch"),
                    **{key: record["best"][key] for key in AGG_KEYS},
                    "high_wss_nrmse_range": record["extra"]["high_wss_nrmse_range"],
                    "physical_p99_ratio": record["extra"]["physical_p99_ratio"],
                    "AG_r2": record["domains"]["AG"],
                    "AAA_r2": record["domains"]["AAA"],
                    "ILO_r2": record["domains"]["ILO"],
                    "last_minus_best_r2cb": record["last_minus_best"][
                        "physical_r2_casebalanced"
                    ],
                    "last_minus_best_high_wss_nrmse_range": (
                        record["extra_last"]["high_wss_nrmse_range"]
                        - record["extra"]["high_wss_nrmse_range"]
                    ),
                    "integrity": record["integrity"],
                }
            )

    with OUT_PAIRS.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "comparison",
            "control",
            "treatment",
            "decision",
            "delta_r2cb",
            "delta_normalized_r2cb",
            "delta_mae",
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
            "case_iou_mean_delta",
            "case_iou_ci95_low",
            "case_iou_ci95_high",
            "case_iou_wins",
            "case_iou_losses",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in comparisons:
            delta = item["aggregate_treatment_minus_control"]
            extra = item["extra_treatment_minus_control"]
            domains = item["domain_r2_delta"]
            cases = item["paired_case_stats"]
            r2 = cases["case_r2"]
            iou = cases["case_top10_iou"]
            writer.writerow(
                {
                    "comparison": item["comparison"],
                    "control": item["control"],
                    "treatment": item["treatment"],
                    "decision": decisions[item["treatment"]]["decision"],
                    "delta_r2cb": delta["physical_r2_casebalanced"],
                    "delta_normalized_r2cb": delta["normalized_r2_casebalanced"],
                    "delta_mae": delta["physical_mae"],
                    "delta_high_wss_nrmse_range": extra[
                        "high_wss_nrmse_range"
                    ],
                    "delta_high_wss_r2": delta["high_wss_r2"],
                    "delta_top10_iou": delta["physical_top10_iou"],
                    "delta_p99_ratio": extra["physical_p99_ratio"],
                    "delta_AG": domains["AG"],
                    "delta_AAA": domains["AAA"],
                    "delta_ILO": domains["ILO"],
                    "case_r2_mean_delta": r2["mean_delta"],
                    "case_r2_ci95_low": r2["ci95_low"],
                    "case_r2_ci95_high": r2["ci95_high"],
                    "case_r2_wins": r2["wins"],
                    "case_r2_losses": r2["losses"],
                    "case_iou_mean_delta": iou["mean_delta"],
                    "case_iou_ci95_low": iou["ci95_low"],
                    "case_iou_ci95_high": iou["ci95_high"],
                    "case_iou_wins": iou["wins"],
                    "case_iou_losses": iou["losses"],
                }
            )

    print(
        json.dumps(
            {
                "status": "passed",
                "slurm_states": states,
                "decisions": decisions,
                "json": str(OUT_JSON),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
