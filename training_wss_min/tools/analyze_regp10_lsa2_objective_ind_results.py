#!/usr/bin/env python3
"""Audit and summarize REG-P10-LSA2 SAME/IND × MSE/H1/H2 results."""

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
MANIFEST = (
    PREFLIGHT / "regp10_lsa2_objective_ind_matrix_20260728_prepared.json"
)
GPU_AUDIT = (
    PREFLIGHT / "regp10_lsa2_objective_ind_matrix_20260728_gpu_smoke.json"
)
SUBMISSION = (
    PREFLIGHT / "regp10_lsa2_objective_ind_matrix_20260728_submission.json"
)
OUT_JSON = (
    PREFLIGHT / "regp10_lsa2_objective_ind_matrix_20260728_results_analysis.json"
)
OUT_SUMMARY = (
    PREFLIGHT / "regp10_lsa2_objective_ind_matrix_20260728_results_summary.csv"
)
OUT_PAIRS = (
    PREFLIGHT
    / "regp10_lsa2_objective_ind_matrix_20260728_paired_case_stats.csv"
)
HISTORICAL_ANCHOR_RUN = (
    ROOT
    / "training_wss_min/runs/pointnetpp_regp10_transformer/outputs/"
    "localtf_sa2_s1234"
)

ORDER = (
    "lsa2_same_mse_s1234",
    "lsa2_same_h1_bce_q90_lam020_s1234",
    "lsa2_same_h2_pinball_q90_lam020_s1234",
    "lsa2_ind_mse_s1234",
    "lsa2_ind_h1_bce_q90_lam020_s1234",
    "lsa2_ind_h2_pinball_q90_lam020_s1234",
)
LABELS = {
    "lsa2_same_mse_s1234": "LSA2 SAME MSE",
    "lsa2_same_h1_bce_q90_lam020_s1234": "LSA2 SAME H1 q90 λ0.20",
    "lsa2_same_h2_pinball_q90_lam020_s1234": "LSA2 SAME H2 q90 λ0.20",
    "lsa2_ind_mse_s1234": "LSA2 IND MSE",
    "lsa2_ind_h1_bce_q90_lam020_s1234": "LSA2 IND H1 q90 λ0.20",
    "lsa2_ind_h2_pinball_q90_lam020_s1234": "LSA2 IND H2 q90 λ0.20",
}
PAIR_SPECS = (
    (
        "same_h1_vs_same_mse",
        "lsa2_same_mse_s1234",
        "lsa2_same_h1_bce_q90_lam020_s1234",
        "h1",
    ),
    (
        "same_h2_vs_same_mse",
        "lsa2_same_mse_s1234",
        "lsa2_same_h2_pinball_q90_lam020_s1234",
        "h2",
    ),
    (
        "ind_mse_vs_same_mse",
        "lsa2_same_mse_s1234",
        "lsa2_ind_mse_s1234",
        "ind",
    ),
    (
        "ind_h1_vs_ind_mse",
        "lsa2_ind_mse_s1234",
        "lsa2_ind_h1_bce_q90_lam020_s1234",
        "h1",
    ),
    (
        "ind_h2_vs_ind_mse",
        "lsa2_ind_mse_s1234",
        "lsa2_ind_h2_pinball_q90_lam020_s1234",
        "h2",
    ),
    (
        "ind_h1_vs_same_h1",
        "lsa2_same_h1_bce_q90_lam020_s1234",
        "lsa2_ind_h1_bce_q90_lam020_s1234",
        "diagnostic",
    ),
    (
        "ind_h2_vs_same_h2",
        "lsa2_same_h2_pinball_q90_lam020_s1234",
        "lsa2_ind_h2_pinball_q90_lam020_s1234",
        "diagnostic",
    ),
)
DOMAINS = ("AG", "AAA", "ILO")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def raw_metrics(record: dict, checkpoint: str = "ckpt_best") -> dict:
    return read_json(
        Path(record["run_dir"]) / f"eval/{checkpoint}/metrics.json"
    )["test"]


def extra_metrics(record: dict, checkpoint: str = "ckpt_best") -> dict[str, float]:
    raw = raw_metrics(record, checkpoint)
    high = raw["regional_field"]["high_wss"]
    return {
        "high_wss_nrmse_range": high["nrmse_range"],
        "physical_top10_ratio": raw["calibration"]["top10_pred_true_ratio"],
        "physical_p99_ratio": raw["calibration"]["p99_pred_true_ratio"],
    }


def slurm_states() -> list[dict[str, str]]:
    output = subprocess.check_output(
        [
            "/public/slurm/bin/sacct",
            "-n",
            "-P",
            "-j",
            "11019,11020",
            "--format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS",
        ],
        text=True,
    )
    expected = {"11019", *(f"11020_{index}" for index in range(6))}
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        job_id, job_name, state, exit_code, elapsed, max_rss = line.split("|")
        if job_id not in expected:
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
    if {row["job_id"] for row in rows} != expected:
        raise RuntimeError(f"missing Slurm rows: expected={sorted(expected)} got={rows}")
    return rows


def objective_decision(kind: str, item: dict, records: dict) -> dict:
    delta = item["aggregate_treatment_minus_control"]
    control = records[item["control"]]
    treatment = records[item["treatment"]]
    delta_high_nrmse = (
        treatment["extra"]["high_wss_nrmse_range"]
        - control["extra"]["high_wss_nrmse_range"]
    )
    if kind == "h1":
        primary = {
            "metric": "delta_top10_iou",
            "delta": delta["physical_top10_iou"],
            "threshold": 0.02,
            "pass": delta["physical_top10_iou"] >= 0.02,
        }
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
    elif kind == "h2":
        primary = {
            "metric": "delta_high_wss_nrmse_range",
            "delta": delta_high_nrmse,
            "threshold": -0.002,
            "pass": delta_high_nrmse <= -0.002,
        }
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
    elif kind == "ind":
        primary = {
            "metric": "delta_physical_r2_casebalanced",
            "delta": delta["physical_r2_casebalanced"],
            "threshold": 0.012,
            "pass": delta["physical_r2_casebalanced"] >= 0.012,
        }
        guards = {
            "normalized_r2_casebalanced": {
                "delta": delta["normalized_r2_casebalanced"],
                "threshold": -0.01,
                "pass": delta["normalized_r2_casebalanced"] >= -0.01,
            },
            "physical_mae_casebalanced": {
                "delta": delta["physical_mae"],
                "threshold": 0.03,
                "pass": delta["physical_mae"] <= 0.03,
            },
            "high_wss_nrmse_range": {
                "delta": delta_high_nrmse,
                "threshold": 0.001,
                "pass": delta_high_nrmse <= 0.001,
            },
            "top10_iou": {
                "delta": delta["physical_top10_iou"],
                "threshold": -0.005,
                "pass": delta["physical_top10_iou"] >= -0.005,
            },
        }
    else:
        return {"decision": "diagnostic_only"}
    return {
        "primary": primary,
        "guards": guards,
        "decision": (
            "go"
            if primary["pass"] and all(guard["pass"] for guard in guards.values())
            else "no_go"
        ),
    }


def main() -> None:
    manifest = read_json(MANIFEST)
    gpu_audit = read_json(GPU_AUDIT)
    submission = read_json(SUBMISSION)
    rows = manifest["configs"]
    if (
        manifest.get("status") != "prepared"
        or [row["experiment_id"] for row in rows] != list(ORDER)
        or gpu_audit.get("status") != "passed"
        or len(gpu_audit.get("jobs", [])) != 6
        or submission.get("status") != "submitted"
    ):
        raise RuntimeError("prepared/submission/GPU matrix contract failed")

    records = {}
    row_by_id = {row["experiment_id"]: row for row in rows}
    for experiment_id in ORDER:
        row = row_by_id[experiment_id]
        record = build_record(
            experiment_id,
            Path(row["run_dir"]),
            1234,
            f"{row['query_mode']}_{row['objective']}",
            Path(row["config"]),
            row["config_sha256"],
        )
        record["label"] = LABELS[experiment_id]
        record["query_mode"] = row["query_mode"]
        record["objective"] = row["objective"]
        record["extra"] = extra_metrics(record)
        record["extra_last"] = extra_metrics(record, "ckpt_last")
        if len(record["per_case"]) != 36:
            record["integrity_errors"].append(
                f"expected_36_cases_got_{len(record['per_case'])}"
            )
            record["integrity"] = "failed"
        records[experiment_id] = record

    failed = {
        experiment_id: record["integrity_errors"]
        for experiment_id, record in records.items()
        if record["integrity"] != "passed"
    }
    if failed:
        raise RuntimeError(f"integrity failures: {failed}")

    states = slurm_states()
    if any(
        row["state"] != "COMPLETED" or row["exit_code"] != "0:0"
        for row in states
    ):
        raise RuntimeError(f"Slurm jobs not cleanly completed: {states}")

    comparisons = []
    comparison_kind = {}
    for offset, (name, control, treatment, kind) in enumerate(PAIR_SPECS):
        item = comparison(
            name,
            records[control],
            records[treatment],
            20260729 + offset * 100,
        )
        item["extra_treatment_minus_control"] = {
            key: records[treatment]["extra"][key] - records[control]["extra"][key]
            for key in records[treatment]["extra"]
        }
        comparisons.append(item)
        comparison_kind[name] = kind

    decisions = {
        item["comparison"]: objective_decision(
            comparison_kind[item["comparison"]], item, records
        )
        for item in comparisons
    }

    historical = build_record(
        "historical_regp10_lsa2_s1234",
        HISTORICAL_ANCHOR_RUN,
        1234,
        "historical_anchor",
    )
    historical["extra"] = extra_metrics(historical)
    same_mse = records["lsa2_same_mse_s1234"]
    historical_drift = {
        "physical_r2_casebalanced": (
            same_mse["best"]["physical_r2_casebalanced"]
            - historical["best"]["physical_r2_casebalanced"]
        ),
        "normalized_r2_casebalanced": (
            same_mse["best"]["normalized_r2_casebalanced"]
            - historical["best"]["normalized_r2_casebalanced"]
        ),
        "physical_mae": (
            same_mse["best"]["physical_mae"] - historical["best"]["physical_mae"]
        ),
        "high_wss_nrmse_range": (
            same_mse["extra"]["high_wss_nrmse_range"]
            - historical["extra"]["high_wss_nrmse_range"]
        ),
        "physical_top10_iou": (
            same_mse["best"]["physical_top10_iou"]
            - historical["best"]["physical_top10_iou"]
        ),
    }

    payload = {
        "schema_version": 1,
        "analysis_date": "2026-07-29",
        "scope": (
            "REG-P10-LSA2 historical development anchor; SAME/IND x MSE/H1/H2; "
            "mixed test36; seed1234; legacy_vertex"
        ),
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha256(MANIFEST),
        "gpu_audit": str(GPU_AUDIT.resolve()),
        "gpu_audit_sha256": sha256(GPU_AUDIT),
        "submission": submission,
        "slurm_states": states,
        "records": records,
        "paired_comparisons": comparisons,
        "registered_gate_decisions": decisions,
        "historical_anchor": historical,
        "concurrent_same_mse_minus_historical_anchor": historical_drift,
        "matrix_decision": {
            "same_h1": decisions["same_h1_vs_same_mse"]["decision"],
            "same_h2": decisions["same_h2_vs_same_mse"]["decision"],
            "ind_main": decisions["ind_mse_vs_same_mse"]["decision"],
            "ind_h1": decisions["ind_h1_vs_ind_mse"]["decision"],
            "ind_h2": decisions["ind_h2_vs_ind_mse"]["decision"],
            "provisional_anchor_pending_same_seed_rerun": (
                "lsa2_same_h2_pinball_q90_lam020_s1234"
            ),
            "formal_anchor_promotion_status": "pending_same_seed_paired_rerun",
            "do_not_promote": [
                "query_mode=independent",
                "H1 hotspot BCE",
                "H1+H2 combination",
            ],
        },
        "interpretation_limits": [
            "All arms use seed1234 only; this is a single-seed engineering screen.",
            "test36 is a repeatedly reused engineering development set, not an independent final test set.",
            "ckpt_best is selected by train loss because the frozen split has no validation set; ckpt_last is sensitivity-only.",
            "The concurrent SAME MSE rerun is materially below the historical identical-configuration run, so all Gate decisions use only concurrent controls.",
            "The SAME-H2 primary improvement exceeds its registered threshold by only about 0.00005 nRMSE; promotion is provisional despite best/last agreement.",
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
            "query_mode",
            "objective",
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
        for experiment_id in ORDER:
            record = records[experiment_id]
            writer.writerow(
                {
                    "experiment_id": experiment_id,
                    "label": record["label"],
                    "query_mode": record["query_mode"],
                    "objective": record["objective"],
                    "best_epoch": record["history"]["best_epoch"],
                    **{key: record["best"][key] for key in AGG_KEYS},
                    "high_wss_nrmse_range": record["extra"][
                        "high_wss_nrmse_range"
                    ],
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
            "kind",
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
            "case_high_wss_r2_mean_delta",
            "case_high_wss_r2_ci95_low",
            "case_high_wss_r2_ci95_high",
            "case_top10_iou_mean_delta",
            "case_top10_iou_ci95_low",
            "case_top10_iou_ci95_high",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in comparisons:
            delta = item["aggregate_treatment_minus_control"]
            extra = item["extra_treatment_minus_control"]
            domains = item["domain_r2_delta"]
            cases = item["paired_case_stats"]
            r2 = cases["case_r2"]
            high = cases["case_high_wss_r2"]
            iou = cases["case_top10_iou"]
            decision = decisions[item["comparison"]]["decision"]
            writer.writerow(
                {
                    "comparison": item["comparison"],
                    "control": item["control"],
                    "treatment": item["treatment"],
                    "kind": comparison_kind[item["comparison"]],
                    "decision": decision,
                    "delta_r2cb": delta["physical_r2_casebalanced"],
                    "delta_normalized_r2cb": delta[
                        "normalized_r2_casebalanced"
                    ],
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
                    "case_high_wss_r2_mean_delta": high["mean_delta"],
                    "case_high_wss_r2_ci95_low": high["ci95_low"],
                    "case_high_wss_r2_ci95_high": high["ci95_high"],
                    "case_top10_iou_mean_delta": iou["mean_delta"],
                    "case_top10_iou_ci95_low": iou["ci95_low"],
                    "case_top10_iou_ci95_high": iou["ci95_high"],
                }
            )

    print(
        json.dumps(
            {
                "status": "passed",
                "records": len(records),
                "matrix_decision": payload["matrix_decision"],
                "json": str(OUT_JSON),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
