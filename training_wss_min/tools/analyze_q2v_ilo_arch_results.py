#!/usr/bin/env python3
"""Audit and summarize the 2026-07-18 Q2V/ILO and Point++ matrix results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PREFLIGHT = REPO / "training_wss_min/preflight"
SUBMISSION = PREFLIGHT / "q2v_ilo_arch_matrix_submission.json"
OUT_JSON = PREFLIGHT / "q2v_ilo_arch_matrix_results_analysis.json"
OUT_CSV = PREFLIGHT / "q2v_ilo_arch_matrix_results_summary.csv"
DATA_ROOT = REPO / "training_wss_min/runs/pointnetpp_q2v_ilo/outputs"
ARCH_ROOT = REPO / "training_wss_min/runs/pointnetpp_q2v_arch_dev/outputs"
ANCHOR = (
    REPO
    / "training_wss_min/runs/pointnetpp_v4/outputs"
    / "ag_aaa_v4_stratified_sa3_random5000_fpscenter_sep"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_root(path: Path) -> tuple[str, dict]:
    payload = read_json(path)
    if len(payload) != 1:
        raise RuntimeError(f"expected one evaluation partition in {path}")
    name = next(iter(payload))
    return name, payload[name]


def finite_numbers(value, where: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            finite_numbers(child, f"{where}.{key}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_numbers(child, f"{where}[{index}]", errors)
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(f"nonfinite:{where}")


def flatten_metrics(metrics: dict) -> dict:
    aggregate = metrics["aggregate"]
    field = metrics["field"]
    balanced = metrics["field_casebalanced"]
    regional = metrics["regional_field"]
    hotspot = metrics["hotspot"]
    calibration = metrics["calibration"]
    normalized = metrics["normalized"]
    norm_aggregate = normalized["aggregate"]
    norm_field = normalized["field"]
    norm_balanced = normalized["field_casebalanced"]
    norm_hotspot = normalized["hotspot"]
    norm_calibration = normalized["calibration"]
    return {
        "physical_r2_casebalanced": balanced["r2"],
        "physical_r2_pooled": field["r2"],
        "physical_r2_case_mean": aggregate["r2_casemean"],
        "physical_r2_case_median": aggregate["r2_casemed"],
        "physical_r2_case_p10": aggregate["r2_casep10"],
        "physical_r2_negative_cases": aggregate["r2_negative_cases"],
        "physical_rmse": field["rmse"],
        "physical_mae": field["mae"],
        "physical_nrmse_range_pooled": field["nrmse_range"],
        "physical_nrmse_range_case_mean": aggregate["nrmse_casemean"],
        "physical_nmae_pooled": field.get("nmae_range"),
        "physical_nmae_case_mean": aggregate.get("nmae_casemean"),
        "high_wss_r2": regional["high_wss"]["r2"],
        "high_wss_mae": regional["high_wss"]["mae"],
        "physical_top10_amplitude_ratio": calibration["top10_pred_true_ratio"],
        "physical_top10_iou": hotspot["top10_iou_casemean"],
        "normalized_r2_casebalanced": norm_balanced["r2"],
        "normalized_r2_pooled": norm_field["r2"],
        "normalized_r2_case_mean": norm_aggregate["r2_casemean"],
        "normalized_r2_case_median": norm_aggregate["r2_casemed"],
        "normalized_r2_case_p10": norm_aggregate["r2_casep10"],
        "normalized_mae": norm_field["mae"],
        "normalized_rmse": norm_field["rmse"],
        "normalized_nrmse_range_pooled": norm_field["nrmse_range"],
        "normalized_nrmse_range_case_mean": norm_aggregate["nrmse_casemean"],
        "normalized_nmae_pooled": norm_field.get("nmae_range"),
        "normalized_nmae_case_mean": norm_aggregate.get("nmae_casemean"),
        "spearman_case_mean": hotspot["spearman_all_casemean"],
        "high_wss_spearman_case_mean": hotspot["spearman_high_wss_casemean"],
        "normalized_top10_amplitude_ratio": norm_calibration["top10_pred_true_ratio"],
        "physical_p99_amplitude_ratio": calibration["p99_pred_true_ratio"],
        "parameters": metrics.get("efficiency", {}).get("parameters"),
        "groups": metrics.get("group_casebalanced", {}),
    }


def history_summary(run_dir: Path) -> dict:
    rows = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines() if line]
    best = min(rows, key=lambda row: row["train_loss"])
    return {
        "epochs": len(rows),
        "first_epoch": rows[0]["epoch"],
        "last_epoch": rows[-1]["epoch"],
        "best_epoch": best["epoch"],
        "best_train_loss": best["train_loss"],
    }


def postview_summary(run_dir: Path) -> dict:
    verification = run_dir / "postview/verification.json"
    if verification.exists():
        payload = read_json(verification)
        return {
            "status": "passed",
            "expected_cases": payload.get("expected_cases"),
            "verified_cases": payload.get("verified_cases", payload.get("n_cases")),
            "path": str(verification),
        }
    bad = run_dir / (
        "postview/ckpt_best/test/ILO__YU_XIANG_SHENG-1__before__peak_wss/"
        "before__mapping_report.json"
    )
    if bad.exists():
        coverage = read_json(bad)["coverage"]
        return {
            "status": "failed_integrity_gate",
            "failed_case": "ILO/YU_XIANG_SHENG-1/before",
            "reason": "PostView Gaussian mapping coverage below required threshold",
            "coverage": coverage,
            "path": str(bad),
        }
    return {"status": "not_requested"}


def run_record(item: dict, job_id: str, slurm_state: str) -> dict:
    run_dir = Path(item["expected_run_dir"])
    best_path = run_dir / "eval/ckpt_best/metrics.json"
    last_path = run_dir / "eval/ckpt_last/metrics.json"
    partition, best_metrics = metric_root(best_path)
    last_partition, last_metrics = metric_root(last_path)
    errors: list[str] = []
    finite_numbers(best_metrics, f"{item['experiment_id']}.best", errors)
    finite_numbers(last_metrics, f"{item['experiment_id']}.last", errors)
    history = history_summary(run_dir)
    if history["epochs"] != 400 or history["first_epoch"] != 0 or history["last_epoch"] != 399:
        errors.append("history_not_400_epochs")
    for name in ("ckpt_best.pt", "ckpt_last.pt", "eval/best_last_comparison.json"):
        if not (run_dir / name).exists():
            errors.append(f"missing:{name}")
    if partition != last_partition:
        errors.append("best_last_partition_mismatch")
    config_path = Path(item["config"])
    if sha256(config_path) != item["config_sha256"]:
        errors.append("submitted_config_hash_changed")
    best = flatten_metrics(best_metrics)
    last = flatten_metrics(last_metrics)
    return {
        "experiment_id": item["experiment_id"],
        "family": item["family"],
        "job_id": job_id,
        "slurm_state": slurm_state,
        "evaluation_partition": partition,
        "run_dir": str(run_dir),
        "config": item["config"],
        "split": item["split"],
        "control_group": item["control_group"],
        "unique_change": item["unique_change"],
        "history": history,
        "best": best,
        "last": last,
        "last_minus_best_r2_casebalanced": (
            last["physical_r2_casebalanced"] - best["physical_r2_casebalanced"]
        ),
        "postview": postview_summary(run_dir) if item["family"] == "data" else {"status": "not_requested"},
        "integrity": "passed" if not errors else "failed",
        "integrity_errors": errors,
    }


def anchor_record(experiment_id: str, path: Path, partition_label: str) -> dict:
    partition, metrics = metric_root(path)
    errors: list[str] = []
    finite_numbers(metrics, experiment_id, errors)
    return {
        "experiment_id": experiment_id,
        "family": "read_only_evaluation",
        "evaluation_partition": partition,
        "partition_label": partition_label,
        "metrics_path": str(path),
        "best": flatten_metrics(metrics),
        "integrity": "passed" if not errors else "failed",
        "integrity_errors": errors,
    }


def delta(records: dict[str, dict], treatment: str, control: str) -> dict:
    treatment_metrics = records[treatment]["best"]
    control_metrics = records[control]["best"]
    keys = (
        "physical_r2_casebalanced", "physical_r2_pooled", "physical_r2_case_mean",
        "physical_r2_negative_cases", "physical_mae", "physical_rmse", "high_wss_r2",
        "high_wss_mae", "spearman_case_mean", "physical_top10_iou",
        "physical_top10_amplitude_ratio", "physical_p99_amplitude_ratio",
    )
    return {
        "treatment": treatment,
        "control": control,
        "treatment_minus_control": {
            key: treatment_metrics[key] - control_metrics[key] for key in keys
        },
    }


def write_csv(records: list[dict], anchors: list[dict]) -> None:
    fieldnames = [
        "experiment_id", "family", "job_id", "slurm_state", "evaluation_partition",
        "best_epoch", "best_train_loss", "physical_r2_casebalanced", "physical_r2_pooled",
        "physical_r2_case_mean", "physical_r2_case_median", "physical_r2_case_p10",
        "physical_r2_negative_cases", "physical_mae", "physical_rmse", "high_wss_r2",
        "high_wss_mae", "spearman_case_mean", "physical_top10_iou",
        "physical_top10_amplitude_ratio", "physical_p99_amplitude_ratio", "parameters",
        "last_minus_best_r2_casebalanced", "postview_status", "integrity",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in records + anchors:
            best = item["best"]
            history = item.get("history", {})
            writer.writerow({
                "experiment_id": item["experiment_id"],
                "family": item["family"],
                "job_id": item.get("job_id", ""),
                "slurm_state": item.get("slurm_state", "COMPLETED"),
                "evaluation_partition": item["evaluation_partition"],
                "best_epoch": history.get("best_epoch", ""),
                "best_train_loss": history.get("best_train_loss", ""),
                **{key: best[key] for key in fieldnames if key in best},
                "last_minus_best_r2_casebalanced": item.get("last_minus_best_r2_casebalanced", ""),
                "postview_status": item.get("postview", {}).get("status", "not_requested"),
                "integrity": item["integrity"],
            })


def main() -> None:
    submission = read_json(SUBMISSION)
    if submission.get("status") != "submitted":
        raise RuntimeError("submission manifest is not in submitted state")
    jobs = {
        "data": next(job["job_id"] for job in submission["jobs"] if job["role"] == "data_training_array"),
        "architecture": next(
            job["job_id"] for job in submission["jobs"]
            if job["role"] == "architecture_dev_array_val_only"
        ),
    }
    data_states = {0: "COMPLETED", 1: "COMPLETED", 2: "COMPLETED", 3: "FAILED", 4: "FAILED", 5: "FAILED"}
    records = []
    for item in submission["configs"]:
        task = item["array_task_id"]
        state = data_states[task] if item["family"] == "data" else "COMPLETED"
        records.append(run_record(item, f"{jobs[item['family']]}_{task}", state))
    anchors = [
        anchor_record("Q2V-10477", ANCHOR / "eval/ckpt_best/metrics.json", "original test27"),
        anchor_record(
            "Q2V-10477-zero-shot-D2-test36",
            ANCHOR / "eval_external/q2v_extended_test36/ckpt_best/metrics.json",
            "D2 extended test36",
        ),
    ]
    by_id = {item["experiment_id"]: item for item in records + anchors}
    comparisons = {
        "d1_add_ilo41_frozen_same_test27": delta(by_id, "q2v_ilo_d1_fixed_frozen", "Q2V-10477"),
        "d1_refit_stats": delta(by_id, "q2v_ilo_d1_fixed_refit", "q2v_ilo_d1_fixed_frozen"),
        "d2_add_ilo32_vs_zero_shot_same_test36": delta(
            by_id, "q2v_ilo_d2_extended_frozen", "Q2V-10477-zero-shot-D2-test36"
        ),
        "d3_add_ilo32_frozen_same_test36": delta(
            by_id, "q2v_ilo_d3_pool2025_frozen", "q2v_ilo_d3_pool2025_control"
        ),
        "d3_refit_stats": delta(
            by_id, "q2v_ilo_d3_pool2025_refit", "q2v_ilo_d3_pool2025_frozen"
        ),
    }
    architecture = [item for item in records if item["family"] == "architecture"]
    rank = sorted(
        architecture, key=lambda item: item["best"]["physical_r2_casebalanced"], reverse=True
    )
    payload = {
        "schema_version": 1,
        "analysis_date": "2026-07-18",
        "source_submission": str(SUBMISSION),
        "source_submission_sha256": sha256(SUBMISSION),
        "job_summary": {
            "cpu_prepare_10484": "COMPLETED",
            "gpu_preflight_10485": "COMPLETED; 12/12 passed",
            "finalizer_10486": "FAILED after successful submission; local xlsx UNO dependency missing on node03",
            "data_array_10487": "3 COMPLETED; 3 quantitative-complete but PostView-gate FAILED",
            "architecture_array_10488": "6/6 COMPLETED",
            "q2v_postview_10489": "COMPLETED; 27/27 passed",
            "q2v_zero_shot_10490": "COMPLETED",
            "queue_at_analysis": "empty",
        },
        "records": records,
        "anchors": anchors,
        "paired_comparisons": comparisons,
        "architecture_rank_by_val_r2_casebalanced": [
            {
                "rank": index,
                "experiment_id": item["experiment_id"],
                "r2_casebalanced": item["best"]["physical_r2_casebalanced"],
                "parameters": item["best"]["parameters"],
            }
            for index, item in enumerate(rank, start=1)
        ],
        "architecture_dev_winner": rank[0]["experiment_id"],
        "interpretation_limits": [
            "all training comparisons are single-seed and are prioritization evidence, not confirmation",
            "architecture matrix used val21 only; original test27 remained untouched",
            "D3 quantitative metrics are valid, but PostView closure failed for one shared ILO case",
            "all high-WSS R2 values remain negative",
        ],
    }
    if any(item["integrity"] != "passed" for item in records + anchors):
        raise RuntimeError("one or more quantitative artifacts failed integrity audit")
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(records, anchors)
    print(json.dumps({
        "status": "passed",
        "records": len(records),
        "anchors": len(anchors),
        "architecture_winner": rank[0]["experiment_id"],
        "json": str(OUT_JSON),
        "csv": str(OUT_CSV),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
