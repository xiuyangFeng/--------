#!/usr/bin/env python3
"""Audit the completed Q2V/test27 sampling-and-radius exploratory matrix.

This is deliberately a result-audit tool, rather than a leaderboard selector:
the four jobs reused Q2V's historical test27 under explicit user authorization.
It therefore records paired exploratory deltas and keeps Q1V/Q2V as historical
anchors instead of treating any row as independent confirmation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PREFLIGHT = REPO / "training_wss_min/preflight"
SUBMISSION = PREFLIGHT / "q2v_sampling_radius_test27_submission.json"
OUT_JSON = PREFLIGHT / "q2v_sampling_radius_test27_results_analysis.json"
OUT_CSV = PREFLIGHT / "q2v_sampling_radius_test27_results_summary.csv"
Q1V = REPO / "training_wss_min/runs/pointnetpp_v4/outputs/ag_aaa_v4_stratified_sa3_random5000_fpscenter_same"
Q2V = REPO / "training_wss_min/runs/pointnetpp_v4/outputs/ag_aaa_v4_stratified_sa3_random5000_fpscenter_sep"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_numbers(value, where: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            finite_numbers(child, f"{where}.{key}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_numbers(child, f"{where}[{index}]", errors)
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(f"nonfinite:{where}")


def metric_root(path: Path) -> tuple[str, dict]:
    payload = read_json(path)
    if len(payload) != 1:
        raise RuntimeError(f"expected one evaluation partition in {path}")
    partition = next(iter(payload))
    return partition, payload[partition]


def flatten_metrics(metrics: dict) -> dict:
    aggregate = metrics["aggregate"]
    field = metrics["field"]
    balanced = metrics["field_casebalanced"]
    regional = metrics["regional_field"]
    hotspot = metrics["hotspot"]
    calibration = metrics["calibration"]
    normalized = metrics["normalized"]
    return {
        "physical_r2_casebalanced": balanced["r2"],
        "physical_r2_pooled": field["r2"],
        "physical_r2_case_mean": aggregate["r2_casemean"],
        "physical_r2_case_median": aggregate["r2_casemed"],
        "physical_r2_case_p10": aggregate["r2_casep10"],
        "physical_r2_negative_cases": aggregate["r2_negative_cases"],
        "physical_mae": field["mae"],
        "physical_rmse": field["rmse"],
        "physical_nrmse_range_pooled": field["nrmse_range"],
        "physical_nrmse_range_case_mean": aggregate.get("nrmse_casemean"),
        "physical_nmae_pooled": field.get("nmae_range"),
        "physical_nmae_case_mean": aggregate.get("nmae_casemean"),
        "high_wss_r2": regional["high_wss"]["r2"],
        "high_wss_mae": regional["high_wss"]["mae"],
        "physical_top10_iou": hotspot["top10_iou_casemean"],
        "spearman_case_mean": hotspot["spearman_all_casemean"],
        "high_wss_spearman_case_mean": hotspot["spearman_high_wss_casemean"],
        "physical_p99_amplitude_ratio": calibration["p99_pred_true_ratio"],
        "normalized_r2_casebalanced": normalized["field_casebalanced"]["r2"],
        "normalized_r2_pooled": normalized["field"]["r2"],
        "normalized_r2_case_mean": normalized["aggregate"]["r2_casemean"],
        "normalized_r2_case_median": normalized["aggregate"]["r2_casemed"],
        "normalized_r2_case_p10": normalized["aggregate"]["r2_casep10"],
        "normalized_mae": normalized["field"]["mae"],
        "normalized_rmse": normalized["field"]["rmse"],
        "normalized_nrmse_range_pooled": normalized["field"]["nrmse_range"],
        "normalized_nrmse_range_case_mean": normalized["aggregate"].get("nrmse_casemean"),
        "normalized_nmae_pooled": normalized["field"].get("nmae_range"),
        "normalized_nmae_case_mean": normalized["aggregate"].get("nmae_casemean"),
        "normalized_top10_amplitude_ratio": normalized["calibration"]["top10_pred_true_ratio"],
    }


def history_summary(run_dir: Path) -> dict:
    rows = [json.loads(line) for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines() if line]
    best = min(rows, key=lambda row: row["train_loss"])
    return {
        "epochs": len(rows),
        "first_epoch": rows[0]["epoch"],
        "last_epoch": rows[-1]["epoch"],
        "best_epoch": best["epoch"],
        "best_train_loss": best["train_loss"],
    }


def postview_summary(run_dir: Path) -> dict:
    path = run_dir / "postview/verification.json"
    payload = read_json(path)
    expected = payload.get("expected_cases", payload.get("n_cases"))
    verified = payload.get("verified_cases", payload.get("n_cases"))
    return {
        "status": "passed" if expected == verified == 27 else "failed",
        "expected_cases": expected,
        "verified_cases": verified,
        "path": str(path),
    }


def run_record(item: dict, task_id: int) -> dict:
    run_dir = Path(item["run_dir"])
    best_partition, best_metrics = metric_root(run_dir / "eval/ckpt_best/metrics.json")
    last_partition, last_metrics = metric_root(run_dir / "eval/ckpt_last/metrics.json")
    history = history_summary(run_dir)
    errors: list[str] = []
    finite_numbers(best_metrics, f"{item['experiment_id']}.best", errors)
    finite_numbers(last_metrics, f"{item['experiment_id']}.last", errors)
    if (history["epochs"], history["first_epoch"], history["last_epoch"]) != (400, 0, 399):
        errors.append("history_not_400_epochs")
    if best_partition != last_partition or best_partition != "test":
        errors.append("unexpected_evaluation_partition")
    for name in ("ckpt_best.pt", "ckpt_last.pt", "eval/best_last_comparison.json", "postview/verification.json"):
        if not (run_dir / name).exists():
            errors.append(f"missing:{name}")
    if sha256(Path(item["config"])) != item["sha256"]:
        errors.append("submitted_config_hash_changed")
    initialization = read_json(run_dir / "initialization.json")
    if item["source_checkpoint"] is None:
        if initialization.get("mode") != "random_initialization":
            errors.append("unexpected_initialization")
    elif initialization.get("checkpoint_sha256") != item["source_checkpoint_sha256"]:
        errors.append("warm_start_checkpoint_hash_mismatch")
    best = flatten_metrics(best_metrics)
    last = flatten_metrics(last_metrics)
    return {
        "experiment_id": item["experiment_id"],
        "job_id": f"10506_{task_id}",
        "slurm_state": "COMPLETED",
        "evaluation_partition": best_partition,
        "run_dir": str(run_dir),
        "config": item["config"],
        "declared_change": item["single_change_or_declared_transition"],
        "history": history,
        "best": best,
        "last": last,
        "last_minus_best_r2_casebalanced": last["physical_r2_casebalanced"] - best["physical_r2_casebalanced"],
        "postview": postview_summary(run_dir),
        "initialization": initialization,
        "integrity": "passed" if not errors else "failed",
        "integrity_errors": errors,
    }


def anchor_record(experiment_id: str, run_dir: Path, label: str) -> dict:
    partition, metrics = metric_root(run_dir / "eval/ckpt_best/metrics.json")
    errors: list[str] = []
    finite_numbers(metrics, experiment_id, errors)
    return {
        "experiment_id": experiment_id,
        "role": "historical_anchor",
        "label": label,
        "evaluation_partition": partition,
        "run_dir": str(run_dir),
        "best": flatten_metrics(metrics),
        "integrity": "passed" if not errors else "failed",
        "integrity_errors": errors,
    }


def paired_delta(records: dict[str, dict], treatment: str, control: str) -> dict:
    keys = (
        "physical_r2_casebalanced", "physical_r2_pooled", "physical_r2_case_mean",
        "physical_r2_negative_cases", "physical_mae", "physical_rmse", "high_wss_r2",
        "high_wss_mae", "spearman_case_mean", "physical_top10_iou",
        "physical_p99_amplitude_ratio", "normalized_r2_casebalanced",
    )
    return {
        "treatment": treatment,
        "control": control,
        "treatment_minus_control": {
            key: records[treatment]["best"][key] - records[control]["best"][key]
            for key in keys
        },
    }


def write_csv(records: list[dict]) -> None:
    fieldnames = (
        "experiment_id", "role", "job_id", "evaluation_partition", "best_epoch",
        "physical_r2_casebalanced", "physical_r2_pooled", "physical_r2_case_mean",
        "physical_r2_negative_cases", "physical_mae", "physical_rmse", "high_wss_r2",
        "spearman_case_mean", "physical_top10_iou", "normalized_r2_casebalanced",
        "last_minus_best_r2_casebalanced", "postview_status", "integrity",
    )
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            metric = record["best"]
            writer.writerow({
                "experiment_id": record["experiment_id"],
                "role": record.get("role", "exploratory_training"),
                "job_id": record.get("job_id", "historical"),
                "evaluation_partition": record["evaluation_partition"],
                "best_epoch": record.get("history", {}).get("best_epoch", ""),
                **{key: metric.get(key) for key in fieldnames if key in metric},
                "last_minus_best_r2_casebalanced": record.get("last_minus_best_r2_casebalanced", ""),
                "postview_status": record.get("postview", {}).get("status", "not_requested"),
                "integrity": record["integrity"],
            })


def main() -> None:
    submission = read_json(SUBMISSION)
    if submission.get("status") != "submitted" or submission.get("test27_policy") != "user-authorized historical anchor; not independent confirmation":
        raise RuntimeError("submission policy is not the authorized exploratory test27 protocol")
    records = [run_record(item, task_id) for task_id, item in enumerate(submission["configs"])]
    anchors = [
        anchor_record("Q1V-10476", Q1V, "random5000 + SAME historical control"),
        anchor_record("Q2V-10477", Q2V, "random5000 + SEP historical anchor"),
    ]
    if any(record["integrity"] != "passed" for record in records + anchors):
        raise RuntimeError("one or more artifacts failed the result integrity audit")
    by_id = {record["experiment_id"]: record for record in records + anchors}
    comparisons = {
        "r80_vs_q2v": paired_delta(by_id, "q2v_radius_r80_sep", "Q2V-10477"),
        "r60_vs_q2v": paired_delta(by_id, "q2v_radius_r60_sep", "Q2V-10477"),
        "fps_multistart_vs_q1v": paired_delta(by_id, "q1v_fpsmultistart5000_same", "Q1V-10476"),
        "warm_start_vs_q1v": paired_delta(by_id, "q2v_to_q1v_same_finetune", "Q1V-10476"),
        "warm_start_vs_q2v": paired_delta(by_id, "q2v_to_q1v_same_finetune", "Q2V-10477"),
    }
    payload = {
        "schema_version": 1,
        "analysis_date": "2026-07-18",
        "purpose": "historical test27 exploratory sampling/radius audit",
        "test27_policy": submission["test27_policy"],
        "source_submission": str(SUBMISSION),
        "source_submission_sha256": sha256(SUBMISSION),
        "job_summary": {
            "gpu_preflight_10505": "COMPLETED; 4/4 passed",
            "training_array_10506": "4/4 completed 400 epochs; best/last evaluation and PostView 27/27 passed",
        },
        "records": records,
        "anchors": anchors,
        "paired_comparisons": comparisons,
        "ranking_note": "The numeric ordering is descriptive only because test27 was reused during exploration.",
        "interpretation_limits": [
            "all comparisons are single-seed exploratory evidence on a reused historical test27",
            "r80/r60 are radius-only contrasts against Q2V; fps_multistart is a sampling-only contrast against Q1V",
            "the warm-start arm changes initialization while adopting Q1V SAME and is not a pure sampling contrast",
            "no new arm exceeded the historical Q1V physical R²_cb; high-WSS R² remains negative in every arm",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(records + anchors)
    print(json.dumps({
        "status": "passed", "records": len(records), "anchors": len(anchors),
        "json": str(OUT_JSON), "csv": str(OUT_CSV),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
