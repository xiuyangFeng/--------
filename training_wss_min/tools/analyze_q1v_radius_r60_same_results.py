#!/usr/bin/env python3
"""Audit the completed Q1V/SAME 0.6x-radius historical-test27 follow-up."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training_wss_min.tools.analyze_q1v_radius_test27_results import (
    finite_numbers, flatten_metrics, history_summary, metric_root, postview_summary, sha256,
)


SUBMISSION = REPO / "training_wss_min/preflight/q1v_radius_r60_same_submission.json"
GPU_PREFLIGHT = REPO / "training_wss_min/preflight/q1v_radius_r60_same_gpu_preflight.json"
OUT_JSON = REPO / "training_wss_min/preflight/q1v_radius_r60_same_results_analysis.json"
OUT_CSV = REPO / "training_wss_min/preflight/q1v_radius_r60_same_results_summary.csv"
Q1V = REPO / "training_wss_min/runs/pointnetpp_v4/outputs/ag_aaa_v4_stratified_sa3_random5000_fpscenter_same"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def anchor_record() -> dict:
    partition, metrics = metric_root(Q1V / "eval/ckpt_best/metrics.json")
    errors: list[str] = []
    finite_numbers(metrics, "Q1V-10476", errors)
    return {"experiment_id": "Q1V-10476", "role": "historical_anchor", "evaluation_partition": partition,
            "run_dir": str(Q1V), "best": flatten_metrics(metrics),
            "integrity": "passed" if not errors else "failed", "integrity_errors": errors}


def result_record(submission: dict, preflight: dict) -> dict:
    run_dir, config = Path(submission["run_dir"]), Path(submission["config"])
    errors: list[str] = []
    best_partition, best_metrics = metric_root(run_dir / "eval/ckpt_best/metrics.json")
    last_partition, last_metrics = metric_root(run_dir / "eval/ckpt_last/metrics.json")
    history = history_summary(run_dir)
    finite_numbers(best_metrics, "q1v_radius_r60_same.best", errors)
    finite_numbers(last_metrics, "q1v_radius_r60_same.last", errors)
    if preflight.get("status") != "passed" or len(preflight.get("jobs", [])) != 1:
        errors.append("formal_gpu_preflight_not_passed")
    elif preflight["jobs"][0].get("config_sha256") != submission["config_sha256"]:
        errors.append("preflight_config_hash_mismatch")
    if sha256(config) != submission["config_sha256"]:
        errors.append("submitted_config_hash_changed")
    if (history["epochs"], history["first_epoch"], history["last_epoch"]) != (400, 0, 399):
        errors.append("history_not_400_epochs")
    if best_partition != last_partition or best_partition != "test":
        errors.append("unexpected_evaluation_partition")
    for name in ("ckpt_best.pt", "ckpt_last.pt", "eval/best_last_comparison.json", "postview/verification.json"):
        if not (run_dir / name).is_file():
            errors.append(f"missing:{name}")
    initialization = read_json(run_dir / "initialization.json")
    if initialization.get("mode") != "random_initialization":
        errors.append("unexpected_initialization")
    postview = postview_summary(run_dir)
    if postview["status"] != "passed":
        errors.append("postview_verification_failed")
    best, last = flatten_metrics(best_metrics), flatten_metrics(last_metrics)
    train_job = next(item["job_id"] for item in submission["jobs"] if item["role"] == "training_test27_exploratory")
    return {"experiment_id": submission["experiment_id"], "job_id": train_job, "slurm_state": "COMPLETED",
            "evaluation_partition": best_partition, "run_dir": str(run_dir), "config": str(config),
            "declared_change": submission["single_change"], "history": history, "best": best, "last": last,
            "last_minus_best_r2_casebalanced": last["physical_r2_casebalanced"] - best["physical_r2_casebalanced"],
            "postview": postview, "initialization": initialization,
            "integrity": "passed" if not errors else "failed", "integrity_errors": errors}


def delta(treatment: dict, control: dict) -> dict:
    keys = ("physical_r2_casebalanced", "physical_r2_pooled", "physical_r2_case_mean",
            "physical_r2_negative_cases", "physical_mae", "physical_rmse", "high_wss_r2",
            "high_wss_mae", "spearman_case_mean", "physical_top10_iou",
            "physical_p99_amplitude_ratio", "normalized_r2_casebalanced")
    return {key: treatment["best"][key] - control["best"][key] for key in keys}


def main() -> None:
    submission, preflight = read_json(SUBMISSION), read_json(GPU_PREFLIGHT)
    if submission.get("status") != "submitted" or submission.get("test27_policy") != "user-authorized historical anchor; not independent confirmation":
        raise RuntimeError("submission is not the authorized exploratory Q1V 0.6x protocol")
    record, anchor = result_record(submission, preflight), anchor_record()
    if record["integrity"] != "passed" or anchor["integrity"] != "passed":
        raise RuntimeError("the Q1V 0.6x follow-up failed result integrity audit")
    payload = {
        "schema_version": 1, "analysis_date": "2026-07-20",
        "purpose": "Q1V-10476 SAME anchored 0.6x-radius historical-test27 follow-up audit",
        "test27_policy": submission["test27_policy"], "source_submission": str(SUBMISSION),
        "source_submission_sha256": sha256(SUBMISSION), "source_gpu_preflight": str(GPU_PREFLIGHT),
        "source_gpu_preflight_sha256": sha256(GPU_PREFLIGHT),
        "job_summary": "10555 preflight passed; 10556 completed 400 epochs, best/last test evaluation, and PostView 27/27",
        "record": record, "anchor": anchor, "paired_vs_q1v": delta(record, anchor),
        "ranking_note": "This is single-seed exploratory evidence on a reused historical test27 and cannot select a final radius.",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ("experiment_id", "job_id", "physical_r2_casebalanced", "physical_r2_pooled",
              "physical_r2_case_mean", "physical_r2_negative_cases", "physical_mae", "physical_rmse",
              "high_wss_r2", "physical_top10_iou", "normalized_r2_casebalanced",
              "last_minus_best_r2_casebalanced", "postview_status", "integrity")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerow({"experiment_id": record["experiment_id"], "job_id": record["job_id"],
                         **{key: record["best"].get(key) for key in fields if key in record["best"]},
                         "last_minus_best_r2_casebalanced": record["last_minus_best_r2_casebalanced"],
                         "postview_status": record["postview"]["status"], "integrity": record["integrity"]})
    print(json.dumps({"status": "passed", "json": str(OUT_JSON), "csv": str(OUT_CSV)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
