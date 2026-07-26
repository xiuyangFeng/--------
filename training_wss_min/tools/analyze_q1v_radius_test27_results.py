#!/usr/bin/env python3
"""Audit the completed Q1V/SAME historical-test27 radius-only matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PREFLIGHT = REPO / "training_wss_min/preflight"
SUBMISSION = PREFLIGHT / "q1v_radius_test27_submission.json"
GPU_PREFLIGHT = PREFLIGHT / "q1v_radius_test27_gpu_preflight.json"
OUT_JSON = PREFLIGHT / "q1v_radius_test27_results_analysis.json"
OUT_CSV = PREFLIGHT / "q1v_radius_test27_results_summary.csv"
Q1V = REPO / "training_wss_min/runs/pointnetpp_v4/outputs/ag_aaa_v4_stratified_sa3_random5000_fpscenter_same"


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
        raise RuntimeError(f"expected exactly one evaluation partition in {path}")
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
    return {"epochs": len(rows), "first_epoch": rows[0]["epoch"], "last_epoch": rows[-1]["epoch"],
            "best_epoch": best["epoch"], "best_train_loss": best["train_loss"]}


def postview_summary(run_dir: Path) -> dict:
    payload = read_json(run_dir / "postview/verification.json")
    expected, verified = payload.get("expected_cases"), payload.get("verified_cases")
    return {"status": "passed" if payload.get("status") == "passed" and expected == verified == 27 else "failed",
            "expected_cases": expected, "verified_cases": verified,
            "path": str((run_dir / "postview/verification.json").resolve())}


def run_record(item: dict, job_id: str) -> dict:
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
        if not (run_dir / name).is_file():
            errors.append(f"missing:{name}")
    if sha256(Path(item["config"])) != item["sha256"]:
        errors.append("submitted_config_hash_changed")
    initialization = read_json(run_dir / "initialization.json")
    if initialization.get("mode") != "random_initialization":
        errors.append("unexpected_initialization")
    postview = postview_summary(run_dir)
    if postview["status"] != "passed":
        errors.append("postview_verification_failed")
    best, last = flatten_metrics(best_metrics), flatten_metrics(last_metrics)
    return {
        "experiment_id": item["experiment_id"], "job_id": job_id, "slurm_state": "COMPLETED",
        "evaluation_partition": best_partition, "run_dir": str(run_dir), "config": item["config"],
        "declared_change": item["single_change"], "history": history, "best": best, "last": last,
        "last_minus_best_r2_casebalanced": last["physical_r2_casebalanced"] - best["physical_r2_casebalanced"],
        "postview": postview, "initialization": initialization,
        "integrity": "passed" if not errors else "failed", "integrity_errors": errors,
    }


def anchor_record() -> dict:
    partition, metrics = metric_root(Q1V / "eval/ckpt_best/metrics.json")
    errors: list[str] = []
    finite_numbers(metrics, "Q1V-10476", errors)
    return {"experiment_id": "Q1V-10476", "role": "historical_anchor", "evaluation_partition": partition,
            "run_dir": str(Q1V), "best": flatten_metrics(metrics),
            "integrity": "passed" if not errors else "failed", "integrity_errors": errors}


def paired_delta(treatment: dict, control: dict) -> dict:
    keys = ("physical_r2_casebalanced", "physical_r2_pooled", "physical_r2_case_mean",
            "physical_r2_negative_cases", "physical_mae", "physical_rmse", "high_wss_r2",
            "high_wss_mae", "spearman_case_mean", "physical_top10_iou",
            "physical_p99_amplitude_ratio", "normalized_r2_casebalanced")
    return {key: treatment["best"][key] - control["best"][key] for key in keys}


def write_csv(records: list[dict]) -> None:
    fields = ("experiment_id", "role", "job_id", "evaluation_partition", "best_epoch",
              "physical_r2_casebalanced", "physical_r2_pooled", "physical_r2_case_mean",
              "physical_r2_negative_cases", "physical_mae", "physical_rmse", "high_wss_r2",
              "spearman_case_mean", "physical_top10_iou", "normalized_r2_casebalanced",
              "last_minus_best_r2_casebalanced", "postview_status", "integrity")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metric = record["best"]
            writer.writerow({"experiment_id": record["experiment_id"], "role": record.get("role", "exploratory_training"),
                             "job_id": record.get("job_id", "historical"), "evaluation_partition": record["evaluation_partition"],
                             "best_epoch": record.get("history", {}).get("best_epoch", ""),
                             **{key: metric.get(key) for key in fields if key in metric},
                             "last_minus_best_r2_casebalanced": record.get("last_minus_best_r2_casebalanced", ""),
                             "postview_status": record.get("postview", {}).get("status", "not_requested"),
                             "integrity": record["integrity"]})


def main() -> None:
    submission, preflight = read_json(SUBMISSION), read_json(GPU_PREFLIGHT)
    if submission.get("status") != "submitted" or submission.get("test27_policy") != "user-authorized historical anchor; not independent confirmation":
        raise RuntimeError("submission is not the authorized exploratory Q1V radius protocol")
    if preflight.get("status") != "passed" or len(preflight.get("jobs", [])) != 3:
        raise RuntimeError("formal GPU preflight did not pass all three configs")
    train_job = next(item["job_id"] for item in submission["jobs"] if item["role"] == "training_test27_exploratory_array")
    records = [run_record(item, f"{train_job}_{index}") for index, item in enumerate(submission["configs"])]
    anchor = anchor_record()
    if any(record["integrity"] != "passed" for record in records + [anchor]):
        raise RuntimeError("one or more Q1V radius artifacts failed integrity audit")
    deltas = {record["experiment_id"]: paired_delta(record, anchor) for record in records}
    winner = max(records, key=lambda record: record["best"]["physical_r2_casebalanced"])
    payload = {
        "schema_version": 1, "analysis_date": "2026-07-19",
        "purpose": "Q1V-10476 SAME anchored historical-test27 radius-only audit",
        "test27_policy": submission["test27_policy"], "source_submission": str(SUBMISSION),
        "source_submission_sha256": sha256(SUBMISSION), "source_gpu_preflight": str(GPU_PREFLIGHT),
        "source_gpu_preflight_sha256": sha256(GPU_PREFLIGHT),
        "job_summary": {"gpu_preflight": "10545 COMPLETED; 3/3 passed", "training_array": f"{train_job} 3/3 COMPLETED; 400 epochs, best/last evaluation, PostView 27/27 passed"},
        "records": records, "anchor": anchor, "paired_vs_q1v": deltas,
        "descriptive_best_by_physical_r2_casebalanced": winner["experiment_id"],
        "ranking_note": "Descriptive ordering only: test27 was reused during exploratory work and cannot select a final radius.",
        "interpretation_limits": [
            "all arms are single-seed, historical-test27 exploratory evidence",
            "each new arm changes only the three SA radii relative to Q1V SAME",
            "any apparent improvement needs a pre-specified independent repeat before it can change the Q1V protocol",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(records + [anchor])
    print(json.dumps({"status": "passed", "records": len(records), "winner": winner["experiment_id"],
                      "json": str(OUT_JSON), "csv": str(OUT_CSV)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
