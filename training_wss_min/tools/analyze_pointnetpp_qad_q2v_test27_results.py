#!/usr/bin/env python3
"""Audit and analyze the exact-Q2V interpolate/QAD historical-test27 matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training_wss_min.config import ExpConfig


PREFLIGHT = REPO / "training_wss_min/preflight"
PREPARED = PREFLIGHT / "pointnetpp_qad_q2v_test27_prepared.json"
SUBMISSION = PREFLIGHT / "pointnetpp_qad_q2v_test27_submission.json"
OUT_JSON = PREFLIGHT / "pointnetpp_qad_q2v_test27_results_analysis.json"
OUT_SUMMARY = PREFLIGHT / "pointnetpp_qad_q2v_test27_results_summary.csv"
OUT_SEEDS = PREFLIGHT / "pointnetpp_qad_q2v_test27_seed_deltas.csv"
OUT_CASES = PREFLIGHT / "pointnetpp_qad_q2v_test27_per_case_deltas.csv"
SACCT = Path("/public/slurm/bin/sacct")
SEEDS = (1234, 7, 2025)
BOOTSTRAP_DRAWS = 20_000


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


def metric_root(path: Path) -> dict:
    payload = read_json(path)
    if set(payload) != {"test"}:
        raise RuntimeError(f"expected test-only metrics: {path}")
    return payload["test"]


def flatten(metrics: dict) -> dict:
    physical = metrics
    normalized = metrics["normalized"]
    physical_nmae = physical["field"].get("nmae_range")
    if physical_nmae is None:
        physical_nmae = physical["field"]["mae"] * physical["field"]["nrmse_range"] / physical["field"]["rmse"]
    normalized_nmae = normalized["field"].get("nmae_range")
    if normalized_nmae is None:
        normalized_nmae = normalized["field"]["mae"] * normalized["field"]["nrmse_range"] / normalized["field"]["rmse"]
    return {
        "physical_r2_casebalanced": physical["field_casebalanced"]["r2"],
        "physical_r2_pooled": physical["field"]["r2"],
        "physical_r2_case_mean": physical["aggregate"]["r2_casemean"],
        "physical_r2_case_median": physical["aggregate"]["r2_casemed"],
        "physical_r2_case_p10": physical["aggregate"]["r2_casep10"],
        "physical_r2_negative_cases": physical["aggregate"]["r2_negative_cases"],
        "physical_mae": physical["field"]["mae"],
        "physical_rmse": physical["field"]["rmse"],
        "physical_nrmse_range_pooled": physical["field"]["nrmse_range"],
        "physical_nrmse_range_case_mean": physical["aggregate"]["nrmse_casemean"],
        "physical_nmae_pooled": physical_nmae,
        "physical_nmae_case_mean": physical["aggregate"].get("nmae_casemean"),
        "high_wss_r2": physical["regional_field"]["high_wss"]["r2"],
        "high_wss_mae": physical["regional_field"]["high_wss"]["mae"],
        "physical_top10_amplitude_ratio": physical["calibration"]["top10_pred_true_ratio"],
        "physical_p99_amplitude_ratio": physical["calibration"]["p99_pred_true_ratio"],
        "physical_top10_iou": physical["hotspot"]["top10_iou_casemean"],
        "normalized_r2_casebalanced": normalized["field_casebalanced"]["r2"],
        "normalized_r2_pooled": normalized["field"]["r2"],
        "normalized_r2_case_mean": normalized["aggregate"]["r2_casemean"],
        "normalized_r2_case_median": normalized["aggregate"]["r2_casemed"],
        "normalized_r2_case_p10": normalized["aggregate"]["r2_casep10"],
        "normalized_r2_negative_cases": normalized["aggregate"]["r2_negative_cases"],
        "normalized_mae": normalized["field"]["mae"],
        "normalized_rmse": normalized["field"]["rmse"],
        "normalized_nrmse_range_pooled": normalized["field"]["nrmse_range"],
        "normalized_nrmse_range_case_mean": normalized["aggregate"]["nrmse_casemean"],
        "normalized_nmae_pooled": normalized_nmae,
        "normalized_nmae_case_mean": normalized["aggregate"].get("nmae_casemean"),
        "spearman_case_mean": normalized["hotspot"]["spearman_all_casemean"],
        "high_wss_spearman_case_mean": normalized["hotspot"]["spearman_high_wss_casemean"],
        "normalized_top10_iou": normalized["hotspot"]["top10_iou_casemean"],
        "normalized_top10_amplitude_ratio": normalized["calibration"]["top10_pred_true_ratio"],
        "normalized_p99_amplitude_ratio": normalized["calibration"]["p99_pred_true_ratio"],
        "parameters": metrics.get("efficiency", {}).get("parameters"),
        "physical_groups": physical.get("group_casebalanced", {}),
        "normalized_groups": normalized.get("group_casebalanced", {}),
    }


def history_summary(run_dir: Path) -> dict:
    rows = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines() if line]
    best = min(rows, key=lambda row: row["train_loss"])
    return {
        "epochs": len(rows), "first_epoch": rows[0]["epoch"], "last_epoch": rows[-1]["epoch"],
        "best_epoch": best["epoch"], "best_train_loss": best["train_loss"],
    }


def per_case(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for row in rows:
        case = row["case"]
        if case in out:
            raise RuntimeError(f"duplicate per-case row {case}: {path}")
        out[case] = {
            "normalized_r2": float(row["normalized_overall_r2"]),
            "physical_r2": float(row["physical_overall_r2"]),
            "normalized_spearman": float(row["normalized_hotspot_spearman_all"]),
            "normalized_top10_iou": float(row["normalized_hotspot_top10_iou"]),
            "normalized_p99_ratio": float(row["normalized_calibration_p99_pred_true_ratio"]),
            "physical_mae": float(row["physical_overall_mae"]),
            "physical_rmse": float(row["physical_overall_rmse"]),
            "physical_nmae": float(row["physical_overall_mae"]) / float(row["physical_distribution_true_dynamic_range"]),
            "normalized_nmae": float(row["normalized_overall_mae"]) / float(row["normalized_distribution_true_dynamic_range"]),
        }
    if len(out) != 27:
        raise RuntimeError(f"expected 27 per-case rows, got {len(out)}: {path}")
    return out


def slurm_states() -> dict[int, dict]:
    proc = subprocess.run(
        [str(SACCT), "-j", "10550", "-X", "--noheader", "--format=JobID,State,ExitCode", "--parsable2"],
        text=True, capture_output=True, check=True,
    )
    out = {}
    for line in proc.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 3 or "_" not in fields[0]:
            continue
        suffix = fields[0].rsplit("_", 1)[-1]
        if suffix.isdigit():
            out[int(suffix)] = {"state": fields[1], "exit_code": fields[2]}
    return out


def make_record(row: dict, run_dir: Path, job_id: str, slurm: dict, anchor: bool = False) -> dict:
    errors = []
    required = (
        "config.json", "history.jsonl", "ckpt_best.pt", "ckpt_last.pt",
        "eval/ckpt_best/metrics.json", "eval/ckpt_last/metrics.json",
        "eval/ckpt_best/per_case_metrics.csv", "eval/ckpt_last/per_case_metrics.csv",
        "eval/best_last_comparison.json",
    )
    for rel in required:
        if not (run_dir / rel).is_file():
            errors.append(f"missing:{rel}")
    config = Path(row["config"])
    expected_hash = row["config_sha256"] if anchor else row["sha256"]
    if sha256(config) != expected_hash:
        errors.append("config_hash_changed")
    runtime_cfg = run_dir / "config.json"
    source_payload = ExpConfig.from_json(config).to_dict()
    runtime_payload = ExpConfig.from_json(runtime_cfg).to_dict()
    for key in ("curv_q02", "curv_q98", "invr_q02", "invr_q98", "y_norm_q02", "y_norm_q98"):
        source_payload["train"].pop(key, None)
        runtime_payload["train"].pop(key, None)
    if source_payload != runtime_payload:
        errors.append("runtime_config_differs_from_submission")
    history = history_summary(run_dir)
    if (history["epochs"], history["first_epoch"], history["last_epoch"]) != (400, 0, 399):
        errors.append("history_not_400_epochs")
    best_raw = metric_root(run_dir / "eval/ckpt_best/metrics.json")
    last_raw = metric_root(run_dir / "eval/ckpt_last/metrics.json")
    finite_numbers(best_raw, f"{row['experiment_id']}.best", errors)
    finite_numbers(last_raw, f"{row['experiment_id']}.last", errors)
    best, last = flatten(best_raw), flatten(last_raw)
    cases = per_case(run_dir / "eval/ckpt_best/per_case_metrics.csv")
    last_cases = per_case(run_dir / "eval/ckpt_last/per_case_metrics.csv")
    if best["physical_nmae_case_mean"] is None:
        best["physical_nmae_case_mean"] = float(np.mean([item["physical_nmae"] for item in cases.values()]))
    if best["normalized_nmae_case_mean"] is None:
        best["normalized_nmae_case_mean"] = float(np.mean([item["normalized_nmae"] for item in cases.values()]))
    if last["physical_nmae_case_mean"] is None:
        last["physical_nmae_case_mean"] = float(np.mean([item["physical_nmae"] for item in last_cases.values()]))
    if last["normalized_nmae_case_mean"] is None:
        last["normalized_nmae_case_mean"] = float(np.mean([item["normalized_nmae"] for item in last_cases.values()]))
    if not anchor and (slurm.get("state"), slurm.get("exit_code")) != ("COMPLETED", "0:0"):
        errors.append(f"slurm:{slurm}")
    if errors:
        raise RuntimeError(f"{row['experiment_id']}: {errors}")
    return {
        "experiment_id": row["experiment_id"], "seed": int(row["seed"]),
        "decoder": row["decoder"], "job_id": job_id, "slurm": slurm,
        "run_dir": str(run_dir), "config": str(config), "config_sha256": expected_hash,
        "history": history, "best": best, "last": last,
        "last_minus_best": {
            "normalized_r2_casebalanced": last["normalized_r2_casebalanced"] - best["normalized_r2_casebalanced"],
            "physical_r2_casebalanced": last["physical_r2_casebalanced"] - best["physical_r2_casebalanced"],
        },
        "per_case": cases,
    }


def paired_stats(values: np.ndarray, rng: np.random.Generator) -> dict:
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(axis=1)
    nonzero = values[np.abs(values) > 1e-15]
    p_value = float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
        "wins": int((values > 1e-15).sum()), "ties": int((np.abs(values) <= 1e-15).sum()),
        "losses": int((values < -1e-15).sum()), "wilcoxon_p": p_value,
    }


PAIR_KEYS = (
    "normalized_r2_casebalanced", "physical_r2_casebalanced", "physical_mae", "physical_rmse",
    "spearman_case_mean", "normalized_top10_iou", "normalized_p99_amplitude_ratio",
    "high_wss_r2", "high_wss_spearman_case_mean",
)
CASE_KEYS = (
    "normalized_r2", "physical_r2", "normalized_spearman", "normalized_top10_iou",
    "normalized_p99_ratio", "physical_mae", "physical_rmse",
)


def mean_metrics(records: list[dict]) -> dict:
    keys = [key for key, value in records[0]["best"].items() if isinstance(value, (int, float)) and value is not None]
    return {key: float(np.mean([record["best"][key] for record in records])) for key in keys}


def main() -> None:
    prepared, submission = read_json(PREPARED), read_json(SUBMISSION)
    states = slurm_states()
    if len(states) != 5:
        raise RuntimeError(f"expected five Slurm array states, got {states}")
    records = []
    anchor = prepared["anchor"]
    anchor_row = {
        "experiment_id": "Q2V-10477", "seed": 1234, "decoder": "interpolate",
        "config": anchor["config"], "config_sha256": anchor["config_sha256"],
    }
    records.append(make_record(anchor_row, Path(anchor["run_dir"]), "10477", {"state": "COMPLETED", "exit_code": "0:0"}, True))
    for row in prepared["configs"]:
        records.append(make_record(row, Path(row["expected_run_dir"]), f"10550_{row['task_id']}", states[row["task_id"]]))

    by_pair = {(record["seed"], record["decoder"]): record for record in records}
    rng = np.random.default_rng(20260719)
    seed_pairs = []
    case_delta_rows = []
    for seed in SEEDS:
        control, treatment = by_pair[(seed, "interpolate")], by_pair[(seed, "qad_lite")]
        delta = {key: treatment["best"][key] - control["best"][key] for key in PAIR_KEYS}
        cases = sorted(control["per_case"])
        if cases != sorted(treatment["per_case"]):
            raise RuntimeError(f"per-case IDs differ for seed {seed}")
        stats = {}
        for key in CASE_KEYS:
            values = np.array([treatment["per_case"][case][key] - control["per_case"][case][key] for case in cases])
            stats[key] = paired_stats(values, rng)
            for case, value in zip(cases, values):
                case_delta_rows.append({"seed": seed, "case": case, "metric": key, "delta": float(value)})
        seed_pairs.append({
            "seed": seed, "control": control["experiment_id"], "treatment": treatment["experiment_id"],
            "delta": delta, "paired_case_stats": stats,
        })

    controls = [by_pair[(seed, "interpolate")] for seed in SEEDS]
    treatments = [by_pair[(seed, "qad_lite")] for seed in SEEDS]
    control_mean, qad_mean = mean_metrics(controls), mean_metrics(treatments)
    cross_seed = {}
    for key in PAIR_KEYS:
        values = np.array([pair["delta"][key] for pair in seed_pairs])
        cross_seed[key] = {
            "mean_delta": float(values.mean()), "median_delta": float(np.median(values)),
            "min_delta": float(values.min()), "max_delta": float(values.max()),
            "positive_seeds": int((values > 0).sum()), "negative_seeds": int((values < 0).sum()),
            "values_by_seed": {str(seed): float(value) for seed, value in zip(SEEDS, values)},
        }

    averaged_case_stats = {}
    cases = sorted(controls[0]["per_case"])
    for key in CASE_KEYS:
        values = np.array([
            np.mean([by_pair[(seed, "qad_lite")]["per_case"][case][key] - by_pair[(seed, "interpolate")]["per_case"][case][key] for seed in SEEDS])
            for case in cases
        ])
        averaged_case_stats[key] = paired_stats(values, rng)

    gates = {
        "prior_normalized_r2_primary": {
            "rule": "continuity check only: mean delta >= +0.015 and at least 2/3 seeds positive",
            "mean_delta": cross_seed["normalized_r2_casebalanced"]["mean_delta"],
            "positive_seeds": cross_seed["normalized_r2_casebalanced"]["positive_seeds"],
            "passed": cross_seed["normalized_r2_casebalanced"]["mean_delta"] >= 0.015 and cross_seed["normalized_r2_casebalanced"]["positive_seeds"] >= 2,
        },
        "spearman_noninferiority": {
            "rule": "no seed delta < -0.01", "passed": min(pair["delta"]["spearman_case_mean"] for pair in seed_pairs) >= -0.01,
        },
        "top10_iou_noninferiority": {
            "rule": "no seed delta < -0.01", "passed": min(pair["delta"]["normalized_top10_iou"] for pair in seed_pairs) >= -0.01,
        },
        "physical_rmse_noninferiority": {
            "rule": "no seed relative worsening > 1%",
            "passed": all(pair["delta"]["physical_rmse"] / by_pair[(pair["seed"], "interpolate")]["best"]["physical_rmse"] <= 0.01 for pair in seed_pairs),
        },
        "tail_direction": {
            "rule": "normalized p99 ratio should not regress across the majority of seeds",
            "passed": sum(pair["delta"]["normalized_p99_amplitude_ratio"] >= 0 for pair in seed_pairs) >= 2,
        },
    }
    result = {
        "schema_version": 1, "analysis_date": "2026-07-19",
        "source_prepared": str(PREPARED), "source_prepared_sha256": sha256(PREPARED),
        "source_submission": str(SUBMISSION), "source_submission_sha256": sha256(SUBMISSION),
        "jobs": {"10549": "COMPLETED (0:0); 5/5 GPU preflight passed", "10550": "5/5 COMPLETED (0:0)"},
        "comparison_scope": "user-authorized exact-Q2V historical test27 comparison; not independent confirmation",
        "records": records, "seed_pairs": seed_pairs,
        "control_three_seed_mean": control_mean, "qad_three_seed_mean": qad_mean,
        "control_seed_std": {key: float(np.std([record["best"][key] for record in controls])) for key in PAIR_KEYS},
        "qad_seed_std": {key: float(np.std([record["best"][key] for record in treatments])) for key in PAIR_KEYS},
        "cross_seed_deltas": cross_seed, "averaged_case_stats": averaged_case_stats,
        "gates": gates, "all_gates_passed": all(item["passed"] for item in gates.values()),
        "decision": "NO_GO_exact_Q2V; QAD does not improve historical test27 under the matched protocol",
        "interpretation": [
            "QAD regresses normalized and physical R2_cb strongly for seed1234",
            "seeds 7 and 2025 show only small or mixed normalized/physical changes",
            "the three-seed mean normalized and physical R2_cb both regress",
            "top10 IoU improves on average, but RMSE and p99 tail direction are not stable",
            "QAD compresses normalized R2 seed dispersion at a lower mean, consistent with a bias/underfit tradeoff rather than a robust gain",
            "test27 is a historical development benchmark and cannot support an independent confirmation claim",
        ],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_keys = [
        "physical_r2_casebalanced", "physical_r2_pooled", "physical_r2_case_mean", "physical_rmse", "physical_mae",
        "high_wss_r2", "physical_top10_iou", "physical_p99_amplitude_ratio",
        "normalized_r2_casebalanced", "normalized_r2_pooled", "normalized_r2_case_mean", "normalized_rmse", "normalized_mae",
        "spearman_case_mean", "high_wss_spearman_case_mean", "normalized_top10_iou", "normalized_p99_amplitude_ratio",
    ]
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_id", "seed", "decoder", "job_id", *summary_keys])
        writer.writeheader()
        for record in records:
            writer.writerow({"experiment_id": record["experiment_id"], "seed": record["seed"], "decoder": record["decoder"], "job_id": record["job_id"], **{key: record["best"][key] for key in summary_keys}})
    with OUT_SEEDS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", *PAIR_KEYS])
        writer.writeheader()
        for pair in seed_pairs:
            writer.writerow({"seed": pair["seed"], **pair["delta"]})
    with OUT_CASES.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "case", "metric", "delta"])
        writer.writeheader()
        writer.writerows(case_delta_rows)
    print(json.dumps({
        "status": "passed", "records": len(records), "decision": result["decision"],
        "mean_normalized_r2_delta": cross_seed["normalized_r2_casebalanced"]["mean_delta"],
        "mean_physical_r2_delta": cross_seed["physical_r2_casebalanced"]["mean_delta"],
        "json": str(OUT_JSON), "summary_csv": str(OUT_SUMMARY),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
