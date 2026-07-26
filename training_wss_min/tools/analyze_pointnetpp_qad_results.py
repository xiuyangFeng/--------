#!/usr/bin/env python3
"""Audit and analyze the paired PointNet++ R0/R1 QAD val21 matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


REPO = Path(__file__).resolve().parents[2]
PREFLIGHT = REPO / "training_wss_min/preflight"
SUBMISSION = PREFLIGHT / "pointnetpp_qad_matrix_submission.json"
OUT_JSON = PREFLIGHT / "pointnetpp_qad_results_analysis.json"
OUT_SUMMARY = PREFLIGHT / "pointnetpp_qad_results_summary.csv"
OUT_SEEDS = PREFLIGHT / "pointnetpp_qad_seed_deltas.csv"
OUT_CASES = PREFLIGHT / "pointnetpp_qad_per_case_deltas.csv"
ANCHOR_RUN = REPO / (
    "training_wss_min/runs/pointnetpp_q2v_arch_dev/outputs/"
    "q2v_arch_dev_n32_w32"
)
SBATCH_SACCT = Path("/public/slurm/bin/sacct")
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


def metric_root(path: Path) -> dict:
    payload = read_json(path)
    if set(payload) != {"val"}:
        raise RuntimeError(f"expected val-only metrics: {path}")
    return payload["val"]


def finite_numbers(value, where: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            finite_numbers(child, f"{where}.{key}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_numbers(child, f"{where}[{index}]", errors)
    elif isinstance(value, float) and not math.isfinite(value):
        errors.append(f"nonfinite:{where}")


def flatten(metrics: dict) -> dict:
    physical = metrics
    normalized = metrics["normalized"]
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
        "physical_nmae_pooled": physical["field"].get("nmae_range"),
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
        "normalized_nmae_pooled": normalized["field"].get("nmae_range"),
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
        "epochs": len(rows), "first_epoch": rows[0]["epoch"],
        "last_epoch": rows[-1]["epoch"], "best_epoch": best["epoch"],
        "best_train_loss": best["train_loss"],
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
        }
    if len(out) != 21:
        raise RuntimeError(f"expected 21 per-case rows, got {len(out)}: {path}")
    return out


def slurm_states() -> dict[int, dict]:
    proc = subprocess.run(
        [str(SBATCH_SACCT), "-j", "10540", "-X", "--noheader",
         "--format=JobID,State,ExitCode", "--parsable2"],
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


def record(experiment_id: str, run_dir: Path, config: Path, expected_hash: str,
           seed: int, decoder: str, job_id: str, slurm: dict) -> dict:
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
    if sha256(config) != expected_hash:
        errors.append("submitted_config_hash_changed")
    history = history_summary(run_dir)
    if (history["epochs"], history["first_epoch"], history["last_epoch"]) != (400, 0, 399):
        errors.append("history_not_400_epochs")
    best_raw = metric_root(run_dir / "eval/ckpt_best/metrics.json")
    last_raw = metric_root(run_dir / "eval/ckpt_last/metrics.json")
    finite_numbers(best_raw, f"{experiment_id}.best", errors)
    finite_numbers(last_raw, f"{experiment_id}.last", errors)
    best, last = flatten(best_raw), flatten(last_raw)
    cases = per_case(run_dir / "eval/ckpt_best/per_case_metrics.csv")
    if slurm.get("state") != "COMPLETED" or slurm.get("exit_code") != "0:0":
        errors.append(f"slurm:{slurm}")
    return {
        "experiment_id": experiment_id, "seed": seed, "decoder": decoder,
        "job_id": job_id, "slurm": slurm, "run_dir": str(run_dir),
        "config": str(config), "config_sha256": expected_hash,
        "history": history, "best": best, "last": last,
        "last_minus_best": {
            "normalized_r2_casebalanced": last["normalized_r2_casebalanced"] - best["normalized_r2_casebalanced"],
            "physical_r2_casebalanced": last["physical_r2_casebalanced"] - best["physical_r2_casebalanced"],
        },
        "per_case": cases,
        "integrity": "passed" if not errors else "failed",
        "integrity_errors": errors,
    }


def bootstrap(values: np.ndarray, seed: int = 20260719) -> dict:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def paired_stats(control: dict, treatment: dict) -> dict:
    if set(control) != set(treatment):
        raise RuntimeError("per-case sets differ")
    cases = sorted(control)
    metrics = tuple(next(iter(control.values())))
    out = {}
    for metric in metrics:
        values = np.array([treatment[c][metric] - control[c][metric] for c in cases])
        stats = bootstrap(values)
        stats.update({
            "wins": int((values > 0).sum()), "ties": int((values == 0).sum()),
            "losses": int((values < 0).sum()),
            "wilcoxon_p": float(wilcoxon(values).pvalue) if np.any(values) else 1.0,
        })
        out[metric] = stats
    return out


def metric_delta(treatment: dict, control: dict) -> dict:
    keys = (
        "normalized_r2_casebalanced", "physical_r2_casebalanced",
        "physical_mae", "physical_rmse", "spearman_case_mean",
        "normalized_top10_iou", "normalized_p99_amplitude_ratio",
        "high_wss_r2", "high_wss_spearman_case_mean",
    )
    return {key: treatment[key] - control[key] for key in keys}


def write_csvs(records: list[dict], seed_pairs: list[dict], averaged_cases: list[dict]) -> None:
    fields = [
        "experiment_id", "seed", "decoder", "job_id", "best_epoch", "best_train_loss",
        "physical_r2_casebalanced", "physical_r2_pooled", "physical_r2_case_mean",
        "physical_r2_negative_cases", "physical_mae", "physical_rmse", "high_wss_r2",
        "normalized_r2_casebalanced", "normalized_r2_pooled", "normalized_r2_case_mean",
        "normalized_r2_negative_cases", "normalized_mae", "normalized_rmse",
        "spearman_case_mean", "normalized_top10_iou", "normalized_p99_amplitude_ratio",
        "parameters", "integrity",
    ]
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            writer.writerow({
                "experiment_id": item["experiment_id"], "seed": item["seed"],
                "decoder": item["decoder"], "job_id": item["job_id"],
                "best_epoch": item["history"]["best_epoch"],
                "best_train_loss": item["history"]["best_train_loss"],
                **{key: item["best"].get(key) for key in fields if key in item["best"]},
                "integrity": item["integrity"],
            })
    delta_fields = ["seed", "control", "treatment", *seed_pairs[0]["delta"]]
    with OUT_SEEDS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=delta_fields)
        writer.writeheader()
        for item in seed_pairs:
            writer.writerow({
                "seed": item["seed"], "control": item["control"],
                "treatment": item["treatment"], **item["delta"],
            })
    with OUT_CASES.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(averaged_cases[0]))
        writer.writeheader(); writer.writerows(averaged_cases)


def main() -> None:
    submission = read_json(SUBMISSION)
    if submission.get("status") != "submitted" or submission.get("job", {}).get("job_id") != "10540":
        raise RuntimeError("unexpected submission manifest")
    states = slurm_states()
    if set(states) != set(range(5)):
        raise RuntimeError(f"missing Slurm task states: {states}")

    records = []
    for item in submission["configs"]:
        task = item["task_id"]
        records.append(record(
            item["experiment_id"], Path(item["expected_run_dir"]), Path(item["config"]),
            item["sha256"], item["seed"], item["decoder"], f"10540_{task}", states[task],
        ))
    anchor_config = Path(submission["anchor_config"])
    anchor = record(
        "r0_interpolate_n32_w32_seed1234", ANCHOR_RUN, anchor_config,
        submission["anchor_config_sha256"], 1234, "interpolate", "10488_2",
        {"state": "COMPLETED", "exit_code": "0:0"},
    )
    records.append(anchor)
    if any(item["integrity"] != "passed" for item in records):
        raise RuntimeError({item["experiment_id"]: item["integrity_errors"] for item in records})

    by_key = {(item["seed"], item["decoder"]): item for item in records}
    seed_pairs = []
    for seed in SEEDS:
        control = by_key[(seed, "interpolate")]
        treatment = by_key[(seed, "qad_lite")]
        seed_pairs.append({
            "seed": seed, "control": control["experiment_id"],
            "treatment": treatment["experiment_id"],
            "delta": metric_delta(treatment["best"], control["best"]),
            "paired_case_stats": paired_stats(control["per_case"], treatment["per_case"]),
        })

    metric_keys = tuple(seed_pairs[0]["delta"])
    cross_seed = {}
    for metric in metric_keys:
        values = np.array([item["delta"][metric] for item in seed_pairs])
        cross_seed[metric] = {
            "mean_delta": float(values.mean()), "median_delta": float(np.median(values)),
            "min_delta": float(values.min()), "max_delta": float(values.max()),
            "positive_seeds": int((values > 0).sum()), "negative_seeds": int((values < 0).sum()),
            "values_by_seed": {str(item["seed"]): item["delta"][metric] for item in seed_pairs},
        }

    cases = sorted(anchor["per_case"])
    averaged_cases = []
    for case in cases:
        row = {"case": case}
        for metric in anchor["per_case"][case]:
            deltas = np.array([
                by_key[(seed, "qad_lite")]["per_case"][case][metric]
                - by_key[(seed, "interpolate")]["per_case"][case][metric]
                for seed in SEEDS
            ])
            row[f"mean_delta_{metric}"] = float(deltas.mean())
        averaged_cases.append(row)
    averaged_case_stats = {}
    for metric in anchor["per_case"][cases[0]]:
        values = np.array([row[f"mean_delta_{metric}"] for row in averaged_cases])
        stats = bootstrap(values, seed=20260719 + len(averaged_case_stats))
        stats.update({
            "wins": int((values > 0).sum()), "losses": int((values < 0).sum()),
            "wilcoxon_p": float(wilcoxon(values).pvalue) if np.any(values) else 1.0,
        })
        averaged_case_stats[metric] = stats

    control_mean = {
        metric: float(np.mean([by_key[(seed, "interpolate")]["best"][metric] for seed in SEEDS]))
        for metric in metric_keys
    }
    qad_mean = {
        metric: float(np.mean([by_key[(seed, "qad_lite")]["best"][metric] for seed in SEEDS]))
        for metric in metric_keys
    }
    gate = {
        "normalized_r2_primary": {
            "rule": "mean delta >= +0.015 and at least 2/3 seeds positive",
            "mean_delta": cross_seed["normalized_r2_casebalanced"]["mean_delta"],
            "positive_seeds": cross_seed["normalized_r2_casebalanced"]["positive_seeds"],
            "passed": (
                cross_seed["normalized_r2_casebalanced"]["mean_delta"] >= 0.015
                and cross_seed["normalized_r2_casebalanced"]["positive_seeds"] >= 2
            ),
        },
        "spearman_noninferiority": {
            "rule": "no seed delta < -0.01",
            "passed": min(item["delta"]["spearman_case_mean"] for item in seed_pairs) >= -0.01,
        },
        "top10_iou_noninferiority": {
            "rule": "no seed delta < -0.01",
            "passed": min(item["delta"]["normalized_top10_iou"] for item in seed_pairs) >= -0.01,
        },
        "physical_rmse_noninferiority": {
            "rule": "no seed relative worsening > 1%",
            "passed": all(
                item["delta"]["physical_rmse"]
                / by_key[(item["seed"], "interpolate")]["best"]["physical_rmse"] <= 0.01
                for item in seed_pairs
            ),
        },
        "tail_direction": {
            "rule": "normalized p99 ratio should not regress across the majority of seeds",
            "passed": cross_seed["normalized_p99_amplitude_ratio"]["positive_seeds"] >= 2,
        },
    }
    overall_pass = all(item["passed"] for item in gate.values())
    decision = (
        "GO_to_R2_DUAL" if overall_pass
        else "NO_GO_for_direct_R2; retain QAD as a conditional mechanism and diagnose regularization"
    )

    payload = {
        "schema_version": 1, "analysis_date": "2026-07-19",
        "source_submission": str(SUBMISSION), "source_submission_sha256": sha256(SUBMISSION),
        "jobs": {"10540": "5/5 COMPLETED (0:0); val21 only; test27 not accessed"},
        "records": records, "seed_pairs": seed_pairs,
        "control_three_seed_mean": control_mean, "qad_three_seed_mean": qad_mean,
        "cross_seed_deltas": cross_seed,
        "averaged_case_stats": averaged_case_stats,
        "gates": gate, "all_gates_passed": overall_pass, "decision": decision,
        "interpretation": [
            "QAD improved normalized R2_cb in seeds 1234 and 7 but regressed in seed 2025",
            "the three-seed mean normalized R2_cb gain is positive but falls just below the +0.015 gate",
            "top10 IoU is non-inferior, but Spearman, physical RMSE, and normalized p99 tail gates are not all stable",
            "R2-DUAL remains blocked because the decoder-only parent did not satisfy the no-regression contract",
            "original test27 remains untouched and must not be used to rescue this result",
        ],
    }
    write_csvs(records, seed_pairs, averaged_cases)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "passed", "records": len(records), "decision": decision,
        "mean_normalized_r2_delta": cross_seed["normalized_r2_casebalanced"]["mean_delta"],
        "positive_seeds": cross_seed["normalized_r2_casebalanced"]["positive_seeds"],
        "json": str(OUT_JSON), "summary_csv": str(OUT_SUMMARY),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
