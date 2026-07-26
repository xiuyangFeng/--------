#!/usr/bin/env python3
"""Audit and summarize the 2026-07-23 D2-K64 ILO/structure screening matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PREFLIGHT = REPO / "training_wss_min/preflight"
WAVE1 = PREFLIGHT / "d2_k64_ilo_structure_wave1_prepared.json"
WAVE2 = PREFLIGHT / "d2_k64_ilo_structure_wave2_prepared.json"
RUN_ROOT = REPO / "training_wss_min/runs/pointnetpp_d2_k64_ilo_structure/outputs"
F0_RUN = (
    REPO
    / "training_wss_min/runs/pointnetpp_sa1_scale_single_seed/outputs"
    / "d2_rand5000_c125_k64"
)
OUT_JSON = PREFLIGHT / "d2_k64_ilo_structure_results_analysis.json"
OUT_SUMMARY = PREFLIGHT / "d2_k64_ilo_structure_results_summary.csv"
OUT_PAIRS = PREFLIGHT / "d2_k64_ilo_structure_paired_case_stats.csv"
BOOTSTRAP_DRAWS = 20_000

LABELS = {
    "f0_d2k64_parent_fixed_test27": "F0",
    "f1_d2k64_ilo41_fixed_test27": "F1",
    "m0_d2k64_zeroshot_mixed36": "M0",
    "m1_d2k64_mixed138_test36": "M1/S0",
    "s1_d2k64_mfeat_mixed": "S1-MFEAT",
    "s2_d2k64_pnxr_mixed": "S2-PNXR",
    "s3_d2k64_pnxr_geope_mixed": "S3-GEOPE",
    "s4_d2k64_pnxr_attn_mixed": "S4-ATTN",
    "s5c_d2k64_pnxr_independent_interp_mixed": "S5C-IND-INTERP",
    "s5_d2k64_pnxr_sep_mixed": "S5-SEP",
}

PAIR_SPECS = (
    ("F1_vs_F0", "f0_d2k64_parent_fixed_test27", "f1_d2k64_ilo41_fixed_test27"),
    ("M1_vs_M0", "m0_d2k64_zeroshot_mixed36", "m1_d2k64_mixed138_test36"),
    ("S1_vs_M1", "m1_d2k64_mixed138_test36", "s1_d2k64_mfeat_mixed"),
    ("S2_vs_M1", "m1_d2k64_mixed138_test36", "s2_d2k64_pnxr_mixed"),
    ("S3_vs_S2", "s2_d2k64_pnxr_mixed", "s3_d2k64_pnxr_geope_mixed"),
    ("S4_vs_S2", "s2_d2k64_pnxr_mixed", "s4_d2k64_pnxr_attn_mixed"),
    (
        "S5_vs_S5C",
        "s5c_d2k64_pnxr_independent_interp_mixed",
        "s5_d2k64_pnxr_sep_mixed",
    ),
)

AGG_KEYS = (
    "physical_r2_casebalanced",
    "physical_r2_pooled",
    "physical_r2_case_mean",
    "physical_r2_case_median",
    "physical_r2_case_p10",
    "physical_r2_negative_cases",
    "physical_mae",
    "physical_rmse",
    "high_wss_r2",
    "high_wss_mae",
    "physical_top10_iou",
    "physical_top10_amplitude_ratio",
    "physical_p99_amplitude_ratio",
    "spearman_case_mean",
    "high_wss_spearman_case_mean",
    "normalized_r2_casebalanced",
    "parameters",
)

CASE_METRICS = {
    "case_r2": ("overall", "r2"),
    "case_mae": ("overall", "mae"),
    "case_rmse": ("overall", "rmse"),
    "case_high_wss_r2": ("high_wss", "r2"),
    "case_high_wss_mae": ("high_wss", "mae"),
    "case_top10_iou": ("hotspot", "top10_iou"),
    "case_spearman": ("hotspot", "spearman_all"),
    "case_high_wss_spearman": ("hotspot", "spearman_high_wss"),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_partition(path: Path) -> tuple[str, dict]:
    payload = read_json(path)
    if len(payload) != 1:
        raise RuntimeError(f"expected one evaluation partition in {path}")
    partition = next(iter(payload))
    return partition, payload[partition]


def finite_numbers(value: object, where: str, errors: list[str]) -> None:
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
    groups = metrics.get("group_casebalanced", {})
    return {
        "physical_r2_casebalanced": balanced["r2"],
        "physical_r2_pooled": field["r2"],
        "physical_r2_case_mean": aggregate["r2_casemean"],
        "physical_r2_case_median": aggregate["r2_casemed"],
        "physical_r2_case_p10": aggregate["r2_casep10"],
        "physical_r2_negative_cases": aggregate["r2_negative_cases"],
        # Keep MAE/RMSE on the same equal-case weighting as the primary R2_cb.
        "physical_mae": balanced["mae"],
        "physical_rmse": balanced["rmse"],
        "high_wss_r2": regional["high_wss"]["r2"],
        "high_wss_mae": regional["high_wss"]["mae"],
        "physical_top10_iou": hotspot["top10_iou_casemean"],
        "physical_top10_amplitude_ratio": calibration["top10_pred_true_ratio"],
        "physical_p99_amplitude_ratio": calibration["p99_pred_true_ratio"],
        "spearman_case_mean": hotspot["spearman_all_casemean"],
        "high_wss_spearman_case_mean": hotspot["spearman_high_wss_casemean"],
        "normalized_r2_casebalanced": normalized["field_casebalanced"]["r2"],
        "parameters": metrics.get("efficiency", {}).get("parameters"),
        "groups": groups,
    }


def history_summary(run_dir: Path, errors: list[str]) -> dict:
    path = run_dir / "history.jsonl"
    if not path.exists():
        errors.append("missing:history.jsonl")
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != 400 or rows[0]["epoch"] != 0 or rows[-1]["epoch"] != 399:
        errors.append("history_not_epochs_0_to_399")
    finite_numbers(rows, f"{run_dir.name}.history", errors)
    best = min(rows, key=lambda row: row["train_loss"])
    return {
        "epochs": len(rows),
        "first_epoch": rows[0]["epoch"],
        "last_epoch": rows[-1]["epoch"],
        "best_epoch": best["epoch"],
        "best_train_loss": best["train_loss"],
        "last_train_loss": rows[-1]["train_loss"],
    }


def per_case(metrics: dict) -> dict[str, dict[str, float]]:
    output = {}
    for case, values in metrics["per_case"].items():
        output[case] = {
            name: float(values[section][key])
            for name, (section, key) in CASE_METRICS.items()
        }
    return output


def record(
    experiment_id: str,
    run_dir: Path,
    job_id: str,
    config: Path | None = None,
    expected_config_sha256: str | None = None,
) -> dict:
    errors: list[str] = []
    required = (
        "ckpt_best.pt",
        "ckpt_last.pt",
        "config.json",
        "eval/best_last_comparison.json",
        "eval/best_last_per_case.csv",
        "eval/ckpt_best/metrics.json",
        "eval/ckpt_best/per_case_metrics.csv",
        "eval/ckpt_last/metrics.json",
        "eval/ckpt_last/per_case_metrics.csv",
    )
    for relative in required:
        if not (run_dir / relative).exists():
            errors.append(f"missing:{relative}")
    best_partition, best_metrics = metric_partition(run_dir / "eval/ckpt_best/metrics.json")
    last_partition, last_metrics = metric_partition(run_dir / "eval/ckpt_last/metrics.json")
    if best_partition != last_partition:
        errors.append("best_last_partition_mismatch")
    finite_numbers(best_metrics, f"{experiment_id}.best", errors)
    finite_numbers(last_metrics, f"{experiment_id}.last", errors)
    if config is not None and expected_config_sha256 is not None:
        if sha256(config) != expected_config_sha256:
            errors.append("prepared_config_hash_changed")
    history = history_summary(run_dir, errors)
    best = flatten_metrics(best_metrics)
    last = flatten_metrics(last_metrics)
    return {
        "experiment_id": experiment_id,
        "label": LABELS[experiment_id],
        "job_id": job_id,
        "run_dir": str(run_dir),
        "evaluation_partition": best_partition,
        "history": history,
        "best": best,
        "last": last,
        "last_minus_best": {
            key: last[key] - best[key]
            for key in ("physical_r2_casebalanced", "physical_mae", "physical_rmse")
        },
        "per_case": per_case(best_metrics),
        "integrity": "passed" if not errors else "failed",
        "integrity_errors": errors,
    }


def bootstrap_mean(values: list[float], seed: int) -> dict:
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_DRAWS)
    )
    return {
        "mean_delta": sum(values) / n,
        "median_delta": sorted(values)[n // 2]
        if n % 2
        else (sorted(values)[n // 2 - 1] + sorted(values)[n // 2]) / 2,
        "ci95_low": draws[int(0.025 * BOOTSTRAP_DRAWS)],
        "ci95_high": draws[int(0.975 * BOOTSTRAP_DRAWS) - 1],
        "wins": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "losses": sum(value < 0 for value in values),
        "n_cases": n,
    }


def comparison(name: str, control: dict, treatment: dict, seed: int) -> dict:
    if set(control["per_case"]) != set(treatment["per_case"]):
        raise RuntimeError(f"{name}: paired case sets differ")
    aggregate_delta = {
        key: treatment["best"][key] - control["best"][key]
        for key in AGG_KEYS
        if key != "parameters"
    }
    cases = sorted(control["per_case"])
    case_stats = {}
    for offset, metric in enumerate(CASE_METRICS):
        values = [
            treatment["per_case"][case][metric] - control["per_case"][case][metric]
            for case in cases
        ]
        case_stats[metric] = bootstrap_mean(values, seed + offset)
    return {
        "comparison": name,
        "control": control["experiment_id"],
        "treatment": treatment["experiment_id"],
        "aggregate_treatment_minus_control": aggregate_delta,
        "paired_case_stats": case_stats,
        "note": (
            "Case-level bootstrap CI describes the mean per-case metric delta; "
            "it is not a bootstrap CI for aggregate field_casebalanced R2."
        ),
    }


def slurm_states() -> dict[str, dict[str, str]]:
    completed_ids = {
        "10838_0": "f1_d2k64_ilo41_fixed_test27",
        "10838_1": "m0_d2k64_zeroshot_mixed36",
        "10838_2": "m1_d2k64_mixed138_test36",
        "10838_3": "s1_d2k64_mfeat_mixed",
        "10838_4": "s2_d2k64_pnxr_mixed",
        "10844_0": "s3_d2k64_pnxr_geope_mixed",
        "10844_1": "s4_d2k64_pnxr_attn_mixed",
        "10844_2": "s5c_d2k64_pnxr_independent_interp_mixed",
        "10844_3": "s5_d2k64_pnxr_sep_mixed",
    }
    command = [
        "/public/slurm/bin/sacct",
        "-n",
        "-P",
        "-j",
        "10837,10838,10843,10844",
        "--format=JobID,State,ExitCode,Elapsed,End",
    ]
    rows = subprocess.check_output(command, text=True).splitlines()
    states: dict[str, dict[str, str]] = {}
    for row in rows:
        fields = row.split("|")
        if len(fields) < 5 or "." in fields[0]:
            continue
        job_id, state, exit_code, elapsed, end = fields[:5]
        states[job_id] = {
            "state": state,
            "exit_code": exit_code,
            "elapsed": elapsed,
            "end": end,
        }
    expected = {"10837", "10843", *completed_ids}
    missing = expected - set(states)
    if missing:
        raise RuntimeError(f"missing Slurm states: {sorted(missing)}")
    failed = {
        job_id: states[job_id]
        for job_id in expected
        if states[job_id]["state"] != "COMPLETED"
        or states[job_id]["exit_code"] != "0:0"
    }
    if failed:
        raise RuntimeError(f"non-completed Slurm jobs: {failed}")
    return states


def write_csvs(records: list[dict], comparisons: list[dict]) -> None:
    fields = [
        "experiment_id",
        "label",
        "job_id",
        "evaluation_partition",
        "best_epoch",
        "best_train_loss",
        *AGG_KEYS,
        "last_minus_best_r2_casebalanced",
        "integrity",
    ]
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            writer.writerow(
                {
                    "experiment_id": item["experiment_id"],
                    "label": item["label"],
                    "job_id": item["job_id"],
                    "evaluation_partition": item["evaluation_partition"],
                    "best_epoch": item["history"]["best_epoch"],
                    "best_train_loss": item["history"]["best_train_loss"],
                    **{key: item["best"][key] for key in AGG_KEYS},
                    "last_minus_best_r2_casebalanced": item["last_minus_best"][
                        "physical_r2_casebalanced"
                    ],
                    "integrity": item["integrity"],
                }
            )
    pair_fields = [
        "comparison",
        "control",
        "treatment",
        "metric",
        "aggregate_delta",
        "mean_case_delta",
        "median_case_delta",
        "ci95_low",
        "ci95_high",
        "wins",
        "ties",
        "losses",
        "n_cases",
    ]
    with OUT_PAIRS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        for item in comparisons:
            for metric, stats in item["paired_case_stats"].items():
                aggregate_key = {
                    "case_r2": "physical_r2_case_mean",
                    "case_mae": "physical_mae",
                    "case_rmse": "physical_rmse",
                    "case_high_wss_r2": "high_wss_r2",
                    "case_high_wss_mae": "high_wss_mae",
                    "case_top10_iou": "physical_top10_iou",
                    "case_spearman": "spearman_case_mean",
                    "case_high_wss_spearman": "high_wss_spearman_case_mean",
                }[metric]
                writer.writerow(
                    {
                        "comparison": item["comparison"],
                        "control": item["control"],
                        "treatment": item["treatment"],
                        "metric": metric,
                        "aggregate_delta": item["aggregate_treatment_minus_control"][
                            aggregate_key
                        ],
                        "mean_case_delta": stats["mean_delta"],
                        "median_case_delta": stats["median_delta"],
                        "ci95_low": stats["ci95_low"],
                        "ci95_high": stats["ci95_high"],
                        "wins": stats["wins"],
                        "ties": stats["ties"],
                        "losses": stats["losses"],
                        "n_cases": stats["n_cases"],
                    }
                )


def main() -> None:
    wave1 = read_json(WAVE1)
    wave2 = read_json(WAVE2)
    states = slurm_states()
    task_jobs = [
        "10838_0",
        "10838_1",
        "10838_2",
        "10838_3",
        "10838_4",
        "10844_0",
        "10844_1",
        "10844_2",
        "10844_3",
    ]
    items = wave1["configs"] + wave2["configs"]
    if len(items) != len(task_jobs):
        raise RuntimeError("prepared config count does not match Slurm task count")
    records = [
        record(
            item["experiment_id"],
            Path(item["run_dir"]),
            job_id,
            Path(item["config"]),
            item["sha256"],
        )
        for item, job_id in zip(items, task_jobs)
    ]
    f0_config = Path(wave1["parent"]["config"])
    records.insert(
        0,
        record(
            "f0_d2k64_parent_fixed_test27",
            F0_RUN,
            "historical-parent",
            f0_config,
            wave1["parent"]["config_sha256"],
        ),
    )
    if any(item["integrity"] != "passed" for item in records):
        raise RuntimeError(
            {
                item["experiment_id"]: item["integrity_errors"]
                for item in records
                if item["integrity"] != "passed"
            }
        )
    by_id = {item["experiment_id"]: item for item in records}
    comparisons = [
        comparison(name, by_id[control], by_id[treatment], 20260723 + index * 100)
        for index, (name, control, treatment) in enumerate(PAIR_SPECS)
    ]
    mixed_rank = sorted(
        [item for item in records if item["evaluation_partition"] == "test"],
        key=lambda item: item["best"]["physical_r2_casebalanced"],
        reverse=True,
    )
    payload = {
        "schema_version": 1,
        "analysis_date": "2026-07-23",
        "scope": "single-seed screening; seed=1234",
        "sources": {
            "wave1_manifest": str(WAVE1),
            "wave1_manifest_sha256": sha256(WAVE1),
            "wave2_manifest": str(WAVE2),
            "wave2_manifest_sha256": sha256(WAVE2),
        },
        "slurm": states,
        "records": records,
        "paired_comparisons": comparisons,
        "mixed_test36_rank_by_physical_r2_casebalanced": [
            {
                "rank": rank,
                "experiment_id": item["experiment_id"],
                "label": item["label"],
                "physical_r2_casebalanced": item["best"]["physical_r2_casebalanced"],
            }
            for rank, item in enumerate(mixed_rank, 1)
        ],
        "interpretation_limits": [
            "All new training arms use only seed 1234; results support screening decisions, not stability claims.",
            "F0/F1 are comparable only on the fixed historical AG/AAA test27.",
            "M0/M1 and S1-S5 are comparable only on the common mixed AG/AAA/ILO test36.",
            "S5 is a decoder-only comparison against S5C, not against S2, because S5/S5C use independent query.",
            "Test sets were used as preauthorized frozen screening endpoints; no hyperparameter tuning claim is made.",
            "Case-level bootstrap intervals do not replace future three-seed confirmation.",
        ],
    }
    write_csvs(records, comparisons)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "records": len(records),
                "new_runs": len(records) - 1,
                "comparisons": len(comparisons),
                "mixed_test36_winner": mixed_rank[0]["label"],
                "json": str(OUT_JSON),
                "summary_csv": str(OUT_SUMMARY),
                "paired_csv": str(OUT_PAIRS),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
