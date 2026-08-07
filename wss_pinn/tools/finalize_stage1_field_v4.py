"""Assemble the Stage-1 field-v4 multi-seed and promotion evidence."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wss_pinn.config import ExperimentConfig
from wss_pinn.tools.gate_stage1_raw_field_v4 import _trained_derivative_probe
from wss_pinn.utils import (
    ROOT,
    atomic_write_json,
    git_state,
    guard_write_path,
    sha256_file,
    utc_now,
)


ORIGINAL_CONFIG_ROOT = ROOT / "wss_pinn/configs/volume_uvwp_peak_field_v4"
CONFIRM_CONFIG_ROOT = ROOT / "wss_pinn/configs/volume_uvwp_peak_field_v4_multiseed_top2"
OUTPUT_ROOT = ROOT / "outputs/wss_pinn/volume_uvwp_peak_field_v4"
REPORT_OUTPUT = OUTPUT_ROOT / "stage1_multiseed_and_promotion_report.json"
CASE_CSV_OUTPUT = OUTPUT_ROOT / "stage1_b0_case_deltas.csv"
DERIVATIVE_OUTPUT = OUTPUT_ROOT / "stage1_trained_derivative_support_gate.json"
ARTIFACT_MANIFEST_OUTPUT = OUTPUT_ROOT / "stage1_artifact_manifest.json"
B0_EVALUATION = OUTPUT_ROOT / "b0/evaluation_val15.json"
FIELD_STATS = ROOT / "data_wss_pinn/volume_uvwp_peak_qs_smooth_v3_train123_val15_test35/field_stats.json"
DECISION_CONTRACT = ORIGINAL_CONFIG_ROOT / "decision_contract.json"
SINGLE_SEED_GATE = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage1_single_seed_gate/report.json"
RAW_GATE = ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage1_raw_gate/report.json"


CONFIG_PATHS = {
    "G-Raw": {
        1234: ORIGINAL_CONFIG_ROOT / "g_raw_s1234.json",
        2345: CONFIRM_CONFIG_ROOT / "g_raw_s2345.json",
        3456: CONFIRM_CONFIG_ROOT / "g_raw_s3456.json",
    },
    "G-PE": {
        1234: ORIGINAL_CONFIG_ROOT / "g_pe_s1234.json",
        2345: CONFIRM_CONFIG_ROOT / "g_pe_s2345.json",
        3456: CONFIRM_CONFIG_ROOT / "g_pe_s3456.json",
    },
    "L-Raw": {1234: ORIGINAL_CONFIG_ROOT / "l_raw_s1234.json"},
    "L-PE": {1234: ORIGINAL_CONFIG_ROOT / "l_pe_s1234.json"},
}
TOP_ARMS = ("G-Raw", "G-PE")
COMPONENTS = ("u", "v", "w", "pressure")
REPORT_METRICS = ("u", "v", "w", "speed", "pressure")


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _case_score(row: dict[str, Any], standard_deviations: dict[str, float]) -> float:
    return float(
        np.mean(
            [
                (float(row["metrics"][name]["rmse"]) / standard_deviations[name]) ** 2
                for name in COMPONENTS
            ]
        )
    )


def _mean_std(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": [float(value) for value in array],
        "mean": float(np.mean(array)),
        "std_sample": float(np.std(array, ddof=1)) if len(array) > 1 else None,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _load_run(config_path: Path, standard_deviations: dict[str, float]) -> dict[str, Any]:
    config = ExperimentConfig.from_json(config_path)
    run_dir = config.run_dir
    summary_path = run_dir / "training_summary.json"
    audit_path = run_dir / "selection_audit.json"
    evaluation_path = run_dir / "evaluation_val15_best_validation_field_cb.json"
    checkpoint_path = run_dir / "checkpoints/best_validation_field_cb.pt"
    for path in (summary_path, audit_path, evaluation_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (run_dir / "error.json").exists():
        raise RuntimeError(f"run has error.json: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or evaluation.get("status") != "completed":
        raise RuntimeError(f"incomplete run: {run_dir}")
    cases = {str(row["case_id"]): row for row in evaluation["cases"]}
    scores = {case_id: _case_score(row, standard_deviations) for case_id, row in cases.items()}
    best_epoch = int(audit["best_epoch_zero_based"]["case_balanced"])
    legacy_epoch = int(audit["best_epoch_zero_based"]["legacy_batch_weighted"])
    history = {int(row["epoch"]): row for row in audit["history"]}
    return {
        "config": config,
        "config_source": _source(config_path),
        "run_dir": run_dir,
        "summary": summary,
        "audit": audit,
        "evaluation": evaluation,
        "cases": cases,
        "case_scores": scores,
        "checkpoint": _source(checkpoint_path),
        "selection_comparison": {
            "fixed_case_balanced_epoch_zero_based": best_epoch,
            "legacy_batch_weighted_epoch_zero_based": legacy_epoch,
            "legacy_minus_fixed_epoch": legacy_epoch - best_epoch,
            "fixed_case_balanced_score": float(history[best_epoch]["case_balanced"]),
            "case_balanced_score_at_legacy_epoch": float(history[legacy_epoch]["case_balanced"]),
            "legacy_epoch_score_penalty": float(
                history[legacy_epoch]["case_balanced"] - history[best_epoch]["case_balanced"]
            ),
        },
    }


def _bootstrap_case_delta(case_deltas: dict[str, float]) -> dict[str, Any]:
    values = np.asarray([case_deltas[key] for key in sorted(case_deltas)], dtype=np.float64)
    rng = np.random.default_rng(20260806)
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    estimates = values[indices].mean(axis=1)
    return {
        "method": "paired validation-case bootstrap; seed means formed before resampling",
        "bootstrap_seed": 20260806,
        "replicates": 10_000,
        "estimate": float(np.mean(values)),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
    }


def _metric_seed_mean(runs: dict[int, dict[str, Any]], metric: str, stat: str) -> float:
    return float(
        np.mean(
            [run["evaluation"]["case_balanced_regression"][metric][stat] for run in runs.values()]
        )
    )


def _cohort_scores(runs: dict[int, dict[str, Any]]) -> dict[str, float]:
    result = {}
    for cohort in ("AAA", "AG", "ILO"):
        per_seed = []
        for run in runs.values():
            values = [
                score
                for case_id, score in run["case_scores"].items()
                if case_id.startswith(cohort + "/")
            ]
            per_seed.append(float(np.mean(values)))
        result[cohort] = float(np.mean(per_seed))
    return result


def _write_case_csv(
    b0: dict[str, Any], runs: dict[str, dict[int, dict[str, Any]]]
) -> None:
    b0_cases = {str(row["case_id"]): row for row in b0["cases"]}
    handle = io.StringIO()
    fieldnames = [
        "arm",
        "seed_scope",
        "case_id",
        "cohort",
        "metric",
        "delta_r2_vs_b0",
        "delta_mae_vs_b0",
        "delta_rmse_vs_b0",
        "delta_validation_field_score_vs_b0",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for arm, arm_runs in runs.items():
        seeds = sorted(arm_runs)
        for case_id in sorted(b0_cases):
            b0_case = b0_cases[case_id]
            score_delta = float(
                np.mean([arm_runs[seed]["case_scores"][case_id] for seed in seeds])
                - float(b0_case["validation_field_score"])
            )
            for metric in REPORT_METRICS:
                model_values = {
                    stat: float(
                        np.mean(
                            [arm_runs[seed]["cases"][case_id]["metrics"][metric][stat] for seed in seeds]
                        )
                    )
                    for stat in ("r2", "mae", "rmse")
                }
                baseline = b0_case["metrics"][metric]
                writer.writerow(
                    {
                        "arm": arm,
                        "seed_scope": "+".join(str(seed) for seed in seeds),
                        "case_id": case_id,
                        "cohort": case_id.split("/", 1)[0],
                        "metric": metric,
                        "delta_r2_vs_b0": model_values["r2"] - float(baseline["r2"]),
                        "delta_mae_vs_b0": model_values["mae"] - float(baseline["mae"]),
                        "delta_rmse_vs_b0": model_values["rmse"] - float(baseline["rmse"]),
                        "delta_validation_field_score_vs_b0": score_delta,
                    }
                )
    target = guard_write_path(CASE_CSV_OUTPUT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(handle.getvalue(), encoding="utf-8")


def _derivative_gate(
    runs: dict[str, dict[int, dict[str, Any]]], device: torch.device
) -> dict[str, Any]:
    arms = []
    for arm in ("G-Raw", "G-PE", "L-Raw", "L-PE"):
        for seed, run in sorted(runs[arm].items()):
            derivative = _trained_derivative_probe(
                run["config"], Path(run["checkpoint"]["path"]), device
            )
            arms.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "experiment_id": run["config"]["experiment"]["id"],
                    "checkpoint": run["checkpoint"],
                    **derivative,
                }
            )
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "route": "volume_uvwp_peak_field_v4",
        "gate_result": "pass" if all(row["gate_result"] == "pass" for row in arms) else "fail",
        "arms": arms,
        "test35_cases_read": 0,
        "git": git_state(),
    }
    atomic_write_json(DERIVATIVE_OUTPUT, report)
    return report


def _full_volume_section(
    runs: dict[str, dict[int, dict[str, Any]]], standard_deviations: dict[str, float]
) -> dict[str, Any]:
    arms = {}
    missing = []
    for arm in TOP_ARMS:
        seed_reports = []
        for seed, run in sorted(runs[arm].items()):
            path = run["run_dir"] / "evaluation_fullvolume_best_validation_field_cb.json"
            if not path.is_file():
                missing.append(str(path.resolve()))
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "completed" or len(payload.get("cases", [])) != 15:
                raise RuntimeError(f"invalid full-volume evaluation: {path}")
            scores = [_case_score(row, standard_deviations) for row in payload["cases"]]
            seed_reports.append(
                {
                    "seed": seed,
                    "source": _source(path),
                    "validation_field_score_proxy_cb": float(np.mean(scores)),
                    "evaluated_points": int(sum(row["evaluated_points"] for row in payload["cases"])),
                    "case_balanced_regression": payload["case_balanced_regression"],
                    "cohort_case_balanced_regression": payload["cohort_case_balanced_regression"],
                }
            )
        if seed_reports:
            arms[arm] = {
                "seeds": seed_reports,
                "score_proxy": _mean_std(
                    [row["validation_field_score_proxy_cb"] for row in seed_reports]
                ),
                "case_balanced_regression_seed_mean": {
                    metric: {
                        stat: float(
                            np.mean(
                                [row["case_balanced_regression"][metric][stat] for row in seed_reports]
                            )
                        )
                        for stat in ("r2", "mae", "rmse")
                    }
                    for metric in REPORT_METRICS
                },
            }
    return {
        "status": "completed" if not missing else "pending",
        "protocol": "all strict-volume val15 points; checkpoint remains fixed-validation selected",
        "arms": arms,
        "missing": missing,
        "test35_cases_read": 0,
    }


def run(*, device_name: str = "cuda:0", rerun_derivatives: bool = False) -> dict[str, Any]:
    stats = json.loads(FIELD_STATS.read_text(encoding="utf-8"))
    standard_deviations = {
        "u": float(stats["velocity_m_s"]["std"][0]),
        "v": float(stats["velocity_m_s"]["std"][1]),
        "w": float(stats["velocity_m_s"]["std"][2]),
        "pressure": float(stats["pressure_relative_pa"]["std"][0]),
    }
    b0 = json.loads(B0_EVALUATION.read_text(encoding="utf-8"))
    decision = json.loads(DECISION_CONTRACT.read_text(encoding="utf-8"))
    runs = {
        arm: {
            seed: _load_run(config_path, standard_deviations)
            for seed, config_path in seed_paths.items()
        }
        for arm, seed_paths in CONFIG_PATHS.items()
    }
    _write_case_csv(b0, runs)

    if rerun_derivatives or not DERIVATIVE_OUTPUT.is_file():
        derivative_report = _derivative_gate(runs, torch.device(device_name))
    else:
        derivative_report = json.loads(DERIVATIVE_OUTPUT.read_text(encoding="utf-8"))

    multiseed = {}
    for arm in TOP_ARMS:
        arm_runs = runs[arm]
        training_scores = [
            float(arm_runs[seed]["summary"]["best_validation_field_score_cb"])
            for seed in sorted(arm_runs)
        ]
        case_mean_scores = {
            case_id: float(np.mean([run["case_scores"][case_id] for run in arm_runs.values()]))
            for case_id in next(iter(arm_runs.values()))["case_scores"]
        }
        b0_case_scores = {
            str(row["case_id"]): float(row["validation_field_score"])
            for row in b0["cases"]
        }
        b0_delta = {
            case_id: case_mean_scores[case_id] - b0_case_scores[case_id]
            for case_id in case_mean_scores
        }
        multiseed[arm] = {
            "training_selection_score": _mean_std(training_scores),
            "fixed_evaluation_score_proxy": _mean_std(
                [float(np.mean(list(run["case_scores"].values()))) for run in arm_runs.values()]
            ),
            "relative_to_b0": {
                "b0_validation_field_score_cb": float(b0["validation_field_score_cb"]),
                "training_score_mean_delta": float(np.mean(training_scores) - b0["validation_field_score_cb"]),
                "evaluation_case_delta_mean": float(np.mean(list(b0_delta.values()))),
                "improved_cases": int(sum(value < 0 for value in b0_delta.values())),
                "total_cases": len(b0_delta),
                "case_deltas": b0_delta,
            },
            "cohort_score_proxy": _cohort_scores(arm_runs),
            "case_balanced_regression_seed_mean": {
                metric: {
                    stat: _metric_seed_mean(arm_runs, metric, stat)
                    for stat in ("r2", "mae", "rmse")
                }
                for metric in REPORT_METRICS
            },
            "near_wall_regression_seed_mean": {
                metric: {
                    stat: _metric_seed_mean(arm_runs, f"near_wall_{metric}", stat)
                    for stat in ("r2", "mae", "rmse")
                }
                for metric in REPORT_METRICS
            },
        }

    raw_runs = runs["G-Raw"]
    pe_runs = runs["G-PE"]
    paired_training_delta = {
        str(seed): float(
            pe_runs[seed]["summary"]["best_validation_field_score_cb"]
            - raw_runs[seed]["summary"]["best_validation_field_score_cb"]
        )
        for seed in sorted(raw_runs)
    }
    case_deltas = {
        case_id: float(
            np.mean(
                [
                    pe_runs[seed]["case_scores"][case_id]
                    - raw_runs[seed]["case_scores"][case_id]
                    for seed in sorted(raw_runs)
                ]
            )
        )
        for case_id in raw_runs[1234]["case_scores"]
    }
    channel_degradation = {
        metric: (
            multiseed["G-PE"]["case_balanced_regression_seed_mean"][metric]["rmse"]
            / multiseed["G-Raw"]["case_balanced_regression_seed_mean"][metric]["rmse"]
            - 1.0
        )
        for metric in COMPONENTS
    }
    near_wall_degradation = {
        metric: (
            multiseed["G-PE"]["near_wall_regression_seed_mean"][metric]["rmse"]
            / multiseed["G-Raw"]["near_wall_regression_seed_mean"][metric]["rmse"]
            - 1.0
        )
        for metric in COMPONENTS
    }
    raw_cohorts = multiseed["G-Raw"]["cohort_score_proxy"]
    pe_cohorts = multiseed["G-PE"]["cohort_score_proxy"]
    cohort_degradation = {
        cohort: pe_cohorts[cohort] / raw_cohorts[cohort] - 1.0
        for cohort in raw_cohorts
    }
    guard = decision["equivalence_guard"]
    stable_direction = all(value < 0 for value in paired_training_delta.values())
    channel_guard = all(value <= float(guard["channel_relative_degradation"]) for value in channel_degradation.values())
    near_wall_guard = all(value <= float(guard["near_wall_relative_degradation"]) for value in near_wall_degradation.values())
    worst_cohort_guard = max(cohort_degradation.values()) <= float(guard["worst_cohort_relative_degradation"])
    derivative_guard = derivative_report.get("gate_result") == "pass"
    pe_go = stable_direction and channel_guard and near_wall_guard and worst_cohort_guard and derivative_guard

    selection_audits = []
    for arm, arm_runs in runs.items():
        for seed, run_data in sorted(arm_runs.items()):
            selection_audits.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "experiment_id": run_data["config"]["experiment"]["id"],
                    **run_data["selection_comparison"],
                }
            )

    full_volume = _full_volume_section(runs, standard_deviations)
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "completed" if full_volume["status"] == "completed" else "awaiting_full_volume",
        "route": "volume_uvwp_peak_field_v4",
        "primary_selection": "validation_field_score_cb",
        "b0": {
            "source": _source(B0_EVALUATION),
            "validation_field_score_cb": float(b0["validation_field_score_cb"]),
        },
        "multiseed": multiseed,
        "paired_g_pe_minus_g_raw": {
            "training_selection_score_by_seed": paired_training_delta,
            "training_selection_score": _mean_std(list(paired_training_delta.values())),
            "direction_stable_improvement": stable_direction,
            "improved_seeds": int(sum(value < 0 for value in paired_training_delta.values())),
            "total_seeds": len(paired_training_delta),
            "fixed_evaluation_case_score_delta": case_deltas,
            "improved_cases": int(sum(value < 0 for value in case_deltas.values())),
            "total_cases": len(case_deltas),
            "case_bootstrap": _bootstrap_case_delta(case_deltas),
        },
        "equivalence_guards_g_pe_vs_g_raw": {
            "thresholds": guard,
            "channel_rmse_relative_degradation": channel_degradation,
            "channel_gate": "pass" if channel_guard else "fail",
            "near_wall_rmse_relative_degradation": near_wall_degradation,
            "near_wall_gate": "pass" if near_wall_guard else "fail",
            "cohort_score_relative_degradation": cohort_degradation,
            "worst_cohort_gate": "pass" if worst_cohort_guard else "fail",
        },
        "trained_derivative_and_support_gate": {
            "source": _source(DERIVATIVE_OUTPUT),
            "gate_result": derivative_report.get("gate_result"),
        },
        "checkpoint_selection_fix_audit": selection_audits,
        "full_volume_promotion_audit": full_volume,
        "decision": {
            "g_pe_promotion": "go" if pe_go else "no_go",
            "g_pe_reasons": {
                "multiseed_direction_stable": stable_direction,
                "channel_guard": channel_guard,
                "near_wall_guard": near_wall_guard,
                "worst_cohort_guard": worst_cohort_guard,
                "derivative_support_guard": derivative_guard,
            },
            "stage1_recommendation": (
                "promote_G_PE" if pe_go else "retain_G_Raw_and_close_PE_frequency_sweep"
            ),
            "local_conditioner": "stop_after_single_seed_no_increment",
            "next_scientific_hypothesis": "patient_specific_conditioning_or_missing_BC",
        },
        "sources": {
            "decision_contract": _source(DECISION_CONTRACT),
            "single_seed_gate": _source(SINGLE_SEED_GATE),
            "raw_gate": _source(RAW_GATE),
            "field_stats": _source(FIELD_STATS),
            "case_delta_csv": _source(CASE_CSV_OUTPUT),
        },
        "forbidden_inputs": {
            "test35_cases_read": 0,
            "wss_used": False,
            "bc_input_or_loss_used": False,
            "pde_loss_used": False,
            "physics_residual_used_for_selection": False,
        },
        "limitations": {
            "bifurcation": "not available because no topology-backed label exists in the Stage-0 contract",
            "scope": "Stage 1 compares representation under missing deployable patient BC; it is not a full patient-specific surrogate ceiling",
        },
        "git": git_state(),
    }
    atomic_write_json(REPORT_OUTPUT, report)
    artifact_paths = [
        ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage0a/report.json",
        ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage0a/probes/manufactured_and_derivative_gate.json",
        ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage0a/probes/v3_support_latent_gradient_probe.json",
        ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage0a/probes/bulk_knn_truth_oracle_val15.json",
        ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage0b/report.json",
        ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage0b/outlet_comparison.csv",
        ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_stage0b/wall_zones.csv",
        ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_preflight_final/report.json",
        ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_multiseed_preflight_final/report.json",
        ROOT / "outputs/wss_pinn/audits/volume_uvwp_peak_field_v4_cpu_smoke_final/report.json",
        B0_EVALUATION,
        OUTPUT_ROOT / "b0/case_metrics_val15.csv",
        RAW_GATE,
        SINGLE_SEED_GATE,
        DERIVATIVE_OUTPUT,
        REPORT_OUTPUT,
        CASE_CSV_OUTPUT,
        OUTPUT_ROOT / "submission_stage1_raw.json",
        OUTPUT_ROOT / "submission_stage1_pe.json",
        OUTPUT_ROOT / "submission_stage1_multiseed.json",
        DECISION_CONTRACT,
        ORIGINAL_CONFIG_ROOT / "matrix.json",
        CONFIRM_CONFIG_ROOT / "matrix.json",
    ]
    for arm_runs in runs.values():
        for run_data in arm_runs.values():
            artifact_paths.extend(
                [
                    Path(run_data["config_source"]["path"]),
                    run_data["run_dir"] / "training_summary.json",
                    run_data["run_dir"] / "selection_audit.json",
                    Path(run_data["checkpoint"]["path"]),
                    run_data["run_dir"] / "evaluation_val15_best_validation_field_cb.json",
                ]
            )
            full_volume_path = (
                run_data["run_dir"] / "evaluation_fullvolume_best_validation_field_cb.json"
            )
            if full_volume_path.is_file():
                artifact_paths.append(full_volume_path)
    unique_paths = sorted({path.resolve() for path in artifact_paths})
    missing_artifacts = [str(path) for path in unique_paths if not path.is_file()]
    if missing_artifacts:
        raise FileNotFoundError(f"final artifact manifest missing files: {missing_artifacts}")
    atomic_write_json(
        ARTIFACT_MANIFEST_OUTPUT,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "route": "volume_uvwp_peak_field_v4",
            "status": "completed",
            "artifacts": {
                str(path.relative_to(ROOT)): sha256_file(path) for path in unique_paths
            },
            "artifact_count": len(unique_paths),
            "test35_cases_read": 0,
            "wss_used": False,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rerun-derivatives", action="store_true")
    args = parser.parse_args()
    report = run(device_name=args.device, rerun_derivatives=args.rerun_derivatives)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(REPORT_OUTPUT),
                "artifact_manifest": str(ARTIFACT_MANIFEST_OUTPUT),
                "decision": report["decision"],
                "full_volume_status": report["full_volume_promotion_audit"]["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
