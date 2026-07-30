#!/usr/bin/env python3
"""Audit and summarize the completed LSA2-H2 log-radius paired screen."""

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
MANIFEST = PREFLIGHT / "lsa2_h2_logradius_matrix_20260729_prepared.json"
STATIC_AUDIT = (
    PREFLIGHT / "lsa2_h2_logradius_matrix_20260729_static_audit.json"
)
GPU_AUDIT = PREFLIGHT / "lsa2_h2_logradius_matrix_20260729_gpu_smoke.json"
SUBMISSION = PREFLIGHT / "lsa2_h2_logradius_matrix_20260729_submission.json"
OUT_JSON = (
    PREFLIGHT / "lsa2_h2_logradius_matrix_20260729_results_analysis.json"
)
OUT_SUMMARY = (
    PREFLIGHT / "lsa2_h2_logradius_matrix_20260729_results_summary.csv"
)
OUT_PAIRS = (
    PREFLIGHT / "lsa2_h2_logradius_matrix_20260729_paired_case_stats.csv"
)

CONTROL = "lsa2_h2_same_repro_s1234"
TREATMENT = "lsa2_h2_logradius_s1234"
ORDER = (CONTROL, TREATMENT)
LABELS = {
    CONTROL: "LSA2 SAME H2 concurrent repro s1234",
    TREATMENT: "LSA2 SAME H2 + log(local_radius) s1234",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def raw_metrics(record: dict, checkpoint: str = "ckpt_best") -> dict:
    return read_json(
        Path(record["run_dir"]) / f"eval/{checkpoint}/metrics.json"
    )["test"]


def extra_metrics(record: dict, checkpoint: str = "ckpt_best") -> dict:
    metrics = raw_metrics(record, checkpoint)
    return {
        "high_wss_nrmse_range": metrics["regional_field"]["high_wss"][
            "nrmse_range"
        ],
        "physical_p99_ratio": metrics["calibration"]["p99_pred_true_ratio"],
        "surface_metric_mode": metrics["surface_metric_mode"],
        "n_cases": metrics["aggregate"]["n_cases"],
    }


def slurm_states() -> list[dict[str, str]]:
    output = subprocess.check_output(
        [
            "/public/slurm/bin/sacct",
            "-n",
            "-P",
            "-j",
            "11032,11033",
            "--format=JobID,JobName,State,ExitCode,Elapsed,Start,End",
        ],
        text=True,
    )
    expected = {"11032", "11033_0", "11033_1"}
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        job_id, job_name, state, exit_code, elapsed, start, end = line.split(
            "|"
        )
        if job_id not in expected:
            continue
        rows.append(
            {
                "job_id": job_id,
                "job_name": job_name,
                "state": state,
                "exit_code": exit_code,
                "elapsed": elapsed,
                "start": start,
                "end": end,
            }
        )
    if {row["job_id"] for row in rows} != expected:
        raise RuntimeError(
            f"missing Slurm rows: expected={sorted(expected)} got={rows}"
        )
    return rows


def registered_gate(result: dict, records: dict) -> dict:
    delta = result["aggregate_treatment_minus_control"]
    control = records[CONTROL]
    treatment = records[TREATMENT]
    delta_high_nrmse = (
        treatment["extra"]["high_wss_nrmse_range"]
        - control["extra"]["high_wss_nrmse_range"]
    )
    primary = {
        "physical_r2_casebalanced": {
            "delta": delta["physical_r2_casebalanced"],
            "threshold": 0.012,
            "pass": delta["physical_r2_casebalanced"] >= 0.012,
        },
        "high_wss_nrmse_range": {
            "delta": delta_high_nrmse,
            "threshold": -0.002,
            "pass": delta_high_nrmse <= -0.002,
        },
    }
    guards = {
        "normalized_r2_casebalanced": {
            "delta": delta["normalized_r2_casebalanced"],
            "threshold": -0.01,
            "pass": delta["normalized_r2_casebalanced"] >= -0.01,
        },
        "physical_mae": {
            "delta": delta["physical_mae"],
            "threshold": 0.05,
            "pass": delta["physical_mae"] <= 0.05,
        },
        "physical_top10_iou": {
            "delta": delta["physical_top10_iou"],
            "threshold": -0.005,
            "pass": delta["physical_top10_iou"] >= -0.005,
        },
    }
    primary_pass = any(item["pass"] for item in primary.values())
    guards_pass = all(item["pass"] for item in guards.values())
    return {
        "primary_rule": (
            "delta physical R2_casebalanced >= +0.012 OR "
            "delta high-WSS nRMSE_range <= -0.002"
        ),
        "primary": primary,
        "primary_pass": primary_pass,
        "guards": guards,
        "guards_pass": guards_pass,
        "decision": (
            "go_promote_logradius_as_single_seed_development_anchor"
            if primary_pass and guards_pass
            else "no_go_keep_same_h2_anchor"
        ),
    }


def main() -> None:
    manifest = read_json(MANIFEST)
    static_audit = read_json(STATIC_AUDIT)
    gpu_audit = read_json(GPU_AUDIT)
    submission = read_json(SUBMISSION)
    rows = manifest["configs"]
    if (
        manifest.get("status") != "prepared"
        or [row["experiment_id"] for row in rows] != list(ORDER)
        or static_audit.get("status") != "passed"
        or gpu_audit.get("status") != "passed"
        or submission.get("status") != "submitted"
    ):
        raise RuntimeError("prepared/static/GPU/submission contract failed")
    if static_audit["manifest_sha256"] != sha256(MANIFEST):
        raise RuntimeError("static audit references a different manifest")
    if gpu_audit["static_audit_sha256"] != sha256(STATIC_AUDIT):
        raise RuntimeError("GPU audit references a different static audit")

    records = {}
    for row in rows:
        experiment_id = row["experiment_id"]
        record = build_record(
            experiment_id,
            Path(row["run_dir"]),
            1234,
            row["role"],
            Path(row["config"]),
            row["config_sha256"],
        )
        record["label"] = LABELS[experiment_id]
        record["role"] = row["role"]
        record["input_features"] = row["input_features"]
        record["input_dim"] = row["input_dim"]
        record["extra"] = extra_metrics(record)
        record["extra_last"] = extra_metrics(record, "ckpt_last")
        if len(record["per_case"]) != 36:
            record["integrity_errors"].append(
                f"expected_36_cases_got_{len(record['per_case'])}"
            )
        if record["extra"]["surface_metric_mode"] != "legacy_vertex":
            record["integrity_errors"].append("unexpected_surface_metric_mode")
        if record["extra"]["n_cases"] != 36:
            record["integrity_errors"].append("aggregate_n_cases_not_36")
        record["integrity"] = (
            "passed" if not record["integrity_errors"] else "failed"
        )
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

    paired = comparison(
        "logradius_vs_concurrent_same_h2",
        records[CONTROL],
        records[TREATMENT],
        20260729,
    )
    paired["extra_treatment_minus_control"] = {
        key: (
            records[TREATMENT]["extra"][key] - records[CONTROL]["extra"][key]
        )
        for key in ("high_wss_nrmse_range", "physical_p99_ratio")
    }
    gate = registered_gate(paired, records)

    source_anchor = manifest["source_anchor_best"]
    historical = {
        "experiment_id": "lsa2_same_h2_pinball_q90_lam020_s1234",
        "role": "reference_only",
        "metrics": source_anchor,
    }
    historical_deltas = {}
    for experiment_id in ORDER:
        record = records[experiment_id]
        historical_deltas[experiment_id] = {
            "physical_r2_casebalanced": (
                record["best"]["physical_r2_casebalanced"]
                - source_anchor["physical_r2_casebalanced"]
            ),
            "normalized_r2_casebalanced": (
                record["best"]["normalized_r2_casebalanced"]
                - source_anchor["normalized_r2_casebalanced"]
            ),
            "physical_mae": (
                record["best"]["physical_mae"]
                - source_anchor["physical_mae_casebalanced"]
            ),
            "high_wss_nrmse_range": (
                record["extra"]["high_wss_nrmse_range"]
                - source_anchor["high_wss_nrmse_range"]
            ),
            "physical_top10_iou": (
                record["best"]["physical_top10_iou"]
                - source_anchor["top10_iou"]
            ),
        }

    payload = {
        "schema_version": 1,
        "analysis_date": "2026-07-29",
        "scope": (
            "REG-P10-LSA2 SAME H2 exact seed1234 concurrent reproduction "
            "versus one added train-only standardized log(local_radius) input; "
            "mixed 138/0/36; legacy_vertex"
        ),
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha256(MANIFEST),
        "static_audit": str(STATIC_AUDIT.resolve()),
        "static_audit_sha256": sha256(STATIC_AUDIT),
        "gpu_audit": str(GPU_AUDIT.resolve()),
        "gpu_audit_sha256": sha256(GPU_AUDIT),
        "submission": submission,
        "slurm_states": states,
        "records": records,
        "paired_comparison": paired,
        "registered_gate": gate,
        "historical_h2_reference": historical,
        "current_minus_historical_h2_reference": historical_deltas,
        "matrix_decision": {
            "winner": TREATMENT,
            "formal_paired_gate": gate["decision"],
            "new_single_seed_development_anchor": TREATMENT,
            "keep_historical_h2_as_reference": True,
            "do_not_start_multiseed_yet": True,
        },
        "interpretation_limits": [
            "Both arms use seed1234 only; this is a single-seed engineering screen.",
            "test36 is a repeatedly reused engineering development set, not an independent final test set.",
            "ckpt_best is selected by train loss because the frozen split has no validation partition; ckpt_last is sensitivity-only.",
            "Formal promotion uses the concurrent same-seed H2 control; the historical H2 run is reference-only because training-trajectory drift is present.",
            "No area-weighted metric is used because some cases do not meet the area-definition requirements.",
            "The added feature is the natural logarithm of physical-mm local_radius and is standardized from the control106 training-only source.",
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
            "role",
            "input_dim",
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
                    "role": record["role"],
                    "input_dim": record["input_dim"],
                    "best_epoch": record["history"]["best_epoch"],
                    **{key: record["best"][key] for key in AGG_KEYS},
                    "high_wss_nrmse_range": record["extra"][
                        "high_wss_nrmse_range"
                    ],
                    "physical_p99_ratio": record["extra"][
                        "physical_p99_ratio"
                    ],
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
            "decision",
            "delta_r2cb",
            "delta_normalized_r2cb",
            "delta_mae",
            "delta_rmse",
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
        delta = paired["aggregate_treatment_minus_control"]
        extra = paired["extra_treatment_minus_control"]
        domains = paired["domain_r2_delta"]
        cases = paired["paired_case_stats"]
        writer.writerow(
            {
                "comparison": paired["comparison"],
                "decision": gate["decision"],
                "delta_r2cb": delta["physical_r2_casebalanced"],
                "delta_normalized_r2cb": delta[
                    "normalized_r2_casebalanced"
                ],
                "delta_mae": delta["physical_mae"],
                "delta_rmse": delta["physical_rmse"],
                "delta_high_wss_nrmse_range": extra[
                    "high_wss_nrmse_range"
                ],
                "delta_high_wss_r2": delta["high_wss_r2"],
                "delta_top10_iou": delta["physical_top10_iou"],
                "delta_p99_ratio": extra["physical_p99_ratio"],
                "delta_AG": domains["AG"],
                "delta_AAA": domains["AAA"],
                "delta_ILO": domains["ILO"],
                "case_r2_mean_delta": cases["case_r2"]["mean_delta"],
                "case_r2_ci95_low": cases["case_r2"]["ci95_low"],
                "case_r2_ci95_high": cases["case_r2"]["ci95_high"],
                "case_r2_wins": cases["case_r2"]["wins"],
                "case_r2_losses": cases["case_r2"]["losses"],
                "case_high_wss_r2_mean_delta": cases["case_high_wss_r2"][
                    "mean_delta"
                ],
                "case_high_wss_r2_ci95_low": cases["case_high_wss_r2"][
                    "ci95_low"
                ],
                "case_high_wss_r2_ci95_high": cases["case_high_wss_r2"][
                    "ci95_high"
                ],
                "case_top10_iou_mean_delta": cases["case_top10_iou"][
                    "mean_delta"
                ],
                "case_top10_iou_ci95_low": cases["case_top10_iou"][
                    "ci95_low"
                ],
                "case_top10_iou_ci95_high": cases["case_top10_iou"][
                    "ci95_high"
                ],
            }
        )

    print(
        json.dumps(
            {
                "status": "passed",
                "records": len(records),
                "decision": gate["decision"],
                "delta_r2cb": paired["aggregate_treatment_minus_control"][
                    "physical_r2_casebalanced"
                ],
                "delta_high_wss_nrmse_range": paired[
                    "extra_treatment_minus_control"
                ]["high_wss_nrmse_range"],
                "json": str(OUT_JSON),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
