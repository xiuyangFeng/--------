#!/usr/bin/env python3
"""Analyze the fixed-split D2 c125×k64 PointNeXt-R + LocalGeoPE control."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from training_wss_min.tools.analyze_d2_k64_ilo_structure_results import (
    AGG_KEYS,
    LABELS,
    CASE_METRICS,
    comparison,
    record,
    sha256,
)


ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = ROOT / "training_wss_min/preflight"
MANIFEST = PREFLIGHT / "d2_c125_k64_pnxr_geope_20260726_prepared.json"
SUBMISSION = PREFLIGHT / "d2_c125_k64_pnxr_geope_20260726_resubmission.json"
PARENT_CONFIG = ROOT / "training_wss_min/configs/pointnetpp_sa1_scale_single_seed_20260721/d2_rand5000_c125_k64.json"
PARENT_RUN = ROOT / "training_wss_min/runs/pointnetpp_sa1_scale_single_seed/outputs/d2_rand5000_c125_k64"
OUT_JSON = PREFLIGHT / "d2_c125_k64_pnxr_geope_20260726_results_analysis.json"
OUT_SUMMARY = PREFLIGHT / "d2_c125_k64_pnxr_geope_20260726_results_summary.csv"
OUT_PAIRS = PREFLIGHT / "d2_c125_k64_pnxr_geope_20260726_paired_case_stats.csv"

PARENT_ID = "d2_c125_k64_parent"
TREATMENT_ID = "d2_c125_k64_pnxr_geope"
LABELS[PARENT_ID] = "D2 c125×k64"
LABELS[TREATMENT_ID] = "D2 c125×k64 + PointNeXt-R + LocalGeoPE"


def slurm_state() -> dict[str, dict[str, str]]:
    cmd = [
        "/public/slurm/bin/sacct", "-n", "-P", "-j", "10958,10959",
        "--format=JobIDRaw,State,ExitCode,Elapsed,End",
    ]
    rows = subprocess.check_output(cmd, text=True).splitlines()
    states: dict[str, dict[str, str]] = {}
    for line in rows:
        fields = line.split("|")
        if len(fields) < 5 or "." in fields[0]:
            continue
        states[fields[0]] = {
            "state": fields[1], "exit_code": fields[2], "elapsed": fields[3], "end": fields[4],
        }
    for job_id in ("10958", "10959"):
        if states.get(job_id, {}).get("state") != "COMPLETED" or states[job_id].get("exit_code") != "0:0":
            raise RuntimeError(f"job {job_id} not completed cleanly: {states.get(job_id)}")
    return states


def domain_delta(control: dict, treatment: dict) -> dict[str, float]:
    groups = ("AG", "AAA")
    return {group: treatment["best"]["groups"][group]["r2"] - control["best"]["groups"][group]["r2"] for group in groups}


def write_csv(records: list[dict], paired: dict) -> None:
    fields = ["experiment_id", "label", "job_id", "best_epoch", "best_train_loss", *AGG_KEYS, "integrity"]
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            writer.writerow({
                "experiment_id": item["experiment_id"], "label": item["label"], "job_id": item["job_id"],
                "best_epoch": item["history"]["best_epoch"], "best_train_loss": item["history"]["best_train_loss"],
                **{key: item["best"][key] for key in AGG_KEYS}, "integrity": item["integrity"],
            })
    with OUT_PAIRS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "metric", "aggregate_delta", "mean_case_delta", "ci95_low", "ci95_high", "wins", "ties", "losses", "n_cases",
        ])
        writer.writeheader()
        key_map = {
            "case_r2": "physical_r2_case_mean", "case_mae": "physical_mae", "case_rmse": "physical_rmse",
            "case_high_wss_r2": "high_wss_r2", "case_high_wss_mae": "high_wss_mae",
            "case_top10_iou": "physical_top10_iou", "case_spearman": "spearman_case_mean",
            "case_high_wss_spearman": "high_wss_spearman_case_mean",
        }
        for metric, stats in paired["paired_case_stats"].items():
            writer.writerow({
                "metric": metric, "aggregate_delta": paired["aggregate_treatment_minus_control"][key_map[metric]],
                "mean_case_delta": stats["mean_delta"], "ci95_low": stats["ci95_low"], "ci95_high": stats["ci95_high"],
                "wins": stats["wins"], "ties": stats["ties"], "losses": stats["losses"], "n_cases": stats["n_cases"],
            })


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    submitted = json.loads(SUBMISSION.read_text(encoding="utf-8"))
    if manifest.get("status") != "prepared" or submitted.get("status") != "submitted":
        raise RuntimeError("prepared/resubmission contract failed")
    child = manifest["configs"][0]
    states = slurm_state()
    parent = record(PARENT_ID, PARENT_RUN, "historical-parent", PARENT_CONFIG, sha256(PARENT_CONFIG))
    treatment = record(TREATMENT_ID, Path(child["run_dir"]), "10959", Path(child["config"]), child["sha256"])
    records = [parent, treatment]
    if any(item["integrity"] != "passed" for item in records):
        raise RuntimeError({item["experiment_id"]: item["integrity_errors"] for item in records if item["integrity"] != "passed"})
    paired = comparison("D2_PNXR_GEOPE_vs_D2", parent, treatment, 20260726)
    payload = {
        "schema_version": 1,
        "analysis_date": "2026-07-26",
        "scope": "D2 c125×k64 fixed AG/AAA 106/0/27; single seed=1234; PointNeXt-R + 7D LocalGeoPE vs original D2",
        "sources": {
            "manifest": str(MANIFEST.resolve()), "manifest_sha256": sha256(MANIFEST),
            "resubmission": str(SUBMISSION.resolve()), "resubmission_sha256": sha256(SUBMISSION),
        },
        "slurm": states,
        "records": {item["experiment_id"]: item for item in records},
        "paired_comparison": paired,
        "domain_r2_delta": domain_delta(parent, treatment),
        "interpretation_limits": [
            "One seed only; this is a fixed-protocol screening comparison, not a stability claim.",
            "Comparison is valid only against D2 c125×k64 on the same frozen AG/AAA 106/0/27 protocol.",
            "It must not be numerically ranked against mixed138/test36 S2/S3 results.",
        ],
    }
    write_csv(records, paired)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    delta = paired["aggregate_treatment_minus_control"]
    print(json.dumps({
        "status": "passed", "r2cb": treatment["best"]["physical_r2_casebalanced"],
        "delta_r2cb": delta["physical_r2_casebalanced"], "delta_mae": delta["physical_mae"],
        "json": str(OUT_JSON), "summary_csv": str(OUT_SUMMARY), "paired_csv": str(OUT_PAIRS),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
