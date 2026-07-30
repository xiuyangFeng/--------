#!/usr/bin/env python3
"""Analyze fixed-test27 D2 PNXR-GeoPE local-Transformer SA1/SA2 controls."""
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
MANIFEST = (
    PREFLIGHT / "d2_c125_k64_pnxr_geope_transformer_20260726_prepared.json"
)
SUBMISSION = (
    PREFLIGHT / "d2_c125_k64_pnxr_geope_transformer_20260726_submission.json"
)
PARENT_RUN = (
    ROOT
    / "training_wss_min/runs/pointnetpp_d2_c125_k64_pnxr_geope/"
    "outputs/d2_c125_k64_pnxr_geope"
)
OUT_JSON = (
    PREFLIGHT
    / "d2_c125_k64_pnxr_geope_transformer_20260726_results_analysis.json"
)
OUT_SUMMARY = (
    PREFLIGHT
    / "d2_c125_k64_pnxr_geope_transformer_20260726_results_summary.csv"
)
OUT_PAIRS = (
    PREFLIGHT
    / "d2_c125_k64_pnxr_geope_transformer_20260726_paired_case_stats.csv"
)
PARENT_ID = "d2_c125_k64_pnxr_geope_parent"
ORDER = (
    "d2_pnxr_geope_localtf_sa1_s1234",
    "d2_pnxr_geope_localtf_sa2_s1234",
)
LABELS = {
    PARENT_ID: "D2 PNXR+GeoPE",
    "d2_pnxr_geope_localtf_sa1_s1234": "D2-L-SA1",
    "d2_pnxr_geope_localtf_sa2_s1234": "D2-L-SA2",
}


def slurm_states() -> dict[str, dict[str, str]]:
    command = [
        "/public/slurm/bin/sacct",
        "-j",
        "10979,10980",
        "-n",
        "-P",
        "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End",
    ]
    states: dict[str, dict[str, str]] = {}
    for line in subprocess.check_output(command, text=True).splitlines():
        fields = line.split("|")
        if len(fields) < 7 or "." in fields[0]:
            continue
        job_id, name, state, exit_code, elapsed, start, end = fields[:7]
        states[job_id] = {
            "name": name,
            "state": state,
            "exit_code": exit_code,
            "elapsed": elapsed,
            "start": start,
            "end": end,
        }
    clean_gate = [
        value
        for value in states.values()
        if value["name"] == "d2tf_pf"
        and value["state"] == "COMPLETED"
        and value["exit_code"] == "0:0"
    ]
    clean_train = [
        value
        for value in states.values()
        if value["name"] == "d2tf"
        and value["state"] == "COMPLETED"
        and value["exit_code"] == "0:0"
    ]
    if len(clean_gate) != 1 or len(clean_train) != 2:
        raise RuntimeError(f"Slurm completion contract failed: {states}")
    return states


def last_delta(control: dict, treatment: dict) -> dict[str, float]:
    keys = (
        "physical_r2_casebalanced",
        "physical_r2_pooled",
        "physical_r2_case_mean",
        "physical_mae",
        "physical_rmse",
        "high_wss_r2",
        "physical_top10_iou",
        "normalized_r2_casebalanced",
    )
    return {key: treatment["last"][key] - control["last"][key] for key in keys}


def decision(result: dict) -> str:
    delta = result["aggregate_treatment_minus_control"]
    case = result["paired_case_stats"]["case_r2"]
    if (
        delta["physical_r2_casebalanced"] < -0.01
        or delta["physical_mae"] > 0.03
        or delta["high_wss_r2"] < -0.02
    ):
        return "no_go"
    if (
        delta["physical_r2_casebalanced"] >= 0.012
        and case["ci95_low"] > 0
    ):
        return "positive_screen"
    return "flat_no_promotion"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    submission = json.loads(SUBMISSION.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "prepared"
        or len(manifest.get("configs", [])) != 2
        or submission.get("status") != "submitted"
    ):
        raise RuntimeError("prepared/submission contract failed")
    states = slurm_states()

    parent = build_record(PARENT_ID, PARENT_RUN, 1234, "parent")
    parent["label"] = LABELS[PARENT_ID]
    records = {PARENT_ID: parent}
    comparisons = []
    rows_by_id = {row["experiment_id"]: row for row in manifest["configs"]}
    for index, experiment_id in enumerate(ORDER):
        row = rows_by_id[experiment_id]
        record = build_record(
            experiment_id,
            Path(row["run_dir"]),
            row["seed"],
            row["variant"],
            Path(row["config"]),
            row["sha256"],
        )
        record["label"] = LABELS[experiment_id]
        record["local_transformer_stages"] = row["local_transformer_stages"]
        records[experiment_id] = record
        result = comparison(
            f"{experiment_id}_vs_d2_parent",
            parent,
            record,
            20260727 + index * 100,
        )
        result["decision"] = decision(result)
        result["last_treatment_minus_parent_last"] = last_delta(parent, record)
        comparisons.append(result)

    failed = {
        experiment_id: record["integrity_errors"]
        for experiment_id, record in records.items()
        if record["integrity"] != "passed"
    }
    if failed:
        raise RuntimeError(f"integrity failures: {failed}")

    direct = comparison(
        "d2_localtf_sa2_vs_sa1",
        records[ORDER[0]],
        records[ORDER[1]],
        20260927,
    )
    by_treatment = {item["treatment"]: item for item in comparisons}
    payload = {
        "schema_version": 1,
        "analysis_date": "2026-07-27",
        "scope": (
            "Fixed AG/AAA 106/0/27 D2 c125×k64 PointNeXt-R + 7D "
            "LocalGeoPE; seed1234 local-Transformer SA1/SA2 controls"
        ),
        "sources": {
            "manifest": str(MANIFEST.resolve()),
            "manifest_sha256": sha256(MANIFEST),
            "submission": str(SUBMISSION.resolve()),
            "submission_sha256": sha256(SUBMISSION),
        },
        "slurm": states,
        "records": records,
        "paired_comparisons_vs_parent": comparisons,
        "direct_sa2_minus_sa1": direct,
        "primary_checkpoint": "ckpt_best selected by train_loss",
        "interpretation": {
            "sa1": "No-Go: physical R2, errors, high-WSS, hotspot, AG and AAA all regress.",
            "sa2": (
                "Flat/no promotion on the registered best checkpoint: physical "
                "R2_cb is nearly unchanged, case CI crosses zero, normalized R2 "
                "declines, and high-WSS slightly regresses."
            ),
            "last_sensitivity": (
                "SA2 ckpt_last is numerically better than its ckpt_best and the "
                "parent last checkpoint, but last is sensitivity-only and cannot "
                "replace the registered train-loss-selected primary result."
            ),
        },
        "interpretation_limits": [
            "All treatment arms use seed1234 only.",
            "test27 is a historical fixed-protocol engineering set.",
            "Comparisons are valid only against the same D2 PNXR-GeoPE parent.",
            "Physical and normalized R2 are reported separately and must not be interchanged.",
            "No checkpoint is reselected from test metrics.",
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
            "local_transformer_stages",
            "best_epoch",
            *AGG_KEYS,
            "AG_r2",
            "AAA_r2",
            "last_minus_best_r2cb",
            "decision",
            "integrity",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for experiment_id, record in records.items():
            paired = by_treatment.get(experiment_id)
            writer.writerow(
                {
                    "experiment_id": experiment_id,
                    "label": record["label"],
                    "local_transformer_stages": json.dumps(
                        record.get("local_transformer_stages", [])
                    ),
                    "best_epoch": record["history"]["best_epoch"],
                    **{key: record["best"][key] for key in AGG_KEYS},
                    "AG_r2": record["domains"]["AG"],
                    "AAA_r2": record["domains"]["AAA"],
                    "last_minus_best_r2cb": record["last_minus_best"][
                        "physical_r2_casebalanced"
                    ],
                    "decision": "parent" if paired is None else paired["decision"],
                    "integrity": record["integrity"],
                }
            )

    with OUT_PAIRS.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "comparison",
            "label",
            "delta_r2cb",
            "delta_normalized_r2cb",
            "delta_mae",
            "delta_rmse",
            "delta_high_wss_r2",
            "delta_top10_iou",
            "delta_AG",
            "delta_AAA",
            "case_r2_mean_delta",
            "case_r2_ci95_low",
            "case_r2_ci95_high",
            "case_r2_wins",
            "case_r2_losses",
            "last_vs_parent_last_r2cb",
            "decision",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in comparisons:
            delta = result["aggregate_treatment_minus_control"]
            domains = result["domain_r2_delta"]
            case = result["paired_case_stats"]["case_r2"]
            writer.writerow(
                {
                    "comparison": result["comparison"],
                    "label": LABELS[result["treatment"]],
                    "delta_r2cb": delta["physical_r2_casebalanced"],
                    "delta_normalized_r2cb": delta[
                        "normalized_r2_casebalanced"
                    ],
                    "delta_mae": delta["physical_mae"],
                    "delta_rmse": delta["physical_rmse"],
                    "delta_high_wss_r2": delta["high_wss_r2"],
                    "delta_top10_iou": delta["physical_top10_iou"],
                    "delta_AG": domains["AG"],
                    "delta_AAA": domains["AAA"],
                    "case_r2_mean_delta": case["mean_delta"],
                    "case_r2_ci95_low": case["ci95_low"],
                    "case_r2_ci95_high": case["ci95_high"],
                    "case_r2_wins": case["wins"],
                    "case_r2_losses": case["losses"],
                    "last_vs_parent_last_r2cb": result[
                        "last_treatment_minus_parent_last"
                    ]["physical_r2_casebalanced"],
                    "decision": result["decision"],
                }
            )

    print(
        json.dumps(
            {
                "status": "passed",
                "records": len(records),
                "sa1_r2cb": records[ORDER[0]]["best"][
                    "physical_r2_casebalanced"
                ],
                "sa2_r2cb": records[ORDER[1]]["best"][
                    "physical_r2_casebalanced"
                ],
                "decisions": {
                    item["treatment"]: item["decision"] for item in comparisons
                },
                "json": str(OUT_JSON),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
