#!/usr/bin/env python3
"""Audit and summarize the completed REG-P10 Transformer matrix."""
from __future__ import annotations

import csv
import json
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
MANIFEST = PREFLIGHT / "regp10_transformer_matrix_20260726_prepared.json"
PARENT_RUN = (
    ROOT
    / "training_wss_min/runs/pointnetpp_s3_regularization/outputs/"
    "droppath010_s1234"
)
OUT_JSON = PREFLIGHT / "regp10_transformer_matrix_20260726_results_analysis.json"
OUT_SUMMARY = (
    PREFLIGHT / "regp10_transformer_matrix_20260726_results_summary.csv"
)
OUT_PAIRS = (
    PREFLIGHT / "regp10_transformer_matrix_20260726_paired_case_stats.csv"
)

LABELS = {
    "localtf_sa1_s1234": "L-SA1",
    "localtf_sa2_s1234": "L-SA2",
    "localtf_sa3_s1234": "L-SA3",
    "localtf_sa12_s1234": "L-SA12",
    "localtf_sa13_s1234": "L-SA13",
    "localtf_sa23_s1234": "L-SA23",
    "localtf_sa123_s1234": "L-SA123",
    "globaltf_sa3_s1234": "G-SA3",
}

GATE = {
    "min_delta_r2cb": 0.012,
    "max_delta_mae_pa": 0.03,
    "min_delta_high_wss_r2": -0.01,
    "min_delta_each_domain_r2": -0.02,
}


def gate_decision(result: dict) -> dict:
    delta = result["aggregate_treatment_minus_control"]
    domains = result["domain_r2_delta"]
    checks = {
        "r2cb": delta["physical_r2_casebalanced"] >= GATE["min_delta_r2cb"],
        "mae": delta["physical_mae"] <= GATE["max_delta_mae_pa"],
        "high_wss": delta["high_wss_r2"] >= GATE["min_delta_high_wss_r2"],
        "domains": min(domains.values()) >= GATE["min_delta_each_domain_r2"],
    }
    if all(checks.values()):
        decision = "go_to_three_seed_confirmation"
    elif (
        delta["physical_r2_casebalanced"] > 0
        and checks["mae"]
        and checks["high_wss"]
        and checks["domains"]
    ):
        decision = "below_primary_gate"
    else:
        decision = "no_go_single_seed"
    return {"decision": decision, "checks": checks}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = manifest["configs"]
    if manifest.get("status") != "prepared" or len(rows) != 8:
        raise RuntimeError("expected exact prepared 8-arm REG-P10 matrix")

    parent = build_record("regp10_parent_s1234", PARENT_RUN, 1234, "parent")
    records = {parent["experiment_id"]: parent}
    comparisons = []
    for index, row in enumerate(rows):
        experiment_id = row["experiment_id"]
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
        record["coarse_attention"] = row["coarse_attention"]
        records[experiment_id] = record
        result = comparison(
            f"{experiment_id}_vs_regp10",
            parent,
            record,
            20260726 + index * 100,
        )
        result["gate"] = gate_decision(result)
        comparisons.append(result)

    failed = {
        experiment_id: record["integrity_errors"]
        for experiment_id, record in records.items()
        if record["integrity"] != "passed"
    }
    if failed:
        raise RuntimeError(f"integrity failures: {failed}")

    by_treatment = {item["treatment"]: item for item in comparisons}
    ranking = sorted(
        (record for key, record in records.items() if key != parent["experiment_id"]),
        key=lambda item: item["best"]["physical_r2_casebalanced"],
        reverse=True,
    )
    promoted = [
        item["treatment"]
        for item in comparisons
        if item["gate"]["decision"] == "go_to_three_seed_confirmation"
    ]

    payload = {
        "schema_version": 1,
        "analysis_date": "2026-07-26",
        "scope": (
            "REG-P10 seed1234 local-SA Transformer full non-empty stage "
            "factorial plus SA3 global-attention control; historical mixed test36"
        ),
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha256(MANIFEST),
        "parent": parent,
        "records": records,
        "paired_comparisons": comparisons,
        "gate": GATE,
        "promoted_to_three_seed_confirmation": promoted,
        "rank_by_physical_r2_casebalanced": [
            {
                "rank": rank,
                "experiment_id": record["experiment_id"],
                "label": record["label"],
                "physical_r2_casebalanced": record["best"][
                    "physical_r2_casebalanced"
                ],
                "delta_r2cb": by_treatment[record["experiment_id"]][
                    "aggregate_treatment_minus_control"
                ]["physical_r2_casebalanced"],
                "decision": by_treatment[record["experiment_id"]]["gate"][
                    "decision"
                ],
            }
            for rank, record in enumerate(ranking, 1)
        ],
        "interpretation_limits": [
            "All eight treatment arms and the REG-P10 parent use seed1234 only.",
            "test36 is a historical engineering screening set, not an independent confirmation set.",
            "ckpt_best selected by train loss is the registered result; ckpt_last is sensitivity-only.",
            "Only an arm passing the R2, MAE, high-WSS, and per-domain guards is promoted to seeds 7/2025.",
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
            "variant",
            "local_transformer_stages",
            "coarse_attention",
            "best_epoch",
            *AGG_KEYS,
            "AG_r2",
            "AAA_r2",
            "ILO_r2",
            "last_minus_best_r2cb",
            "decision",
            "integrity",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for experiment_id, record in records.items():
            result = by_treatment.get(experiment_id)
            writer.writerow(
                {
                    "experiment_id": experiment_id,
                    "label": record.get("label", "REG-P10"),
                    "variant": record["variant"],
                    "local_transformer_stages": json.dumps(
                        record.get("local_transformer_stages", [])
                    ),
                    "coarse_attention": record.get("coarse_attention", False),
                    "best_epoch": record["history"].get("best_epoch"),
                    **{key: record["best"][key] for key in AGG_KEYS},
                    "AG_r2": record["domains"]["AG"],
                    "AAA_r2": record["domains"]["AAA"],
                    "ILO_r2": record["domains"]["ILO"],
                    "last_minus_best_r2cb": record["last_minus_best"][
                        "physical_r2_casebalanced"
                    ],
                    "decision": (
                        "parent" if result is None else result["gate"]["decision"]
                    ),
                    "integrity": record["integrity"],
                }
            )

    with OUT_PAIRS.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "comparison",
            "label",
            "delta_r2cb",
            "delta_mae",
            "delta_rmse",
            "delta_high_wss_r2",
            "delta_top10_iou",
            "delta_AG",
            "delta_AAA",
            "delta_ILO",
            "case_r2_mean_delta",
            "case_r2_ci95_low",
            "case_r2_ci95_high",
            "case_r2_wins",
            "case_r2_losses",
            "decision",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in comparisons:
            delta = result["aggregate_treatment_minus_control"]
            domains = result["domain_r2_delta"]
            cases = result["paired_case_stats"]["case_r2"]
            writer.writerow(
                {
                    "comparison": result["comparison"],
                    "label": LABELS[result["treatment"]],
                    "delta_r2cb": delta["physical_r2_casebalanced"],
                    "delta_mae": delta["physical_mae"],
                    "delta_rmse": delta["physical_rmse"],
                    "delta_high_wss_r2": delta["high_wss_r2"],
                    "delta_top10_iou": delta["physical_top10_iou"],
                    "delta_AG": domains["AG"],
                    "delta_AAA": domains["AAA"],
                    "delta_ILO": domains["ILO"],
                    "case_r2_mean_delta": cases["mean_delta"],
                    "case_r2_ci95_low": cases["ci95_low"],
                    "case_r2_ci95_high": cases["ci95_high"],
                    "case_r2_wins": cases["wins"],
                    "case_r2_losses": cases["losses"],
                    "decision": result["gate"]["decision"],
                }
            )

    print(
        json.dumps(
            {
                "status": "passed",
                "records": len(records),
                "winner": ranking[0]["experiment_id"],
                "winner_r2cb": ranking[0]["best"][
                    "physical_r2_casebalanced"
                ],
                "promoted": promoted,
                "json": str(OUT_JSON),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
