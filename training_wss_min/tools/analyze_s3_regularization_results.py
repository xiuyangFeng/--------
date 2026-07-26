#!/usr/bin/env python3
"""Audit and summarize the S3-GEOPE three-seed regularization matrix."""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from training_wss_min.tools.analyze_d2_k64_ilo_structure_results import (
    AGG_KEYS,
    sha256,
)
from training_wss_min.tools.analyze_s3_rootcause_results import (
    PARENT,
    build_record,
    comparison,
)


REPO = Path(__file__).resolve().parents[2]
PREFLIGHT = REPO / "training_wss_min/preflight"
MANIFEST = PREFLIGHT / "s3_regularization_matrix_prepared.json"
OUT_JSON = PREFLIGHT / "s3_regularization_results_analysis.json"
OUT_SUMMARY = PREFLIGHT / "s3_regularization_results_summary.csv"
OUT_PAIRS = PREFLIGHT / "s3_regularization_paired_case_stats.csv"
VARIANTS = (
    "head_dropout01",
    "droppath005",
    "droppath010",
    "neighbordrop005",
)
SEEDS = (1234, 7, 2025)
DOMAINS = ("AG", "AAA", "ILO")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarize_variant(variant: str, comparisons: dict[tuple[str, int], dict]) -> dict:
    rows = [comparisons[(variant, seed)] for seed in SEEDS]
    metric_keys = (
        "physical_r2_casebalanced",
        "physical_mae",
        "physical_rmse",
        "high_wss_r2",
        "physical_top10_iou",
        "spearman_case_mean",
    )
    metric_deltas = {
        key: [r["aggregate_treatment_minus_control"][key] for r in rows]
        for key in metric_keys
    }
    domain_deltas = {
        domain: [r["domain_r2_delta"][domain] for r in rows]
        for domain in DOMAINS
    }
    r2 = metric_deltas["physical_r2_casebalanced"]
    r2_mean = mean(r2)
    wins = sum(value > 0 for value in r2)
    mae_ok = mean(metric_deltas["physical_mae"]) <= 0
    high_ok = mean(metric_deltas["high_wss_r2"]) >= -0.01
    domain_means = {domain: mean(values) for domain, values in domain_deltas.items()}
    domain_ok = min(domain_means.values()) >= -0.02
    if r2_mean >= 0.015 and wins >= 2 and mae_ok and high_ok and domain_ok:
        decision = "Go"
    elif r2_mean >= 0.005 and wins >= 2 and sum((mae_ok, high_ok, domain_ok)) >= 2:
        decision = "Weak-Go"
    else:
        decision = "No-Go"
    return {
        "variant": variant,
        "seeds": list(SEEDS),
        "delta_r2cb_per_seed": dict(zip(map(str, SEEDS), r2)),
        "delta_r2cb_mean": r2_mean,
        "delta_r2cb_sd": statistics.stdev(r2),
        "r2_wins": wins,
        "r2_losses": sum(value < 0 for value in r2),
        "metric_deltas_per_seed": metric_deltas,
        "metric_delta_means": {
            key: mean(values) for key, values in metric_deltas.items()
        },
        "domain_deltas_per_seed": domain_deltas,
        "domain_delta_means": domain_means,
        "guardrails": {
            "mae_mean_nonworse": mae_ok,
            "high_wss_mean_ge_minus_0p01": high_ok,
            "all_domain_means_ge_minus_0p02": domain_ok,
        },
        "decision": decision,
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records: dict[str, dict] = {}
    for seed, run_dir in PARENT.items():
        records[f"s3_parent_s{seed}"] = build_record(
            f"s3_parent_s{seed}", run_dir, seed, "s3_parent"
        )
    for row in manifest["configs"]:
        records[row["experiment_id"]] = build_record(
            row["experiment_id"],
            Path(row["run_dir"]),
            int(row["seed"]),
            row["variant"],
            Path(row["config"]),
            row["sha256"],
        )
    failed = {
        experiment_id: record["integrity_errors"]
        for experiment_id, record in records.items()
        if record["integrity"] != "passed"
    }
    if failed:
        raise RuntimeError(f"integrity failures: {failed}")

    comparisons = []
    comparison_index: dict[tuple[str, int], dict] = {}
    for row in manifest["configs"]:
        seed = int(row["seed"])
        result = comparison(
            f"{row['experiment_id']}_vs_s3parent",
            records[f"s3_parent_s{seed}"],
            records[row["experiment_id"]],
            20260724 + len(comparisons) * 100,
        )
        comparisons.append(result)
        comparison_index[(row["variant"], seed)] = result
    summaries = [
        summarize_variant(variant, comparison_index) for variant in VARIANTS
    ]
    ranking = sorted(
        summaries, key=lambda row: row["delta_r2cb_mean"], reverse=True
    )

    payload = {
        "schema_version": 1,
        "analysis_date": "2026-07-24",
        "scope": (
            "S3-GEOPE strict single-variable regularization matrix; "
            "three paired seeds; historical mixed test36 engineering screen"
        ),
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha256(MANIFEST),
        "s3_parent_r2cb": {
            str(seed): records[f"s3_parent_s{seed}"]["best"][
                "physical_r2_casebalanced"
            ]
            for seed in SEEDS
        },
        "records": records,
        "paired_comparisons": comparisons,
        "three_seed_summaries": summaries,
        "rank_by_mean_delta_r2cb": [
            {
                "rank": index,
                "variant": row["variant"],
                "delta_r2cb_mean": row["delta_r2cb_mean"],
                "delta_r2cb_sd": row["delta_r2cb_sd"],
                "decision": row["decision"],
            }
            for index, row in enumerate(ranking, 1)
        ],
        "decision_rule": {
            "Go": (
                "mean ΔR²_cb>=+0.015; >=2/3 positive; mean MAE nonworse; "
                "mean high-WSS Δ>=-0.01; all domain mean Δ>=-0.02"
            ),
            "Weak-Go": (
                "mean ΔR²_cb>=+0.005; >=2/3 positive; at least two of "
                "MAE/high-WSS/domain guardrails pass"
            ),
            "No-Go": "otherwise",
        },
        "interpretation_limits": [
            "test36 is a repeatedly used engineering screening set.",
            "checkpoint selection remains ckpt_best(train_loss).",
            "a Go result is a candidate anchor, not independent confirmation.",
        ],
    }
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "experiment_id",
            "variant",
            "seed",
            "best_epoch",
            *AGG_KEYS,
            "AG_r2",
            "AAA_r2",
            "ILO_r2",
            "integrity",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records.values():
            writer.writerow({
                "experiment_id": record["experiment_id"],
                "variant": record["variant"],
                "seed": record["seed"],
                "best_epoch": record["history"].get("best_epoch"),
                **{key: record["best"][key] for key in AGG_KEYS},
                "AG_r2": record["domains"]["AG"],
                "AAA_r2": record["domains"]["AAA"],
                "ILO_r2": record["domains"]["ILO"],
                "integrity": record["integrity"],
            })

    with OUT_PAIRS.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "comparison",
            "control",
            "treatment",
            "delta_r2cb",
            "delta_mae",
            "delta_high_wss_r2",
            "delta_AG",
            "delta_AAA",
            "delta_ILO",
            "case_r2_mean_delta",
            "case_r2_ci95_low",
            "case_r2_ci95_high",
            "case_r2_wins",
            "case_r2_losses",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in comparisons:
            case_r2 = result["paired_case_stats"]["case_r2"]
            delta = result["aggregate_treatment_minus_control"]
            writer.writerow({
                "comparison": result["comparison"],
                "control": result["control"],
                "treatment": result["treatment"],
                "delta_r2cb": delta["physical_r2_casebalanced"],
                "delta_mae": delta["physical_mae"],
                "delta_high_wss_r2": delta["high_wss_r2"],
                "delta_AG": result["domain_r2_delta"]["AG"],
                "delta_AAA": result["domain_r2_delta"]["AAA"],
                "delta_ILO": result["domain_r2_delta"]["ILO"],
                "case_r2_mean_delta": case_r2["mean_delta"],
                "case_r2_ci95_low": case_r2["ci95_low"],
                "case_r2_ci95_high": case_r2["ci95_high"],
                "case_r2_wins": case_r2["wins"],
                "case_r2_losses": case_r2["losses"],
            })
    print(json.dumps({
        "status": "passed",
        "records": len(records),
        "winner": ranking[0]["variant"],
        "winner_mean_delta_r2cb": ranking[0]["delta_r2cb_mean"],
        "winner_decision": ranking[0]["decision"],
        "json": str(OUT_JSON),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
