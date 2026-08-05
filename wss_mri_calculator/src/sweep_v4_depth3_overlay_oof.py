#!/usr/bin/env python3
"""Select a train-only overlay of depth-3 physics gain on held-out V4 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from batch_validate_cfd import summarize
from calculate_wss_cfd import wss_from_gradient
from calibrate_v4_high_tail_oof import case_tail_metrics, smooth_gate
from calibrate_v4_oof import EPS, load_cache
from validate_v4_high_tail_holdout import pooled_high_metrics


def scalar_case_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    variance = float(np.sum((truth - np.mean(truth)) ** 2))
    alpha = float(
        np.dot(truth, prediction) / max(float(np.dot(prediction, prediction)), EPS)
    )
    high = truth >= np.quantile(truth, 0.90)
    error = prediction - truth
    return {
        "raw_r2": 1.0 - float(np.sum(error**2)) / max(variance, EPS),
        "scaled_r2": 1.0
        - float(np.sum((alpha * prediction - truth) ** 2)) / max(variance, EPS),
        "alpha": alpha,
        "mae_pa": float(np.mean(np.abs(error))),
        "nrmse": float(
            np.sqrt(np.mean(error**2)) / max(float(np.mean(np.abs(truth))), EPS)
        ),
        "high_wss_nrmse": float(
            np.sqrt(np.mean(error[high] ** 2))
            / max(float(np.mean(np.abs(truth[high]))), EPS)
        ),
        **case_tail_metrics(truth, prediction),
    }


def summarize_prediction(
    truth: np.ndarray, prediction: np.ndarray, groups: np.ndarray
) -> dict[str, Any]:
    rows = [
        scalar_case_metrics(truth[groups == case], prediction[groups == case])
        for case in np.unique(groups)
    ]
    metrics = (
        "raw_r2",
        "scaled_r2",
        "alpha",
        "mae_pa",
        "nrmse",
        "high_wss_nrmse",
        "high_r2",
        "high_mae_pa",
        "high_nrmse",
        "high_bias_pa",
        "high_spearman",
        "peak_underestimate_fraction",
        "peak_ratio",
        "top10_jaccard",
    )
    result = {metric: summarize([row[metric] for row in rows]) for metric in metrics}
    high_r2 = np.asarray([row["high_r2"] for row in rows])
    peak_under = np.asarray([row["peak_underestimate_fraction"] for row in rows])
    result["target_gates"] = {
        "cases_high_r2_at_least_0p9": int(np.sum(high_r2 >= 0.90)),
        "cases_peak_underestimate_at_most_10pct": int(np.sum(peak_under <= 0.10)),
        "cases_meeting_both": int(np.sum((high_r2 >= 0.90) & (peak_under <= 0.10))),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-predictions", required=True)
    parser.add_argument("--fusion-cache-dir", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--predictions-out", required=True)
    args = parser.parse_args()

    source = np.load(args.oof_predictions, allow_pickle=False)
    groups = np.asarray(source["point_case"])
    truth = np.asarray(source["truth_mag"], dtype=np.float64)
    base_vector = np.asarray(source["base_wss_vec"], dtype=np.float64)
    base_mag = np.linalg.norm(base_vector, axis=1)
    predicted_rank = np.asarray(source["case_pred_rank"], dtype=np.float64)

    fusion = load_cache(Path(args.fusion_cache_dir))
    fusion_groups = np.asarray(fusion["point_case"])
    if not np.array_equal(groups, fusion_groups):
        raise RuntimeError("OOF predictions and fusion cache point order differ")
    physics_base = np.linalg.norm(
        wss_from_gradient(np.asarray(fusion["gradient_v4_base"], dtype=np.float64), "carreau"),
        axis=1,
    )
    physics_fusion = np.linalg.norm(
        wss_from_gradient(np.asarray(fusion["gradient_v4"], dtype=np.float64), "carreau"),
        axis=1,
    )
    physics_ratio = np.maximum(
        physics_fusion / np.maximum(physics_base, EPS), 1.0
    )

    source_names = (
        "current_safe_oof",
        "tail_w10_s80_oof",
        "tail_w10_peak30_s80_oof",
    )
    candidates = []
    candidate_predictions: dict[str, np.ndarray] = {}
    for source_name in source_names:
        source_prediction = np.asarray(source[source_name], dtype=np.float64)
        settings = [(0.80, 0.0, 1.0)]
        settings.extend(
            (gate_start, strength, cap)
            for gate_start in (0.80, 0.90, 0.95)
            for strength in (0.25, 0.50, 0.75, 1.00)
            for cap in (1.05, 1.10, 1.20)
        )
        for gate_start, strength, cap in settings:
            gate = smooth_gate(predicted_rank, gate_start)
            capped_ratio = np.clip(physics_ratio, 1.0, cap)
            overlay = 1.0 + strength * gate * (capped_ratio - 1.0)
            prediction = source_prediction * overlay
            name = (
                f"{source_name}__g{gate_start:.2f}_s{strength:.2f}_c{cap:.2f}"
            )
            case_balanced = summarize_prediction(truth, prediction, groups)
            pooled = pooled_high_metrics(truth, prediction, groups)
            row = {
                "name": name,
                "source": source_name,
                "gate_start": gate_start,
                "strength": strength,
                "ratio_cap": cap,
                "case_balanced": case_balanced,
                "pooled_high_wss": pooled,
            }
            candidates.append(row)
            candidate_predictions[name] = prediction

    eligible = [
        row
        for row in candidates
        if row["case_balanced"]["raw_r2"]["mean"] >= 0.9535
        and row["pooled_high_wss"]["r2"] >= 0.90
        and row["pooled_high_wss"]["mean_underestimate_fraction"] <= 0.10
    ]
    if not eligible:
        raise RuntimeError("no overlay candidate passed the fixed safety constraints")

    def selection_key(row: dict[str, Any]) -> tuple[float, ...]:
        gates = row["case_balanced"]["target_gates"]
        return (
            float(gates["cases_meeting_both"]),
            float(gates["cases_high_r2_at_least_0p9"]),
            float(gates["cases_peak_underestimate_at_most_10pct"]),
            float(row["case_balanced"]["high_r2"]["mean"]),
            -float(row["case_balanced"]["peak_underestimate_fraction"]["mean"]),
            float(row["case_balanced"]["raw_r2"]["mean"]),
        )

    selected = max(eligible, key=selection_key)
    selected_prediction = candidate_predictions[selected["name"]]
    ranked = sorted(eligible, key=selection_key, reverse=True)
    payload = {
        "status": "train_only_depth3_overlay_oof_selection",
        "source_predictions": str(Path(args.oof_predictions).resolve()),
        "fusion_cache_dir": str(Path(args.fusion_cache_dir).resolve()),
        "selection_rule": {
            "constraints": {
                "case_balanced_overall_raw_r2_mean_at_least": 0.9535,
                "pooled_high_r2_at_least": 0.90,
                "pooled_mean_underestimate_at_most": 0.10,
            },
            "lexicographic_objective": [
                "cases_meeting_both",
                "cases_high_r2_at_least_0p9",
                "cases_peak_underestimate_at_most_10pct",
                "case_balanced_high_r2_mean",
                "negative_case_balanced_peak_underestimate_mean",
                "case_balanced_overall_raw_r2_mean",
            ],
        },
        "physics_ratio": {
            "mean": float(np.mean(physics_ratio)),
            "p95": float(np.quantile(physics_ratio, 0.95)),
            "max": float(np.max(physics_ratio)),
            "fraction_above_one": float(np.mean(physics_ratio > 1.0 + 1e-12)),
        },
        "selected": selected,
        "top20": ranked[:20],
        "n_candidates": len(candidates),
        "n_eligible": len(eligible),
    }
    output = Path(args.json_out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    prediction_output = Path(args.predictions_out).resolve()
    prediction_output.parent.mkdir(parents=True, exist_ok=True)
    with prediction_output.open("wb") as handle:
        np.savez_compressed(
            handle,
            point_case=groups,
            truth_mag=truth.astype(np.float32),
            base_wss_vec=base_vector.astype(np.float32),
            source_prediction=np.asarray(source[selected["source"]], dtype=np.float32),
            physics_ratio=physics_ratio.astype(np.float32),
            selected_prediction=selected_prediction.astype(np.float32),
        )
    print(json.dumps(selected, indent=2, ensure_ascii=False), flush=True)
    print(output)


if __name__ == "__main__":
    main()
