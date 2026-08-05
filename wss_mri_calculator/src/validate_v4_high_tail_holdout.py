"""Validate the selected high-tail V4 calibrator on fixed holdout65 cases."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from calculate_wss_cfd import wss_from_gradient
from calibrate_v4_high_tail_oof import (
    aggregate,
    current_safe_ratio,
    fit_case_scale_models,
    smooth_gate,
    tail_ratio,
    within_case_rank,
)
from calibrate_v4_oof import (
    EPS,
    build_case_scale_features,
    build_features,
    load_cache,
)


def pooled_high_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    high = np.zeros(len(truth), dtype=bool)
    for case in np.unique(groups):
        mask = groups == case
        high[mask] = truth[mask] >= np.quantile(truth[mask], 0.90)
    y = truth[high]
    p = prediction[high]
    variance = float(np.sum((y - np.mean(y)) ** 2))
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    mean_ratio = float(np.mean(p) / max(float(np.mean(y)), EPS))
    return {
        "n_points": int(np.sum(high)),
        "r2": 1.0 - float(np.sum((p - y) ** 2)) / max(variance, EPS),
        "rmse_pa": rmse,
        "nrmse": rmse / max(float(np.mean(np.abs(y))), EPS),
        "bias_pa": float(np.mean(p - y)),
        "mean_prediction_over_truth": mean_ratio,
        "mean_underestimate_fraction": max(0.0, 1.0 - mean_ratio),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--case-order-result", required=True)
    parser.add_argument("--development-cases", type=int, default=73)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--predictions-out", default="")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--profile-features", action="store_true")
    parser.add_argument("--peak-anchor-strength", type=float, default=0.30)
    args = parser.parse_args()
    started = time.time()

    data = load_cache(Path(args.cache_dir))
    if args.profile_features:
        from v4_profile_features import base_v4_view, build_profile_features

        x, feature_names = build_profile_features(data)
        prediction_data = base_v4_view(data)
    else:
        x, feature_names = build_features(data)
        prediction_data = data
    order_payload = json.loads(
        Path(args.case_order_result).read_text(encoding="utf-8")
    )
    ordered_cases = [row["canonical_id"] for row in order_payload["cases"]]
    development = set(ordered_cases[: args.development_cases])
    groups = np.asarray(data["point_case"])
    train_mask = np.isin(groups, list(development))
    valid_mask = ~train_mask
    if len(np.unique(groups[train_mask])) != args.development_cases:
        raise RuntimeError("development case cache is incomplete")

    truth = np.asarray(data["truth_mag"], dtype=np.float64)
    base_vector = wss_from_gradient(
        np.asarray(prediction_data["gradient_v4"], dtype=np.float64), "carreau"
    )
    base_mag = np.linalg.norm(base_vector, axis=1)
    target = np.log(np.maximum(truth, EPS) / np.maximum(base_mag, EPS))
    target = np.clip(target, np.log(0.5), np.log(2.5))
    truth_rank = within_case_rank(truth, groups)
    predicted_rank = x[:, feature_names.index("case_pred_rank")]

    parameters = {
        "loss": "squared_error",
        "learning_rate": 0.05,
        "max_iter": 350,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 120,
        "l2_regularization": 5.0,
    }
    current_model = HistGradientBoostingRegressor(
        **parameters, random_state=args.seed
    )
    current_model.fit(x[train_mask], target[train_mask])
    current_raw = np.clip(np.exp(current_model.predict(x[valid_mask])), 0.7, 1.8)

    tail_gate_train = smooth_gate(truth_rank[train_mask], 0.80)
    underprediction = np.clip(
        target[train_mask] / max(float(np.log(2.5)), EPS), 0.0, 1.0
    )
    sample_weight = (
        1.0 + 10.0 * tail_gate_train + 5.0 * tail_gate_train * underprediction
    )
    tail_model = HistGradientBoostingRegressor(
        **parameters, random_state=args.seed + 1000
    )
    tail_model.fit(x[train_mask], target[train_mask], sample_weight=sample_weight)
    tail_raw = np.clip(np.exp(tail_model.predict(x[valid_mask])), 0.7, 2.2)

    overall_model, tail_scale_model, peak_scale_model = fit_case_scale_models(
        x[train_mask],
        feature_names,
        groups[train_mask],
        truth[train_mask],
        base_mag[train_mask],
    )
    valid_cases_unique, valid_case_x = build_case_scale_features(
        x[valid_mask], feature_names, groups[valid_mask]
    )
    overall_alpha = np.clip(
        np.exp(overall_model.predict(valid_case_x)), 1.00, 1.40
    )
    tail_alpha = np.clip(
        np.exp(tail_scale_model.predict(valid_case_x)), 1.00, 1.80
    )
    peak_alpha = np.clip(
        np.exp(peak_scale_model.predict(valid_case_x)), 1.00, 2.20
    )

    valid_groups = groups[valid_mask]
    valid_base = base_mag[valid_mask]
    valid_rank = predicted_rank[valid_mask]
    current_prediction = np.zeros_like(valid_base)
    tail_prediction = np.zeros_like(valid_base)
    for case, case_alpha, case_tail, case_peak in zip(
        valid_cases_unique, overall_alpha, tail_alpha, peak_alpha
    ):
        local = valid_groups == case
        current_ratio = current_safe_ratio(
            current_raw[local], valid_base[local], float(case_alpha)
        )
        selected_ratio = tail_ratio(
            current_ratio,
            tail_raw[local],
            valid_base[local],
            valid_rank[local],
            float(case_tail),
            float(case_peak),
            gate_start=0.80,
            use_peak_anchor=args.peak_anchor_strength,
        )
        current_prediction[local] = valid_base[local] * current_ratio
        tail_prediction[local] = valid_base[local] * selected_ratio

    subset_data: dict[str, np.ndarray | list[str]] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and len(value) == len(groups):
            subset_data[key] = value[valid_mask]
        else:
            subset_data[key] = value
    predictions = {
        "v4_base": valid_base,
        "current_safe": current_prediction,
        "high_tail_w10_s80": tail_prediction,
    }
    aggregate_result, case_rows = aggregate(subset_data, predictions)
    pooled = {
        name: pooled_high_metrics(
            truth[valid_mask], magnitude, valid_groups
        )
        for name, magnitude in predictions.items()
    }
    payload: dict[str, Any] = {
        "status": "fixed_development73_holdout65_high_tail_validation",
        "development_cases": int(len(np.unique(groups[train_mask]))),
        "holdout_cases": int(len(np.unique(valid_groups))),
        "candidate": (
            "tail_weight=10, prediction-rank gate start=0.80, "
            f"peak anchor strength={args.peak_anchor_strength:.2f}"
        ),
        "peak_anchor_strength": args.peak_anchor_strength,
        "feature_names": feature_names,
        "feature_variant": "depth3_profile" if args.profile_features else "base",
        "aggregate_case_balanced": aggregate_result,
        "pooled_case_q90_high_wss": pooled,
        "target_gate": {
            "pooled_high_r2_at_least_0p9": bool(
                pooled["high_tail_w10_s80"]["r2"] >= 0.90
            ),
            "pooled_mean_underestimate_at_most_10pct": bool(
                pooled["high_tail_w10_s80"]["mean_underestimate_fraction"] <= 0.10
            ),
        },
        "cases": case_rows,
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.json_out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.predictions_out:
        prediction_output = Path(args.predictions_out).resolve()
        prediction_output.parent.mkdir(parents=True, exist_ok=True)
        with prediction_output.open("wb") as handle:
            np.savez_compressed(
                handle,
                point_case=valid_groups,
                truth_mag=truth[valid_mask].astype(np.float32),
                base_wss_vec=base_vector[valid_mask].astype(np.float32),
                current_safe=current_prediction.astype(np.float32),
                high_tail_w10_s80=tail_prediction.astype(np.float32),
            )
    print(
        json.dumps(
            {
                "pooled_case_q90_high_wss": pooled,
                "case_balanced_high_r2": {
                    name: aggregate_result[name]["high_r2"]
                    for name in predictions
                },
                "target_gate": payload["target_gate"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(output)


if __name__ == "__main__":
    main()
