"""Apply the frozen high-WSS-tail V4 calibrator to complete cached cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from calculate_wss_cfd import wss_from_gradient
from calibrate_v4_high_tail_oof import (
    aggregate,
    current_safe_ratio,
    tail_ratio,
)
from calibrate_v4_oof import (
    EPS,
    build_case_scale_features,
    build_features,
    load_cache,
)
from validate_v4_high_tail_holdout import pooled_high_metrics


def validate_package(package: dict[str, Any], feature_names: list[str]) -> None:
    if int(package.get("schema_version", 0)) != 2:
        raise RuntimeError("high-tail calibrator schema mismatch")
    if list(package.get("feature_names", [])) != feature_names:
        raise RuntimeError("high-tail calibrator feature schema mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--predictions-out", required=True)
    parser.add_argument("--peak-anchor-strength", type=float, default=None)
    args = parser.parse_args()

    package = joblib.load(args.model)
    data = load_cache(Path(args.cache_dir))
    if package.get("feature_variant", "base") == "depth3_profile":
        from v4_profile_features import base_v4_view, build_profile_features

        x, feature_names = build_profile_features(data)
        prediction_data = base_v4_view(data)
    else:
        x, feature_names = build_features(data)
        prediction_data = data
    validate_package(package, feature_names)
    peak_anchor_strength = (
        float(args.peak_anchor_strength)
        if args.peak_anchor_strength is not None
        else float(
            package.get(
                "peak_anchor_strength",
                package.get("use_peak_anchor", False),
            )
        )
    )
    if not 0.0 <= peak_anchor_strength <= 1.0:
        raise ValueError("peak anchor strength must be in [0, 1]")
    groups = np.asarray(data["point_case"])
    truth = np.asarray(data["truth_mag"], dtype=np.float64)
    base_vector = wss_from_gradient(
        np.asarray(prediction_data["gradient_v4"], dtype=np.float64), "carreau"
    )
    base_mag = np.linalg.norm(base_vector, axis=1)
    predicted_rank = x[:, feature_names.index("case_pred_rank")]

    current_raw = np.exp(package["current_model"].predict(x))
    current_raw = np.clip(current_raw, *package["current_raw_ratio_clip"])
    tail_raw = np.exp(package["tail_model"].predict(x))
    tail_raw = np.clip(tail_raw, *package["tail_raw_ratio_clip"])
    cases, case_x = build_case_scale_features(x, feature_names, groups)
    overall_alpha = np.clip(
        np.exp(package["overall_case_model"].predict(case_x)),
        *package["case_alpha_clip"],
    )
    tail_alpha = np.clip(
        np.exp(package["tail_case_model"].predict(case_x)),
        *package["tail_alpha_clip"],
    )
    peak_alpha = np.clip(
        np.exp(package["peak_case_model"].predict(case_x)),
        *package["peak_alpha_clip"],
    )

    current_prediction = np.zeros_like(base_mag)
    tail_prediction = np.zeros_like(base_mag)
    current_ratio_all = np.zeros_like(base_mag)
    tail_ratio_all = np.zeros_like(base_mag)
    for case, case_alpha, case_tail, case_peak in zip(
        cases, overall_alpha, tail_alpha, peak_alpha
    ):
        mask = groups == case
        current_ratio = current_safe_ratio(
            current_raw[mask], base_mag[mask], float(case_alpha)
        )
        selected_ratio = tail_ratio(
            current_ratio,
            tail_raw[mask],
            base_mag[mask],
            predicted_rank[mask],
            float(case_tail),
            float(case_peak),
            gate_start=float(package["tail_gate_start"]),
            use_peak_anchor=peak_anchor_strength,
        )
        current_ratio_all[mask] = current_ratio
        tail_ratio_all[mask] = selected_ratio
        current_prediction[mask] = base_mag[mask] * current_ratio
        tail_prediction[mask] = base_mag[mask] * selected_ratio

    predictions = {
        "v4_base": base_mag,
        "current_safe": current_prediction,
        "high_tail_w10_s80": tail_prediction,
    }
    aggregate_result, case_rows = aggregate(data, predictions)
    pooled = {
        name: pooled_high_metrics(truth, magnitude, groups)
        for name, magnitude in predictions.items()
    }
    payload = {
        "status": "frozen_high_tail_calibrator_evaluation",
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "model": str(Path(args.model).resolve()),
        "method_name": package["method_name"],
        "feature_variant": package.get("feature_variant", "base"),
        "peak_anchor_strength": peak_anchor_strength,
        "n_cases": int(len(np.unique(groups))),
        "n_points": int(len(groups)),
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
    }
    json_out = Path(args.json_out).resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    current_vector = base_vector * current_ratio_all[:, None]
    tail_vector = base_vector * tail_ratio_all[:, None]
    prediction_out = Path(args.predictions_out).resolve()
    prediction_out.parent.mkdir(parents=True, exist_ok=True)
    with prediction_out.open("wb") as handle:
        np.savez_compressed(
            handle,
            point_case=groups,
            point_cohort=np.asarray(data["point_cohort"]),
            wall_mm=np.asarray(data["wall_mm"], dtype=np.float32),
            truth_mag=truth.astype(np.float32),
            base_wss_vec=base_vector.astype(np.float32),
            current_wss_vec=current_vector.astype(np.float32),
            high_tail_wss_vec=tail_vector.astype(np.float32),
            current_ratio=current_ratio_all.astype(np.float32),
            high_tail_ratio=tail_ratio_all.astype(np.float32),
        )
    print(
        json.dumps(
            {
                "pooled_case_q90_high_wss": pooled,
                "case_balanced_high_r2": {
                    name: aggregate_result[name]["high_r2"]
                    for name in predictions
                },
                "overall_raw_r2": {
                    name: aggregate_result[name]["raw_r2"]
                    for name in predictions
                },
                "target_gate": payload["target_gate"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(json_out)


if __name__ == "__main__":
    main()
