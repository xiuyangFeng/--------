"""Apply a frozen V4 residual calibrator and evaluate cached cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from batch_validate_cfd import summarize
from calculate_wss_cfd import evaluate, wss_from_gradient
from calibrate_v4_oof import (
    EPS,
    build_case_scale_features,
    build_features,
    load_cache,
)


def _validate_feature_schema(
    package: dict[str, object], feature_names: list[str]
) -> None:
    """Reject inference when diagnostics do not match the frozen model."""
    if list(package.get("feature_names", [])) != feature_names:
        raise RuntimeError("calibrator feature schema mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--predictions-out", required=True)
    args = parser.parse_args()

    data = load_cache(Path(args.cache_dir))
    x, feature_names = build_features(data)
    package = joblib.load(args.model)
    _validate_feature_schema(package, feature_names)
    raw_ratio = np.exp(package["model"].predict(x))
    raw_lower, raw_upper = package["raw_ratio_clip"]
    raw_ratio = np.clip(raw_ratio, float(raw_lower), float(raw_upper))
    point_case = np.asarray(data["point_case"])
    case_names, case_x = build_case_scale_features(x, feature_names, point_case)
    case_lower, case_upper = package["case_alpha_clip"]
    case_alpha = np.clip(
        np.exp(package["case_model"].predict(case_x)),
        float(case_lower),
        float(case_upper),
    )
    lower, upper = package["final_ratio_clip"]
    ratio = np.zeros_like(raw_ratio)
    gradient = np.asarray(data["gradient_v4"], dtype=np.float64)
    base_vector = wss_from_gradient(gradient, "carreau")
    base_mag = np.linalg.norm(base_vector, axis=1)
    for case, alpha in zip(case_names, case_alpha):
        mask = point_case == case
        weight = base_mag[mask] ** 2
        effective = float(
            np.sum(weight * raw_ratio[mask]) / max(float(np.sum(weight)), EPS)
        )
        anchored = np.clip(
            raw_ratio[mask] * alpha / max(effective, EPS),
            float(lower),
            float(upper),
        )
        strength_config = package["local_strength"]
        full_at = float(strength_config["full_at_case_alpha"])
        minimum = float(strength_config["minimum"])
        strength = float(
            np.clip(
                minimum
                + (1.0 - minimum) * (alpha - 1.0) / max(full_at - 1.0, EPS),
                minimum,
                1.0,
            )
        )
        if (
            strength_config.get("fallback_to_physics_at_lower_bound", False)
            and alpha <= float(case_lower) + 1e-6
        ):
            ratio[mask] = 1.0
        else:
            ratio[mask] = np.clip(
                alpha + strength * (anchored - alpha),
                float(lower),
                float(upper),
            )

    point_cohort = np.asarray(data["point_cohort"])
    truth = np.asarray(data["truth_mag"], dtype=np.float64)
    truth_vec = np.asarray(data["truth_vec"], dtype=np.float64)
    wall_mm = np.asarray(data["wall_mm"], dtype=np.float64)
    calibrated_vector = base_vector * ratio[:, None]

    case_rows = []
    for case in np.unique(point_case):
        mask = point_case == case
        case_rows.append(
            {
                "canonical_id": str(case),
                "cohort": str(point_cohort[np.flatnonzero(mask)[0]]),
                "n_points": int(mask.sum()),
                "ratio_p05": float(np.quantile(ratio[mask], 0.05)),
                "ratio_p50": float(np.quantile(ratio[mask], 0.50)),
                "ratio_p95": float(np.quantile(ratio[mask], 0.95)),
                "methods": {
                    "v4_base": evaluate(
                        truth[mask], truth_vec[mask], base_vector[mask]
                    ),
                    "v4_calibrated": evaluate(
                        truth[mask], truth_vec[mask], calibrated_vector[mask]
                    ),
                },
            }
        )
    aggregate = {}
    for name in ("v4_base", "v4_calibrated"):
        aggregate[name] = {
            metric: summarize([row["methods"][name][metric] for row in case_rows])
            for metric in (
                "raw_r2",
                "scaled_r2",
                "spearman",
                "alpha",
                "mae_pa",
                "nrmse",
                "high_wss_nrmse",
                "direction_cosine_p50",
                "coverage",
            )
        }
    delta = np.asarray(
        [
            row["methods"]["v4_calibrated"]["raw_r2"]
            - row["methods"]["v4_base"]["raw_r2"]
            for row in case_rows
        ]
    )
    payload = {
        "status": "frozen_calibrator_evaluation",
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "model": str(Path(args.model).resolve()),
        "n_cases": len(case_rows),
        "n_points": int(len(point_case)),
        "aggregate": aggregate,
        "paired": {
            "raw_r2_delta": summarize(delta.tolist()),
            "wins": int(np.sum(delta > 0)),
            "losses": int(np.sum(delta < 0)),
        },
        "cases": case_rows,
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    prediction_out = Path(args.predictions_out)
    prediction_out.parent.mkdir(parents=True, exist_ok=True)
    with prediction_out.open("wb") as handle:
        np.savez_compressed(
            handle,
            point_case=point_case,
            point_cohort=point_cohort,
            wall_mm=wall_mm.astype(np.float32),
            truth_mag=truth.astype(np.float32),
            base_wss_vec=base_vector.astype(np.float32),
            calibrated_wss_vec=calibrated_vector.astype(np.float32),
            calibration_ratio=ratio.astype(np.float32),
        )
    print(json.dumps(payload["aggregate"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
