"""Fixed V4 residual calibrator validation on cases untouched by its design."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from batch_validate_cfd import summarize
from calculate_wss_cfd import evaluate, wss_from_gradient
from calibrate_v4_oof import (
    EPS,
    build_case_scale_features,
    build_features,
    load_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--case-order-result", required=True)
    parser.add_argument("--development-cases", type=int, default=73)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    started = time.time()

    data = load_cache(Path(args.cache_dir))
    x, feature_names = build_features(data)
    order_payload = json.loads(
        Path(args.case_order_result).read_text(encoding="utf-8")
    )
    ordered_cases = [row["canonical_id"] for row in order_payload["cases"]]
    development = set(ordered_cases[: args.development_cases])
    point_case = np.asarray(data["point_case"])
    train_mask = np.isin(point_case, list(development))
    valid_mask = ~train_mask
    if len(np.unique(point_case[train_mask])) != args.development_cases:
        raise RuntimeError("development case cache is incomplete")
    if not valid_mask.any():
        raise RuntimeError("no holdout cases remain")

    truth = np.asarray(data["truth_mag"], dtype=np.float64)
    truth_vec = np.asarray(data["truth_vec"], dtype=np.float64)
    gradient = np.asarray(data["gradient_v4"], dtype=np.float64)
    base_vector = wss_from_gradient(gradient, "carreau")
    base_mag = np.linalg.norm(base_vector, axis=1)
    target = np.log(np.maximum(truth, EPS) / np.maximum(base_mag, EPS))
    target = np.clip(target, np.log(0.5), np.log(2.5))

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=350,
        max_leaf_nodes=15,
        min_samples_leaf=120,
        l2_regularization=5.0,
        random_state=args.seed,
    )
    model.fit(x[train_mask], target[train_mask])
    learned_ratio = np.exp(model.predict(x[valid_mask]))
    learned_ratio = np.clip(learned_ratio, 0.7, 1.8)
    alpha = float(
        np.dot(truth[train_mask], base_mag[train_mask])
        / max(float(np.dot(base_mag[train_mask], base_mag[train_mask])), EPS)
    )

    train_cases, train_case_x = build_case_scale_features(
        x[train_mask], feature_names, point_case[train_mask]
    )
    valid_cases_unique, valid_case_x = build_case_scale_features(
        x[valid_mask], feature_names, point_case[valid_mask]
    )
    train_case_alpha = []
    for case in train_cases:
        mask = point_case[train_mask] == case
        local_truth = truth[train_mask][mask]
        local_base = base_mag[train_mask][mask]
        train_case_alpha.append(
            float(
                np.dot(local_truth, local_base)
                / max(float(np.dot(local_base, local_base)), EPS)
            )
        )
    case_model = make_pipeline(
        StandardScaler(), RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0, 1000.0))
    )
    case_model.fit(train_case_x, np.log(np.asarray(train_case_alpha)))
    predicted_case_alpha = np.clip(
        np.exp(case_model.predict(valid_case_x)), 1.00, 1.40
    )

    valid_cases = point_case[valid_mask]
    valid_truth = truth[valid_mask]
    valid_truth_vec = truth_vec[valid_mask]
    valid_base_vector = base_vector[valid_mask]
    valid_base_mag = base_mag[valid_mask]
    anchored_ratio = np.zeros_like(learned_ratio)
    safe_ratio = np.zeros_like(learned_ratio)
    case_scale_ratio = np.zeros_like(learned_ratio)
    for case, case_alpha in zip(valid_cases_unique, predicted_case_alpha):
        mask = valid_cases == case
        weight = valid_base_mag[mask] ** 2
        effective = float(
            np.sum(weight * learned_ratio[mask]) / max(float(np.sum(weight)), EPS)
        )
        anchored_ratio[mask] = np.clip(
            learned_ratio[mask] * case_alpha / max(effective, EPS), 0.85, 1.60
        )
        local_strength = float(
            np.clip(0.25 + 0.75 * (case_alpha - 1.0) / 0.08, 0.25, 1.0)
        )
        if case_alpha <= 1.000001:
            safe_ratio[mask] = 1.0
        else:
            safe_ratio[mask] = np.clip(
                case_alpha + local_strength * (anchored_ratio[mask] - case_alpha),
                0.85,
                1.60,
            )
        case_scale_ratio[mask] = case_alpha
    predictions = {
        "v4_base": valid_base_vector,
        "global_scale": valid_base_vector * alpha,
        "hist_calibrated": valid_base_vector * learned_ratio[:, None],
        "hist_guarded": valid_base_vector
        * np.clip(alpha + 0.75 * (learned_ratio - alpha), 0.85, 1.60)[:, None],
        "case_ridge_scale": valid_base_vector * case_scale_ratio[:, None],
        "hist_case_anchored": valid_base_vector * anchored_ratio[:, None],
        "hist_case_safe": valid_base_vector * safe_ratio[:, None],
    }
    aggregate = {}
    case_rows = []
    for case in np.unique(valid_cases):
        mask = valid_cases == case
        reports = {
            name: evaluate(valid_truth[mask], valid_truth_vec[mask], vector[mask])
            for name, vector in predictions.items()
        }
        case_rows.append({"canonical_id": str(case), "methods": reports})
    for name in predictions:
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
            )
        }

    payload = {
        "status": "fixed_model_development73_holdout65",
        "development_cases": int(len(np.unique(point_case[train_mask]))),
        "holdout_cases": int(len(np.unique(valid_cases))),
        "feature_names": feature_names,
        "global_alpha": alpha,
        "model": {
            "type": "HistGradientBoostingRegressor",
            "learning_rate": 0.05,
            "max_iter": 350,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 120,
            "l2_regularization": 5.0,
            "ratio_clip": [0.7, 1.8],
            "seed": args.seed,
        },
        "aggregate": aggregate,
        "cases": case_rows,
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
