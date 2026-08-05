"""Grouped train-only OOF screen for high-WSS-tail V4 calibration.

The frozen V4 calibrator optimizes pointwise squared log-ratio error, so the
case-wise top 10% WSS tail contributes only a small fraction of its objective.
This study keeps the current safe calibration for the bulk and tests a second
tail-weighted point model plus case-level tail/peak scale anchors.  Tail gates
use prediction rank only; truth is used solely for training weights/targets and
held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedGroupKFold
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


def within_case_rank(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    rank = np.zeros(len(values), dtype=np.float64)
    for case in np.unique(groups):
        mask = groups == case
        local = np.asarray(values[mask], dtype=np.float64)
        order = np.argsort(local, kind="stable")
        local_rank = np.empty(len(local), dtype=np.float64)
        local_rank[order] = np.linspace(0.0, 1.0, len(local), endpoint=True)
        rank[mask] = local_rank
    return rank


def smooth_gate(rank: np.ndarray, start: float) -> np.ndarray:
    t = np.clip((rank - start) / max(1.0 - start, EPS), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def case_r2_emphasis(truth: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Approximate equal-case high-tail R² loss while fitting log ratios."""
    emphasis = np.zeros(len(truth), dtype=np.float64)
    for case in np.unique(groups):
        mask = groups == case
        local = np.asarray(truth[mask], dtype=np.float64)
        high = local >= np.quantile(local, 0.90)
        variance = float(np.sum((local[high] - np.mean(local[high])) ** 2))
        local_emphasis = np.zeros(len(local), dtype=np.float64)
        local_emphasis[high] = local[high] ** 2 / max(variance, EPS)
        emphasis[mask] = local_emphasis
    positive = emphasis[emphasis > 0]
    if len(positive):
        emphasis /= max(float(np.median(positive)), EPS)
    return np.clip(emphasis, 0.0, 30.0)


def fit_case_scale_models(
    x_train: np.ndarray,
    feature_names: list[str],
    groups_train: np.ndarray,
    truth_train: np.ndarray,
    base_train: np.ndarray,
) -> tuple[Any, Any, Any]:
    cases, case_x = build_case_scale_features(x_train, feature_names, groups_train)
    overall_alpha = []
    tail_alpha = []
    peak_alpha = []
    for case in cases:
        mask = groups_train == case
        truth = truth_train[mask]
        base = base_train[mask]
        high = truth >= np.quantile(truth, 0.90)
        overall_alpha.append(
            float(np.dot(truth, base) / max(float(np.dot(base, base)), EPS))
        )
        tail_alpha.append(
            float(
                np.dot(truth[high], base[high])
                / max(float(np.dot(base[high], base[high])), EPS)
            )
        )
        peak_alpha.append(float(np.max(truth) / max(float(np.max(base)), EPS)))

    def fit(target: list[float]) -> Any:
        model = make_pipeline(
            StandardScaler(), RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0, 1000.0))
        )
        model.fit(case_x, np.log(np.clip(np.asarray(target), 0.5, 3.0)))
        return model

    return fit(overall_alpha), fit(tail_alpha), fit(peak_alpha)


def current_safe_ratio(
    raw_ratio: np.ndarray,
    base: np.ndarray,
    case_alpha: float,
) -> np.ndarray:
    weight = base**2
    effective = float(np.sum(weight * raw_ratio) / max(float(np.sum(weight)), EPS))
    anchored = np.clip(raw_ratio * case_alpha / max(effective, EPS), 0.85, 1.60)
    strength = float(np.clip(0.25 + 0.75 * (case_alpha - 1.0) / 0.08, 0.25, 1.0))
    if case_alpha <= 1.000001:
        return np.ones_like(anchored)
    return np.clip(case_alpha + strength * (anchored - case_alpha), 0.85, 1.60)


def tail_ratio(
    current_ratio: np.ndarray,
    learned_tail_ratio: np.ndarray,
    base: np.ndarray,
    predicted_rank: np.ndarray,
    tail_alpha: float,
    peak_alpha: float,
    *,
    gate_start: float,
    use_peak_anchor: bool | float,
) -> np.ndarray:
    gate = smooth_gate(predicted_rank, gate_start)
    tail_weight = base**2 * np.maximum(gate, 1e-6)
    effective = float(
        np.sum(tail_weight * learned_tail_ratio)
        / max(float(np.sum(tail_weight)), EPS)
    )
    anchored = learned_tail_ratio * tail_alpha / max(effective, EPS)
    peak_strength = float(use_peak_anchor)
    if peak_strength > 0:
        peak_gate = smooth_gate(predicted_rank, 0.95)
        peak_factor = max(peak_alpha, tail_alpha) / max(tail_alpha, EPS)
        anchored = anchored * peak_factor ** (peak_strength * peak_gate)
    blended = current_ratio * (1.0 - gate) + anchored * gate
    return np.clip(blended, 0.85, 2.20)


def case_tail_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    high = truth >= np.quantile(truth, 0.90)
    y = truth[high]
    p = prediction[high]
    variance = float(np.sum((y - np.mean(y)) ** 2))
    error = p - y
    peak_index = int(np.argmax(truth))
    peak_under = max(
        0.0, 1.0 - float(prediction[peak_index]) / max(float(truth[peak_index]), EPS)
    )
    predicted_high = prediction >= np.quantile(prediction, 0.90)
    intersection = int(np.sum(high & predicted_high))
    union = int(np.sum(high | predicted_high))
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "high_r2": 1.0 - float(np.sum(error**2)) / max(variance, EPS),
        "high_mae_pa": float(np.mean(np.abs(error))),
        "high_nrmse": rmse / max(float(np.mean(np.abs(y))), EPS),
        "high_bias_pa": float(np.mean(error)),
        "high_spearman": float(spearmanr(y, p).statistic),
        "peak_underestimate_fraction": peak_under,
        "peak_ratio": float(prediction[peak_index] / max(truth[peak_index], EPS)),
        "top10_jaccard": float(intersection / max(union, 1)),
    }


def aggregate(
    data: dict[str, np.ndarray | list[str]],
    predictions: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups = np.asarray(data["point_case"])
    truth = np.asarray(data["truth_mag"], dtype=np.float64)
    truth_vec = np.asarray(data["truth_vec"], dtype=np.float64)
    base_vector = wss_from_gradient(
        np.asarray(data["gradient_v4"], dtype=np.float64), "carreau"
    )
    base_mag = np.linalg.norm(base_vector, axis=1)
    case_rows = []
    for case in np.unique(groups):
        mask = groups == case
        methods = {}
        for name, magnitude in predictions.items():
            ratio = magnitude[mask] / np.maximum(base_mag[mask], EPS)
            standard = evaluate(
                truth[mask], truth_vec[mask], base_vector[mask] * ratio[:, None]
            )
            methods[name] = {
                **standard,
                **case_tail_metrics(truth[mask], magnitude[mask]),
            }
        case_rows.append({"canonical_id": str(case), "methods": methods})

    result: dict[str, Any] = {}
    standard_metrics = (
        "raw_r2",
        "scaled_r2",
        "spearman",
        "alpha",
        "mae_pa",
        "nrmse",
        "high_wss_nrmse",
    )
    tail_metrics = (
        "high_r2",
        "high_mae_pa",
        "high_nrmse",
        "high_bias_pa",
        "high_spearman",
        "peak_underestimate_fraction",
        "peak_ratio",
        "top10_jaccard",
    )
    for name in predictions:
        result[name] = {
            metric: summarize([row["methods"][name][metric] for row in case_rows])
            for metric in (*standard_metrics, *tail_metrics)
        }
        peak_under = np.asarray(
            [
                row["methods"][name]["peak_underestimate_fraction"]
                for row in case_rows
            ]
        )
        high_r2 = np.asarray([row["methods"][name]["high_r2"] for row in case_rows])
        result[name]["peak_underestimate_fraction"]["max"] = float(
            np.max(peak_under)
        )
        result[name]["target_gates"] = {
            "mean_high_r2_at_least_0p9": bool(np.mean(high_r2) >= 0.90),
            "cases_high_r2_at_least_0p9": int(np.sum(high_r2 >= 0.90)),
            "cases_peak_underestimate_at_most_10pct": int(np.sum(peak_under <= 0.10)),
            "all_cases_peak_underestimate_at_most_10pct": bool(np.max(peak_under) <= 0.10),
        }
    return result, case_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--predictions-out", default="")
    parser.add_argument("--profile-features", action="store_true")
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
    truth = np.asarray(data["truth_mag"], dtype=np.float64)
    groups = np.asarray(data["point_case"])
    strata = np.asarray(data["point_cohort"])
    base_vector = wss_from_gradient(
        np.asarray(prediction_data["gradient_v4"], dtype=np.float64), "carreau"
    )
    base_mag = np.linalg.norm(base_vector, axis=1)
    target = np.log(np.maximum(truth, EPS) / np.maximum(base_mag, EPS))
    target = np.clip(target, np.log(0.5), np.log(2.5))
    truth_rank = within_case_rank(truth, groups)
    predicted_rank = x[:, feature_names.index("case_pred_rank")]

    prediction_names = (
        "v4_base",
        "current_safe_oof",
        "tail_scale_only_s80_oof",
        "tail_w4_s80_oof",
        "tail_w10_s80_oof",
        "tail_w10_s90_oof",
        "tail_w10_peak10_s80_oof",
        "tail_w10_peak20_s80_oof",
        "tail_w10_peak30_s80_oof",
        "tail_case_r2_w2_s80_oof",
        "tail_case_r2_w5_s80_oof",
        "tail_case_r2_w10_s80_oof",
        "tail_w10_peak_s80_oof",
    )
    predictions = {name: np.zeros_like(base_mag) for name in prediction_names}
    predictions["v4_base"] = base_mag.copy()
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    fold_rows = []

    for fold, (train_index, valid_index) in enumerate(
        splitter.split(x, strata, groups), start=1
    ):
        model_parameters = {
            "loss": "squared_error",
            "learning_rate": 0.05,
            "max_iter": 350,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 120,
            "l2_regularization": 5.0,
        }
        current_model = HistGradientBoostingRegressor(
            **model_parameters, random_state=args.seed + fold
        )
        current_model.fit(x[train_index], target[train_index])
        current_raw = np.clip(
            np.exp(current_model.predict(x[valid_index])), 0.7, 1.8
        )

        tail_models = {}
        for tail_weight in (4.0, 10.0):
            model = HistGradientBoostingRegressor(
                **model_parameters, random_state=args.seed + 100 * fold + int(tail_weight)
            )
            tail_emphasis = smooth_gate(truth_rank[train_index], 0.80)
            underprediction = np.clip(
                target[train_index] / max(float(np.log(2.5)), EPS), 0.0, 1.0
            )
            sample_weight = (
                1.0
                + tail_weight * tail_emphasis
                + 0.5 * tail_weight * tail_emphasis * underprediction
            )
            model.fit(x[train_index], target[train_index], sample_weight=sample_weight)
            tail_models[tail_weight] = np.clip(
                np.exp(model.predict(x[valid_index])), 0.7, 2.2
            )

        r2_models = {}
        r2_emphasis = case_r2_emphasis(
            truth[train_index], groups[train_index]
        )
        for r2_weight in (2.0, 5.0, 10.0):
            model = HistGradientBoostingRegressor(
                **model_parameters,
                random_state=args.seed + 10000 * fold + int(r2_weight),
            )
            sample_weight = 1.0 + r2_weight * r2_emphasis
            model.fit(x[train_index], target[train_index], sample_weight=sample_weight)
            r2_models[r2_weight] = np.clip(
                np.exp(model.predict(x[valid_index])), 0.7, 2.2
            )

        overall_model, tail_scale_model, peak_scale_model = fit_case_scale_models(
            x[train_index],
            feature_names,
            groups[train_index],
            truth[train_index],
            base_mag[train_index],
        )
        valid_cases, valid_case_x = build_case_scale_features(
            x[valid_index], feature_names, groups[valid_index]
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

        valid_groups = groups[valid_index]
        for case, case_alpha, case_tail_alpha, case_peak_alpha in zip(
            valid_cases, overall_alpha, tail_alpha, peak_alpha
        ):
            local = valid_groups == case
            index = valid_index[local]
            local_base = base_mag[index]
            local_rank = predicted_rank[index]
            current = current_safe_ratio(
                current_raw[local], local_base, float(case_alpha)
            )
            predictions["current_safe_oof"][index] = local_base * current
            predictions["tail_scale_only_s80_oof"][index] = local_base * tail_ratio(
                current,
                current,
                local_base,
                local_rank,
                float(case_tail_alpha),
                float(case_peak_alpha),
                gate_start=0.80,
                use_peak_anchor=False,
            )
            predictions["tail_w4_s80_oof"][index] = local_base * tail_ratio(
                current,
                tail_models[4.0][local],
                local_base,
                local_rank,
                float(case_tail_alpha),
                float(case_peak_alpha),
                gate_start=0.80,
                use_peak_anchor=False,
            )
            predictions["tail_w10_s80_oof"][index] = local_base * tail_ratio(
                current,
                tail_models[10.0][local],
                local_base,
                local_rank,
                float(case_tail_alpha),
                float(case_peak_alpha),
                gate_start=0.80,
                use_peak_anchor=False,
            )
            predictions["tail_w10_s90_oof"][index] = local_base * tail_ratio(
                current,
                tail_models[10.0][local],
                local_base,
                local_rank,
                float(case_tail_alpha),
                float(case_peak_alpha),
                gate_start=0.90,
                use_peak_anchor=False,
            )
            for peak_strength in (0.10, 0.20, 0.30):
                predictions[
                    f"tail_w10_peak{int(100 * peak_strength):02d}_s80_oof"
                ][index] = local_base * tail_ratio(
                    current,
                    tail_models[10.0][local],
                    local_base,
                    local_rank,
                    float(case_tail_alpha),
                    float(case_peak_alpha),
                    gate_start=0.80,
                    use_peak_anchor=peak_strength,
                )
            for r2_weight in (2.0, 5.0, 10.0):
                predictions[
                    f"tail_case_r2_w{int(r2_weight)}_s80_oof"
                ][index] = local_base * tail_ratio(
                    current,
                    r2_models[r2_weight][local],
                    local_base,
                    local_rank,
                    float(case_tail_alpha),
                    float(case_peak_alpha),
                    gate_start=0.80,
                    use_peak_anchor=False,
                )
            predictions["tail_w10_peak_s80_oof"][index] = local_base * tail_ratio(
                current,
                tail_models[10.0][local],
                local_base,
                local_rank,
                float(case_tail_alpha),
                float(case_peak_alpha),
                gate_start=0.80,
                use_peak_anchor=True,
            )

        fold_rows.append(
            {
                "fold": fold,
                "train_cases": int(len(np.unique(groups[train_index]))),
                "valid_cases": int(len(np.unique(groups[valid_index]))),
                "overall_alpha_p50": float(np.median(overall_alpha)),
                "tail_alpha_p50": float(np.median(tail_alpha)),
                "peak_alpha_p50": float(np.median(peak_alpha)),
            }
        )
        print(
            f"fold={fold}/{args.folds} train_cases={fold_rows[-1]['train_cases']} "
            f"valid_cases={fold_rows[-1]['valid_cases']} "
            f"tail_alpha_p50={fold_rows[-1]['tail_alpha_p50']:.3f} "
            f"peak_alpha_p50={fold_rows[-1]['peak_alpha_p50']:.3f}",
            flush=True,
        )

    aggregate_result, case_rows = aggregate(data, predictions)
    payload = {
        "status": "grouped_case_oof_train_only_high_tail_screen",
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "n_cases": int(len(np.unique(groups))),
        "n_points": int(len(groups)),
        "folds": args.folds,
        "seed": args.seed,
        "high_wss_definition": "case truth top 10%; truth is used only in training-fold weights/targets and held-out evaluation",
        "inference_tail_gate": "case_pred_rank derived from physics V4 predictions",
        "feature_names": feature_names,
        "feature_variant": "depth3_profile" if args.profile_features else "base",
        "fold_rows": fold_rows,
        "aggregate": aggregate_result,
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
                point_case=groups,
                point_cohort=strata,
                truth_mag=truth.astype(np.float32),
                truth_vec=np.asarray(data["truth_vec"], dtype=np.float32),
                base_wss_vec=base_vector.astype(np.float32),
                case_pred_rank=predicted_rank.astype(np.float32),
                **{
                    name: magnitude.astype(np.float32)
                    for name, magnitude in predictions.items()
                },
            )
    concise = {
        name: {
            "raw_r2_mean": values["raw_r2"]["mean"],
            "high_r2_mean": values["high_r2"]["mean"],
            "high_nrmse_mean": values["high_nrmse"]["mean"],
            "peak_under_p95": values["peak_underestimate_fraction"]["p95"],
            "peak_under_max": values["peak_underestimate_fraction"]["max"],
            "target_gates": values["target_gates"],
        }
        for name, values in aggregate_result.items()
    }
    print(json.dumps(concise, indent=2, ensure_ascii=False), flush=True)
    print(output)


if __name__ == "__main__":
    main()
