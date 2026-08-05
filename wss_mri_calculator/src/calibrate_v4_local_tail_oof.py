"""Grouped OOF study adding invariant local wall-neighborhood context to V4."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedGroupKFold

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


def build_local_context(
    data: dict[str, np.ndarray | list[str]],
    base_mag: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    groups = np.asarray(data["point_case"])
    wall = np.asarray(data["wall_mm"], dtype=np.float64)
    radius = np.asarray(data["local_radius_mm"], dtype=np.float64)
    columns: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, values: np.ndarray) -> None:
        names.append(name)
        columns.append(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0))

    for k in (8, 16, 32):
        local_mean = np.zeros(len(base_mag), dtype=np.float64)
        local_max = np.zeros(len(base_mag), dtype=np.float64)
        local_std = np.zeros(len(base_mag), dtype=np.float64)
        local_rank = np.zeros(len(base_mag), dtype=np.float64)
        local_distance = np.zeros(len(base_mag), dtype=np.float64)
        local_radius_mean = np.zeros(len(base_mag), dtype=np.float64)
        for case in np.unique(groups):
            mask = groups == case
            index = np.flatnonzero(mask)
            coords = wall[mask]
            values = base_mag[mask]
            local_radius = radius[mask]
            distance, neighbor = cKDTree(coords).query(coords, k=min(k, len(coords)))
            if neighbor.ndim == 1:
                neighbor = neighbor[:, None]
                distance = distance[:, None]
            neighbor_values = values[neighbor]
            local_mean[index] = np.mean(neighbor_values, axis=1)
            local_max[index] = np.max(neighbor_values, axis=1)
            local_std[index] = np.std(neighbor_values, axis=1)
            local_rank[index] = np.mean(
                neighbor_values <= values[:, None], axis=1
            )
            local_distance[index] = np.median(distance[:, 1:], axis=1)
            local_radius_mean[index] = np.mean(local_radius[neighbor], axis=1)
        add(f"local{k}_log_mean_pred", np.log(np.maximum(local_mean, EPS)))
        add(f"local{k}_log_max_pred", np.log(np.maximum(local_max, EPS)))
        add(
            f"local{k}_log_pred_over_mean",
            np.log(np.maximum(base_mag, EPS)) - np.log(np.maximum(local_mean, EPS)),
        )
        add(
            f"local{k}_pred_over_max",
            base_mag / np.maximum(local_max, EPS),
        )
        add(
            f"local{k}_coefficient_of_variation",
            local_std / np.maximum(local_mean, EPS),
        )
        add(f"local{k}_rank", local_rank)
        add(f"local{k}_median_neighbor_distance_mm", local_distance)
        add(
            f"local{k}_radius_over_neighbor_mean",
            radius / np.maximum(local_radius_mean, EPS),
        )
    return np.column_stack(columns), names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--predictions-out", default="")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--profile-features", action="store_true")
    args = parser.parse_args()
    started = time.time()

    data = load_cache(Path(args.cache_dir))
    if args.profile_features:
        from v4_profile_features import base_v4_view, build_profile_features

        x_base, base_feature_names = build_profile_features(data)
        prediction_data = base_v4_view(data)
    else:
        x_base, base_feature_names = build_features(data)
        prediction_data = data
    truth = np.asarray(data["truth_mag"], dtype=np.float64)
    groups = np.asarray(data["point_case"])
    strata = np.asarray(data["point_cohort"])
    base_vector = wss_from_gradient(
        np.asarray(prediction_data["gradient_v4"], dtype=np.float64), "carreau"
    )
    base_mag = np.linalg.norm(base_vector, axis=1)
    local_x, local_names = build_local_context(data, base_mag)
    x = np.column_stack([x_base, local_x])
    feature_names = [*base_feature_names, *local_names]
    ratio_target = np.log(np.maximum(truth, EPS) / np.maximum(base_mag, EPS))
    ratio_target = np.clip(ratio_target, np.log(0.5), np.log(2.5))
    log_truth_target = np.log(np.maximum(truth, EPS))
    truth_rank = within_case_rank(truth, groups)
    predicted_rank = x_base[:, base_feature_names.index("case_pred_rank")]

    names = (
        "v4_base",
        "local_current_safe_oof",
        "local_tail_safe_oof",
        "local_tail_case_safe_oof",
        "local_dual_tail_oof",
        "local_dual_peak10_oof",
        "local_dual_peak20_oof",
        "local_dual_peak30_oof",
        "local_direct_oof",
        "local_direct_safe_oof",
    )
    predictions = {name: np.zeros_like(base_mag) for name in names}
    predictions["v4_base"] = base_mag.copy()
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    fold_rows = []

    for fold, (train_index, valid_index) in enumerate(
        splitter.split(x, strata, groups), start=1
    ):
        current = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=19,
            min_samples_leaf=100,
            l2_regularization=7.0,
            random_state=args.seed + fold,
        )
        current.fit(x[train_index], ratio_target[train_index])
        current_raw = np.clip(np.exp(current.predict(x[valid_index])), 0.7, 1.9)

        tail_gate_train = smooth_gate(truth_rank[train_index], 0.75)
        underprediction = np.clip(
            ratio_target[train_index] / max(float(np.log(2.5)), EPS), 0.0, 1.0
        )
        sample_weight = (
            1.0 + 12.0 * tail_gate_train + 8.0 * tail_gate_train * underprediction
        )
        tail = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.04,
            max_iter=320,
            max_leaf_nodes=23,
            min_samples_leaf=60,
            l2_regularization=10.0,
            random_state=args.seed + 100 + fold,
        )
        tail.fit(
            x[train_index], ratio_target[train_index], sample_weight=sample_weight
        )
        tail_raw = np.clip(np.exp(tail.predict(x[valid_index])), 0.7, 2.3)

        direct = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.04,
            max_iter=320,
            max_leaf_nodes=23,
            min_samples_leaf=60,
            l2_regularization=10.0,
            random_state=args.seed + 1000 + fold,
        )
        direct.fit(
            x[train_index],
            log_truth_target[train_index],
            sample_weight=sample_weight,
        )
        direct_prediction = np.exp(direct.predict(x[valid_index]))
        direct_ratio = np.clip(
            direct_prediction / np.maximum(base_mag[valid_index], EPS), 0.7, 2.3
        )

        overall_model, tail_scale_model, peak_scale_model = fit_case_scale_models(
            x_base[train_index],
            base_feature_names,
            groups[train_index],
            truth[train_index],
            base_mag[train_index],
        )
        valid_cases, valid_case_x = build_case_scale_features(
            x_base[valid_index], base_feature_names, groups[valid_index]
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
        for case, case_alpha, case_tail, case_peak in zip(
            valid_cases, overall_alpha, tail_alpha, peak_alpha
        ):
            local = valid_groups == case
            index = valid_index[local]
            local_base = base_mag[index]
            rank = predicted_rank[index]
            current_ratio = current_safe_ratio(
                current_raw[local], local_base, float(case_alpha)
            )
            tail_safe = current_safe_ratio(
                tail_raw[local], local_base, float(case_alpha)
            )
            tail_case_safe = current_safe_ratio(
                tail_raw[local], local_base, float(case_tail)
            )
            dual = tail_ratio(
                current_ratio,
                tail_raw[local],
                local_base,
                rank,
                float(case_tail),
                float(case_peak),
                gate_start=0.75,
                use_peak_anchor=False,
            )
            predictions["local_current_safe_oof"][index] = local_base * current_ratio
            predictions["local_tail_safe_oof"][index] = local_base * tail_safe
            predictions["local_tail_case_safe_oof"][index] = (
                local_base * tail_case_safe
            )
            predictions["local_dual_tail_oof"][index] = local_base * dual
            predictions["local_direct_oof"][index] = (
                local_base * direct_ratio[local]
            )
            direct_safe = current_safe_ratio(
                direct_ratio[local], local_base, float(case_alpha)
            )
            predictions["local_direct_safe_oof"][index] = local_base * direct_safe
            for peak_strength in (0.10, 0.20, 0.30):
                peak_dual = tail_ratio(
                    current_ratio,
                    tail_raw[local],
                    local_base,
                    rank,
                    float(case_tail),
                    float(case_peak),
                    gate_start=0.75,
                    use_peak_anchor=peak_strength,
                )
                predictions[
                    f"local_dual_peak{int(100 * peak_strength):02d}_oof"
                ][index] = local_base * peak_dual

        fold_rows.append(
            {
                "fold": fold,
                "train_cases": int(len(np.unique(groups[train_index]))),
                "valid_cases": int(len(np.unique(groups[valid_index]))),
            }
        )
        print(
            f"fold={fold}/{args.folds} train_cases={fold_rows[-1]['train_cases']} "
            f"valid_cases={fold_rows[-1]['valid_cases']}",
            flush=True,
        )

    aggregate_result, case_rows = aggregate(data, predictions)
    payload: dict[str, Any] = {
        "status": "grouped_case_oof_train_only_local_tail_screen",
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "n_cases": int(len(np.unique(groups))),
        "n_points": int(len(groups)),
        "folds": args.folds,
        "seed": args.seed,
        "base_feature_names": base_feature_names,
        "local_feature_names": local_names,
        "feature_variant": (
            "depth3_profile_plus_local" if args.profile_features else "base_plus_local"
        ),
        "local_feature_contract": (
            "Euclidean kNN summaries of physics-V4 predictions, wall spacing, and "
            "local radius; no raw coordinates or truth-derived inference features"
        ),
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
            "high_r2_p05": values["high_r2"]["p05"],
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
