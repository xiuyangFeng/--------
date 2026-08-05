"""Grouped OOF tail calibration with local and intrinsic geometry features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
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
from calibrate_v4_local_tail_oof import build_local_context
from calibrate_v4_oof import (
    EPS,
    build_case_scale_features,
    build_features,
    load_cache,
)


def build_intrinsic_geometry(
    data: dict[str, np.ndarray | list[str]],
) -> tuple[np.ndarray, list[str]]:
    groups = np.asarray(data["point_case"])
    wall = np.asarray(data["wall_mm"], dtype=np.float64)
    normal = np.asarray(data["normal"], dtype=np.float64)
    radius = np.asarray(data["local_radius_mm"], dtype=np.float64)
    values = np.zeros((len(groups), 11), dtype=np.float64)
    for case in np.unique(groups):
        mask = groups == case
        coords = wall[mask]
        centered = coords - np.mean(coords, axis=0, keepdims=True)
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ axes.T
        scale = np.quantile(projected, 0.95, axis=0) - np.quantile(
            projected, 0.05, axis=0
        )
        scale = np.maximum(scale, 1e-6)
        normalized = projected / scale
        projected_normal = normal[mask] @ axes.T
        local_radius = radius[mask]
        order = np.argsort(local_radius, kind="stable")
        radius_rank = np.empty(len(local_radius), dtype=np.float64)
        radius_rank[order] = np.linspace(0.0, 1.0, len(local_radius), endpoint=True)
        values[mask, 0:3] = normalized
        values[mask, 3:6] = np.abs(normalized)
        values[mask, 6:9] = projected_normal
        values[mask, 9] = np.linalg.norm(normalized, axis=1)
        values[mask, 10] = radius_rank
    names = [
        "intrinsic_pca1",
        "intrinsic_pca2",
        "intrinsic_pca3",
        "intrinsic_abs_pca1",
        "intrinsic_abs_pca2",
        "intrinsic_abs_pca3",
        "intrinsic_normal_pca1",
        "intrinsic_normal_pca2",
        "intrinsic_normal_pca3",
        "intrinsic_radius_from_case_center",
        "intrinsic_local_radius_rank",
    ]
    return values, names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--current-oof-predictions", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--predictions-out", default="")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    started = time.time()

    data = load_cache(Path(args.cache_dir))
    x_base, base_names = build_features(data)
    truth = np.asarray(data["truth_mag"], dtype=np.float64)
    groups = np.asarray(data["point_case"])
    strata = np.asarray(data["point_cohort"])
    base_vector = wss_from_gradient(
        np.asarray(data["gradient_v4"], dtype=np.float64), "carreau"
    )
    base_mag = np.linalg.norm(base_vector, axis=1)
    local_x, local_names = build_local_context(data, base_mag)
    intrinsic_x, intrinsic_names = build_intrinsic_geometry(data)
    x = np.column_stack([x_base, local_x, intrinsic_x])
    ratio_target = np.log(np.maximum(truth, EPS) / np.maximum(base_mag, EPS))
    ratio_target = np.clip(ratio_target, np.log(0.5), np.log(2.5))
    truth_rank = within_case_rank(truth, groups)
    predicted_rank = x_base[:, base_names.index("case_pred_rank")]

    with np.load(args.current_oof_predictions, allow_pickle=False) as source:
        if not np.array_equal(groups, np.asarray(source["point_case"])):
            raise RuntimeError("current OOF predictions do not match cache order")
        current_prediction = np.asarray(source["current_safe_oof"], dtype=np.float64)

    names = (
        "v4_base",
        "current_safe_oof",
        "intrinsic_extra20_safe_oof",
        "intrinsic_extra10_safe_oof",
        "intrinsic_extra20_dual_oof",
        "intrinsic_extra10_dual_oof",
    )
    predictions = {name: np.zeros_like(base_mag) for name in names}
    predictions["v4_base"] = base_mag.copy()
    predictions["current_safe_oof"] = current_prediction.copy()
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    fold_rows = []

    for fold, (train_index, valid_index) in enumerate(
        splitter.split(x, strata, groups), start=1
    ):
        tail_emphasis = smooth_gate(truth_rank[train_index], 0.75)
        underprediction = np.clip(
            ratio_target[train_index] / max(float(np.log(2.5)), EPS), 0.0, 1.0
        )
        sample_weight = (
            1.0 + 12.0 * tail_emphasis + 8.0 * tail_emphasis * underprediction
        )
        raw_predictions: dict[int, np.ndarray] = {}
        for minimum_leaf in (20, 10):
            model = ExtraTreesRegressor(
                n_estimators=400,
                max_features=0.75,
                min_samples_leaf=minimum_leaf,
                n_jobs=16,
                random_state=args.seed + 100 * fold + minimum_leaf,
            )
            model.fit(
                x[train_index], ratio_target[train_index], sample_weight=sample_weight
            )
            raw_predictions[minimum_leaf] = np.clip(
                np.exp(model.predict(x[valid_index])), 0.7, 2.3
            )

        overall_model, tail_scale_model, peak_scale_model = fit_case_scale_models(
            x_base[train_index],
            base_names,
            groups[train_index],
            truth[train_index],
            base_mag[train_index],
        )
        valid_cases, valid_case_x = build_case_scale_features(
            x_base[valid_index], base_names, groups[valid_index]
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
            current_ratio = current_prediction[index] / np.maximum(local_base, EPS)
            for minimum_leaf in (20, 10):
                learned = raw_predictions[minimum_leaf][local]
                safe = current_safe_ratio(learned, local_base, float(case_alpha))
                dual = tail_ratio(
                    current_ratio,
                    learned,
                    local_base,
                    rank,
                    float(case_tail),
                    float(case_peak),
                    gate_start=0.75,
                    use_peak_anchor=False,
                )
                predictions[f"intrinsic_extra{minimum_leaf}_safe_oof"][index] = (
                    local_base * safe
                )
                predictions[f"intrinsic_extra{minimum_leaf}_dual_oof"][index] = (
                    local_base * dual
                )

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
        "status": "grouped_case_oof_train_only_intrinsic_tail_screen",
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "current_oof_predictions": str(
            Path(args.current_oof_predictions).resolve()
        ),
        "n_cases": int(len(np.unique(groups))),
        "n_points": int(len(groups)),
        "folds": args.folds,
        "seed": args.seed,
        "base_feature_names": base_names,
        "local_feature_names": local_names,
        "intrinsic_feature_names": intrinsic_names,
        "intrinsic_feature_contract": (
            "case-centered, robustly scaled PCA coordinates and normals; no case ID, "
            "raw absolute coordinates, or truth-derived inference features"
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
