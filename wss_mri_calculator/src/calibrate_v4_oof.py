"""Grouped out-of-fold calibration study for cached V4 point predictions.

All folds are split by case.  The script quantifies how much of the remaining
WSS error is predictable from estimator/geometry diagnostics without using
truth from the held-out cases.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from batch_validate_cfd import summarize
from calculate_wss_cfd import evaluate, wss_from_gradient


EPS = 1e-8


def _safe(values: np.ndarray, fill: float = 0.0) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    return np.nan_to_num(result, nan=fill, posinf=fill, neginf=fill)


def _log1p(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(_safe(values), 0.0))


def load_cache(cache_dir: Path) -> dict[str, np.ndarray | list[str]]:
    files = sorted(cache_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no cache files under {cache_dir}")
    blocks: list[dict[str, np.ndarray]] = []
    cases: list[str] = []
    cohorts: list[str] = []
    for path in files:
        with np.load(path, allow_pickle=False) as source:
            block = {key: np.asarray(source[key]) for key in source.files}
            case = str(block.pop("canonical_id"))
            cohort = str(block.pop("cohort"))
        blocks.append(block)
        cases.append(case)
        cohorts.append(cohort)

    lengths = np.asarray([len(block["truth_mag"]) for block in blocks])
    point_case = np.repeat(np.asarray(cases), lengths)
    point_cohort = np.repeat(np.asarray(cohorts), lengths)
    merged: dict[str, np.ndarray | list[str]] = {
        "case_names": cases,
        "case_cohorts": cohorts,
        "point_case": point_case,
        "point_cohort": point_cohort,
    }
    common = sorted(set.intersection(*(set(block) for block in blocks)))
    for key in common:
        value = blocks[0][key]
        if value.ndim == 0:
            continue
        merged[key] = np.concatenate([block[key] for block in blocks], axis=0)
    return merged


def build_features(data: dict[str, np.ndarray | list[str]]) -> tuple[np.ndarray, list[str]]:
    gradient_v1 = np.asarray(data["gradient_v1"], dtype=np.float64)
    gradient_v3 = np.asarray(data["gradient_v3"], dtype=np.float64)
    gradient_v4 = np.asarray(data["gradient_v4"], dtype=np.float64)
    grad_v1 = np.linalg.norm(gradient_v1, axis=1)
    grad_v3 = np.linalg.norm(gradient_v3, axis=1)
    grad_v4 = np.linalg.norm(gradient_v4, axis=1)
    pred_v1 = np.linalg.norm(wss_from_gradient(gradient_v1, "carreau"), axis=1)
    pred_v3 = np.linalg.norm(wss_from_gradient(gradient_v3, "carreau"), axis=1)
    pred_v4 = np.linalg.norm(wss_from_gradient(gradient_v4, "carreau"), axis=1)
    radius = np.asarray(data["local_radius_mm"], dtype=np.float64)

    raw_surface = np.asarray(data["surface__raw_scalar_gradient"], dtype=np.float64)
    surface_ratio = raw_surface / np.maximum(grad_v1, EPS)
    ray_ratio = np.asarray(data["v3__ray_to_fallback_norm"], dtype=np.float64)
    eta = np.asarray(data["v3__eta_p50_mm"], dtype=np.float64)
    lateral = np.asarray(data["v3__lateral_p50_mm"], dtype=np.float64)

    feature_values: list[np.ndarray] = []
    feature_names: list[str] = []

    def add(name: str, values: np.ndarray) -> None:
        feature_names.append(name)
        feature_values.append(_safe(values))

    add("log_pred_v1", np.log(np.maximum(pred_v1, EPS)))
    add("log_pred_v3", np.log(np.maximum(pred_v3, EPS)))
    add("log_pred_v4", np.log(np.maximum(pred_v4, EPS)))
    add("log_grad_v1", np.log(np.maximum(grad_v1, EPS)))
    add("log_grad_v4", np.log(np.maximum(grad_v4, EPS)))
    add("v3_correction", np.asarray(data["v3_correction"]))
    add("combined_correction", np.asarray(data["combined_correction"]))
    add("surface_ratio", surface_ratio)
    add("ray_ratio", ray_ratio)
    add("surface_valid", np.isfinite(raw_surface).astype(np.float64))
    add("ray_valid", np.isfinite(ray_ratio).astype(np.float64))
    add("surface_minus_ray_ratio", surface_ratio - ray_ratio)
    add("ray_used", np.asarray(data["v3__ray_used"]))
    add("surface_applied", np.asarray(data["surface__applied"]))
    add("log_radius", np.log(np.maximum(radius, EPS)))
    add("v3_depth_over_radius", np.asarray(data["v3__depth_limit_mm"]) / np.maximum(radius, EPS))
    add("surface_depth_over_radius", np.asarray(data["surface__depth_limit_mm"]) / np.maximum(radius, EPS))
    add("eta_over_radius", eta / np.maximum(radius, EPS))
    add("lateral_over_eta", lateral / np.maximum(eta, EPS))
    add("log_v3_re", _log1p(np.asarray(data["v3__local_reynolds"])))
    add("log_surface_re", _log1p(np.asarray(data["surface__local_reynolds"])))
    add("edge_velocity", np.asarray(data["v3__edge_velocity_m_s"]))
    add("v3_scale", np.asarray(data["v3__selected_scale_fraction"]))
    add("surface_scale", np.asarray(data["surface__selected_fraction"]))
    add("log_v3_cv", np.log(np.maximum(_safe(np.asarray(data["v3__cv_score"]), 1.0), EPS)))
    add("log_surface_cv", np.log(np.maximum(_safe(np.asarray(data["surface__cv_score"]), 1.0), EPS)))
    add("log_surface_condition", _log1p(np.asarray(data["surface__condition"])))
    add("surface_effective", _log1p(np.asarray(data["surface__effective_samples"])))
    add("surface_anchor_groups", _log1p(np.asarray(data["surface__anchor_groups"])))
    add("surface_owner_rejected", _log1p(np.asarray(data["surface__ownership_rejected"])))
    add("v3_owner_rejected", _log1p(np.asarray(data["v3__ownership_rejected"])))
    add("v3_tube_rejected", _log1p(np.asarray(data["v3__tube_rejected"])))

    point_cohort = np.asarray(data["point_cohort"])
    for cohort in ("AG", "AAA", "ILO"):
        add(f"cohort_{cohort}", (point_cohort == cohort).astype(np.float64))

    # Inference-time case summaries help estimate case-wide residual scale.
    point_case = np.asarray(data["point_case"])
    for name, values in (
        ("case_pred_p50", pred_v4),
        ("case_pred_p90", pred_v4),
        ("case_radius_p50", radius),
        ("case_surface_ratio_p50", surface_ratio),
        ("case_combined_correction_p50", np.asarray(data["combined_correction"])),
        ("case_re_p50", np.asarray(data["surface__local_reynolds"])),
    ):
        repeated = np.zeros(len(point_case), dtype=np.float64)
        for case in np.unique(point_case):
            mask = point_case == case
            q = 0.90 if name.endswith("p90") else 0.50
            finite = _safe(values[mask])
            repeated[mask] = float(np.quantile(finite, q))
        if "re_" in name or "pred_" in name:
            repeated = _log1p(repeated)
        add(name, repeated)

    # Relative within-case position is available at inference and separates
    # local high-shear regions from globally high-flow cases.
    for name, values in (
        ("case_pred_rank", pred_v4),
        ("case_radius_rank", radius),
        ("case_surface_ratio_rank", _safe(surface_ratio)),
    ):
        rank = np.zeros(len(point_case), dtype=np.float64)
        for case in np.unique(point_case):
            mask = point_case == case
            local = np.asarray(values[mask], dtype=np.float64)
            order = np.argsort(local, kind="stable")
            local_rank = np.empty(len(local), dtype=np.float64)
            local_rank[order] = np.linspace(0.0, 1.0, len(local), endpoint=True)
            rank[mask] = local_rank
        add(name, rank)

    case_pred_p50 = feature_values[feature_names.index("case_pred_p50")]
    add(
        "log_pred_over_case_p50",
        np.log1p(np.maximum(pred_v4, 0.0)) - case_pred_p50,
    )

    return np.column_stack(feature_values), feature_names


def aggregate_predictions(
    data: dict[str, np.ndarray | list[str]],
    predictions: dict[str, np.ndarray],
) -> dict[str, object]:
    point_case = np.asarray(data["point_case"])
    truth_mag = np.asarray(data["truth_mag"], dtype=np.float64)
    truth_vec = np.asarray(data["truth_vec"], dtype=np.float64)
    gradient_v4 = np.asarray(data["gradient_v4"], dtype=np.float64)
    base_vector = wss_from_gradient(gradient_v4, "carreau")
    base_mag = np.linalg.norm(base_vector, axis=1)
    result: dict[str, object] = {}
    for name, magnitude in predictions.items():
        ratio = magnitude / np.maximum(base_mag, EPS)
        vector = base_vector * ratio[:, None]
        reports = []
        for case in np.unique(point_case):
            mask = point_case == case
            reports.append(evaluate(truth_mag[mask], truth_vec[mask], vector[mask]))
        metrics = (
            "raw_r2",
            "scaled_r2",
            "spearman",
            "alpha",
            "mae_pa",
            "nrmse",
            "high_wss_nrmse",
        )
        result[name] = {
            metric: summarize([report[metric] for report in reports])
            for metric in metrics
        }
    return result


def build_case_features(
    x: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cases = np.unique(groups)
    rows = []
    for case in cases:
        local = x[groups == case]
        rows.append(
            np.concatenate(
                [
                    np.mean(local, axis=0),
                    np.std(local, axis=0),
                    np.quantile(local, 0.10, axis=0),
                    np.quantile(local, 0.50, axis=0),
                    np.quantile(local, 0.90, axis=0),
                ]
            )
        )
    return cases, np.stack(rows)


CASE_SCALE_FEATURES = (
    "case_pred_p50",
    "case_pred_p90",
    "case_radius_p50",
    "case_surface_ratio_p50",
    "case_combined_correction_p50",
    "case_re_p50",
    "combined_correction",
    "surface_ratio",
    "surface_applied",
    "ray_used",
    "log_pred_v4",
    "log_radius",
)


def build_case_scale_features(
    x: np.ndarray,
    feature_names: list[str],
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = [feature_names.index(name) for name in CASE_SCALE_FEATURES]
    cases = np.unique(groups)
    rows = []
    for case in cases:
        local = x[groups == case][:, selected]
        rows.append(
            np.concatenate(
                [
                    np.mean(local, axis=0),
                    np.quantile(local, 0.10, axis=0),
                    np.quantile(local, 0.50, axis=0),
                    np.quantile(local, 0.90, axis=0),
                ]
            )
        )
    return cases, np.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    started = time.time()

    data = load_cache(Path(args.cache_dir))
    x, feature_names = build_features(data)
    truth = np.asarray(data["truth_mag"], dtype=np.float64)
    gradient_v4 = np.asarray(data["gradient_v4"], dtype=np.float64)
    base_mag = np.linalg.norm(wss_from_gradient(gradient_v4, "carreau"), axis=1)
    log_ratio_target = np.log(np.maximum(truth, EPS) / np.maximum(base_mag, EPS))
    log_ratio_target = np.clip(log_ratio_target, np.log(0.5), np.log(2.5))
    groups = np.asarray(data["point_case"])
    strata = np.asarray(data["point_cohort"])
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )

    predictions = {
        "v4_base": base_mag.copy(),
        "global_scale_oof": np.zeros_like(base_mag),
        "power_law_oof": np.zeros_like(base_mag),
        "isotonic_oof": np.zeros_like(base_mag),
        "hist_gbdt_oof": np.zeros_like(base_mag),
        "hist_blend75_oof": np.zeros_like(base_mag),
        "hist_blend50_oof": np.zeros_like(base_mag),
        "hist_guarded_oof": np.zeros_like(base_mag),
        "case_ridge_scale_oof": np.zeros_like(base_mag),
        "hist_case_anchored_oof": np.zeros_like(base_mag),
        "hist_case_safe_oof": np.zeros_like(base_mag),
        "extra_trees_oof": np.zeros_like(base_mag),
    }
    fold_rows = []
    for fold, (train_index, valid_index) in enumerate(
        splitter.split(x, strata, groups), start=1
    ):
        train_truth = truth[train_index]
        train_base = base_mag[train_index]
        alpha = float(
            np.dot(train_truth, train_base)
            / max(float(np.dot(train_base, train_base)), EPS)
        )
        predictions["global_scale_oof"][valid_index] = alpha * base_mag[valid_index]

        power = LinearRegression()
        power.fit(
            np.log(np.maximum(train_base, EPS))[:, None],
            np.log(np.maximum(train_truth, EPS)),
        )
        predictions["power_law_oof"][valid_index] = np.exp(
            power.predict(np.log(np.maximum(base_mag[valid_index], EPS))[:, None])
        )

        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0)
        isotonic.fit(train_base, train_truth)
        predictions["isotonic_oof"][valid_index] = isotonic.predict(
            base_mag[valid_index]
        )

        hist = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=350,
            max_leaf_nodes=15,
            min_samples_leaf=120,
            l2_regularization=5.0,
            random_state=args.seed + fold,
        )
        hist.fit(x[train_index], log_ratio_target[train_index])
        hist_ratio = np.exp(hist.predict(x[valid_index]))
        hist_ratio = np.clip(hist_ratio, 0.7, 1.8)
        predictions["hist_gbdt_oof"][valid_index] = (
            base_mag[valid_index] * hist_ratio
        )
        predictions["hist_blend75_oof"][valid_index] = base_mag[valid_index] * (
            alpha + 0.75 * (hist_ratio - alpha)
        )
        predictions["hist_blend50_oof"][valid_index] = base_mag[valid_index] * (
            alpha + 0.50 * (hist_ratio - alpha)
        )
        guarded_ratio = np.clip(alpha + 0.75 * (hist_ratio - alpha), 0.85, 1.60)
        predictions["hist_guarded_oof"][valid_index] = (
            base_mag[valid_index] * guarded_ratio
        )

        train_groups = groups[train_index]
        valid_groups = groups[valid_index]
        train_cases, train_case_x = build_case_scale_features(
            x[train_index], feature_names, train_groups
        )
        valid_cases, valid_case_x = build_case_scale_features(
            x[valid_index], feature_names, valid_groups
        )
        train_case_alpha = []
        for case in train_cases:
            mask = train_groups == case
            local_truth = truth[train_index][mask]
            local_base = base_mag[train_index][mask]
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
        for case, case_alpha in zip(valid_cases, predicted_case_alpha):
            local_in_valid = valid_groups == case
            local_index = valid_index[local_in_valid]
            predictions["case_ridge_scale_oof"][local_index] = (
                base_mag[local_index] * case_alpha
            )
            local_ratio = hist_ratio[local_in_valid]
            local_weight = base_mag[local_index] ** 2
            effective = float(
                np.sum(local_weight * local_ratio)
                / max(float(np.sum(local_weight)), EPS)
            )
            anchored = np.clip(
                local_ratio * case_alpha / max(effective, EPS), 0.85, 1.60
            )
            predictions["hist_case_anchored_oof"][local_index] = (
                base_mag[local_index] * anchored
            )
            local_strength = float(
                np.clip(0.25 + 0.75 * (case_alpha - 1.0) / 0.08, 0.25, 1.0)
            )
            if case_alpha <= 1.000001:
                safe_ratio = np.ones_like(anchored)
            else:
                safe_ratio = np.clip(
                    case_alpha + local_strength * (anchored - case_alpha),
                    0.85,
                    1.60,
                )
            predictions["hist_case_safe_oof"][local_index] = (
                base_mag[local_index] * safe_ratio
            )

        extra = ExtraTreesRegressor(
            n_estimators=300,
            max_features=0.8,
            min_samples_leaf=80,
            n_jobs=16,
            random_state=args.seed + fold,
        )
        extra.fit(x[train_index], log_ratio_target[train_index])
        extra_ratio = np.exp(extra.predict(x[valid_index]))
        predictions["extra_trees_oof"][valid_index] = base_mag[valid_index] * np.clip(
            extra_ratio, 0.7, 1.8
        )
        fold_rows.append(
            {
                "fold": fold,
                "train_cases": int(len(np.unique(groups[train_index]))),
                "valid_cases": int(len(np.unique(groups[valid_index]))),
                "global_alpha": alpha,
                "power_exponent": float(power.coef_[0]),
            }
        )
        print(
            f"fold={fold}/{args.folds} train_cases={fold_rows[-1]['train_cases']} "
            f"valid_cases={fold_rows[-1]['valid_cases']} alpha={alpha:.4f}",
            flush=True,
        )

    aggregate = aggregate_predictions(data, predictions)
    case_rows = []
    point_case = np.asarray(data["point_case"])
    truth = np.asarray(data["truth_mag"], dtype=np.float64)
    truth_vec = np.asarray(data["truth_vec"], dtype=np.float64)
    base_vector = wss_from_gradient(gradient_v4, "carreau")
    for case in np.unique(point_case):
        mask = point_case == case
        methods = {}
        for name, magnitude in predictions.items():
            ratio = magnitude[mask] / np.maximum(base_mag[mask], EPS)
            methods[name] = evaluate(
                truth[mask], truth_vec[mask], base_vector[mask] * ratio[:, None]
            )
        case_rows.append({"canonical_id": str(case), "methods": methods})
    payload = {
        "status": "grouped_case_oof_train_only",
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "n_cases": int(len(np.unique(groups))),
        "n_points": int(len(groups)),
        "folds": args.folds,
        "seed": args.seed,
        "feature_names": feature_names,
        "fold_rows": fold_rows,
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
