"""Inference-safe near-wall profile features for high-WSS calibration studies."""

from __future__ import annotations

import numpy as np

from calibrate_v4_high_tail_oof import within_case_rank
from calibrate_v4_oof import EPS, build_features


def base_v4_view(data: dict[str, np.ndarray | list[str]]) -> dict[str, np.ndarray | list[str]]:
    """Return a shallow view whose prediction remains the original V4 base."""
    restored = dict(data)
    restored["gradient_v4"] = np.asarray(data["gradient_v4_base"])
    restored["combined_correction"] = np.asarray(data["combined_correction_base"])
    return restored


def _safe(values: np.ndarray, fill: float = 0.0) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result[~np.isfinite(result)] = fill
    return result


def build_profile_features(
    data: dict[str, np.ndarray | list[str]],
) -> tuple[np.ndarray, list[str]]:
    """Append local depth resolution, curvature, and fit-residual diagnostics."""
    restored = base_v4_view(data)
    base_x, base_names = build_features(restored)
    depth_limit = np.maximum(
        _safe(np.asarray(data["depth3__depth_limit_mm"]), 1.0), 1e-6
    )
    grad_v1 = np.linalg.norm(
        np.asarray(data["gradient_v1"], dtype=np.float64), axis=1
    )
    raw_gradient = _safe(np.asarray(data["depth3__raw_scalar_gradient"]))
    depth3_ratio = raw_gradient / np.maximum(grad_v1, EPS)
    depth3_correction = _safe(np.asarray(data["depth3__correction"]), 1.0)
    surface_correction = _safe(np.asarray(data["surface__correction"]), 1.0)
    point_case = np.asarray(data["point_case"])

    values: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, array: np.ndarray) -> None:
        names.append(name)
        values.append(_safe(np.asarray(array, dtype=np.float64)))

    add("profile_depth3_ratio", np.clip(depth3_ratio, -5.0, 5.0))
    add("profile_depth3_correction", depth3_correction)
    add(
        "profile_depth3_minus_surface_correction",
        depth3_correction - surface_correction,
    )
    add("profile_depth3_applied", np.asarray(data["depth3__applied"]))
    add("profile_fusion_applied", np.asarray(data["depth3_fusion_applied"]))
    add("profile_depth3_fraction", np.asarray(data["depth3__selected_fraction"]))
    add(
        "profile_log_depth3_cv",
        np.log(np.maximum(_safe(np.asarray(data["depth3__cv_score"]), 1.0), EPS)),
    )
    add(
        "profile_log_depth3_condition",
        np.log1p(np.maximum(_safe(np.asarray(data["depth3__condition"])), 0.0)),
    )
    add(
        "profile_log_effective_samples",
        np.log1p(np.maximum(_safe(np.asarray(data["depth3__effective_samples"])), 0.0)),
    )
    add(
        "profile_log_anchor_groups",
        np.log1p(np.maximum(_safe(np.asarray(data["depth3__anchor_groups"])), 0.0)),
    )
    add("profile_log_depth_limit", np.log(np.maximum(depth_limit, EPS)))

    for source, short_name in (
        ("depth3__min_cell_depth_mm", "min_depth"),
        ("depth3__depth_p10_mm", "depth_p10"),
        ("depth3__depth_p50_mm", "depth_p50"),
        ("depth3__depth_p90_mm", "depth_p90"),
        ("depth3__max_cell_depth_mm", "max_depth"),
        ("depth3__depth_span_mm", "depth_span"),
    ):
        raw = _safe(np.asarray(data[source]))
        add(f"profile_{short_name}_over_limit", raw / depth_limit)
        if short_name in {"min_depth", "depth_span"}:
            add(f"profile_log_{short_name}", np.log(np.maximum(raw, EPS)))

    coefficient_2 = np.clip(
        _safe(np.asarray(data["depth3__depth_coefficient_2_ratio"])), -10.0, 10.0
    )
    coefficient_3 = np.clip(
        _safe(np.asarray(data["depth3__depth_coefficient_3_ratio"])), -10.0, 10.0
    )
    residual_nrmse = np.clip(
        _safe(np.asarray(data["depth3__fit_residual_nrmse"]), 1.0), 0.0, 10.0
    )
    gradient_spread = np.clip(
        _safe(np.asarray(data["depth3__fit_gradient_spread"]), 1.0), 0.0, 10.0
    )
    add("profile_depth_coefficient_2_ratio", coefficient_2)
    add("profile_depth_coefficient_3_ratio", coefficient_3)
    add("profile_log_residual_nrmse", np.log1p(residual_nrmse))
    add("profile_log_gradient_spread", np.log1p(gradient_spread))
    add("profile_fit_count", np.asarray(data["depth3__fit_count"]))

    secant_values: dict[str, np.ndarray] = {}
    for source, short_name in (
        ("depth3__secant_gradient_p25", "secant_p25"),
        ("depth3__secant_gradient_p50", "secant_p50"),
        ("depth3__secant_gradient_p75", "secant_p75"),
        ("depth3__secant_gradient_p90", "secant_p90"),
        ("depth3__near_secant_gradient_median", "near_secant"),
    ):
        secant = _safe(np.asarray(data[source]))
        secant_values[short_name] = secant
        add(
            f"profile_{short_name}_over_fallback",
            np.clip(secant / np.maximum(grad_v1, EPS), -5.0, 5.0),
        )
    secant_iqr = _safe(np.asarray(data["depth3__secant_gradient_iqr"]))
    add(
        "profile_secant_iqr_over_median",
        np.clip(
            secant_iqr / np.maximum(np.abs(secant_values["secant_p50"]), EPS),
            -10.0,
            10.0,
        ),
    )
    add(
        "profile_log_near_cell_count",
        np.log1p(np.maximum(_safe(np.asarray(data["depth3__near_cell_count"])), 0.0)),
    )

    for name, array in (
        ("profile_min_depth_rank", _safe(np.asarray(data["depth3__min_cell_depth_mm"]))),
        ("profile_curvature2_rank", coefficient_2),
        ("profile_curvature3_rank", coefficient_3),
        ("profile_residual_rank", residual_nrmse),
        ("profile_gradient_spread_rank", gradient_spread),
        ("profile_depth3_correction_rank", depth3_correction),
        ("profile_near_secant_rank", secant_values["near_secant"]),
        ("profile_secant_p90_rank", secant_values["secant_p90"]),
    ):
        add(name, within_case_rank(array, point_case))

    return np.column_stack([base_x, *values]), [*base_names, *names]
