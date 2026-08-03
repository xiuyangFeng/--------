"""Multi-scale point-cloud velocity -> WSS wall-gradient reconstruction.

This module keeps the frozen ``adaptive_cv`` v1 estimator intact and adds a
candidate v2 correction for its dominant residual error: a finite-neighbourhood
underestimate of the wall derivative.

The estimator uses only wall/interior coordinates, wall normals and velocity.
It never reads CFD wall-shear truth.  For each wall point it:

1. fits the same no-slip quadratic profile as v1 at several neighbourhoods;
2. keeps the v1 adaptive-CV gradient as the stable direction estimate;
3. extrapolates the multi-scale gradient to zero sampling depth;
4. converts the extrapolation into a positive magnitude correction; and
5. robustly regularizes that correction on the sampled wall point cloud.

The final correction is scalar, so v1's highly accurate WSS direction is
preserved.  Carreau-Yasuda viscosity must be applied after this function.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

MM_TO_M = 1e-3


def _profile_fit(
    eta_m: np.ndarray,
    tangent_velocity: np.ndarray,
    degree: int,
) -> tuple[float, np.ndarray, int, float, float] | None:
    """Fit one normalized no-slip profile and return its diagnostics."""
    min_samples = degree + 3
    keep = eta_m > 1e-7
    if int(keep.sum()) < min_samples:
        return None

    eta = eta_m[keep]
    velocity = tangent_velocity[keep]
    scale = max(float(np.median(eta)), 1e-8)
    z = eta / scale
    design = np.column_stack([z**power for power in range(1, degree + 1)])

    try:
        coefficient, *_ = np.linalg.lstsq(design, velocity, rcond=1e-10)
        gram_inverse = np.linalg.pinv(design.T @ design)
        leverage = np.einsum("ij,jk,ik->i", design, gram_inverse, design)
        leverage = np.clip(leverage, 0.0, 0.999)
        residual = velocity - design @ coefficient
        loo_residual = residual / (1.0 - leverage[:, None])
        signal = float(np.mean(np.sum(velocity**2, axis=1)))
        score = float(np.mean(np.sum(loo_residual**2, axis=1))) / max(
            signal, 1e-16
        )
        condition = float(np.linalg.cond(design.T @ design))
    except np.linalg.LinAlgError:
        return None

    gradient = coefficient[0] / scale
    return score, gradient, int(keep.sum()), condition, scale


def _zero_bandwidth_gradient(
    fits: list[dict],
    *,
    min_score: float,
    cv_tolerance: float,
    extrapolation_power: float,
) -> tuple[np.ndarray | None, int, float, float]:
    """Extrapolate gradient(depth) to zero depth using stable profile fits."""
    stable = [
        fit
        for fit in fits
        if np.isfinite(fit["score"])
        and fit["score"] <= cv_tolerance * min_score
    ]
    if len(stable) < 3:
        return None, len(stable), float("nan"), float("nan")

    depth = np.asarray([fit["scale"] for fit in stable], dtype=np.float64)
    if float(depth.max() / max(depth.min(), 1e-12)) < 1.05:
        return None, len(stable), float("nan"), float("nan")

    gradient = np.stack([fit["gradient"] for fit in stable])
    magnitude = np.linalg.norm(gradient, axis=1)
    if float(np.std(depth)) > 0 and float(np.std(magnitude)) > 0:
        trend_correlation = float(np.corrcoef(depth, magnitude)[0, 1])
    else:
        trend_correlation = 0.0
    reference = gradient[0]
    reference_norm = max(float(np.linalg.norm(reference)), 1e-12)
    cosine = np.einsum("ij,j->i", gradient, reference) / np.maximum(
        np.linalg.norm(gradient, axis=1) * reference_norm, 1e-12
    )
    direction_cosine_p10 = float(np.quantile(cosine, 0.10))
    x = depth**float(extrapolation_power)
    x /= max(float(x.max()), 1e-30)
    design = np.column_stack([np.ones(len(x)), x])

    scores = np.asarray([fit["score"] for fit in stable], dtype=np.float64)
    # Relative weighting is intentionally bounded: LOOCV should identify
    # reliable scales without allowing one near-zero score to dominate.
    floor = max(float(min_score) * 0.5, 1e-12)
    weight = 1.0 / np.maximum(scores, floor)
    weight /= max(float(weight.max()), 1e-12)
    sqrt_weight = np.sqrt(weight)

    try:
        coefficient, *_ = np.linalg.lstsq(
            design * sqrt_weight[:, None],
            gradient * sqrt_weight[:, None],
            rcond=1e-10,
        )
    except np.linalg.LinAlgError:
        return None, len(stable), trend_correlation, direction_cosine_p10
    return coefficient[0], len(stable), trend_correlation, direction_cosine_p10


def _regularize_correction(
    wall_mm: np.ndarray,
    raw_correction: np.ndarray,
    cv_score: np.ndarray,
    *,
    smooth_neighbors: int,
    global_blend: float,
    correction_min: float,
    correction_max: float,
) -> tuple[np.ndarray, float]:
    """Robustly regularize a positive correction on the wall point cloud."""
    finite = np.isfinite(raw_correction) & np.isfinite(cv_score)
    if not finite.any():
        return np.ones(len(wall_mm), dtype=np.float64), 1.0

    clipped = np.clip(raw_correction, correction_min, correction_max)
    quality_cutoff = float(np.quantile(cv_score[finite], 0.70))
    reliable = finite & (cv_score <= quality_cutoff)
    if not reliable.any():
        reliable = finite
    case_correction = float(np.median(clipped[reliable]))

    k = min(max(int(smooth_neighbors), 1), len(wall_mm))
    if k == 1:
        local = clipped
    else:
        _, neighbor_index = cKDTree(wall_mm).query(wall_mm, k=k)
        local = np.nanmedian(clipped[neighbor_index], axis=1)

    blend = float(global_blend)
    regularized = (1.0 - blend) * local + blend * case_correction
    regularized = np.clip(regularized, correction_min, correction_max)
    regularized[~np.isfinite(regularized)] = case_correction
    return regularized, case_correction


def fit_wall_gradient_multiscale(
    wall_mm: np.ndarray,
    normals: np.ndarray,
    interior_mm: np.ndarray,
    velocity: np.ndarray,
    tree: cKDTree,
    *,
    adaptive_neighbors: tuple[int, ...],
    degree: int = 2,
    base_cv_tolerance: float = 1.25,
    extrapolation_cv_tolerance: float = 1.5,
    extrapolation_power: float = 1.0,
    smooth_neighbors: int = 24,
    global_blend: float = 0.0,
    correction_min: float = 1.0,
    correction_max: float = 1.35,
    correction_shrink: float = 0.5,
    max_trend_correlation: float = 0.0,
    min_direction_cosine: float = 0.98,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
    """Estimate wall gradient with adaptive-CV plus multi-scale correction.

    Returns the corrected gradient, the v1 selected sample count, and
    diagnostics containing both the uncorrected base gradient and the final
    positive correction.
    """
    candidates = tuple(
        sorted({int(value) for value in adaptive_neighbors if int(value) > 0})
    )
    if len(candidates) < 3:
        raise ValueError("multiscale extrapolation requires at least 3 candidates")
    if base_cv_tolerance < 1.0 or extrapolation_cv_tolerance < 1.0:
        raise ValueError("CV tolerances must be >= 1.0")
    if extrapolation_power <= 0:
        raise ValueError("extrapolation_power must be positive")
    if not 0.0 <= global_blend <= 1.0:
        raise ValueError("global_blend must be in [0, 1]")
    if not 0 < correction_min <= correction_max:
        raise ValueError("invalid correction bounds")
    if not 0.0 <= correction_shrink <= 1.0:
        raise ValueError("correction_shrink must be in [0, 1]")

    query_k = min(max(candidates), len(interior_mm))
    _, neighbor_index = tree.query(wall_mm, k=query_k)

    n_wall = len(wall_mm)
    base_gradient = np.full((n_wall, 3), np.nan, dtype=np.float64)
    raw_correction = np.ones(n_wall, dtype=np.float64)
    used = np.zeros(n_wall, dtype=np.int32)
    selected_neighbors = np.zeros(n_wall, dtype=np.int32)
    cv_score = np.full(n_wall, np.nan, dtype=np.float64)
    design_condition = np.full(n_wall, np.nan, dtype=np.float64)
    stable_scale_count = np.zeros(n_wall, dtype=np.int32)
    trend_correlation = np.full(n_wall, np.nan, dtype=np.float64)
    scale_direction_cosine = np.full(n_wall, np.nan, dtype=np.float64)

    for row, (point, normal, index) in enumerate(
        zip(wall_mm, normals, neighbor_index)
    ):
        delta_m = (interior_mm[index] - point) * MM_TO_M
        eta_all = delta_m @ normal
        local_velocity = velocity[index]
        tangent_velocity = local_velocity - np.outer(
            local_velocity @ normal, normal
        )

        fits: list[dict] = []
        for candidate in candidates:
            candidate = min(candidate, query_k)
            result = _profile_fit(
                eta_all[:candidate], tangent_velocity[:candidate], degree
            )
            if result is None:
                continue
            score, gradient, n_used, condition, scale = result
            fits.append(
                {
                    "neighbors": candidate,
                    "score": score,
                    "gradient": gradient,
                    "used": n_used,
                    "condition": condition,
                    "scale": scale,
                }
            )

        finite_scores = [fit["score"] for fit in fits if np.isfinite(fit["score"])]
        if not finite_scores:
            continue
        min_score = float(min(finite_scores))
        threshold = base_cv_tolerance * min_score
        chosen = next(
            fit
            for fit in fits
            if np.isfinite(fit["score"]) and fit["score"] <= threshold
        )

        base_gradient[row] = chosen["gradient"]
        used[row] = chosen["used"]
        selected_neighbors[row] = chosen["neighbors"]
        cv_score[row] = chosen["score"]
        design_condition[row] = chosen["condition"]

        extrapolated, count, trend, direction_cosine = _zero_bandwidth_gradient(
            fits,
            min_score=min_score,
            cv_tolerance=extrapolation_cv_tolerance,
            extrapolation_power=extrapolation_power,
        )
        stable_scale_count[row] = count
        trend_correlation[row] = trend
        scale_direction_cosine[row] = direction_cosine
        if extrapolated is None:
            continue

        base_norm = float(np.linalg.norm(chosen["gradient"]))
        extrapolated_norm = float(np.linalg.norm(extrapolated))
        if base_norm <= 1e-12 or not np.isfinite(extrapolated_norm):
            continue
        if float(np.dot(extrapolated, chosen["gradient"])) <= 0:
            continue
        correction_ratio = extrapolated_norm / base_norm
        # A positive zero-bandwidth correction is physically credible only
        # when finite-depth gradients decrease as the sampled layer thickens
        # and their vector directions remain coherent across scales.
        if correction_ratio > 1.0:
            if not np.isfinite(trend) or trend > max_trend_correlation:
                continue
            if not np.isfinite(direction_cosine) or direction_cosine < min_direction_cosine:
                continue
        raw_correction[row] = correction_ratio

    correction, case_correction = _regularize_correction(
        wall_mm,
        raw_correction,
        cv_score,
        smooth_neighbors=smooth_neighbors,
        global_blend=global_blend,
        correction_min=correction_min,
        correction_max=correction_max,
    )
    correction = 1.0 + float(correction_shrink) * (correction - 1.0)
    gradient = base_gradient * correction[:, None]
    diagnostics: dict[str, np.ndarray | float] = {
        "base_gradient": base_gradient,
        "raw_correction": raw_correction,
        "correction": correction,
        "case_correction": case_correction,
        "selected_neighbors": selected_neighbors,
        "cv_score": cv_score,
        "design_condition": design_condition,
        "stable_scale_count": stable_scale_count,
        "trend_correlation": trend_correlation,
        "scale_direction_cosine": scale_direction_cosine,
    }
    return gradient, used, diagnostics
