"""Closest-surface moving-least-squares WSS estimator (experimental V4).

V3 is frozen after its blind test and must remain unchanged.  This module
tests a different observation model on train cases only:

* every interior cell is attached to its own nearest wall anchor;
* depth is measured from that local curved wall, rather than from the target
  point's tangent plane;
* no slip is imposed on the whole local surface by multiplying every basis
  function by the local wall distance; and
* the well-established V1 direction is retained while MLS estimates only the
  derivative magnitude, which is the part that remains systematically biased.

The local scalar model is

    u_q = z (b0 + b_eta z + b1 s1 + b2 s2),

where ``u_q`` is velocity projected on the fallback shear direction,
``z`` is local normal depth and ``s1,s2`` locate the sample's wall anchor in
the target tangent plane.  Therefore ``b0 / depth_scale`` is the target wall
gradient without an interpolation-to-ray stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


MM_TO_M = 1e-3


@dataclass(frozen=True)
class SurfaceMLSV4Config:
    query_neighbors: int = 384
    depth_fractions: tuple[float, ...] = (0.45, 0.65, 0.82, 1.0)
    min_samples: int = 16
    min_anchor_groups: int = 6
    density_kg_m3: float = 1060.0
    viscosity_pa_s: float = 0.0035
    re_depth_coefficient: float = 5.0
    depth_min_mm: float = 0.20
    depth_max_mm: float = 2.50
    depth_max_radius_fraction: float = 0.70
    fallback_radius_mm: float = 5.0
    edge_velocity_quantile: float = 0.80
    edge_velocity_min_m_s: float = 0.02
    surface_spacing_multiplier: float = 3.0
    surface_depth_multiplier: float = 0.75
    surface_min_radius_mm: float = 0.30
    surface_max_radius_mm: float = 2.0
    surface_max_radius_fraction: float = 0.50
    ownership_min_normal_cosine: float = 0.50
    support_depth_overshoot: float = 1.15
    cv_tolerance: float = 1.25
    huber_delta: float = 1.5
    robust_iterations: int = 3
    ridge: float = 1e-8
    parallel_transport: bool = False
    depth_degree: int = 2
    tangential_degree: int = 1
    group_balance_power: float = 0.0
    correction_min: float = 1.0
    correction_max: float = 1.50
    correction_shrink: float = 1.0
    max_cv_score: float = 0.20
    max_condition: float = 1.0e6

    def validate(self) -> None:
        fractions = np.asarray(self.depth_fractions, dtype=np.float64)
        if self.query_neighbors < self.min_samples:
            raise ValueError("query_neighbors must be >= min_samples")
        if np.any(fractions <= 0) or np.any(fractions > 1):
            raise ValueError("depth_fractions must be in (0, 1]")
        if np.any(np.diff(fractions) <= 0):
            raise ValueError("depth_fractions must be strictly increasing")
        if self.min_samples < 8 or self.min_anchor_groups < 2:
            raise ValueError("insufficient sample/group minimum")
        if not 0 < self.depth_min_mm <= self.depth_max_mm:
            raise ValueError("invalid depth bounds")
        if not 0 < self.depth_max_radius_fraction <= 1:
            raise ValueError("invalid radius depth fraction")
        if not -1 <= self.ownership_min_normal_cosine <= 1:
            raise ValueError("ownership normal cosine must be in [-1, 1]")
        if self.cv_tolerance < 1 or self.huber_delta <= 0:
            raise ValueError("invalid CV/Huber setting")
        if self.robust_iterations < 1 or self.ridge < 0:
            raise ValueError("invalid robust/ridge setting")
        if self.depth_degree not in {1, 2, 3}:
            raise ValueError("depth_degree must be 1, 2, or 3")
        if self.tangential_degree not in {0, 1, 2}:
            raise ValueError("tangential_degree must be 0, 1, or 2")
        if not 0 <= self.group_balance_power <= 1:
            raise ValueError("group_balance_power must be in [0, 1]")
        if not 0 < self.correction_min <= self.correction_max:
            raise ValueError("invalid correction bounds")
        if not 0 <= self.correction_shrink <= 1:
            raise ValueError("correction_shrink must be in [0, 1]")


@dataclass(frozen=True)
class SurfaceColumnV4Config(SurfaceMLSV4Config):
    """Two-level prism-column estimator configuration."""

    parallel_transport: bool = True
    min_column_samples: int = 4
    min_column_depth_span_mm: float = 0.12
    column_depth_degree: int = 2
    surface_gradient_degree: int = 1
    column_score_floor: float = 1e-4

    def validate(self) -> None:
        super().validate()
        if self.min_column_samples < 3:
            raise ValueError("min_column_samples must be >= 3")
        if self.min_column_depth_span_mm <= 0:
            raise ValueError("min_column_depth_span_mm must be positive")
        if self.column_depth_degree not in {1, 2, 3}:
            raise ValueError("column_depth_degree must be 1, 2, or 3")
        if self.surface_gradient_degree not in {0, 1, 2}:
            raise ValueError("surface_gradient_degree must be 0, 1, or 2")
        if self.column_score_floor <= 0:
            raise ValueError("column_score_floor must be positive")


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = int(np.argmin(np.abs(normal)))
    seed = np.zeros(3, dtype=np.float64)
    seed[axis] = 1.0
    first = np.cross(normal, seed)
    first /= max(float(np.linalg.norm(first)), 1e-12)
    second = np.cross(normal, first)
    second /= max(float(np.linalg.norm(second)), 1e-12)
    return first, second


def _depth_limit_mm(
    radius_mm: float,
    edge_velocity_m_s: float,
    config: SurfaceMLSV4Config,
) -> tuple[float, float]:
    radius = (
        float(radius_mm)
        if np.isfinite(radius_mm) and radius_mm > 0
        else config.fallback_radius_mm
    )
    upper = min(config.depth_max_mm, config.depth_max_radius_fraction * radius)
    upper = max(upper, min(config.depth_min_mm, radius))
    lower = min(config.depth_min_mm, upper)
    speed = max(float(edge_velocity_m_s), 0.0)
    reynolds = (
        config.density_kg_m3
        * speed
        * radius
        * MM_TO_M
        / config.viscosity_pa_s
    )
    if speed < config.edge_velocity_min_m_s or reynolds <= 1e-12:
        depth = upper
    else:
        depth = config.re_depth_coefficient * radius / np.sqrt(reynolds)
    return float(np.clip(depth, lower, upper)), float(reynolds)


def _weighted_ridge(
    design: np.ndarray,
    target: np.ndarray,
    base_weight: np.ndarray,
    config: SurfaceMLSV4Config,
) -> dict[str, Any] | None:
    weight = np.maximum(np.asarray(base_weight, dtype=np.float64), 1e-10)
    coefficient: np.ndarray | None = None
    for _ in range(config.robust_iterations):
        sqrt_weight = np.sqrt(weight)
        weighted_design = design * sqrt_weight[:, None]
        weighted_target = target * sqrt_weight
        gram = weighted_design.T @ weighted_design
        ridge_scale = max(float(np.trace(gram)) / max(len(gram), 1), 1e-12)
        regularized = gram + config.ridge * ridge_scale * np.eye(gram.shape[0])
        try:
            coefficient = np.linalg.solve(
                regularized, weighted_design.T @ weighted_target
            )
        except np.linalg.LinAlgError:
            return None
        residual = target - design @ coefficient
        center = float(np.median(residual))
        mad = 1.4826 * float(np.median(np.abs(residual - center)))
        scale = max(mad, 1e-6 * max(float(np.std(target)), 1.0), 1e-12)
        normalized = np.abs(residual - center) / (config.huber_delta * scale)
        robust = np.ones_like(normalized)
        outside = normalized > 1.0
        robust[outside] = 1.0 / normalized[outside]
        weight = np.maximum(base_weight * robust, 1e-10)
    if coefficient is None:
        return None

    sqrt_weight = np.sqrt(weight)
    weighted_design = design * sqrt_weight[:, None]
    weighted_target = target * sqrt_weight
    gram = weighted_design.T @ weighted_design
    ridge_scale = max(float(np.trace(gram)) / max(len(gram), 1), 1e-12)
    regularized = gram + config.ridge * ridge_scale * np.eye(gram.shape[0])
    try:
        gram_inverse = np.linalg.pinv(regularized)
        leverage = weight * np.einsum(
            "ij,jk,ik->i", design, gram_inverse, design
        )
        leverage = np.clip(leverage, 0.0, 0.995)
        weighted_residual = weighted_target - weighted_design @ coefficient
        loo = weighted_residual / (1.0 - leverage)
        signal = float(np.sum(weighted_target**2) / max(np.sum(weight), 1e-12))
        score = (
            float(np.sum(loo**2) / max(np.sum(weight), 1e-12))
            / max(signal, 1e-16)
        )
        condition = float(np.linalg.cond(regularized))
    except np.linalg.LinAlgError:
        return None
    residual = target - design @ coefficient
    residual_rmse = float(
        np.sqrt(np.sum(weight * residual**2) / max(float(np.sum(weight)), 1e-12))
    )
    target_rms = float(
        np.sqrt(np.sum(weight * target**2) / max(float(np.sum(weight)), 1e-12))
    )
    return {
        "coefficient": coefficient,
        "score": score,
        "condition": condition,
        "residual_rmse": residual_rmse,
        "residual_nrmse": residual_rmse / max(target_rms, 1e-12),
        "effective_samples": float(
            np.sum(weight) ** 2 / max(float(np.sum(weight**2)), 1e-12)
        ),
    }


def _parallel_transport_to_target(
    velocity: np.ndarray,
    anchor_normal: np.ndarray,
    target_normal: np.ndarray,
) -> np.ndarray:
    """Minimally rotate anchor-tangent velocity into the target tangent plane."""
    tangent = velocity - np.einsum("ij,ij->i", velocity, anchor_normal)[:, None] * anchor_normal
    target = np.broadcast_to(target_normal, anchor_normal.shape)
    cross = np.cross(anchor_normal, target)
    sine = np.linalg.norm(cross, axis=1)
    cosine = np.clip(np.einsum("ij,ij->i", anchor_normal, target), -1.0, 1.0)
    axis = cross / np.maximum(sine[:, None], 1e-12)
    rotated = (
        tangent * cosine[:, None]
        + np.cross(axis, tangent) * sine[:, None]
        + axis * np.einsum("ij,ij->i", axis, tangent)[:, None] * (1.0 - cosine)[:, None]
    )
    nearly_parallel = sine <= 1e-10
    rotated[nearly_parallel] = tangent[nearly_parallel]
    return rotated


def _surface_design(
    z: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    config: SurfaceMLSV4Config,
) -> np.ndarray:
    columns = [z**power for power in range(1, config.depth_degree + 1)]
    if config.tangential_degree >= 1:
        columns.extend([z * q1, z * q2])
    if config.tangential_degree >= 2:
        columns.extend([z * q1**2, z * q1 * q2, z * q2**2])
    return np.column_stack(columns)


def _wall_gradient_design(
    q1: np.ndarray,
    q2: np.ndarray,
    degree: int,
) -> np.ndarray:
    columns = [np.ones(len(q1), dtype=np.float64)]
    if degree >= 1:
        columns.extend([q1, q2])
    if degree >= 2:
        columns.extend([q1**2, q1 * q2, q2**2])
    return np.column_stack(columns)


def fit_wall_gradient_surface_mls(
    target_wall_mm: np.ndarray,
    target_normals: np.ndarray,
    full_wall_mm: np.ndarray,
    full_wall_normals: np.ndarray,
    interior_mm: np.ndarray,
    velocity: np.ndarray,
    interior_tree: cKDTree,
    *,
    local_radius_mm: np.ndarray | None,
    fallback_gradient: np.ndarray,
    config: SurfaceMLSV4Config = SurfaceMLSV4Config(),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Estimate a conservative magnitude correction in closest-surface coordinates."""
    config.validate()
    n_target = len(target_wall_mm)
    if target_normals.shape != (n_target, 3):
        raise ValueError("target_normals shape mismatch")
    if fallback_gradient.shape != (n_target, 3):
        raise ValueError("fallback_gradient shape mismatch")
    if local_radius_mm is not None and len(local_radius_mm) != n_target:
        raise ValueError("local_radius_mm length mismatch")

    wall_tree = cKDTree(full_wall_mm)
    spacing_distance, _ = wall_tree.query(
        target_wall_mm, k=min(8, len(full_wall_mm))
    )
    if spacing_distance.ndim == 1:
        wall_spacing = np.maximum(spacing_distance, 1e-3)
    else:
        wall_spacing = np.median(spacing_distance[:, 1:], axis=1)
    query_k = min(config.query_neighbors, len(interior_mm))
    _, neighbor_index = interior_tree.query(target_wall_mm, k=query_k)
    if neighbor_index.ndim == 1:
        neighbor_index = neighbor_index[:, None]

    gradient = np.asarray(fallback_gradient, dtype=np.float64).copy()
    raw_gradient = np.full(n_target, np.nan, dtype=np.float64)
    correction = np.ones(n_target, dtype=np.float64)
    applied = np.zeros(n_target, dtype=np.int8)
    selected_fraction = np.full(n_target, np.nan, dtype=np.float64)
    cv_score = np.full(n_target, np.nan, dtype=np.float64)
    condition = np.full(n_target, np.nan, dtype=np.float64)
    selected_samples = np.zeros(n_target, dtype=np.int32)
    effective_samples = np.zeros(n_target, dtype=np.float64)
    anchor_groups = np.zeros(n_target, dtype=np.int32)
    depth_limit = np.full(n_target, np.nan, dtype=np.float64)
    surface_radius = np.full(n_target, np.nan, dtype=np.float64)
    local_reynolds = np.full(n_target, np.nan, dtype=np.float64)
    owner_rejected = np.zeros(n_target, dtype=np.int32)
    min_cell_depth = np.full(n_target, np.nan, dtype=np.float64)
    depth_p10 = np.full(n_target, np.nan, dtype=np.float64)
    depth_p50 = np.full(n_target, np.nan, dtype=np.float64)
    depth_p90 = np.full(n_target, np.nan, dtype=np.float64)
    max_cell_depth = np.full(n_target, np.nan, dtype=np.float64)
    depth_span = np.full(n_target, np.nan, dtype=np.float64)
    depth_coefficient_2_ratio = np.full(n_target, np.nan, dtype=np.float64)
    depth_coefficient_3_ratio = np.full(n_target, np.nan, dtype=np.float64)
    fit_residual_rmse = np.full(n_target, np.nan, dtype=np.float64)
    fit_residual_nrmse = np.full(n_target, np.nan, dtype=np.float64)
    fit_gradient_spread = np.full(n_target, np.nan, dtype=np.float64)
    fit_count = np.zeros(n_target, dtype=np.int32)
    secant_gradient_p25 = np.full(n_target, np.nan, dtype=np.float64)
    secant_gradient_p50 = np.full(n_target, np.nan, dtype=np.float64)
    secant_gradient_p75 = np.full(n_target, np.nan, dtype=np.float64)
    secant_gradient_p90 = np.full(n_target, np.nan, dtype=np.float64)
    near_secant_gradient_median = np.full(n_target, np.nan, dtype=np.float64)
    secant_gradient_iqr = np.full(n_target, np.nan, dtype=np.float64)
    near_cell_count = np.zeros(n_target, dtype=np.int32)

    for row, (point, normal, index) in enumerate(
        zip(target_wall_mm, target_normals, neighbor_index)
    ):
        fallback = fallback_gradient[row]
        fallback_norm = float(np.linalg.norm(fallback))
        if not np.isfinite(fallback_norm) or fallback_norm <= 1e-12:
            continue
        direction = fallback / fallback_norm
        candidate = interior_mm[index]
        _, anchor = wall_tree.query(candidate, k=1)
        anchor_point = full_wall_mm[anchor]
        anchor_normal = full_wall_normals[anchor]
        anchor_to_cell = candidate - anchor_point
        local_depth = np.einsum("ij,ij->i", anchor_to_cell, anchor_normal)
        owner_cosine = anchor_normal @ normal
        anchor_delta = anchor_point - point
        tangent_1, tangent_2 = _tangent_basis(normal)
        s1 = anchor_delta @ tangent_1
        s2 = anchor_delta @ tangent_2
        surface_distance = np.sqrt(s1**2 + s2**2)

        radius = (
            float(local_radius_mm[row])
            if local_radius_mm is not None and np.isfinite(local_radius_mm[row])
            else config.fallback_radius_mm
        )
        radius = max(radius, config.depth_min_mm)
        provisional_depth = min(
            config.depth_max_mm, config.depth_max_radius_fraction * radius
        )
        bandwidth = max(
            config.surface_min_radius_mm,
            config.surface_spacing_multiplier * float(wall_spacing[row]),
            config.surface_depth_multiplier * provisional_depth,
        )
        bandwidth = min(
            bandwidth,
            config.surface_max_radius_mm,
            max(config.surface_min_radius_mm, config.surface_max_radius_fraction * radius),
        )
        surface_radius[row] = bandwidth
        preliminary = (
            (local_depth > 1e-4)
            & (local_depth <= provisional_depth)
            & (surface_distance <= bandwidth)
            & (owner_cosine >= config.ownership_min_normal_cosine)
        )
        if int(preliminary.sum()) >= config.min_samples:
            speed = np.linalg.norm(
                velocity[index][preliminary]
                - np.outer(velocity[index][preliminary] @ normal, normal),
                axis=1,
            )
            edge_velocity = float(np.quantile(speed, config.edge_velocity_quantile))
        else:
            edge_velocity = 0.0
        depth_limit[row], local_reynolds[row] = _depth_limit_mm(
            radius, edge_velocity, config
        )

        owner_base = (
            (local_depth > 1e-4)
            & (surface_distance <= bandwidth)
            & (owner_cosine >= config.ownership_min_normal_cosine)
        )
        owner_rejected[row] = int(
            np.sum(
                (local_depth > 1e-4)
                & (local_depth <= config.support_depth_overshoot * depth_limit[row])
                & ~owner_base
            )
        )
        if config.parallel_transport:
            transported_velocity = _parallel_transport_to_target(
                velocity[index], anchor_normal, normal
            )
            scalar_velocity = transported_velocity @ direction
        else:
            scalar_velocity = velocity[index] @ direction
        fits: list[dict[str, Any]] = []
        for fraction in config.depth_fractions:
            fit_depth = float(fraction * depth_limit[row])
            keep = owner_base & (
                local_depth <= config.support_depth_overshoot * fit_depth
            )
            selected = np.flatnonzero(keep)
            if len(selected) < config.min_samples:
                continue
            groups = int(len(np.unique(anchor[selected])))
            if groups < config.min_anchor_groups:
                continue
            depth_scale_m = max(fit_depth * MM_TO_M, 1e-8)
            z = local_depth[selected] * MM_TO_M / depth_scale_m
            q1 = s1[selected] / max(bandwidth, 1e-6)
            q2 = s2[selected] / max(bandwidth, 1e-6)
            design = _surface_design(z, q1, q2, config)
            radial = surface_distance[selected] / max(bandwidth, 1e-6)
            normalized_depth = local_depth[selected] / max(fit_depth, 1e-6)
            base_weight = (
                np.exp(-0.5 * radial**2)
                * np.exp(-0.5 * normalized_depth**2)
                * np.clip(owner_cosine[selected], 0.0, 1.0) ** 2
            )
            if config.group_balance_power > 0:
                _, inverse, counts = np.unique(
                    anchor[selected], return_inverse=True, return_counts=True
                )
                base_weight *= counts[inverse] ** (-config.group_balance_power)
            fit = _weighted_ridge(
                design, scalar_velocity[selected], base_weight, config
            )
            if fit is None or not np.isfinite(fit["score"]):
                continue
            fit.update(
                {
                    "fraction": float(fraction),
                    "gradient": float(fit["coefficient"][0] / depth_scale_m),
                    "n_used": int(len(selected)),
                    "anchor_groups": groups,
                    "min_depth_mm": float(np.min(local_depth[selected])),
                    "depth_p10_mm": float(np.quantile(local_depth[selected], 0.10)),
                    "depth_p50_mm": float(np.quantile(local_depth[selected], 0.50)),
                    "depth_p90_mm": float(np.quantile(local_depth[selected], 0.90)),
                    "max_depth_mm": float(np.max(local_depth[selected])),
                }
            )
            secant_gradient = scalar_velocity[selected] / np.maximum(
                local_depth[selected] * MM_TO_M, 1e-12
            )
            near_cutoff = float(np.quantile(local_depth[selected], 0.25))
            near = local_depth[selected] <= near_cutoff
            fit.update(
                {
                    "secant_gradient_p25": float(np.quantile(secant_gradient, 0.25)),
                    "secant_gradient_p50": float(np.quantile(secant_gradient, 0.50)),
                    "secant_gradient_p75": float(np.quantile(secant_gradient, 0.75)),
                    "secant_gradient_p90": float(np.quantile(secant_gradient, 0.90)),
                    "near_secant_gradient_median": float(
                        np.median(secant_gradient[near])
                    ),
                    "near_cell_count": int(np.sum(near)),
                }
            )
            fits.append(fit)
        if not fits:
            continue
        min_score = min(float(fit["score"]) for fit in fits)
        eligible = [
            fit
            for fit in fits
            if float(fit["score"]) <= config.cv_tolerance * min_score
        ]
        chosen = min(eligible, key=lambda fit: float(fit["fraction"]))
        estimate = float(chosen["gradient"])
        ratio = estimate / fallback_norm
        raw_gradient[row] = estimate
        selected_fraction[row] = float(chosen["fraction"])
        cv_score[row] = float(chosen["score"])
        condition[row] = float(chosen["condition"])
        selected_samples[row] = int(chosen["n_used"])
        effective_samples[row] = float(chosen["effective_samples"])
        anchor_groups[row] = int(chosen["anchor_groups"])
        min_cell_depth[row] = float(chosen["min_depth_mm"])
        depth_p10[row] = float(chosen["depth_p10_mm"])
        depth_p50[row] = float(chosen["depth_p50_mm"])
        depth_p90[row] = float(chosen["depth_p90_mm"])
        max_cell_depth[row] = float(chosen["max_depth_mm"])
        depth_span[row] = max_cell_depth[row] - min_cell_depth[row]
        coefficient = np.asarray(chosen["coefficient"], dtype=np.float64)
        coefficient_scale = max(abs(float(coefficient[0])), 1e-12)
        if config.depth_degree >= 2:
            depth_coefficient_2_ratio[row] = float(coefficient[1] / coefficient_scale)
        if config.depth_degree >= 3:
            depth_coefficient_3_ratio[row] = float(coefficient[2] / coefficient_scale)
        fit_residual_rmse[row] = float(chosen["residual_rmse"])
        fit_residual_nrmse[row] = float(chosen["residual_nrmse"])
        fit_gradient_spread[row] = float(
            np.std([float(fit["gradient"]) for fit in fits])
            / max(abs(estimate), 1e-12)
        )
        fit_count[row] = int(len(fits))
        secant_gradient_p25[row] = float(chosen["secant_gradient_p25"])
        secant_gradient_p50[row] = float(chosen["secant_gradient_p50"])
        secant_gradient_p75[row] = float(chosen["secant_gradient_p75"])
        secant_gradient_p90[row] = float(chosen["secant_gradient_p90"])
        near_secant_gradient_median[row] = float(
            chosen["near_secant_gradient_median"]
        )
        secant_gradient_iqr[row] = (
            secant_gradient_p75[row] - secant_gradient_p25[row]
        )
        near_cell_count[row] = int(chosen["near_cell_count"])
        if (
            np.isfinite(ratio)
            and ratio > 0
            and cv_score[row] <= config.max_cv_score
            and condition[row] <= config.max_condition
        ):
            clipped = float(
                np.clip(ratio, config.correction_min, config.correction_max)
            )
            clipped = 1.0 + config.correction_shrink * (clipped - 1.0)
            correction[row] = clipped
            gradient[row] = fallback * clipped
            applied[row] = int(abs(clipped - 1.0) > 1e-12)

    diagnostics = {
        "raw_scalar_gradient": raw_gradient,
        "correction": correction,
        "applied": applied,
        "selected_fraction": selected_fraction,
        "cv_score": cv_score,
        "condition": condition,
        "selected_samples": selected_samples,
        "effective_samples": effective_samples,
        "anchor_groups": anchor_groups,
        "depth_limit_mm": depth_limit,
        "surface_radius_mm": surface_radius,
        "local_reynolds": local_reynolds,
        "ownership_rejected": owner_rejected,
        "min_cell_depth_mm": min_cell_depth,
        "depth_p10_mm": depth_p10,
        "depth_p50_mm": depth_p50,
        "depth_p90_mm": depth_p90,
        "max_cell_depth_mm": max_cell_depth,
        "depth_span_mm": depth_span,
        "depth_coefficient_2_ratio": depth_coefficient_2_ratio,
        "depth_coefficient_3_ratio": depth_coefficient_3_ratio,
        "fit_residual_rmse": fit_residual_rmse,
        "fit_residual_nrmse": fit_residual_nrmse,
        "fit_gradient_spread": fit_gradient_spread,
        "fit_count": fit_count,
        "secant_gradient_p25": secant_gradient_p25,
        "secant_gradient_p50": secant_gradient_p50,
        "secant_gradient_p75": secant_gradient_p75,
        "secant_gradient_p90": secant_gradient_p90,
        "near_secant_gradient_median": near_secant_gradient_median,
        "secant_gradient_iqr": secant_gradient_iqr,
        "near_cell_count": near_cell_count,
    }
    return gradient, diagnostics


def fit_wall_gradient_surface_columns(
    target_wall_mm: np.ndarray,
    target_normals: np.ndarray,
    full_wall_mm: np.ndarray,
    full_wall_normals: np.ndarray,
    interior_mm: np.ndarray,
    velocity: np.ndarray,
    interior_tree: cKDTree,
    *,
    local_radius_mm: np.ndarray | None,
    fallback_gradient: np.ndarray,
    config: SurfaceColumnV4Config = SurfaceColumnV4Config(),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fit each wall-anchor prism column first, then regress gradients on the wall."""
    config.validate()
    n_target = len(target_wall_mm)
    wall_tree = cKDTree(full_wall_mm)
    spacing_distance, _ = wall_tree.query(
        target_wall_mm, k=min(8, len(full_wall_mm))
    )
    wall_spacing = (
        np.maximum(spacing_distance, 1e-3)
        if spacing_distance.ndim == 1
        else np.median(spacing_distance[:, 1:], axis=1)
    )
    query_k = min(config.query_neighbors, len(interior_mm))
    _, neighbor_index = interior_tree.query(target_wall_mm, k=query_k)
    if neighbor_index.ndim == 1:
        neighbor_index = neighbor_index[:, None]

    gradient = np.asarray(fallback_gradient, dtype=np.float64).copy()
    raw_gradient = np.full(n_target, np.nan, dtype=np.float64)
    correction = np.ones(n_target, dtype=np.float64)
    applied = np.zeros(n_target, dtype=np.int8)
    cv_score = np.full(n_target, np.nan, dtype=np.float64)
    condition = np.full(n_target, np.nan, dtype=np.float64)
    selected_fraction = np.full(n_target, np.nan, dtype=np.float64)
    selected_columns = np.zeros(n_target, dtype=np.int32)
    depth_limit = np.full(n_target, np.nan, dtype=np.float64)

    for row, (point, normal, index) in enumerate(
        zip(target_wall_mm, target_normals, neighbor_index)
    ):
        fallback = fallback_gradient[row]
        fallback_norm = float(np.linalg.norm(fallback))
        if not np.isfinite(fallback_norm) or fallback_norm <= 1e-12:
            continue
        direction = fallback / fallback_norm
        candidate = interior_mm[index]
        _, anchor = wall_tree.query(candidate, k=1)
        anchor_point = full_wall_mm[anchor]
        anchor_normal = full_wall_normals[anchor]
        anchor_to_cell = candidate - anchor_point
        local_depth = np.einsum("ij,ij->i", anchor_to_cell, anchor_normal)
        owner_cosine = anchor_normal @ normal
        anchor_delta = anchor_point - point
        tangent_1, tangent_2 = _tangent_basis(normal)
        s1 = anchor_delta @ tangent_1
        s2 = anchor_delta @ tangent_2
        surface_distance = np.sqrt(s1**2 + s2**2)
        radius = (
            float(local_radius_mm[row])
            if local_radius_mm is not None and np.isfinite(local_radius_mm[row])
            else config.fallback_radius_mm
        )
        radius = max(radius, config.depth_min_mm)
        provisional_depth = min(
            config.depth_max_mm, config.depth_max_radius_fraction * radius
        )
        bandwidth = min(
            max(
                config.surface_min_radius_mm,
                config.surface_spacing_multiplier * float(wall_spacing[row]),
                config.surface_depth_multiplier * provisional_depth,
            ),
            config.surface_max_radius_mm,
            max(config.surface_min_radius_mm, config.surface_max_radius_fraction * radius),
        )
        owner_base = (
            (local_depth > 1e-4)
            & (surface_distance <= bandwidth)
            & (owner_cosine >= config.ownership_min_normal_cosine)
        )
        tangent_velocity = _parallel_transport_to_target(
            velocity[index], anchor_normal, normal
        )
        scalar_velocity = tangent_velocity @ direction
        preliminary = owner_base & (local_depth <= provisional_depth)
        if int(preliminary.sum()) >= config.min_samples:
            speed = np.linalg.norm(tangent_velocity[preliminary], axis=1)
            edge_velocity = float(np.quantile(speed, config.edge_velocity_quantile))
        else:
            edge_velocity = 0.0
        depth_limit[row], _ = _depth_limit_mm(radius, edge_velocity, config)

        fits: list[dict[str, Any]] = []
        for fraction in config.depth_fractions:
            fit_depth = float(fraction * depth_limit[row])
            selected = np.flatnonzero(
                owner_base
                & (local_depth <= config.support_depth_overshoot * fit_depth)
            )
            if len(selected) < config.min_samples:
                continue
            group_gradients: list[float] = []
            group_scores: list[float] = []
            group_s1: list[float] = []
            group_s2: list[float] = []
            group_cosine: list[float] = []
            for anchor_id in np.unique(anchor[selected]):
                group = selected[anchor[selected] == anchor_id]
                if len(group) < config.min_column_samples:
                    continue
                group_depth = local_depth[group]
                if float(np.ptp(group_depth)) < config.min_column_depth_span_mm:
                    continue
                scale_m = max(float(np.max(group_depth)) * MM_TO_M, 1e-8)
                z = group_depth * MM_TO_M / scale_m
                design = np.column_stack(
                    [z**power for power in range(1, config.column_depth_degree + 1)]
                )
                weight = np.exp(-0.5 * (group_depth / max(fit_depth, 1e-6)) ** 2)
                column_fit = _weighted_ridge(
                    design, scalar_velocity[group], weight, config
                )
                if column_fit is None or not np.isfinite(column_fit["score"]):
                    continue
                group_gradients.append(
                    float(column_fit["coefficient"][0] / scale_m)
                )
                group_scores.append(float(column_fit["score"]))
                group_s1.append(float(np.median(s1[group])))
                group_s2.append(float(np.median(s2[group])))
                group_cosine.append(float(np.median(owner_cosine[group])))
            if len(group_gradients) < config.min_anchor_groups:
                continue
            group_gradient = np.asarray(group_gradients, dtype=np.float64)
            q1 = np.asarray(group_s1, dtype=np.float64) / max(bandwidth, 1e-6)
            q2 = np.asarray(group_s2, dtype=np.float64) / max(bandwidth, 1e-6)
            design = _wall_gradient_design(q1, q2, config.surface_gradient_degree)
            radial = np.sqrt(q1**2 + q2**2)
            score = np.asarray(group_scores, dtype=np.float64)
            weight = (
                np.exp(-0.5 * radial**2)
                * np.asarray(group_cosine, dtype=np.float64) ** 2
                / np.maximum(score, config.column_score_floor)
            )
            weight /= max(float(np.max(weight)), 1e-12)
            surface_fit = _weighted_ridge(design, group_gradient, weight, config)
            if surface_fit is None or not np.isfinite(surface_fit["score"]):
                continue
            fits.append(
                {
                    "fraction": float(fraction),
                    "gradient": float(surface_fit["coefficient"][0]),
                    "score": float(surface_fit["score"]),
                    "condition": float(surface_fit["condition"]),
                    "columns": int(len(group_gradient)),
                }
            )
        if not fits:
            continue
        min_score = min(float(fit["score"]) for fit in fits)
        eligible = [
            fit
            for fit in fits
            if float(fit["score"]) <= config.cv_tolerance * min_score
        ]
        chosen = min(eligible, key=lambda fit: float(fit["fraction"]))
        estimate = float(chosen["gradient"])
        ratio = estimate / fallback_norm
        raw_gradient[row] = estimate
        cv_score[row] = float(chosen["score"])
        condition[row] = float(chosen["condition"])
        selected_fraction[row] = float(chosen["fraction"])
        selected_columns[row] = int(chosen["columns"])
        if (
            np.isfinite(ratio)
            and ratio > 0
            and condition[row] <= config.max_condition
        ):
            clipped = float(np.clip(ratio, config.correction_min, config.correction_max))
            clipped = 1.0 + config.correction_shrink * (clipped - 1.0)
            correction[row] = clipped
            gradient[row] = fallback * clipped
            applied[row] = int(abs(clipped - 1.0) > 1e-12)

    return gradient, {
        "raw_scalar_gradient": raw_gradient,
        "correction": correction,
        "applied": applied,
        "cv_score": cv_score,
        "condition": condition,
        "selected_fraction": selected_fraction,
        "selected_columns": selected_columns,
        "depth_limit_mm": depth_limit,
    }
