"""Geometry-aware normal-tube point-cloud velocity -> WSS estimator (v3).

V1 and V2 intentionally remain untouched.  This module replaces their
isotropic Euclidean-K sampling stage with a configurable normal-tube sampler:

* physical depth is capped by a local-radius/Reynolds prior;
* candidates are stratified along the inward wall normal;
* lateral distance to the normal is explicitly bounded and weighted;
* optional nearest-wall ownership rejects points from non-local wall patches;
* a no-slip anisotropic MLS model absorbs first-order tangential variation; and
* optional multi-depth extrapolation preserves the stable base direction.

The estimator reads coordinates, normals and interior velocity only.  Local
radius may be supplied by the caller from a geometry sidecar, or estimated
from the wall point cloud before this function is called.  It never reads WSS
truth when selecting points or estimating the wall gradient.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree


MM_TO_M = 1e-3


def _known_keys(mapping: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {label} keys: {unknown}")


def _tuple_float(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{label} must be a list of numbers")
    result = tuple(float(item) for item in value)
    if not result:
        raise ValueError(f"{label} cannot be empty")
    return result


@dataclass(frozen=True)
class NormalMultiscaleV3Config:
    """Validated algorithm configuration for the V3 estimator."""

    query_neighbors: int = 256
    sampling_mode: str = "normal_tube_points"
    depth_fractions: tuple[float, ...] = (0.35, 0.50, 0.70, 0.85, 1.00)
    depth_bins: int = 8
    points_per_bin: int = 2
    min_samples: int = 8
    tube_wall_spacing_multiplier: float = 0.75
    tube_depth_slope: float = 0.25
    tube_min_radius_mm: float = 0.15
    tube_max_radius_fraction: float = 0.35
    fallback_tube_expand: float = 1.50
    ray_stations: int = 12
    ray_interpolation_neighbors: int = 16
    ray_min_interpolation_neighbors: int = 4
    ray_start_fractions: tuple[float, ...] = (0.35,)
    ray_distance_floor_mm: float = 0.03
    ownership_enabled: bool = True
    ownership_max_anchor_spacing_multiplier: float = 2.50
    ownership_max_anchor_separation_mm: float = 2.00
    ownership_min_normal_cosine: float = 0.50
    density_kg_m3: float = 1060.0
    re_viscosity_pa_s: float = 0.0035
    re_depth_coefficient: float = 5.0
    depth_min_mm: float = 0.20
    depth_max_mm: float = 2.50
    depth_max_radius_fraction: float = 0.70
    fallback_radius_mm: float = 5.0
    edge_velocity_quantile: float = 0.80
    edge_velocity_min_m_s: float = 0.02
    pulsatile_frequency_hz: float = 0.0
    fit_model: str = "anisotropic_mls"
    cv_tolerance: float = 1.25
    lateral_weight_power: float = 2.0
    extrapolation_enabled: bool = True
    extrapolation_cv_tolerance: float = 1.50
    extrapolation_power: float = 1.0
    correction_min: float = 1.0
    correction_max: float = 1.35
    correction_shrink: float = 0.50
    max_trend_correlation: float = 0.0
    min_direction_cosine: float = 0.98
    hybrid_enabled: bool = False
    hybrid_lateral_over_eta_min: float = 1.40
    hybrid_max_ownership_rejected: int = -1
    hybrid_min_ray_to_fallback_norm: float = 0.0
    hybrid_max_ray_to_fallback_norm: float = 1.0e6
    hybrid_preserve_fallback_direction: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "NormalMultiscaleV3Config":
        """Build from the nested JSON ``method`` section and reject typos."""
        _known_keys(
            payload,
            {
                "sampling",
                "ownership",
                "depth_prior",
                "fit",
                "multiscale",
                "hybrid",
            },
            "method",
        )
        sampling = dict(payload.get("sampling", {}))
        ownership = dict(payload.get("ownership", {}))
        depth = dict(payload.get("depth_prior", {}))
        fit = dict(payload.get("fit", {}))
        multiscale = dict(payload.get("multiscale", {}))
        hybrid = dict(payload.get("hybrid", {}))
        _known_keys(
            sampling,
            {
                "query_neighbors",
                "mode",
                "depth_fractions",
                "depth_bins",
                "points_per_bin",
                "min_samples",
                "tube_wall_spacing_multiplier",
                "tube_depth_slope",
                "tube_min_radius_mm",
                "tube_max_radius_fraction",
                "fallback_tube_expand",
                "ray_stations",
                "ray_interpolation_neighbors",
                "ray_min_interpolation_neighbors",
                "ray_start_fractions",
                "ray_distance_floor_mm",
            },
            "sampling",
        )
        _known_keys(
            ownership,
            {
                "enabled",
                "max_anchor_spacing_multiplier",
                "max_anchor_separation_mm",
                "min_normal_cosine",
            },
            "ownership",
        )
        _known_keys(
            depth,
            {
                "density_kg_m3",
                "viscosity_pa_s",
                "re_depth_coefficient",
                "min_mm",
                "max_mm",
                "max_radius_fraction",
                "fallback_radius_mm",
                "edge_velocity_quantile",
                "edge_velocity_min_m_s",
                "pulsatile_frequency_hz",
            },
            "depth_prior",
        )
        _known_keys(
            fit,
            {"model", "cv_tolerance", "lateral_weight_power"},
            "fit",
        )
        _known_keys(
            multiscale,
            {
                "enabled",
                "cv_tolerance",
                "extrapolation_power",
                "correction_min",
                "correction_max",
                "correction_shrink",
                "max_trend_correlation",
                "min_direction_cosine",
            },
            "multiscale",
        )
        _known_keys(
            hybrid,
            {
                "enabled",
                "lateral_over_eta_min",
                "max_ownership_rejected",
                "min_ray_to_fallback_norm",
                "max_ray_to_fallback_norm",
                "preserve_fallback_direction",
            },
            "hybrid",
        )
        config = cls(
            query_neighbors=int(sampling.get("query_neighbors", cls.query_neighbors)),
            sampling_mode=str(sampling.get("mode", cls.sampling_mode)),
            depth_fractions=_tuple_float(
                sampling.get("depth_fractions", cls.depth_fractions),
                "sampling.depth_fractions",
            ),
            depth_bins=int(sampling.get("depth_bins", cls.depth_bins)),
            points_per_bin=int(
                sampling.get("points_per_bin", cls.points_per_bin)
            ),
            min_samples=int(sampling.get("min_samples", cls.min_samples)),
            tube_wall_spacing_multiplier=float(
                sampling.get(
                    "tube_wall_spacing_multiplier",
                    cls.tube_wall_spacing_multiplier,
                )
            ),
            tube_depth_slope=float(
                sampling.get("tube_depth_slope", cls.tube_depth_slope)
            ),
            tube_min_radius_mm=float(
                sampling.get("tube_min_radius_mm", cls.tube_min_radius_mm)
            ),
            tube_max_radius_fraction=float(
                sampling.get(
                    "tube_max_radius_fraction", cls.tube_max_radius_fraction
                )
            ),
            fallback_tube_expand=float(
                sampling.get("fallback_tube_expand", cls.fallback_tube_expand)
            ),
            ray_stations=int(sampling.get("ray_stations", cls.ray_stations)),
            ray_interpolation_neighbors=int(
                sampling.get(
                    "ray_interpolation_neighbors",
                    cls.ray_interpolation_neighbors,
                )
            ),
            ray_min_interpolation_neighbors=int(
                sampling.get(
                    "ray_min_interpolation_neighbors",
                    cls.ray_min_interpolation_neighbors,
                )
            ),
            ray_start_fractions=_tuple_float(
                sampling.get("ray_start_fractions", cls.ray_start_fractions),
                "sampling.ray_start_fractions",
            ),
            ray_distance_floor_mm=float(
                sampling.get(
                    "ray_distance_floor_mm", cls.ray_distance_floor_mm
                )
            ),
            ownership_enabled=bool(
                ownership.get("enabled", cls.ownership_enabled)
            ),
            ownership_max_anchor_spacing_multiplier=float(
                ownership.get(
                    "max_anchor_spacing_multiplier",
                    cls.ownership_max_anchor_spacing_multiplier,
                )
            ),
            ownership_max_anchor_separation_mm=float(
                ownership.get(
                    "max_anchor_separation_mm",
                    cls.ownership_max_anchor_separation_mm,
                )
            ),
            ownership_min_normal_cosine=float(
                ownership.get(
                    "min_normal_cosine", cls.ownership_min_normal_cosine
                )
            ),
            density_kg_m3=float(
                depth.get("density_kg_m3", cls.density_kg_m3)
            ),
            re_viscosity_pa_s=float(
                depth.get("viscosity_pa_s", cls.re_viscosity_pa_s)
            ),
            re_depth_coefficient=float(
                depth.get("re_depth_coefficient", cls.re_depth_coefficient)
            ),
            depth_min_mm=float(depth.get("min_mm", cls.depth_min_mm)),
            depth_max_mm=float(depth.get("max_mm", cls.depth_max_mm)),
            depth_max_radius_fraction=float(
                depth.get(
                    "max_radius_fraction", cls.depth_max_radius_fraction
                )
            ),
            fallback_radius_mm=float(
                depth.get("fallback_radius_mm", cls.fallback_radius_mm)
            ),
            edge_velocity_quantile=float(
                depth.get("edge_velocity_quantile", cls.edge_velocity_quantile)
            ),
            edge_velocity_min_m_s=float(
                depth.get("edge_velocity_min_m_s", cls.edge_velocity_min_m_s)
            ),
            pulsatile_frequency_hz=float(
                depth.get("pulsatile_frequency_hz", cls.pulsatile_frequency_hz)
            ),
            fit_model=str(fit.get("model", cls.fit_model)),
            cv_tolerance=float(fit.get("cv_tolerance", cls.cv_tolerance)),
            lateral_weight_power=float(
                fit.get("lateral_weight_power", cls.lateral_weight_power)
            ),
            extrapolation_enabled=bool(
                multiscale.get("enabled", cls.extrapolation_enabled)
            ),
            extrapolation_cv_tolerance=float(
                multiscale.get(
                    "cv_tolerance", cls.extrapolation_cv_tolerance
                )
            ),
            extrapolation_power=float(
                multiscale.get(
                    "extrapolation_power", cls.extrapolation_power
                )
            ),
            correction_min=float(
                multiscale.get("correction_min", cls.correction_min)
            ),
            correction_max=float(
                multiscale.get("correction_max", cls.correction_max)
            ),
            correction_shrink=float(
                multiscale.get("correction_shrink", cls.correction_shrink)
            ),
            max_trend_correlation=float(
                multiscale.get(
                    "max_trend_correlation", cls.max_trend_correlation
                )
            ),
            min_direction_cosine=float(
                multiscale.get(
                    "min_direction_cosine", cls.min_direction_cosine
                )
            ),
            hybrid_enabled=bool(hybrid.get("enabled", cls.hybrid_enabled)),
            hybrid_lateral_over_eta_min=float(
                hybrid.get(
                    "lateral_over_eta_min", cls.hybrid_lateral_over_eta_min
                )
            ),
            hybrid_max_ownership_rejected=int(
                hybrid.get(
                    "max_ownership_rejected",
                    cls.hybrid_max_ownership_rejected,
                )
            ),
            hybrid_min_ray_to_fallback_norm=float(
                hybrid.get(
                    "min_ray_to_fallback_norm",
                    cls.hybrid_min_ray_to_fallback_norm,
                )
            ),
            hybrid_max_ray_to_fallback_norm=float(
                hybrid.get(
                    "max_ray_to_fallback_norm",
                    cls.hybrid_max_ray_to_fallback_norm,
                )
            ),
            hybrid_preserve_fallback_direction=bool(
                hybrid.get(
                    "preserve_fallback_direction",
                    cls.hybrid_preserve_fallback_direction,
                )
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        fractions = np.asarray(self.depth_fractions, dtype=np.float64)
        ray_starts = np.asarray(self.ray_start_fractions, dtype=np.float64)
        if self.query_neighbors < self.min_samples:
            raise ValueError("query_neighbors must be >= min_samples")
        if self.sampling_mode not in {"normal_tube_points", "normal_ray_interpolation"}:
            raise ValueError(
                "sampling.mode must be normal_tube_points or normal_ray_interpolation"
            )
        if self.depth_bins < 2 or self.points_per_bin < 1 or self.min_samples < 5:
            raise ValueError("invalid depth-bin/sample configuration")
        if np.any(fractions <= 0) or np.any(fractions > 1):
            raise ValueError("depth_fractions must be in (0, 1]")
        if np.any(np.diff(fractions) <= 0):
            raise ValueError("depth_fractions must be strictly increasing")
        if self.tube_min_radius_mm <= 0 or self.tube_max_radius_fraction <= 0:
            raise ValueError("tube radii must be positive")
        if self.fallback_tube_expand < 1.0:
            raise ValueError("fallback_tube_expand must be >= 1")
        if self.ray_stations < self.min_samples:
            raise ValueError("ray_stations must be >= min_samples")
        if not 1 <= self.ray_min_interpolation_neighbors <= self.ray_interpolation_neighbors:
            raise ValueError("invalid ray interpolation neighbor counts")
        if np.any(ray_starts <= 0) or np.any(ray_starts >= 1):
            raise ValueError("ray_start_fractions must be in (0, 1)")
        if np.any(np.diff(ray_starts) <= 0):
            raise ValueError("ray_start_fractions must be strictly increasing")
        if self.ray_distance_floor_mm <= 0:
            raise ValueError("ray_distance_floor_mm must be positive")
        if not -1.0 <= self.ownership_min_normal_cosine <= 1.0:
            raise ValueError("ownership min_normal_cosine must be in [-1, 1]")
        if self.density_kg_m3 <= 0 or self.re_viscosity_pa_s <= 0:
            raise ValueError("density and viscosity must be positive")
        if not 0 < self.depth_min_mm <= self.depth_max_mm:
            raise ValueError("invalid absolute depth bounds")
        if not 0 < self.depth_max_radius_fraction <= 1.0:
            raise ValueError("depth max_radius_fraction must be in (0, 1]")
        if not 0 < self.edge_velocity_quantile <= 1.0:
            raise ValueError("edge_velocity_quantile must be in (0, 1]")
        if self.fit_model not in {"normal_profile", "anisotropic_mls"}:
            raise ValueError("fit.model must be normal_profile or anisotropic_mls")
        if (
            self.sampling_mode == "normal_ray_interpolation"
            and self.fit_model != "normal_profile"
        ):
            raise ValueError(
                "normal_ray_interpolation requires fit.model=normal_profile"
            )
        if self.cv_tolerance < 1.0 or self.extrapolation_cv_tolerance < 1.0:
            raise ValueError("CV tolerances must be >= 1")
        if self.extrapolation_power <= 0:
            raise ValueError("extrapolation_power must be positive")
        if not 0 < self.correction_min <= self.correction_max:
            raise ValueError("invalid correction bounds")
        if not 0 <= self.correction_shrink <= 1:
            raise ValueError("correction_shrink must be in [0, 1]")
        if self.hybrid_lateral_over_eta_min <= 0:
            raise ValueError("hybrid lateral_over_eta_min must be positive")
        if self.hybrid_max_ownership_rejected < -1:
            raise ValueError("hybrid max_ownership_rejected must be >= -1")
        if self.hybrid_min_ray_to_fallback_norm < 0:
            raise ValueError("hybrid min_ray_to_fallback_norm must be >= 0")
        if (
            self.hybrid_max_ray_to_fallback_norm
            < self.hybrid_min_ray_to_fallback_norm
        ):
            raise ValueError("hybrid max ray/fallback norm must be >= min")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_local_radius_from_wall(
    target_wall_mm: np.ndarray,
    target_normals: np.ndarray,
    full_wall_mm: np.ndarray,
    full_wall_normals: np.ndarray,
    wall_tree: cKDTree,
    *,
    search_neighbors: int = 512,
    opposite_normal_cosine: float = -0.6,
) -> np.ndarray:
    """Approximate radius as half the nearest opposite-facing wall distance."""
    k = min(max(int(search_neighbors), 8), len(full_wall_mm))
    distance, index = wall_tree.query(target_wall_mm, k=k)
    if index.ndim == 1:
        index = index[:, None]
        distance = distance[:, None]
    dot = np.einsum("nki,ni->nk", full_wall_normals[index], target_normals)
    spacing_distance, _ = wall_tree.query(target_wall_mm, k=min(8, len(full_wall_mm)))
    if spacing_distance.ndim == 1:
        spacing = np.maximum(spacing_distance, 1e-3)
    else:
        spacing = np.median(spacing_distance[:, 1:], axis=1)
    mask = (dot <= float(opposite_normal_cosine)) & (
        distance > np.maximum(3.0 * spacing[:, None], 1.0)
    )
    opposite = np.min(np.where(mask, distance, np.inf), axis=1)
    radius = 0.5 * opposite
    radius[~np.isfinite(radius)] = np.nan
    return radius


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
    config: NormalMultiscaleV3Config,
) -> tuple[float, float]:
    """Return Reynolds-based maximum depth and the local Reynolds number."""
    radius_mm = (
        float(radius_mm)
        if np.isfinite(radius_mm) and radius_mm > 0
        else config.fallback_radius_mm
    )
    radius_m = radius_mm * MM_TO_M
    upper_mm = min(
        config.depth_max_mm,
        config.depth_max_radius_fraction * radius_mm,
    )
    upper_mm = max(upper_mm, min(config.depth_min_mm, radius_mm))
    lower_mm = min(config.depth_min_mm, upper_mm)
    speed = max(float(edge_velocity_m_s), 0.0)
    reynolds = (
        config.density_kg_m3
        * speed
        * radius_m
        / config.re_viscosity_pa_s
    )
    if speed < config.edge_velocity_min_m_s or reynolds <= 1e-12:
        depth_mm = upper_mm
    else:
        depth_mm = (
            config.re_depth_coefficient
            * radius_mm
            / np.sqrt(reynolds)
        )
    if config.pulsatile_frequency_hz > 0:
        omega = 2.0 * np.pi * config.pulsatile_frequency_hz
        stokes_mm = (
            np.sqrt(
                2.0
                * config.re_viscosity_pa_s
                / (config.density_kg_m3 * omega)
            )
            / MM_TO_M
        )
        depth_mm = min(depth_mm, float(stokes_mm))
    return float(np.clip(depth_mm, lower_mm, upper_mm)), float(reynolds)


def _stratified_selection(
    eta_mm: np.ndarray,
    lateral_mm: np.ndarray,
    eligible: np.ndarray,
    depth_mm: float,
    tube_radius_mm: np.ndarray,
    *,
    depth_bins: int,
    points_per_bin: int,
) -> np.ndarray:
    selected: list[int] = []
    edges = np.linspace(0.0, float(depth_mm), int(depth_bins) + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = eligible & (eta_mm > lower) & (eta_mm <= upper)
        candidate = np.flatnonzero(in_bin)
        if not len(candidate):
            continue
        center = 0.5 * (lower + upper)
        half_width = max(0.5 * (upper - lower), 1e-6)
        score = (
            lateral_mm[candidate]
            / np.maximum(tube_radius_mm[candidate], 1e-6)
            + 0.20 * np.abs(eta_mm[candidate] - center) / half_width
        )
        order = candidate[np.argsort(score, kind="stable")]
        selected.extend(order[:points_per_bin].tolist())
    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.asarray(sorted(set(selected), key=selected.index), dtype=np.int64)


def _profile_fit(
    eta_m: np.ndarray,
    lateral_1_m: np.ndarray,
    lateral_2_m: np.ndarray,
    lateral_mm: np.ndarray,
    tube_radius_mm: np.ndarray,
    tangent_velocity: np.ndarray,
    config: NormalMultiscaleV3Config,
) -> dict[str, Any] | None:
    if len(eta_m) < config.min_samples:
        return None
    eta_scale = max(float(np.median(eta_m)), 1e-8)
    lateral_scale = max(
        float(np.median(np.maximum(lateral_mm, config.tube_min_radius_mm)))
        * MM_TO_M,
        1e-8,
    )
    z = eta_m / eta_scale
    q1 = lateral_1_m / lateral_scale
    q2 = lateral_2_m / lateral_scale
    if config.fit_model == "normal_profile":
        design = np.column_stack([z, z**2])
    else:
        design = np.column_stack([z, z**2, z * q1, z * q2])
    ratio = lateral_mm / np.maximum(tube_radius_mm, 1e-6)
    weight = np.exp(-0.5 * np.maximum(ratio, 0.0) ** config.lateral_weight_power)
    sqrt_weight = np.sqrt(np.maximum(weight, 1e-6))
    weighted_design = design * sqrt_weight[:, None]
    weighted_velocity = tangent_velocity * sqrt_weight[:, None]
    try:
        coefficient, *_ = np.linalg.lstsq(
            weighted_design, weighted_velocity, rcond=1e-10
        )
        gram_inverse = np.linalg.pinv(weighted_design.T @ weighted_design)
        leverage = np.einsum(
            "ij,jk,ik->i", weighted_design, gram_inverse, weighted_design
        )
        leverage = np.clip(leverage, 0.0, 0.999)
        residual = weighted_velocity - weighted_design @ coefficient
        loo_residual = residual / (1.0 - leverage[:, None])
        signal = float(np.mean(np.sum(weighted_velocity**2, axis=1)))
        score = float(np.mean(np.sum(loo_residual**2, axis=1))) / max(
            signal, 1e-16
        )
        condition = float(np.linalg.cond(weighted_design.T @ weighted_design))
    except np.linalg.LinAlgError:
        return None
    return {
        "score": score,
        "gradient": coefficient[0] / eta_scale,
        "condition": condition,
        "n_used": int(len(eta_m)),
        "eta_p50_mm": float(np.median(eta_m) / MM_TO_M),
        "eta_max_mm": float(np.max(eta_m) / MM_TO_M),
        "lateral_p50_mm": float(np.median(lateral_mm)),
        "lateral_p95_mm": float(np.quantile(lateral_mm, 0.95)),
    }


def _normal_ray_fits(
    point: np.ndarray,
    normal: np.ndarray,
    depth_mm: float,
    wall_spacing_mm: float,
    full_wall_mm: np.ndarray,
    full_wall_normals: np.ndarray,
    wall_tree: cKDTree,
    interior_mm: np.ndarray,
    velocity: np.ndarray,
    interior_tree: cKDTree,
    config: NormalMultiscaleV3Config,
) -> tuple[list[dict[str, Any]], int]:
    """Interpolate velocities at physical stations on the target normal ray."""
    fits: list[dict[str, Any]] = []
    rejected_total = 0
    interpolation_k = min(config.ray_interpolation_neighbors, len(interior_mm))
    max_anchor_separation = max(
        config.ownership_max_anchor_separation_mm,
        config.ownership_max_anchor_spacing_multiplier * float(wall_spacing_mm),
    )
    for start_fraction in config.ray_start_fractions:
        stations_mm = np.linspace(
            float(start_fraction) * depth_mm,
            depth_mm,
            config.ray_stations,
        )
        query = point + stations_mm[:, None] * normal
        query_distance, query_index = interior_tree.query(query, k=interpolation_k)
        if query_index.ndim == 1:
            query_index = query_index[:, None]
            query_distance = query_distance[:, None]
        support_delta = interior_mm[query_index] - point
        support_eta = np.einsum("nki,i->nk", support_delta, normal)
        support_squared = np.sum(support_delta**2, axis=2)
        support_lateral = np.sqrt(
            np.maximum(support_squared - support_eta**2, 0.0)
        )
        valid = (support_eta > 1e-4) & (support_eta <= 1.25 * depth_mm)
        if config.ownership_enabled:
            flat_index = query_index.reshape(-1)
            _, anchor = wall_tree.query(interior_mm[flat_index], k=1)
            anchor = anchor.reshape(query_index.shape)
            anchor_separation = np.linalg.norm(
                full_wall_mm[anchor] - point, axis=2
            )
            normal_cosine = np.einsum(
                "nki,i->nk", full_wall_normals[anchor], normal
            )
            owner_ok = (
                (anchor_separation <= max_anchor_separation)
                & (normal_cosine >= config.ownership_min_normal_cosine)
            )
            rejected_total += int(np.sum(valid & ~owner_ok))
            valid &= owner_ok
        interpolated_velocity: list[np.ndarray] = []
        retained_station: list[float] = []
        support_radius: list[float] = []
        for station_row in range(config.ray_stations):
            keep = valid[station_row]
            if int(keep.sum()) < config.ray_min_interpolation_neighbors:
                continue
            distance = query_distance[station_row, keep]
            weight = 1.0 / np.maximum(
                distance, config.ray_distance_floor_mm
            ) ** 2
            local_velocity = velocity[query_index[station_row, keep]]
            tangent_velocity = local_velocity - np.outer(
                local_velocity @ normal, normal
            )
            interpolated_velocity.append(
                np.sum(weight[:, None] * tangent_velocity, axis=0)
                / max(float(np.sum(weight)), 1e-12)
            )
            retained_station.append(float(stations_mm[station_row]))
            support_radius.append(
                float(
                    np.sum(weight * support_lateral[station_row, keep])
                    / max(float(np.sum(weight)), 1e-12)
                )
            )
        if len(retained_station) < config.min_samples:
            continue
        station = np.asarray(retained_station, dtype=np.float64)
        pseudo_velocity = np.stack(interpolated_velocity)
        zero_lateral = np.zeros(len(station), dtype=np.float64)
        fit = _profile_fit(
            station * MM_TO_M,
            zero_lateral,
            zero_lateral,
            zero_lateral,
            np.ones(len(station), dtype=np.float64),
            pseudo_velocity,
            config,
        )
        if fit is None:
            continue
        support = np.asarray(support_radius, dtype=np.float64)
        fit["lateral_p50_mm"] = float(np.median(support))
        fit["lateral_p95_mm"] = float(np.quantile(support, 0.95))
        fit["start_fraction"] = float(start_fraction)
        fits.append(fit)
    return fits, rejected_total


def _zero_depth_correction(
    fits: list[dict[str, Any]],
    chosen: dict[str, Any],
    config: NormalMultiscaleV3Config,
) -> tuple[float, int, float, float]:
    if not config.extrapolation_enabled:
        return 1.0, 0, float("nan"), float("nan")
    min_score = min(float(fit["score"]) for fit in fits)
    stable = [
        fit
        for fit in fits
        if np.isfinite(fit["score"])
        and fit["score"] <= config.extrapolation_cv_tolerance * min_score
    ]
    if len(stable) < 3:
        return 1.0, len(stable), float("nan"), float("nan")
    depth = np.asarray([fit["depth_mm"] for fit in stable], dtype=np.float64)
    gradient = np.stack([fit["gradient"] for fit in stable])
    magnitude = np.linalg.norm(gradient, axis=1)
    if float(depth.max() / max(depth.min(), 1e-12)) < 1.10:
        return 1.0, len(stable), float("nan"), float("nan")
    trend = (
        float(np.corrcoef(depth, magnitude)[0, 1])
        if np.std(depth) > 0 and np.std(magnitude) > 0
        else 0.0
    )
    reference = np.asarray(chosen["gradient"], dtype=np.float64)
    reference_norm = max(float(np.linalg.norm(reference)), 1e-12)
    cosine = np.einsum("ij,j->i", gradient, reference) / np.maximum(
        np.linalg.norm(gradient, axis=1) * reference_norm, 1e-12
    )
    direction_p10 = float(np.quantile(cosine, 0.10))
    x = depth**config.extrapolation_power
    x /= max(float(x.max()), 1e-12)
    design = np.column_stack([np.ones(len(x)), x])
    scores = np.asarray([fit["score"] for fit in stable], dtype=np.float64)
    floor = max(min_score * 0.5, 1e-12)
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
        return 1.0, len(stable), trend, direction_p10
    extrapolated = coefficient[0]
    extrapolated_norm = float(np.linalg.norm(extrapolated))
    if not np.isfinite(extrapolated_norm) or float(extrapolated @ reference) <= 0:
        return 1.0, len(stable), trend, direction_p10
    ratio = extrapolated_norm / reference_norm
    if ratio > 1.0:
        if trend > config.max_trend_correlation:
            return 1.0, len(stable), trend, direction_p10
        if direction_p10 < config.min_direction_cosine:
            return 1.0, len(stable), trend, direction_p10
    ratio = float(np.clip(ratio, config.correction_min, config.correction_max))
    ratio = 1.0 + config.correction_shrink * (ratio - 1.0)
    return ratio, len(stable), trend, direction_p10


def fit_wall_gradient_normal_multiscale(
    target_wall_mm: np.ndarray,
    target_normals: np.ndarray,
    full_wall_mm: np.ndarray,
    full_wall_normals: np.ndarray,
    interior_mm: np.ndarray,
    velocity: np.ndarray,
    interior_tree: cKDTree,
    *,
    local_radius_mm: np.ndarray | None,
    config: NormalMultiscaleV3Config,
    fallback_gradient: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Estimate target-wall gradients with the configured V3 method."""
    config.validate()
    n_target = len(target_wall_mm)
    if target_normals.shape != (n_target, 3):
        raise ValueError("target_normals shape mismatch")
    if len(full_wall_mm) != len(full_wall_normals):
        raise ValueError("full wall coordinate/normal length mismatch")
    if local_radius_mm is not None and len(local_radius_mm) != n_target:
        raise ValueError("local_radius_mm length mismatch")
    if fallback_gradient is not None and fallback_gradient.shape != (n_target, 3):
        raise ValueError("fallback_gradient shape mismatch")
    if config.hybrid_enabled and fallback_gradient is None:
        raise ValueError("hybrid V3 requires fallback_gradient")
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

    gradient = np.full((n_target, 3), np.nan, dtype=np.float64)
    used = np.zeros(n_target, dtype=np.int32)
    selected_scale = np.full(n_target, np.nan, dtype=np.float64)
    cv_score = np.full(n_target, np.nan, dtype=np.float64)
    design_condition = np.full(n_target, np.nan, dtype=np.float64)
    depth_limit = np.full(n_target, np.nan, dtype=np.float64)
    local_reynolds = np.full(n_target, np.nan, dtype=np.float64)
    edge_velocity = np.full(n_target, np.nan, dtype=np.float64)
    radius_used = np.full(n_target, np.nan, dtype=np.float64)
    eta_p50 = np.full(n_target, np.nan, dtype=np.float64)
    eta_max = np.full(n_target, np.nan, dtype=np.float64)
    lateral_p50 = np.full(n_target, np.nan, dtype=np.float64)
    lateral_p95 = np.full(n_target, np.nan, dtype=np.float64)
    ownership_rejected = np.zeros(n_target, dtype=np.int32)
    tube_rejected = np.zeros(n_target, dtype=np.int32)
    fallback_expanded = np.zeros(n_target, dtype=np.int8)
    raw_correction = np.ones(n_target, dtype=np.float64)
    stable_scale_count = np.zeros(n_target, dtype=np.int32)
    trend_correlation = np.full(n_target, np.nan, dtype=np.float64)
    scale_direction_cosine = np.full(n_target, np.nan, dtype=np.float64)

    for row, (point, normal, index) in enumerate(
        zip(target_wall_mm, target_normals, neighbor_index)
    ):
        candidate_delta_mm = interior_mm[index] - point
        eta_mm = candidate_delta_mm @ normal
        squared = np.sum(candidate_delta_mm**2, axis=1)
        lateral_mm = np.sqrt(np.maximum(squared - eta_mm**2, 0.0))
        local_velocity = velocity[index]
        tangent_velocity = local_velocity - np.outer(
            local_velocity @ normal, normal
        )
        speed = np.linalg.norm(tangent_velocity, axis=1)
        radius = (
            float(local_radius_mm[row])
            if local_radius_mm is not None and np.isfinite(local_radius_mm[row])
            else config.fallback_radius_mm
        )
        radius = max(radius, config.depth_min_mm)
        radius_used[row] = radius
        provisional_depth = min(
            config.depth_max_mm,
            config.depth_max_radius_fraction * radius,
        )
        positive = (eta_mm > 1e-4) & (eta_mm <= provisional_depth)
        tube_radius = np.maximum(
            config.tube_min_radius_mm,
            config.tube_wall_spacing_multiplier * float(wall_spacing[row])
            + config.tube_depth_slope * np.maximum(eta_mm, 0.0),
        )
        tube_radius = np.minimum(
            tube_radius,
            max(
                config.tube_min_radius_mm,
                config.tube_max_radius_fraction * radius,
            ),
        )

        owner_ok = np.ones(query_k, dtype=bool)
        if config.ownership_enabled:
            _, anchor = wall_tree.query(interior_mm[index], k=1)
            anchor_separation = np.linalg.norm(
                full_wall_mm[anchor] - point, axis=1
            )
            normal_cosine = full_wall_normals[anchor] @ normal
            max_separation = max(
                config.ownership_max_anchor_separation_mm,
                config.ownership_max_anchor_spacing_multiplier
                * float(wall_spacing[row]),
            )
            owner_ok = (
                (anchor_separation <= max_separation)
                & (normal_cosine >= config.ownership_min_normal_cosine)
            )
            ownership_rejected[row] = int(np.sum(positive & ~owner_ok))

        # U_e must represent the target normal tube, not the full isotropic
        # query ball; otherwise the Reynolds prior re-introduces the lateral
        # mixing that V3 is designed to remove.
        preliminary = positive & owner_ok & (
            lateral_mm <= tube_radius * config.fallback_tube_expand
        )
        if int(preliminary.sum()) >= config.min_samples:
            edge_velocity[row] = float(
                np.quantile(speed[preliminary], config.edge_velocity_quantile)
            )
        elif positive.any():
            edge_velocity[row] = float(
                np.quantile(speed[positive], config.edge_velocity_quantile)
            )
        else:
            edge_velocity[row] = 0.0
        depth_limit[row], local_reynolds[row] = _depth_limit_mm(
            radius, edge_velocity[row], config
        )

        within_depth = (eta_mm > 1e-4) & (eta_mm <= depth_limit[row])
        within_tube = lateral_mm <= tube_radius
        tube_rejected[row] = int(np.sum(within_depth & owner_ok & ~within_tube))

        tangent_1, tangent_2 = _tangent_basis(normal)
        delta_m = candidate_delta_mm * MM_TO_M
        lateral_1_m = delta_m @ tangent_1
        lateral_2_m = delta_m @ tangent_2
        fits: list[dict[str, Any]] = []
        for fraction in config.depth_fractions:
            scale_depth = float(fraction * depth_limit[row])
            if config.sampling_mode == "normal_ray_interpolation":
                ray_fits, ray_rejected = _normal_ray_fits(
                    point,
                    normal,
                    scale_depth,
                    float(wall_spacing[row]),
                    full_wall_mm,
                    full_wall_normals,
                    wall_tree,
                    interior_mm,
                    velocity,
                    interior_tree,
                    config,
                )
                ownership_rejected[row] += int(ray_rejected)
                for fit in ray_fits:
                    if np.isfinite(fit["score"]):
                        fit.update(
                            {
                                "fraction": float(fraction),
                                "depth_mm": scale_depth,
                                "expanded": False,
                            }
                        )
                        fits.append(fit)
                continue
            eligible = (
                (eta_mm > 1e-4)
                & (eta_mm <= scale_depth)
                & owner_ok
                & within_tube
            )
            selected = _stratified_selection(
                eta_mm,
                lateral_mm,
                eligible,
                scale_depth,
                tube_radius,
                depth_bins=config.depth_bins,
                points_per_bin=config.points_per_bin,
            )
            expanded = False
            if len(selected) < config.min_samples:
                expanded_radius = tube_radius * config.fallback_tube_expand
                expanded_eligible = (
                    (eta_mm > 1e-4)
                    & (eta_mm <= scale_depth)
                    & owner_ok
                    & (lateral_mm <= expanded_radius)
                )
                selected = _stratified_selection(
                    eta_mm,
                    lateral_mm,
                    expanded_eligible,
                    scale_depth,
                    expanded_radius,
                    depth_bins=config.depth_bins,
                    points_per_bin=config.points_per_bin,
                )
                expanded = len(selected) >= config.min_samples
                fit_tube_radius = expanded_radius
            else:
                fit_tube_radius = tube_radius
            if len(selected) < config.min_samples:
                continue
            fit = _profile_fit(
                eta_mm[selected] * MM_TO_M,
                lateral_1_m[selected],
                lateral_2_m[selected],
                lateral_mm[selected],
                fit_tube_radius[selected],
                tangent_velocity[selected],
                config,
            )
            if fit is None or not np.isfinite(fit["score"]):
                continue
            fit.update(
                {
                    "fraction": float(fraction),
                    "depth_mm": scale_depth,
                    "expanded": expanded,
                }
            )
            fits.append(fit)

        if not fits:
            continue
        min_score = min(float(fit["score"]) for fit in fits)
        threshold = config.cv_tolerance * min_score
        eligible_fit = [
            fit for fit in fits if float(fit["score"]) <= threshold
        ]
        if config.sampling_mode == "normal_ray_interpolation":
            # Near-wall ray interpolation is an extrapolation from cell
            # centres.  Among CV-near-optimal fits prefer the later start and
            # then the shallower physical bandwidth to avoid artificial slope
            # inflation from stations unsupported by the point cloud.
            chosen = max(
                eligible_fit,
                key=lambda fit: (
                    float(fit.get("start_fraction", 0.0)),
                    -float(fit["fraction"]),
                ),
            )
        else:
            chosen = eligible_fit[0]
        correction, stable_count, trend, direction_cosine = _zero_depth_correction(
            fits, chosen, config
        )
        gradient[row] = np.asarray(chosen["gradient"]) * correction
        used[row] = int(chosen["n_used"])
        selected_scale[row] = float(chosen["fraction"])
        cv_score[row] = float(chosen["score"])
        design_condition[row] = float(chosen["condition"])
        eta_p50[row] = float(chosen["eta_p50_mm"])
        eta_max[row] = float(chosen["eta_max_mm"])
        lateral_p50[row] = float(chosen["lateral_p50_mm"])
        lateral_p95[row] = float(chosen["lateral_p95_mm"])
        fallback_expanded[row] = int(bool(chosen["expanded"]))
        raw_correction[row] = correction
        stable_scale_count[row] = stable_count
        trend_correlation[row] = trend
        scale_direction_cosine[row] = direction_cosine

    ray_gradient = gradient.copy()
    ray_used = np.ones(n_target, dtype=np.int8)
    ray_to_fallback_norm = np.full(n_target, np.nan, dtype=np.float64)
    applied_correction = np.ones(n_target, dtype=np.float64)
    if config.hybrid_enabled:
        ratio = lateral_p50 / np.maximum(eta_p50, 1e-6)
        ray_mask = (
            np.isfinite(ray_gradient).all(axis=1)
            & (ratio >= config.hybrid_lateral_over_eta_min)
        )
        if config.hybrid_max_ownership_rejected >= 0:
            ray_mask &= (
                ownership_rejected <= config.hybrid_max_ownership_rejected
            )
        fallback_valid = np.isfinite(fallback_gradient).all(axis=1)
        ray_to_fallback_norm = np.linalg.norm(ray_gradient, axis=1) / np.maximum(
            np.linalg.norm(fallback_gradient, axis=1), 1e-12
        )
        ray_mask &= (
            ray_to_fallback_norm >= config.hybrid_min_ray_to_fallback_norm
        )
        gradient = np.asarray(fallback_gradient, dtype=np.float64).copy()
        applied_correction = np.clip(
            ray_to_fallback_norm,
            config.hybrid_min_ray_to_fallback_norm,
            config.hybrid_max_ray_to_fallback_norm,
        )
        if config.hybrid_preserve_fallback_direction:
            gradient[ray_mask] = (
                fallback_gradient[ray_mask]
                * applied_correction[ray_mask, None]
            )
        else:
            ray_scale = applied_correction / np.maximum(
                ray_to_fallback_norm, 1e-12
            )
            gradient[ray_mask] = (
                ray_gradient[ray_mask] * ray_scale[ray_mask, None]
            )
        gradient[~fallback_valid & np.isfinite(ray_gradient).all(axis=1)] = ray_gradient[
            ~fallback_valid & np.isfinite(ray_gradient).all(axis=1)
        ]
        ray_used = ray_mask.astype(np.int8)

    diagnostics = {
        "ray_gradient": ray_gradient,
        "ray_used": ray_used,
        "ray_to_fallback_norm": ray_to_fallback_norm,
        "applied_correction": applied_correction,
        "selected_scale_fraction": selected_scale,
        "selected_samples": used.copy(),
        "cv_score": cv_score,
        "design_condition": design_condition,
        "depth_limit_mm": depth_limit,
        "local_reynolds": local_reynolds,
        "edge_velocity_m_s": edge_velocity,
        "local_radius_mm": radius_used,
        "eta_p50_mm": eta_p50,
        "eta_max_mm": eta_max,
        "lateral_p50_mm": lateral_p50,
        "lateral_p95_mm": lateral_p95,
        "ownership_rejected": ownership_rejected,
        "tube_rejected": tube_rejected,
        "fallback_tube_expanded": fallback_expanded,
        "correction": raw_correction,
        "stable_scale_count": stable_scale_count,
        "trend_correlation": trend_correlation,
        "scale_direction_cosine": scale_direction_cosine,
    }
    return gradient, used, diagnostics
