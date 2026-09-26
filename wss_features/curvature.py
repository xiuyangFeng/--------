"""Principal curvatures of a wall point cloud from a local Monge quadric.

Verbatim copy of the curvature part of ``wss_v5.views.wall_geom_v2``.  Sign
convention: positive where the surface is convex along the outward normal, so a
straight tube of radius R gives ``k1 = 1/R``, ``k2 = 0``.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

FINE_K = 32          # ~1.3 mm neighbourhood at the 0.41 mm median wall spacing
COARSE_K = 128       # ~2.6 mm neighbourhood
CURV_CLIP_PER_MM = 2.0   # |k| beyond 2 /mm (R < 0.5 mm) is a fit artefact, not anatomy
CHUNK = 8192
BLOCK = 1024          # v0.12 work unit of the threaded fit: (1024, k, 6) blocks stay in cache; per-point results do not depend on it
FINE_MAP = {"curv_k1": "k1", "curv_k2": "k2", "curv_mean": "mean", "curv_gauss": "gauss",
            "curvedness": "curvedness", "shape_index": "shape_index"}


def unit_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, 1e-12, None)


_unit_rows = unit_rows


def _tangent_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Any orthonormal pair spanning the tangent plane (principal curvatures are basis independent)."""
    helper = np.where(
        (np.abs(normals[:, 0]) < 0.9)[:, None],
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    )
    t1 = unit_rows(np.cross(normals, helper))
    t2 = np.cross(normals, t1)
    return t1, t2


def _threads(n_tasks: int) -> int:
    """Worker threads for chunked kernels: ``WSS_FEATURES_THREADS`` or min(8, CPUs); never more than the tasks."""
    try:
        wanted = int(os.environ.get("WSS_FEATURES_THREADS", "0"))
    except ValueError:
        wanted = 0
    return max(1, min(wanted or min(8, os.cpu_count() or 1), n_tasks))


def _monge_chunk(xyz, normals, t1, t2, neighbours, start, stop, k1, k2, residual, radius) -> None:
    nb = neighbours[start:stop]                       # (m, k) excluding self
    delta = xyz[nb] - xyz[start:stop, None, :]        # (m, k, 3)
    u = np.einsum("mkd,md->mk", delta, t1[start:stop])
    v = np.einsum("mkd,md->mk", delta, t2[start:stop])
    w = np.einsum("mkd,md->mk", delta, normals[start:stop])
    # filled in place: the same values as np.stack([...], axis=-1) without the interleaving copy
    design = np.empty(u.shape + (6,))
    design[..., 0] = 0.5 * u * u
    design[..., 1] = u * v
    design[..., 2] = 0.5 * v * v
    design[..., 3] = u
    design[..., 4] = v
    design[..., 5] = np.ones_like(u)
    gram = np.einsum("mki,mkj->mij", design, design)
    rhs = np.einsum("mki,mk->mi", design, w)
    scale = np.trace(gram, axis1=1, axis2=2)[:, None, None] / 6.0
    gram = gram + 1e-9 * np.clip(scale, 1e-12, None) * np.eye(6)
    coef = np.linalg.solve(gram, rhs)                 # (m, 6)
    fit = np.einsum("mki,mi->mk", design, coef)
    residual[start:stop] = np.sqrt(np.mean((fit - w) ** 2, axis=1))
    radius[start:stop] = np.max(np.sqrt(u * u + v * v), axis=1)
    a, b, c, du, dv = coef[:, 0], coef[:, 1], coef[:, 2], coef[:, 3], coef[:, 4]
    first_e, first_f, first_g = 1.0 + du * du, du * dv, 1.0 + dv * dv
    denom = np.sqrt(1.0 + du * du + dv * dv)
    second_l, second_m, second_n = a / denom, b / denom, c / denom
    det_first = np.clip(first_e * first_g - first_f * first_f, 1e-12, None)
    mean_h = (first_e * second_n - 2.0 * first_f * second_m + first_g * second_l) / (2.0 * det_first)
    gauss_k = (second_l * second_n - second_m * second_m) / det_first
    root = np.sqrt(np.clip(mean_h * mean_h - gauss_k, 0.0, None))
    # eigenvalues of the Weingarten map, then flipped so convex-outward is positive
    k1[start:stop] = -(mean_h - root)
    k2[start:stop] = -(mean_h + root)


def principal_curvatures(xyz: np.ndarray, normals: np.ndarray, neighbours: np.ndarray) -> dict[str, np.ndarray]:
    """Least-squares Monge patch ``w = a u^2/2 + b uv + c v^2/2 + d u + e v + f`` per point.

    Chunks are independent (each point's fit reads only its own rows), so they run on a thread pool;
    every output element is computed by the same operations in the same order as the serial loop.
    """
    n_points = xyz.shape[0]
    t1, t2 = _tangent_basis(normals)
    k1 = np.empty(n_points)
    k2 = np.empty(n_points)
    residual = np.empty(n_points)
    radius = np.empty(n_points)
    starts = list(range(0, n_points, BLOCK))
    work = lambda start: _monge_chunk(xyz, normals, t1, t2, neighbours, start, min(start + BLOCK, n_points), k1, k2, residual, radius)
    workers = _threads(len(starts))
    if workers == 1:
        for start in starts:
            work(start)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, starts))
    clipped = float(np.mean((np.abs(k1) > CURV_CLIP_PER_MM) | (np.abs(k2) > CURV_CLIP_PER_MM)))
    k1 = np.clip(k1, -CURV_CLIP_PER_MM, CURV_CLIP_PER_MM)
    k2 = np.clip(k2, -CURV_CLIP_PER_MM, CURV_CLIP_PER_MM)
    k_max = np.maximum(k1, k2)
    k_min = np.minimum(k1, k2)
    curvedness = np.sqrt(0.5 * (k_max * k_max + k_min * k_min))
    spread = k_max - k_min
    shape_index = np.where(spread > 1e-9, (2.0 / np.pi) * np.arctan((k_max + k_min) / np.clip(spread, 1e-9, None)), 0.0)
    return {
        "k1": k_max, "k2": k_min,
        "mean": 0.5 * (k_max + k_min), "gauss": k_max * k_min,
        "curvedness": curvedness, "shape_index": shape_index,
        "_fit_residual_mm": residual, "_neighbourhood_mm": radius, "_clipped_fraction": clipped,
    }


__all__ = ["BLOCK", "CHUNK", "COARSE_K", "CURV_CLIP_PER_MM", "FINE_K", "FINE_MAP", "principal_curvatures", "unit_rows"]
