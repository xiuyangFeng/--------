"""Surface smoothing and resampling for the deployment point-cloud contract.

Verbatim copies of ``taubin_smooth`` / ``gaussian_weights``
(``training_wss_min.tools.deployment_geometry_sensitivity``), ``voxel_rows``
(``training_wss_min.tools.deployment_resample_reeval``) and ``sample_surface``
(``training_wss_min.tools.deployment_stl_simulation``).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .cloud import median_spacing


def gaussian_weights(vertices: np.ndarray, sigma_mm: float) -> sp.csr_matrix:
    """Row-normalised Gaussian weights over vertices within 3 σ, self included (weight 1), so that a vertex with
    no neighbour inside the window is a fixed point of the operator instead of being pulled towards the origin."""
    tree = cKDTree(vertices)
    pairs = tree.query_pairs(r=3.0 * sigma_mm, output_type="ndarray")
    own = np.arange(len(vertices))
    i = np.concatenate([pairs[:, 0], pairs[:, 1], own]); j = np.concatenate([pairs[:, 1], pairs[:, 0], own])
    d2 = ((vertices[i] - vertices[j]) ** 2).sum(axis=1)
    w = np.exp(-0.5 * d2 / (sigma_mm * sigma_mm))
    W = sp.csr_matrix((w, (i, j)), shape=(len(vertices), len(vertices)))
    rs = np.asarray(W.sum(axis=1)).ravel()
    return sp.diags(1.0 / np.maximum(rs, 1e-12)) @ W


def taubin_smooth(vertices: np.ndarray, sigma_mm: float, iterations: int = 20, lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
    W = gaussian_weights(vertices, sigma_mm)
    V = np.array(vertices, dtype=np.float64, copy=True)
    for _ in range(iterations):
        V = V + lam * (W @ V - V)
        V = V + mu * (W @ V - V)
    return V


def voxel_rows(xyz: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """One point per occupied voxel (the one nearest the voxel centroid); voxel size found by bisection."""
    n = len(xyz)
    target = max(int(round(n * frac)), 64)
    if frac >= 1.0:
        return np.arange(n)
    d, _ = cKDTree(xyz).query(xyz, k=2)
    spacing = float(np.median(d[:, 1]))
    lo, hi = spacing * 0.5, spacing * 40.0
    origin = xyz.min(axis=0) + np.random.default_rng(seed).uniform(0.0, spacing, 3)
    best = None
    for _ in range(40):
        h = 0.5 * (lo + hi)
        keys = np.floor((xyz - origin) / h).astype(np.int64)
        keys -= keys.min(axis=0)
        span = keys.max(axis=0) + 1
        # 1-D voxel code (collision-free: strides are the per-axis spans); groups are identical to np.unique(axis=0),
        # and the retained row per group (nearest the centroid, ties by row order) does not depend on group numbering.
        code = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
        _, inv, counts = np.unique(code, return_inverse=True, return_counts=True)
        m = len(counts)
        if best is None or abs(m - target) < abs(best[0] - target):
            best = (m, inv.reshape(-1), counts)
        if abs(m - target) <= max(1, int(0.01 * target)):
            break
        if m > target:
            lo = h
        else:
            hi = h
    m, inv, counts = best
    centroid = np.zeros((m, 3))
    np.add.at(centroid, inv, xyz)
    centroid /= counts[:, None]
    dist = np.linalg.norm(xyz - centroid[inv], axis=1)
    order = np.lexsort((dist, inv))
    first = np.concatenate([[True], inv[order][1:] != inv[order][:-1]])
    return np.sort(order[first])


def sample_surface(vertices: np.ndarray, faces: np.ndarray, spacing: float, seed: int) -> np.ndarray:
    """Area-weighted random points (oversampled) then voxel decimation, calibrated so that the median
    nearest-neighbour distance of the result matches ``spacing`` (two refinement passes)."""
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    total = float(area.sum())
    rng = np.random.default_rng(seed)
    target = int(round(1.155 * total / (spacing * spacing)))   # hexagonal packing at NN distance = spacing
    n_raw = max(target * 6, 20000)
    pick = rng.choice(len(faces), size=n_raw, p=area / total)
    u, v = rng.random(n_raw), rng.random(n_raw)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    pts = tri[pick, 0] + u[:, None] * (tri[pick, 1] - tri[pick, 0]) + v[:, None] * (tri[pick, 2] - tri[pick, 0])
    out = None
    for _ in range(3):
        rows = voxel_rows(pts, min(target / n_raw, 1.0), seed)
        out = pts[rows]
        actual = float(median_spacing(out))
        if abs(actual - spacing) / spacing < 0.05:
            break
        target = max(int(round(target * (actual / spacing) ** 2)), 500)
    return out


__all__ = ["gaussian_weights", "sample_surface", "taubin_smooth", "voxel_rows"]
