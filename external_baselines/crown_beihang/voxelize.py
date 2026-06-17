from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

import numpy as np


def coord_span_mm(points: np.ndarray) -> List[float]:
    span = np.ptp(points, axis=0)
    return [float(v) for v in span]


def subsample_points(
    points: np.ndarray,
    values: np.ndarray,
    wall_mask: np.ndarray,
    max_points: int | None,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = points.shape[0]
    if max_points is None or n <= max_points:
        return points, values, wall_mask
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return points[idx], values[idx], wall_mask[idx]


def filter_points(
    points: np.ndarray,
    values: np.ndarray,
    wall_mask: np.ndarray,
    point_filter: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if point_filter == "all":
        return points, values, wall_mask
    if point_filter == "interior":
        mask = ~wall_mask.astype(bool)
        return points[mask], values[mask], wall_mask[mask]
    raise ValueError(f"未知 point_filter={point_filter}; 可选 all/interior")


def voxelize_and_average(
    points: np.ndarray,
    values: np.ndarray,
    wall_mask: np.ndarray,
    voxel_size_mm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Distance-weighted average within voxels (CROWN source logic).
    points: (N, 3) mm; values: (N, C); wall_mask: (N,)
    Returns centers (M,3), averaged values (M,C), wall_mask (M,)
    """
    voxel_indices = np.floor(points / voxel_size_mm).astype(np.int64)
    unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    centers = (unique_voxels + 0.5) * voxel_size_mm
    m = unique_voxels.shape[0]
    c = values.shape[1]
    voxel_values = np.zeros((m, c), dtype=np.float64)
    voxel_wall = np.zeros(m, dtype=np.float32)

    for i in range(m):
        mask = inverse_indices == i
        voxel_points = points[mask]
        voxel_vals = values[mask]
        voxel_w = wall_mask[mask]
        center = centers[i]
        dists = np.linalg.norm(voxel_points - center, axis=1) + 1e-8
        weights = 1.0 / dists
        weights = weights / weights.sum()
        voxel_values[i] = (voxel_vals * weights[:, None]).sum(axis=0)
        voxel_wall[i] = float(np.max(voxel_w))

    return centers, voxel_values, voxel_wall


def parse_time_index(stem: str) -> int | None:
    match = re.search(r"-(\d+)$", stem)
    if match:
        return int(match.group(1))
    return None
