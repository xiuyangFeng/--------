from __future__ import annotations

import re
import time
from typing import Callable, List, Sequence, Tuple

import numpy as np

ProgressFn = Callable[[str], None]


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
    progress: ProgressFn | None = None,
    progress_every_voxels: int = 5000,
    progress_every_sec: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Distance-weighted average within voxels (CROWN source logic).
    points: (N, 3) mm; values: (N, C); wall_mask: (N,)
    Returns centers (M,3), averaged values (M,C), wall_mask (M,)
    """
    n = points.shape[0]
    if progress:
        progress(f"voxelize start n_points={n} voxel_size_mm={voxel_size_mm}")

    t0 = time.time()
    voxel_indices = np.floor(points / voxel_size_mm).astype(np.int64)
    unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    m = unique_voxels.shape[0]
    if progress:
        progress(
            f"voxelize unique done n_voxel={m} elapsed_sec={time.time() - t0:.1f}"
        )

    centers = (unique_voxels + 0.5) * voxel_size_mm
    c = values.shape[1]
    voxel_values = np.zeros((m, c), dtype=np.float64)
    voxel_wall = np.zeros(m, dtype=np.float32)

    order = np.argsort(inverse_indices, kind="mergesort")
    inv_sorted = inverse_indices[order]
    pts_sorted = points[order]
    vals_sorted = values[order]
    wall_sorted = wall_mask[order]

    boundaries = np.concatenate(
        [np.array([0], dtype=np.int64), np.flatnonzero(np.diff(inv_sorted)) + 1, np.array([n], dtype=np.int64)]
    )

    last_log = t0
    for seg_idx in range(m):
        start, end = int(boundaries[seg_idx]), int(boundaries[seg_idx + 1])
        voxel_points = pts_sorted[start:end]
        voxel_vals = vals_sorted[start:end]
        voxel_w = wall_sorted[start:end]
        center = centers[seg_idx]
        dists = np.linalg.norm(voxel_points - center, axis=1) + 1e-8
        weights = 1.0 / dists
        weights = weights / weights.sum()
        voxel_values[seg_idx] = (voxel_vals * weights[:, None]).sum(axis=0)
        voxel_wall[seg_idx] = float(np.max(voxel_w))

        if progress and (
            (seg_idx + 1) % progress_every_voxels == 0
            or (time.time() - last_log) >= progress_every_sec
            or seg_idx + 1 == m
        ):
            elapsed = time.time() - t0
            pct = 100.0 * (seg_idx + 1) / max(m, 1)
            progress(
                f"voxelize aggregate {seg_idx + 1}/{m} ({pct:.1f}%) elapsed_sec={elapsed:.1f}"
            )
            last_log = time.time()

    if progress:
        progress(f"voxelize done n_voxel={m} total_sec={time.time() - t0:.1f}")

    return centers, voxel_values, voxel_wall


def parse_time_index(stem: str) -> int | None:
    match = re.search(r"-(\d+)$", stem)
    if match:
        return int(match.group(1))
    return None
