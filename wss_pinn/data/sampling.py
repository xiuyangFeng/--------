"""三槽采样：wall / near_wall / core 互不重叠。

学习要点
--------
PINN batch **必须**显式区分三类点：

- **wall**：壁面点，才有直接 WSS 标签；
- **near_wall**：靠近壁的体点（``int_type==1``）；
- **core**：远离壁的体点（``int_type==0``）。

编码约定来自冻结的 ``pipeline_wss_min``：``INTERIOR/core=0``, ``NEAR_WALL=1``。
无 WSS 标签的体点不得进入 ``direct_wss`` loss。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


SLOTS = ("wall", "near_wall", "core")


def sample_indices(
    candidates: Iterable[int] | np.ndarray, count: int, seed: int
) -> np.ndarray:
    """无放回采样最多 ``count`` 个候选下标，排序后返回（稳定便于对照）。"""
    values = np.asarray(list(candidates) if not isinstance(candidates, np.ndarray) else candidates)
    values = values.astype(np.int64, copy=False).reshape(-1)
    if values.size == 0:
        raise ValueError("sampling slot is empty")
    if len(np.unique(values)) != len(values):
        raise ValueError("sampling candidates contain duplicates")
    rng = np.random.default_rng(int(seed))
    size = min(int(count), len(values))
    return np.sort(rng.choice(values, size=size, replace=False)).astype(np.int64)


def build_three_slot_sampling(
    n_wall: int,
    int_type: np.ndarray,
    wall_points: int,
    near_wall_points: int,
    core_points: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """按三槽各自配额采样，并断言 near_wall ∩ core = ∅。"""
    int_type = np.asarray(int_type).reshape(-1)
    near_candidates = np.flatnonzero(int_type == 1)
    # 冻结 pipeline_wss_min 编码：INTERIOR/core=0，NEAR_WALL=1
    core_candidates = np.flatnonzero(int_type == 0)
    # 不同槽用不同种子偏移，避免偶然相关
    result = {
        "wall": sample_indices(np.arange(int(n_wall)), wall_points, seed + 11),
        "near_wall": sample_indices(near_candidates, near_wall_points, seed + 23),
        "core": sample_indices(core_candidates, core_points, seed + 37),
    }
    if set(result) != set(SLOTS) or any(len(result[name]) == 0 for name in SLOTS):
        raise ValueError("all physics sampling slots must be non-empty")
    if np.intersect1d(result["near_wall"], result["core"]).size:
        raise ValueError("near-wall and core slots overlap")
    return result
