"""峰值体域路线的可复现性辅助工具。

学习要点
--------
1. **原子写 ``.npy``**：构建 sidecar 时避免半截数组；与 JSON 原子写同思路。
2. **``tensor_state_sha256``**：对模型 ``state_dict`` 做稳定哈希（按参数名
   排序，写入 dtype/shape/原始字节）。用于：
   - 记录初始化证据 ``initialization.pt``；
   - resume 时核对「是否仍是同一次随机初始化」，防止跨 run 热启动冒充续训。
3. **``StreamingMoments``**：在超大 mmap 数组上在线累计 mean/std，供
   train138-only 严格体域场统计，不把全库装进内存。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from wss_pinn.utils import guard_write_path


def atomic_save_npy(path: str | Path, array: np.ndarray) -> Path:
    """原子保存单个可 mmap 的 ``.npy`` 数组。

    先写同目录临时文件，再 ``os.replace``；失败时清理临时文件。
    """
    target = guard_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    """对张量字典做稳定 SHA256（参数名 + dtype + shape + CPU 连续字节）。

    排序参数名保证同一权重不论 dict 插入顺序如何哈希一致。
    """
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


class StreamingMoments:
    """大数组上的 float64 流式均值/标准差累加器。

    每次 ``update`` 接受 ``[N, width]``（或 ``[N]`` 自动扩成 ``[N,1]``）；
    ``result()`` 返回 count / mean / std 列表。``std`` 有 ``minimum_std``
    下界，避免归一化除零。
    """

    def __init__(self, width: int):
        self.width = int(width)
        self.count = 0
        self.total = np.zeros(self.width, dtype=np.float64)     # Σ x
        self.total_sq = np.zeros(self.width, dtype=np.float64)  # Σ x²

    def update(self, value: np.ndarray) -> None:
        """追加一批样本；拒绝错误宽度或含 NaN/Inf 的输入。"""
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or array.shape[1] != self.width:
            raise ValueError(
                f"streaming moments expected [N,{self.width}], got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("streaming moments received NaN/Inf")
        self.count += int(array.shape[0])
        self.total += array.sum(axis=0, dtype=np.float64)
        self.total_sq += np.square(array).sum(axis=0, dtype=np.float64)

    def result(self, minimum_std: float = 1e-8) -> dict[str, list[float] | int]:
        """用总体方差公式收尾：``Var = E[x²] - (E[x])²``，再取 sqrt。"""
        if self.count <= 0:
            raise ValueError("cannot finalize empty streaming moments")
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - np.square(mean), 0.0)
        std = np.maximum(np.sqrt(variance), float(minimum_std))
        return {
            "count": self.count,
            "mean": mean.tolist(),
            "std": std.tolist(),
        }


def iter_slices(length: int, chunk_size: int = 1_000_000) -> Iterable[slice]:
    """把 ``[0, length)`` 切成固定大小的 ``slice``，用于分块扫 mmap 数组。"""
    for start in range(0, int(length), int(chunk_size)):
        yield slice(start, min(start + int(chunk_size), int(length)))
