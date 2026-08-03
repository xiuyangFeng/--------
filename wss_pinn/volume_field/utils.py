"""Small reproducibility helpers for the volume-field route."""

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
    """Atomically save one mmap-friendly ``.npy`` array."""
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
    """Stable hash of tensor names, dtypes, shapes and raw CPU bytes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


class StreamingMoments:
    """Float64 streaming mean/std for large mmap arrays."""

    def __init__(self, width: int):
        self.width = int(width)
        self.count = 0
        self.total = np.zeros(self.width, dtype=np.float64)
        self.total_sq = np.zeros(self.width, dtype=np.float64)

    def update(self, value: np.ndarray) -> None:
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
    for start in range(0, int(length), int(chunk_size)):
        yield slice(start, min(start + int(chunk_size), int(length)))
