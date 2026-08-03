"""Streaming physical-unit regression metrics."""

from __future__ import annotations

import numpy as np


class RegressionAccumulator:
    def __init__(self):
        self.count = 0
        self.true_sum = 0.0
        self.true_sq_sum = 0.0
        self.error_sq_sum = 0.0
        self.error_abs_sum = 0.0

    def update(self, truth: np.ndarray, prediction: np.ndarray) -> None:
        y = np.asarray(truth, dtype=np.float64).reshape(-1)
        p = np.asarray(prediction, dtype=np.float64).reshape(-1)
        if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError("metric inputs must be finite and shape-matched")
        error = p - y
        self.count += int(len(y))
        self.true_sum += float(y.sum(dtype=np.float64))
        self.true_sq_sum += float(np.square(y).sum(dtype=np.float64))
        self.error_sq_sum += float(np.square(error).sum(dtype=np.float64))
        self.error_abs_sum += float(np.abs(error).sum(dtype=np.float64))

    def result(self) -> dict[str, float | int]:
        if self.count <= 0:
            return {"count": 0, "r2": float("nan"), "mae": float("nan"), "rmse": float("nan")}
        total = self.true_sq_sum - self.true_sum**2 / self.count
        return {
            "count": self.count,
            "r2": 1.0 - self.error_sq_sum / max(total, 1e-12),
            "mae": self.error_abs_sum / self.count,
            "rmse": float(np.sqrt(self.error_sq_sum / self.count)),
        }
