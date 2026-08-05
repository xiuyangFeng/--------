"""流式物理单位回归指标（R² / MAE / RMSE）。

学习要点
--------
1. **流式累计**：评估时分 chunk 解码全场，不能把整例点云一次装进内存；
   本类只保留充分统计量（count、真值一/二阶矩、误差绝对和/平方和）。
2. **R² 定义**：``1 - SS_res / SS_tot``，其中 ``SS_tot`` 用
   ``Σy² - (Σy)²/n``，与病例内均值中心化一致；分母用 ``1e-12`` 下界防除零。
3. **有限性检查**：真值或预测含 NaN/Inf 直接报错，避免脏指标混进汇总表。
"""

from __future__ import annotations

import numpy as np


class RegressionAccumulator:
    """单标量场的流式回归指标累加器。

    用法：多次 ``update(truth, prediction)`` 后调用 ``result()`` 得到
    ``{count, r2, mae, rmse}``。近壁/核心分区各自维护一个实例。
    """

    def __init__(self):
        self.count = 0
        self.true_sum = 0.0       # Σ y
        self.true_sq_sum = 0.0    # Σ y²
        self.prediction_sum = 0.0       # Σ p
        self.prediction_sq_sum = 0.0    # Σ p²
        self.error_sq_sum = 0.0   # Σ (p-y)²
        self.error_abs_sum = 0.0  # Σ |p-y|

    def update(self, truth: np.ndarray, prediction: np.ndarray) -> None:
        """追加一批点；``truth`` 与 ``prediction`` 展平后必须同形状且有限。"""
        y = np.asarray(truth, dtype=np.float64).reshape(-1)
        p = np.asarray(prediction, dtype=np.float64).reshape(-1)
        if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError("metric inputs must be finite and shape-matched")
        error = p - y
        self.count += int(len(y))
        self.true_sum += float(y.sum(dtype=np.float64))
        self.true_sq_sum += float(np.square(y).sum(dtype=np.float64))
        self.prediction_sum += float(p.sum(dtype=np.float64))
        self.prediction_sq_sum += float(np.square(p).sum(dtype=np.float64))
        self.error_sq_sum += float(np.square(error).sum(dtype=np.float64))
        self.error_abs_sum += float(np.abs(error).sum(dtype=np.float64))

    def result(self) -> dict[str, float | int]:
        """返回累计后的 count / R² / MAE / RMSE；无样本时指标为 NaN。"""
        if self.count <= 0:
            return {
                "count": 0,
                "r2": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
                "truth_mean": float("nan"),
                "truth_variance": float("nan"),
                "prediction_mean": float("nan"),
                "prediction_variance": float("nan"),
                "prediction_to_truth_variance_ratio": float("nan"),
            }
        # SS_tot = Σ(y - ȳ)² = Σy² - (Σy)²/n
        total = self.true_sq_sum - self.true_sum**2 / self.count
        prediction_total = (
            self.prediction_sq_sum - self.prediction_sum**2 / self.count
        )
        truth_variance = max(total / self.count, 0.0)
        prediction_variance = max(prediction_total / self.count, 0.0)
        return {
            "count": self.count,
            "r2": 1.0 - self.error_sq_sum / max(total, 1e-12),
            "mae": self.error_abs_sum / self.count,
            "rmse": float(np.sqrt(self.error_sq_sum / self.count)),
            "truth_mean": self.true_sum / self.count,
            "truth_variance": truth_variance,
            "prediction_mean": self.prediction_sum / self.count,
            "prediction_variance": prediction_variance,
            "prediction_to_truth_variance_ratio": (
                prediction_variance / max(truth_variance, 1e-30)
            ),
        }
