"""血液流变：Carreau–Yasuda 表观粘度 μ(γ̇)。

学习要点
--------
主物理口径是非牛顿血：粘度随剪切率变化，不是常数 μ。

公式（与 Fluent UDF 一致）::

    μ(γ̇) = μ∞ + (μ0 − μ∞) · [1 + (λ γ̇)^a]^((n−1)/a)

- γ̇ → 0 时 μ → μ0（高粘）
- γ̇ → ∞ 时 μ → μ∞（低粘，接近牛顿平台）

``UDF_PARAMETERS`` 来自 ``data_new/.../udf-inlet4.c`` 审计，勿随意改数字；
常数粘度消融必须另开命名实验，不能 silently 改这里默认值。
"""

from __future__ import annotations

import numpy as np
import torch


# 与 CFD UDF 对齐的默认参数（Pa·s、s、无量纲）
UDF_PARAMETERS = {
    "mu_inf_pa_s": 0.0035,
    "mu_zero_pa_s": 0.16,
    "lambda_s": 8.2,
    "a": 0.64,
    "n": 0.2128,
}


def carreau_yasuda_numpy(
    shear_rate_s_inv: np.ndarray | float,
    *,
    mu_inf_pa_s: float = UDF_PARAMETERS["mu_inf_pa_s"],
    mu_zero_pa_s: float = UDF_PARAMETERS["mu_zero_pa_s"],
    lambda_s: float = UDF_PARAMETERS["lambda_s"],
    a: float = UDF_PARAMETERS["a"],
    n: float = UDF_PARAMETERS["n"],
) -> np.ndarray:
    """NumPy 版 μ(γ̇)，用于 CFD Oracle / 离线壁面剪切后处理。"""
    gamma = np.asarray(shear_rate_s_inv, dtype=np.float64)
    if np.any(gamma < 0):
        raise ValueError("shear rate must be non-negative")
    return mu_inf_pa_s + (mu_zero_pa_s - mu_inf_pa_s) * np.power(
        1.0 + np.power(lambda_s * gamma, a), (n - 1.0) / a
    )


def carreau_yasuda_torch(
    shear_rate_s_inv: torch.Tensor,
    *,
    mu_inf_pa_s: float = UDF_PARAMETERS["mu_inf_pa_s"],
    mu_zero_pa_s: float = UDF_PARAMETERS["mu_zero_pa_s"],
    lambda_s: float = UDF_PARAMETERS["lambda_s"],
    a: float = UDF_PARAMETERS["a"],
    n: float = UDF_PARAMETERS["n"],
) -> torch.Tensor:
    """Torch 版 μ(γ̇)，供可微物理 loss（F2+）使用。"""
    if bool(torch.any(shear_rate_s_inv < 0)):
        raise ValueError("shear rate must be non-negative")
    return mu_inf_pa_s + (mu_zero_pa_s - mu_inf_pa_s) * (
        1.0 + (lambda_s * shear_rate_s_inv).pow(a)
    ).pow((n - 1.0) / a)
