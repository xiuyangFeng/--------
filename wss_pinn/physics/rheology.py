from __future__ import annotations

import numpy as np
import torch


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
    if bool(torch.any(shear_rate_s_inv < 0)):
        raise ValueError("shear rate must be non-negative")
    return mu_inf_pa_s + (mu_zero_pa_s - mu_inf_pa_s) * (
        1.0 + (lambda_s * shear_rate_s_inv).pow(a)
    ).pow((n - 1.0) / a)

