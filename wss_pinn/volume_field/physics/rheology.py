"""Differentiable Carreau-Yasuda apparent viscosity."""

from __future__ import annotations

import torch


def carreau_yasuda(
    shear_rate_s_inv: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    gamma = shear_rate_s_inv.clamp_min(0.0)
    mu_inf = float(config["mu_inf_pa_s"])
    mu_zero = float(config["mu_zero_pa_s"])
    time_constant = float(config["lambda_s"])
    exponent_a = float(config["a"])
    exponent_n = float(config["n"])
    # a<1 makes d(gamma**a)/dgamma singular exactly at zero.  The physical
    # limit is mu_zero, so use that exact branch and a harmless positive value
    # only for evaluating the non-zero branch.
    safe_gamma = gamma.clamp_min(1e-12)
    value = mu_inf + (mu_zero - mu_inf) * (
        1.0 + (time_constant * safe_gamma).pow(exponent_a)
    ).pow((exponent_n - 1.0) / exponent_a)
    return torch.where(gamma == 0, torch.full_like(value, mu_zero), value)
