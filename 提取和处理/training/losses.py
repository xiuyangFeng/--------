from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    mse_per_dim = F.mse_loss(pred, target, reduction="none").mean(dim=0)
    return (mse_per_dim * weights).sum()

