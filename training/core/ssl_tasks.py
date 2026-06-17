"""G3 · 几何 SSL 代理任务（T1 坐标掩码重建）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import FieldPointNeXt, build_model


@dataclass
class CoordMaskBatch:
    data: object
    target_coords: torch.Tensor
    mask: torch.Tensor


def apply_coord_mask(
    data,
    *,
    mask_ratio: float = 0.25,
    generator: torch.Generator | None = None,
) -> CoordMaskBatch:
    """随机 mask 点坐标（x/y/z），返回 masked 图与重建目标。"""
    coords = data.x[:, :3].clone()
    n = coords.size(0)
    n_mask = max(1, int(round(n * mask_ratio)))
    if coords.is_cuda:
        perm = torch.randperm(n, device=coords.device)
    else:
        perm = torch.randperm(n, generator=generator)
    mask = torch.zeros(n, dtype=torch.bool, device=coords.device)
    mask[perm[:n_mask]] = True

    masked = data.clone()
    masked.x = data.x.clone()
    masked.x[mask, :3] = 0.0
    return CoordMaskBatch(data=masked, target_coords=coords, mask=mask)


def coord_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not mask.any():
        return pred.sum() * 0.0
    return F.mse_loss(pred[mask], target[mask])


class PointNeXtCoordSSL(nn.Module):
    """FieldPointNeXt encoder + 坐标重建头（不含 WSS/压力监督）。"""

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        head_layout: str = "mlp2",
        pool_k_tiers=None,
    ):
        super().__init__()
        self.encoder = build_model(
            model_name="pointnext",
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            heads=4,
            wss_dim=0,
            head_layout=head_layout,
            pool_k_tiers=pool_k_tiers,
        )
        assert isinstance(self.encoder, FieldPointNeXt)
        self.coord_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, data, *, mask_ratio: float = 0.25, generator=None) -> Tuple[torch.Tensor, CoordMaskBatch]:
        batch = apply_coord_mask(data, mask_ratio=mask_ratio, generator=generator)
        h = self.encoder._encode(batch.data)
        pred = self.coord_head(h)
        loss = coord_reconstruction_loss(pred, batch.target_coords, batch.mask)
        return loss, batch

    def encode(self, data) -> torch.Tensor:
        return self.encoder._encode(data)
