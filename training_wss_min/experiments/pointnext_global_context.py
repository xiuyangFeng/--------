#!/usr/bin/env python3
"""B-REP/C2: frozen B1 PointNeXt with one case-global context at the head."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from training_wss_min.pointnext import FeaturePropagation, InvResMLP, SetAbstraction, mlp


class PointNeXtGlobalContextSeg(nn.Module):
    """Append per-case mean+max of decoded point features before the scalar head."""

    def __init__(self, in_dim: int, width: int = 32,
                 sa_ratios=(0.25, 0.25, 0.25, 0.25),
                 sa_radius=(0.05, 0.10, 0.20, 0.40),
                 sa_nsample=(16, 16, 16, 16), sa_blocks=(1, 1, 1, 1),
                 invres_radius_scale: float = 1.0, fp_knn: int = 3,
                 head_hidden: int = 64, out_dim: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.num_sa = len(sa_ratios)
        enc = [width * (2 ** i) for i in range(self.num_sa + 1)]
        self.enc_channels = enc
        self.stem = mlp([in_dim, width, width])
        self.sa = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(self.num_sa):
            self.sa.append(SetAbstraction(
                enc[i], enc[i + 1], sa_ratios[i], sa_radius[i], sa_nsample[i]))
            self.blocks.append(nn.ModuleList([
                InvResMLP(enc[i + 1], sa_radius[i] * invres_radius_scale,
                          sa_nsample[i]) for _ in range(sa_blocks[i])
            ]))
        self.fp = nn.ModuleList([
            FeaturePropagation(enc[i + 1], enc[i], enc[i], k=fp_knn)
            for i in reversed(range(self.num_sa))
        ])
        self.head = nn.Sequential(
            nn.Linear(3 * enc[0], head_hidden), nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden, out_dim),
        )

    def decode_features(self, pos, x, batch):
        x = self.stem(x)
        levels: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = [(pos, x, batch)]
        for i in range(self.num_sa):
            pos, x, batch = self.sa[i](pos, x, batch)
            for block in self.blocks[i]:
                x = block(pos, x, batch)
            levels.append((pos, x, batch))
        pos_c, x_c, batch_c = levels[-1]
        for j, i in enumerate(reversed(range(self.num_sa))):
            pos_f, x_f, batch_f = levels[i]
            x_c = self.fp[j](pos_f, x_f, batch_f, pos_c, x_c, batch_c)
            pos_c, batch_c = pos_f, batch_f
        return x_c

    @staticmethod
    def append_global_context(decoded, batch):
        """Aggregate strictly within each case, then broadcast to its own points."""
        n_cases = int(batch.max().item()) + 1 if batch.numel() else 0
        case_mean = scatter(decoded, batch, dim=0, dim_size=n_cases, reduce="mean")
        case_max = scatter(decoded, batch, dim=0, dim_size=n_cases, reduce="max")
        return torch.cat([decoded, case_mean[batch], case_max[batch]], dim=-1)

    def forward(self, pos, x, batch):
        decoded = self.decode_features(pos, x, batch)
        out = self.head(self.append_global_context(decoded, batch))
        return out.squeeze(-1) if out.size(-1) == 1 else out


def build_global_context_model(model_cfg, in_dim: int) -> PointNeXtGlobalContextSeg:
    m = model_cfg
    return PointNeXtGlobalContextSeg(
        in_dim=in_dim, width=m.width, sa_ratios=tuple(m.sa_ratios),
        sa_radius=tuple(m.sa_radius), sa_nsample=tuple(m.sa_nsample),
        sa_blocks=tuple(m.sa_blocks), invres_radius_scale=m.invres_radius_scale,
        fp_knn=m.fp_knn, head_hidden=m.head_hidden, out_dim=m.out_dim,
        dropout=m.dropout)
