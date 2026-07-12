#!/usr/bin/env python3
"""C1 PointNeXt: normalize only local relative coordinates by layer radius."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import fps
from torch_geometric.utils import scatter

from .pointnext import FeaturePropagation, _group, mlp


class RadiusNormSetAbstraction(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ratio: float, r: float, nsample: int):
        super().__init__()
        self.ratio, self.r, self.nsample = ratio, r, nsample
        self.local = mlp([in_ch + 3, out_ch, out_ch])

    def forward(self, pos, x, batch):
        idx = fps(pos, batch, ratio=self.ratio, random_start=self.training)
        pos_q, batch_q = pos[idx], batch[idx]
        row, col = _group(pos, batch, pos_q, batch_q, self.r, self.nsample)
        rel = (pos[col] - pos_q[row]) / self.r
        grouped = torch.cat([rel, x[col]], dim=-1)
        h = self.local(grouped)
        out = scatter(h, row, dim=0, dim_size=pos_q.size(0), reduce="max")
        return pos_q, out, batch_q


class RadiusNormInvResMLP(nn.Module):
    def __init__(self, ch: int, r: float, nsample: int, expansion: int = 4):
        super().__init__()
        self.r, self.nsample = r, nsample
        self.local = mlp([ch + 3, ch, ch], last_act=False)
        self.pw = mlp([ch, ch * expansion, ch], last_act=False)
        self.act = nn.GELU()

    def forward(self, pos, x, batch):
        row, col = _group(pos, batch, pos, batch, self.r, self.nsample)
        rel = (pos[col] - pos[row]) / self.r
        grouped = torch.cat([rel, x[col]], dim=-1)
        h = self.local(grouped)
        h = scatter(h, row, dim=0, dim_size=pos.size(0), reduce="max")
        h = self.act(x + h)
        h = self.act(h + self.pw(h))
        return h


class PointNeXtRadiusNormSeg(nn.Module):
    def __init__(self, in_dim: int, width: int = 32,
                 sa_ratios=(0.25, 0.25, 0.25, 0.25),
                 sa_radius=(0.05, 0.10, 0.20, 0.40),
                 sa_nsample=(16, 16, 16, 16), sa_blocks=(1, 1, 1, 1),
                 invres_radius_scale: float = 1.0, fp_knn: int = 3,
                 head_hidden: int = 64, out_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_sa = len(sa_ratios)
        enc = [width * (2 ** i) for i in range(self.num_sa + 1)]
        self.enc_channels = enc
        self.stem = mlp([in_dim, width, width])
        self.sa = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(self.num_sa):
            self.sa.append(RadiusNormSetAbstraction(
                enc[i], enc[i + 1], sa_ratios[i], sa_radius[i], sa_nsample[i]))
            self.blocks.append(nn.ModuleList([
                RadiusNormInvResMLP(enc[i + 1],
                                    sa_radius[i] * invres_radius_scale,
                                    sa_nsample[i])
                for _ in range(sa_blocks[i])
            ]))
        self.fp = nn.ModuleList()
        for i in reversed(range(self.num_sa)):
            self.fp.append(FeaturePropagation(enc[i + 1], enc[i], enc[i], k=fp_knn))
        self.head = nn.Sequential(
            nn.Linear(enc[0], head_hidden), nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden, out_dim),
        )

    def forward(self, pos, x, batch):
        x = self.stem(x)
        levels: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = [(pos, x, batch)]
        for i in range(self.num_sa):
            pos, x, batch = self.sa[i](pos, x, batch)
            for blk in self.blocks[i]:
                x = blk(pos, x, batch)
            levels.append((pos, x, batch))
        pos_c, x_c, batch_c = levels[-1]
        for j, i in enumerate(reversed(range(self.num_sa))):
            pos_f, x_f, batch_f = levels[i]
            x_c = self.fp[j](pos_f, x_f, batch_f, pos_c, x_c, batch_c)
            pos_c, batch_c = pos_f, batch_f
        out = self.head(x_c)
        return out.squeeze(-1) if out.size(-1) == 1 else out


def build_radius_norm_model(model_cfg, in_dim: int) -> PointNeXtRadiusNormSeg:
    m = model_cfg
    return PointNeXtRadiusNormSeg(
        in_dim=in_dim, width=m.width, sa_ratios=tuple(m.sa_ratios),
        sa_radius=tuple(m.sa_radius), sa_nsample=tuple(m.sa_nsample),
        sa_blocks=tuple(m.sa_blocks), invres_radius_scale=m.invres_radius_scale,
        fp_knn=m.fp_knn, head_hidden=m.head_hidden, out_dim=m.out_dim,
        dropout=m.dropout,
    )
