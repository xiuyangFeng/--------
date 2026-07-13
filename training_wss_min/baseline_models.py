#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小 WSS 基线模型：MLP、PointNet 和 PointNet++。

三者都遵守训练器的 ``forward(pos, x, batch) -> (N,)`` 接口：

* ``MLP``：每个点独立预测，没有点云上下文；
* ``PointNet``：逐点编码 + 每病例全局 max-pool；
* ``PointNet++``：FPS、ball-query 局部集合抽象和 FP 插值解码，**不含**
  PointNeXt 的倒置瓶颈或残差块。

这是与 ``pointnext.py`` 独立的最简模型组，避免把后续路线中的模块带入
2×3 baseline。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
from torch_geometric.nn import fps, knn_interpolate, radius
from torch_geometric.utils import scatter


def _mlp(channels: List[int], *, last_act: bool = True) -> nn.Sequential:
    """Linear + BatchNorm + ReLU 的基础 PointNet MLP。"""
    layers: List[nn.Module] = []
    for i, (cin, cout) in enumerate(zip(channels[:-1], channels[1:])):
        layers.append(nn.Linear(cin, cout))
        is_last = i == len(channels) - 2
        if not is_last or last_act:
            layers.extend((nn.BatchNorm1d(cout), nn.ReLU(inplace=True)))
    return nn.Sequential(*layers)


class MLPRegressor(nn.Module):
    """无邻域、无全局池化的逐点 MLP 对照。"""

    def __init__(self, in_dim: int, hidden: Tuple[int, ...], out_dim: int = 1):
        super().__init__()
        self.network = _mlp([in_dim, *hidden, out_dim], last_act=False)

    def forward(self, pos: torch.Tensor, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        del pos, batch
        out = self.network(x)
        return out.squeeze(-1) if out.size(-1) == 1 else out


class PointNetRegressor(nn.Module):
    """经典 PointNet 分割式逐点回归：局部编码 + per-case global max。"""

    def __init__(self, in_dim: int, width: int, head_hidden: int, out_dim: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        local_dim = width * 4
        self.local = _mlp([in_dim, width, width * 2, local_dim])
        self.decoder = _mlp([local_dim * 2, width * 4, head_hidden])
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Linear(head_hidden, out_dim)

    def forward(self, pos: torch.Tensor, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        del pos
        local = self.local(x)
        global_feat = scatter(local, batch, dim=0, reduce="max")
        fused = torch.cat([local, global_feat[batch]], dim=-1)
        out = self.out(self.dropout(self.decoder(fused)))
        return out.squeeze(-1) if out.size(-1) == 1 else out


def _group(pos_src, batch_src, pos_query, batch_query, radius_value: float, nsample: int):
    assign = radius(pos_src, pos_query, radius_value, batch_src, batch_query,
                    max_num_neighbors=nsample)
    return assign[0], assign[1]  # query indices, source indices


class PointNetSetAbstraction(nn.Module):
    """PointNet++ 单尺度 set-abstraction；无残差、无倒置瓶颈。"""

    def __init__(self, in_ch: int, out_ch: int, ratio: float, radius_value: float,
                 nsample: int):
        super().__init__()
        self.ratio = ratio
        self.radius_value = radius_value
        self.nsample = nsample
        self.local = _mlp([in_ch + 3, out_ch, out_ch])

    def forward(self, pos, x, batch):
        idx = fps(pos, batch, ratio=self.ratio, random_start=self.training)
        pos_q, batch_q = pos[idx], batch[idx]
        row, col = _group(pos, batch, pos_q, batch_q, self.radius_value, self.nsample)
        grouped = torch.cat([pos[col] - pos_q[row], x[col]], dim=-1)
        h = self.local(grouped)
        x_q = scatter(h, row, dim=0, dim_size=pos_q.size(0), reduce="max")
        return pos_q, x_q, batch_q


class FeaturePropagation(nn.Module):
    """PointNet++ 的 3-NN 插值上采样和 skip 拼接。"""

    def __init__(self, coarse_ch: int, skip_ch: int, out_ch: int, k: int):
        super().__init__()
        self.k = k
        self.mlp = _mlp([coarse_ch + skip_ch, out_ch, out_ch])

    def forward(self, pos_fine, x_skip, batch_fine, pos_coarse, x_coarse, batch_coarse):
        up = knn_interpolate(x_coarse, pos_coarse, pos_fine, batch_coarse, batch_fine,
                             k=self.k)
        return self.mlp(torch.cat([up, x_skip], dim=-1))


class PointNetPlusPlusRegressor(nn.Module):
    """经典 single-scale PointNet++ segmentation backbone，用于逐点标量回归。"""

    def __init__(self, in_dim: int, width: int, sa_ratios, sa_radius, sa_nsample,
                 fp_knn: int, head_hidden: int, out_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        if not (len(sa_ratios) == len(sa_radius) == len(sa_nsample)):
            raise ValueError("PointNet++ SA ratios/radii/nsample must have equal length")
        self.stem = _mlp([in_dim, width, width])
        self.n_stages = len(sa_ratios)
        channels = [width * (2 ** i) for i in range(self.n_stages + 1)]
        self.sa = nn.ModuleList(
            PointNetSetAbstraction(channels[i], channels[i + 1], sa_ratios[i],
                                   sa_radius[i], sa_nsample[i])
            for i in range(self.n_stages)
        )
        self.fp = nn.ModuleList(
            FeaturePropagation(channels[i + 1], channels[i], channels[i], fp_knn)
            for i in reversed(range(self.n_stages))
        )
        self.head = nn.Sequential(
            nn.Linear(channels[0], head_hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden, out_dim),
        )

    def forward(self, pos, x, batch):
        x = self.stem(x)
        levels = [(pos, x, batch)]
        for sa in self.sa:
            pos, x, batch = sa(pos, x, batch)
            levels.append((pos, x, batch))

        pos_c, x_c, batch_c = levels[-1]
        for fp, level_idx in zip(self.fp, reversed(range(self.n_stages))):
            pos_f, x_skip, batch_f = levels[level_idx]
            x_c = fp(pos_f, x_skip, batch_f, pos_c, x_c, batch_c)
            pos_c, batch_c = pos_f, batch_f
        out = self.head(x_c)
        return out.squeeze(-1) if out.size(-1) == 1 else out


def build_baseline_model(model_cfg, in_dim: int) -> nn.Module:
    """按 ``ModelConfig.name`` 构建 2×3 矩阵使用的模型。"""
    name = model_cfg.name
    if name == "mlp":
        return MLPRegressor(in_dim, tuple(model_cfg.mlp_hidden), model_cfg.out_dim)
    if name == "pointnet":
        return PointNetRegressor(in_dim, model_cfg.width, model_cfg.head_hidden,
                                 model_cfg.out_dim, model_cfg.dropout)
    if name == "pointnetpp":
        return PointNetPlusPlusRegressor(
            in_dim, model_cfg.width, tuple(model_cfg.sa_ratios),
            tuple(model_cfg.sa_radius), tuple(model_cfg.sa_nsample), model_cfg.fp_knn,
            model_cfg.head_hidden, model_cfg.out_dim, model_cfg.dropout,
        )
    raise ValueError(f"unknown minimal baseline model {name!r}")
