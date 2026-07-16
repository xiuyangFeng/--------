#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PointNeXt-S（残差版）用于点云逐点回归 WSS。

要点（与 baseline 讨论一致）：
- **ball-query 分组**（半径固定，物理尺度不随点密度变），因此"训练 2000 点、
  推理 9k~200k 点"能迁移——这是选点云网络而非固定-k 图网络的关键。
- **InvResMLP 残差块**：PointNeXt 相对 PointNet++ 的核心改进（倒置瓶颈 + 残差），
  用户点名要的"带残差版本"。
- **FP 插值解码器**上采样回输入分辨率 -> 逐点输出，天然支持全点云推理（A 路）。
- PyG 批：把不同点数的病例 concat 成一张大图，用 batch 索引区分，支持变长点云。

输出 out_dim=1 为标量 WSS；改成 3 即可切矢量（二期），网络主体不变。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import fps, radius, knn_interpolate
from torch_geometric.utils import scatter


def mlp(channels: List[int], act=nn.GELU, bn: bool = True, last_act: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(1, len(channels)):
        layers.append(nn.Linear(channels[i - 1], channels[i]))
        is_last = i == len(channels) - 1
        if not is_last or last_act:
            if bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(act())
    return nn.Sequential(*layers)


def _group(pos_s, batch_s, pos_q, batch_q, r, nsample):
    """ball-query 分组，返回 (row=query idx, col=source idx)。"""
    assign = radius(pos_s, pos_q, r, batch_s, batch_q, max_num_neighbors=nsample)
    row, col = assign[0], assign[1]        # row -> pos_q(中心), col -> pos_s(邻居)
    return row, col


class SetAbstraction(nn.Module):
    """下采样 + ball-query 局部聚合（PointNet++/PointNeXt 的 SA）。"""

    def __init__(self, in_ch: int, out_ch: int, ratio: float, r: float, nsample: int):
        super().__init__()
        self.ratio, self.r, self.nsample = ratio, r, nsample
        # 局部 MLP 输入 = [相对坐标(3), 邻居特征(in_ch)]
        self.local = mlp([in_ch + 3, out_ch, out_ch])

    def forward(self, pos, x, batch):
        idx = fps(pos, batch, ratio=self.ratio, random_start=self.training)
        pos_q, batch_q = pos[idx], batch[idx]
        row, col = _group(pos, batch, pos_q, batch_q, self.r, self.nsample)
        rel = (pos[col] - pos_q[row])
        grouped = torch.cat([rel, x[col]], dim=-1)
        h = self.local(grouped)
        out = scatter(h, row, dim=0, dim_size=pos_q.size(0), reduce="max")
        return pos_q, out, batch_q


class InvResMLP(nn.Module):
    """PointNeXt 残差块：ball-query 深度聚合 + 倒置瓶颈 pointwise MLP + 残差。"""

    def __init__(self, ch: int, r: float, nsample: int, expansion: int = 4):
        super().__init__()
        self.r, self.nsample = r, nsample
        self.local = mlp([ch + 3, ch, ch], last_act=False)          # 局部聚合(同分辨率)
        self.pw = mlp([ch, ch * expansion, ch], last_act=False)     # 倒置瓶颈
        self.act = nn.GELU()

    def forward(self, pos, x, batch):
        row, col = _group(pos, batch, pos, batch, self.r, self.nsample)
        rel = pos[col] - pos[row]
        grouped = torch.cat([rel, x[col]], dim=-1)
        h = self.local(grouped)
        h = scatter(h, row, dim=0, dim_size=pos.size(0), reduce="max")
        h = self.act(x + h)          # 残差 1（局部）
        h = self.act(h + self.pw(h)) # 残差 2（pointwise 倒置瓶颈）
        return h


class FeaturePropagation(nn.Module):
    """FP 解码：kNN 反距离插值上采样 + skip 拼接 + MLP。"""

    def __init__(self, coarse_ch: int, skip_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        self.k = k
        self.mlp = mlp([coarse_ch + skip_ch, out_ch, out_ch])

    def forward(self, pos_fine, x_fine_skip, batch_fine, pos_coarse, x_coarse, batch_coarse):
        up = knn_interpolate(x_coarse, pos_coarse, pos_fine,
                             batch_coarse, batch_fine, k=self.k)
        h = up if x_fine_skip is None else torch.cat([up, x_fine_skip], dim=-1)
        return self.mlp(h)


class PointNeXtSeg(nn.Module):
    """PointNeXt-S 分割/逐点回归主干。"""

    def __init__(self, in_dim: int, width: int = 32,
                 sa_ratios=(0.25, 0.25, 0.25, 0.25),
                 sa_radius=(0.05, 0.10, 0.20, 0.40),
                 sa_nsample=(16, 16, 16, 16),
                 sa_blocks=(1, 1, 1, 1),
                 invres_radius_scale: float = 1.0,
                 fp_knn: int = 3, head_hidden: int = 64,
                 out_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_sa = len(sa_ratios)
        enc = [width * (2 ** i) for i in range(self.num_sa + 1)]   # [32,64,128,256,512]
        self.enc_channels = enc
        self.stem = mlp([in_dim, width, width])

        self.sa = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(self.num_sa):
            self.sa.append(SetAbstraction(enc[i], enc[i + 1],
                                          sa_ratios[i], sa_radius[i], sa_nsample[i]))
            blk = nn.ModuleList([
                InvResMLP(enc[i + 1], sa_radius[i] * invres_radius_scale, sa_nsample[i])
                for _ in range(sa_blocks[i])
            ])
            self.blocks.append(blk)

        # 解码：从最粗层往细层，通道镜像 enc
        self.fp = nn.ModuleList()
        for i in reversed(range(self.num_sa)):
            coarse_ch = enc[i + 1]
            skip_ch = enc[i]
            self.fp.append(FeaturePropagation(coarse_ch, skip_ch, enc[i], k=fp_knn))

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

        # 解码：level top -> level 0
        pos_c, x_c, batch_c = levels[-1]
        for j, i in enumerate(reversed(range(self.num_sa))):
            pos_f, x_f, batch_f = levels[i]
            x_c = self.fp[j](pos_f, x_f, batch_f, pos_c, x_c, batch_c)
            pos_c, batch_c = pos_f, batch_f
        out = self.head(x_c)
        return out.squeeze(-1) if out.size(-1) == 1 else out


if __name__ == "__main__":
    torch.manual_seed(0)
    # 合成两团点云做前反向冒烟
    N1, N2 = 5000, 3000
    pos = torch.rand(N1 + N2, 3) * 2 - 1
    x = torch.rand(N1 + N2, 3)
    batch = torch.cat([torch.zeros(N1, dtype=torch.long), torch.ones(N2, dtype=torch.long)])
    net = PointNeXtSeg(in_dim=3, out_dim=1)
    n_par = sum(p.numel() for p in net.parameters())
    net.train()
    out = net(pos, x, batch)
    loss = (out - torch.randn_like(out)).pow(2).mean()
    loss.backward()
    print(f"params={n_par/1e6:.2f}M  out={tuple(out.shape)}  loss={loss.item():.4f}  bwd OK")
