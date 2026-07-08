"""前沿方向 §3/§15.1 · 两阶段可微剖面 WSS 头。

- ``DifferentiableWSSHead``（K7，§3）：单标量剖面系数 a1 + 切向修正，
  WSS = mu * a1 * t_hat。5907 Gate-1 Go / Gate-2 横向未过——横向输出通道
  结构上被 0.1 倍 dir 修正掐死（判读修正见前沿方向 §15 序言）。
- ``DualProfileWSSHead``（K8，§15.1）：coef_head 输出 2×n_basis（stream/cross
  各一组），在逐点局部切平面基 (t̂_s, t̂_c) 上合成
  WSS_vec = mu * (a1_s·t̂_s + a1_c·t̂_c)——横向第一次拥有结构上专属的输出
  通道 a1_c。frame 由 K9 sidecar 预计算（只依赖图几何，推理期可得）；
  forward 时把 (a1_s, a1_c) 暂存于 ``last_a1`` 供 L_ray（§15.2）直接监督。

输出 ``[wss_mag, wss_x, wss_y, wss_z]``，与 WSS_TARGET_NAMES(global) 对齐。

归一化说明：监督目标 y_wss 各通道独立 z-score，因此物理层输出后接一个
可学习的逐通道仿射校准（对应 oracle 的 calibrated 口径——raw 尺度不可用，
校准后信息可恢复）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableWSSHead(nn.Module):
    """两阶段可微 WSS：剖面系数 (a1..an) + 切向 t_hat → WSS = mu * a1 * t_hat。"""

    def __init__(
        self,
        hidden_dim: int,
        n_basis: int = 3,
        mu: float = 1.0,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        if n_basis < 1:
            raise ValueError(f"n_basis 须 >= 1，收到 {n_basis}")
        self.mu = float(mu)
        self.n_basis = n_basis
        coef_layers: list[nn.Module] = [
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        if head_dropout > 0:
            coef_layers.append(nn.Dropout(p=head_dropout))
        coef_layers.append(nn.Linear(hidden_dim, n_basis))
        self.coef_head = nn.Sequential(*coef_layers)
        # 切向修正：以中心线切向为基，允许小幅可学偏转（保持在切流形附近）
        self.dir_head = nn.Linear(hidden_dim, 3)
        # 逐通道仿射校准（oracle calibrated 口径；z-score 目标下 raw 尺度不可用）
        self.out_scale = nn.Parameter(torch.ones(4))
        self.out_bias = nn.Parameter(torch.zeros(4))

    def forward(self, h: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        coef = self.coef_head(h)  # (N, n_basis)
        a1 = coef[:, 0]  # d u_t / d eta |_{eta=0}
        wss_mag = self.mu * a1  # 解析 WSS（z-score 空间，可为负）
        t_dir = F.normalize(tangent + 0.1 * self.dir_head(h), dim=-1, eps=1e-8)
        wss_vec = wss_mag.unsqueeze(-1) * t_dir  # (N, 3)
        out = torch.cat([wss_mag.unsqueeze(-1), wss_vec], dim=-1)
        return out * self.out_scale + self.out_bias


class DualProfileWSSHead(nn.Module):
    """K8 双分量局部系剖面头：WSS_vec = mu * (a1_s·t̂_s + a1_c·t̂_c)。

    与 K7 的差异：无 dir 修正头——方向完全由预计算局部 frame 决定，
    自由度全部落在两组剖面系数上；幅值通道 = ‖WSS_vec‖（与矢量自洽）。
    ``last_a1`` (N, 2) 每次 forward 更新（带梯度），供 DualDomainLoss 的
    L_ray 读取；单 GPU 训练口径（DataParallel 下该暂存不可用）。
    """

    def __init__(
        self,
        hidden_dim: int,
        n_basis: int = 3,
        mu: float = 1.0,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        if n_basis < 1:
            raise ValueError(f"n_basis 须 >= 1，收到 {n_basis}")
        self.mu = float(mu)
        self.n_basis = n_basis
        coef_layers: list[nn.Module] = [
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        if head_dropout > 0:
            coef_layers.append(nn.Dropout(p=head_dropout))
        coef_layers.append(nn.Linear(hidden_dim, 2 * n_basis))
        self.coef_head = nn.Sequential(*coef_layers)
        self.out_scale = nn.Parameter(torch.ones(4))
        self.out_bias = nn.Parameter(torch.zeros(4))
        self.last_a1: torch.Tensor | None = None

    def forward(self, h: torch.Tensor, t_s: torch.Tensor, t_c: torch.Tensor) -> torch.Tensor:
        coef = self.coef_head(h)  # (N, 2*n_basis)：[:, :n_basis]=stream, [:, n_basis:]=cross
        a1_s = coef[:, 0]
        a1_c = coef[:, self.n_basis]
        self.last_a1 = torch.stack([a1_s, a1_c], dim=-1)
        wss_vec = self.mu * (a1_s.unsqueeze(-1) * t_s + a1_c.unsqueeze(-1) * t_c)  # (N, 3)
        wss_mag = torch.sqrt(wss_vec.square().sum(dim=-1) + 1e-12)
        out = torch.cat([wss_mag.unsqueeze(-1), wss_vec], dim=-1)
        return out * self.out_scale + self.out_bias
