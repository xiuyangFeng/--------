"""前沿方向 §3 · 两阶段可微剖面 WSS 头（DifferentiableWSSHead）。

网络对每个壁面点输出近壁切向速度剖面 u_t(eta) = a1*eta + a2*eta^2 + ... 的
低维系数与切向方向修正，物理层解析求导 WSS = mu * a1 * t_hat。求导发生在
光滑基系数上而非逐点差分（历史 VelGrad/vel_diff 的噪声放大器），对应
Gate-0 oracle `v3p_profile_wss_oracle_20260704.json`（cross calibrated R² 0.632 Go）。

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
