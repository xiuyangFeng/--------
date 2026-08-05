"""病例级几何编码器：把 15 维描述子压成 latent 向量。

学习要点
--------
PINN 场网络对每个查询点预测 (WSS, u, v, w, p)，但不同动脉几何不同。
``GeometryEncoder`` 把「整例血管」的统计描述（见 ``data.sidecar._geometry_descriptor``）
编码成固定维 latent，再与每个点的傅里叶坐标拼接，相当于条件场网络。

输入 15 维通常包含：壁面点云的 mean/std/min/max（各 3）+ log(N)、半径均值/标准差等。
"""

from __future__ import annotations

import torch
from torch import nn


class GeometryEncoder(nn.Module):
    """两层 SiLU MLP：descriptor → geometry_latent。"""

    def __init__(self, input_dim: int = 15, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        # 允许传入一维向量；统一成 [1, D] 或 [B, D]
        if descriptor.ndim == 1:
            descriptor = descriptor.unsqueeze(0)
        return self.network(descriptor)
