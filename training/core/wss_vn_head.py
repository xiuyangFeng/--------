"""G1-a0 · Vector-Neuron 风格等变 WSS 头（VNHeadPlain）。

用标量 backbone 特征调制中心线切向，升维为向量神经元通道，再经 VN-MLP 输出
SO(3) 等变的 ``[wss_x, wss_y, wss_z]``；标量 ``wss`` 走不变分支。不引入法向 /
局部基底 / 切空间投影（对应路径 G · G1-a 第一档）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VNLinear(nn.Module):
    """等变线性：在 channel 维共享权重，保持每个神经元的 3D 向量性质。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, C_in, 3) -> (N, C_out, 3)
        return self.linear(x.transpose(1, 2)).transpose(1, 2)


class VNBatchNorm(nn.Module):
    """按向量模长做 BatchNorm（等变）。"""

    def __init__(self, num_channels: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = x.norm(dim=-1)
        mag_n = self.bn(mag)
        scale = mag_n / mag.clamp_min(1e-8)
        return x * scale.unsqueeze(-1)


class VNLeakyReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=0.1)


class _VNDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0:
            return x
        mask = (torch.rand(x.size(0), x.size(1), 1, device=x.device) > self.p).float()
        return x * mask / (1.0 - self.p)


class EquivariantWSSHeadPlain(nn.Module):
    """G1-a0：仅替换 wss_head；输入标量 hidden + 节点 Tangent_XYZ。"""

    def __init__(
        self,
        hidden_dim: int,
        wss_dim: int = 4,
        vn_channels: int = 32,
        vn_layers: int = 2,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        if wss_dim != 4:
            raise ValueError(f"VNHeadPlain 当前仅支持 wss_dim=4，收到 {wss_dim}")
        self.vn_channels = vn_channels
        self.lift = nn.Sequential(
            nn.Linear(hidden_dim, vn_channels),
            nn.LayerNorm(vn_channels),
            nn.GELU(),
        )
        blocks: list[nn.Module] = []
        ch = vn_channels
        for _ in range(max(1, vn_layers)):
            blocks.extend([VNLinear(ch, ch), VNBatchNorm(ch), VNLeakyReLU()])
            if head_dropout > 0:
                blocks.append(_VNDropout(head_dropout))
        blocks.append(VNLinear(ch, 1))
        self.vn_mlp = nn.Sequential(*blocks)
        self.mag_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            *( [nn.Dropout(p=head_dropout)] if head_dropout > 0 else [] ),
            nn.Linear(hidden_dim, 1),
        )

    def _lift(self, h: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        t = F.normalize(tangent, dim=-1, eps=1e-8)
        w = self.lift(h)
        return w.unsqueeze(-1) * t.unsqueeze(1)

    def forward(self, h: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        vn = self._lift(h, tangent)
        wss_vec = self.vn_mlp(vn).squeeze(1)
        wss_mag = self.mag_head(h).squeeze(-1)
        return torch.cat([wss_mag.unsqueeze(-1), wss_vec], dim=-1)
