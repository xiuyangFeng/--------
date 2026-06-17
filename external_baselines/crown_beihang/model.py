from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrownPointNet(nn.Module):
    """PointNet-style regressor from CROWN/Beihang source (configurable input dim)."""

    def __init__(self, input_dim: int = 3, output_dim: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mlp_convs1 = nn.ModuleList()
        self.mlp_bns1 = nn.ModuleList()
        in_ch = input_dim
        for out_ch in [256, 512]:
            self.mlp_convs1.append(nn.Conv1d(in_ch, out_ch, 1))
            self.mlp_bns1.append(nn.BatchNorm1d(out_ch))
            in_ch = out_ch

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_ch = 1024
        for out_ch in [512, 256, output_dim]:
            self.mlp_convs.append(nn.Conv1d(last_ch, out_ch, 1))
            if out_ch != output_dim:
                self.mlp_bns.append(nn.BatchNorm1d(out_ch))
            last_ch = out_ch

        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.mlp_convs1):
            x = F.relu(self.mlp_bns1[i](conv(x)))

        x_global = torch.max(x, 2, keepdim=True)[0]
        x_global = x_global.expand(-1, -1, x.shape[-1])
        x = torch.cat([x_global, x], dim=1)

        for i, conv in enumerate(self.mlp_convs):
            if i < len(self.mlp_bns):
                x = F.relu(self.mlp_bns[i](conv(x)))
            else:
                x = conv(x)
        return x
