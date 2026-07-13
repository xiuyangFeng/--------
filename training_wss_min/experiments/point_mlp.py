#!/usr/bin/env python3
"""B-REP/M1 point-wise MLP control.

Every output point depends only on that point's frozen input-feature vector.
``pos`` and ``batch`` are accepted for interface compatibility but are never
used, which makes the absence of neighbourhood aggregation explicit.
"""

from __future__ import annotations

import torch
from torch import nn


class PointWiseMLP(nn.Module):
    def __init__(self, in_dim: int, width: int = 32, out_dim: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        channels = [in_dim, width, width * 2, width * 4, width * 8,
                    width * 16, width * 8, width * 4, width * 2]
        layers = []
        for cin, cout in zip(channels[:-1], channels[1:]):
            layers.extend((nn.Linear(cin, cout), nn.BatchNorm1d(cout), nn.GELU()))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(channels[-1], out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, pos, x, batch):
        del pos, batch
        out = self.network(x)
        return out.squeeze(-1) if out.size(-1) == 1 else out


def build_point_mlp(model_cfg, in_dim: int) -> PointWiseMLP:
    if model_cfg.name != "point_mlp":
        raise ValueError(f"M1 requires model.name='point_mlp', got {model_cfg.name!r}")
    return PointWiseMLP(in_dim, width=model_cfg.width,
                        out_dim=model_cfg.out_dim, dropout=model_cfg.dropout)
