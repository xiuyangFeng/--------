from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, TransformerConv

from pipeline.config import GLOBAL_COND_DIM, MODEL_INPUT_DIM, NODE_FEATURE_DIM, TARGET_DIM


def expand_global_cond(data) -> torch.Tensor:
    if not hasattr(data, "global_cond") or data.global_cond is None:
        # MLP 和 GNN 统一走“节点特征 + 广播后的图级条件”输入格式。
        return torch.zeros(
            data.x.size(0),
            GLOBAL_COND_DIM,
            dtype=data.x.dtype,
            device=data.x.device,
        )

    if hasattr(data, "batch") and data.batch is not None:
        return data.global_cond[data.batch]
    return data.global_cond.expand(data.x.size(0), -1)


def build_model(model_name: str, hidden_dim: int, num_layers: int, dropout: float, heads: int):
    registry: Dict[str, nn.Module] = {
        "mlp": FieldMLP(
            in_dim=MODEL_INPUT_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            out_dim=TARGET_DIM,
        ),
        "graphsage": FieldGraphSAGE(
            in_dim=MODEL_INPUT_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            out_dim=TARGET_DIM,
        ),
        "transformer": FieldTransformer(
            in_dim=MODEL_INPUT_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            out_dim=TARGET_DIM,
            heads=heads,
        ),
    }
    if model_name not in registry:
        raise ValueError(f"未知模型: {model_name}")
    return registry[model_name]


class FieldMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, out_dim: int):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, data):
        # 点级基线不使用图结构，专门用来评估“图建模”本身的收益。
        x = torch.cat([data.x, expand_global_cond(data)], dim=-1)
        return self.net(x)


class FieldGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, out_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim) for _ in range(max(1, num_layers))]
        )
        self.dropout = dropout
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x = torch.cat([data.x, expand_global_cond(data)], dim=-1)
        x = self.in_proj(x)
        for conv in self.layers:
            residual = x
            # 保留浅残差，降低小样本场景下深层图网络的训练不稳定。
            x = conv(x, data.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual
        return self.out_proj(x)


class FieldTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        out_dim: int,
        heads: int,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                TransformerConv(
                    hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                )
                for _ in range(max(1, num_layers))
            ]
        )
        self.post_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in self.layers])
        self.dropout = dropout
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x = torch.cat([data.x, expand_global_cond(data)], dim=-1)
        x = self.in_proj(x)
        for conv, linear in zip(self.layers, self.post_layers):
            residual = x
            # 这里故意不把几何特征拆成多个分支，先保持主干简单，便于后续做 feature ablation。
            x = conv(x, data.edge_index)
            x = F.elu(x)
            x = linear(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual
        return self.out_proj(x)
