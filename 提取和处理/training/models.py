from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, TransformerConv

from pipeline.config import GLOBAL_COND_DIM, MODEL_INPUT_DIM, NODE_FEATURE_DIM, TARGET_DIM


def expand_global_cond(data) -> torch.Tensor:
    # 所有模型都统一走“节点特征 + 广播后的图级条件”输入范式。
    # 这样后续无论是 MLP 还是图模型，特征定义始终一致，方便做公平对照。
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
    # 这里用一个极小的 registry，而不是复杂工厂类，方便后续直接插入新 backbone。
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
        "meshgraphnet": FieldMeshGraphNet(
            in_dim=MODEL_INPUT_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            out_dim=TARGET_DIM,
        ),
        "pointnetpp": FieldPointNetPP(
            in_dim=MODEL_INPUT_DIM,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            out_dim=TARGET_DIM,
        ),
    }
    if model_name not in registry:
        raise ValueError(f"未知模型: {model_name}")
    return registry[model_name]


class FieldMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, out_dim: int):
        super().__init__()
        # MLP 是任务 A 的“无图结构下限”，目的是回答图建模本身有没有价值。
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
        # 这里固定用同宽隐藏层，目的是让 GraphSAGE 和 Transformer 的参数量级别接近。
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
        # TransformerConv 这里主要承担“更强消息传递”基线，不在第一版里引入额外分支和注意力花活。
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
            x = conv(x, data.edge_index)
            x = F.elu(x)
            x = linear(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# Literature baselines
# ---------------------------------------------------------------------------


class _EdgeModel(nn.Module):
    """MeshGraphNet 的边更新 MLP。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, src, dst, edge_attr):
        inp = torch.cat([src, dst, edge_attr], dim=-1)
        return self.net(inp)


class _NodeModel(nn.Module):
    """MeshGraphNet 的节点更新 MLP。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x, agg_edge):
        inp = torch.cat([x, agg_edge], dim=-1)
        return self.net(inp)


class FieldMeshGraphNet(nn.Module):
    """Pfaff et al. (ICLR 2021) 的 Encode-Process-Decode 基线。

    边特征采用相对位移加欧氏距离，保持与物理仿真代理模型常见写法一致。
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, out_dim: int):
        super().__init__()
        edge_feat_dim = 4  # dx, dy, dz, ||d||
        self.node_encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.edge_encoder = nn.Sequential(nn.Linear(edge_feat_dim, hidden_dim), nn.ReLU())

        self.edge_models = nn.ModuleList()
        self.node_models = nn.ModuleList()
        for _ in range(max(1, num_layers)):
            self.edge_models.append(
                _EdgeModel(hidden_dim * 3, hidden_dim, hidden_dim)
            )
            self.node_models.append(
                _NodeModel(hidden_dim * 2, hidden_dim, hidden_dim)
            )

        self.dropout = dropout
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _build_edge_attr(self, pos, edge_index):
        row, col = edge_index
        diff = pos[col] - pos[row]
        dist = diff.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.cat([diff, dist], dim=-1)

    def forward(self, data):
        x_in = torch.cat([data.x, expand_global_cond(data)], dim=-1)
        pos = data.x[:, :3]
        edge_index = data.edge_index
        edge_attr = self._build_edge_attr(pos, edge_index)

        x = self.node_encoder(x_in)
        e = self.edge_encoder(edge_attr)

        row, col = edge_index
        for edge_model, node_model in zip(self.edge_models, self.node_models):
            x_res = x
            e = edge_model(x[row], x[col], e)
            agg = torch.zeros_like(x)
            agg.scatter_add_(0, row.unsqueeze(1).expand_as(e), e)
            x = node_model(x, agg)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res

        return self.decoder(x)


class FieldPointNetPP(nn.Module):
    """简化版 PointNet++ 风格基线。

    这里不复现完整的 set abstraction / feature propagation，而是用图邻域
    定义局部点集，再做 max-pooling 聚合，用来回答“简单邻域聚合是否足以
    接近显式消息传递”的问题。
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, out_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        self.local_mlps = nn.ModuleList()
        for _ in range(max(1, num_layers)):
            self.local_mlps.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )

        self.dropout = dropout
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        x_in = torch.cat([data.x, expand_global_cond(data)], dim=-1)
        x = self.in_proj(x_in)
        edge_index = data.edge_index
        row, col = edge_index

        for mlp in self.local_mlps:
            x_res = x
            neighbour_feats = x[col]
            max_pool = torch.full_like(x, float("-inf"))
            max_pool.scatter_reduce_(0, row.unsqueeze(1).expand_as(neighbour_feats), neighbour_feats, reduce="amax")
            max_pool = max_pool.clamp_min(0)
            x = mlp(torch.cat([x, max_pool], dim=-1))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res

        return self.out_proj(x)
