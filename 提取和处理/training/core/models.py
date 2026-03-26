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
    # 如果图里根本没有 global_cond，就补零向量占位。
    if not hasattr(data, "global_cond") or data.global_cond is None:
        # MLP 和 GNN 统一走“节点特征 + 广播后的图级条件”输入格式。
        return torch.zeros(
            data.x.size(0),
            GLOBAL_COND_DIM,
            dtype=data.x.dtype,
            device=data.x.device,
        )

    # 批量图时，batch 向量把每个节点映射到所属图，直接按 batch 广播图级条件。
    if hasattr(data, "batch") and data.batch is not None:
        return data.global_cond[data.batch]
    # 单图场景下，把 [1, d] 的图级条件扩展到 [num_nodes, d]。
    return data.global_cond.expand(data.x.size(0), -1)


class FieldMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, out_dim: int):
        super().__init__()
        # MLP 是任务 A 的“无图结构下限”，目的是回答图建模本身有没有价值。
        # 第一层先把输入映射到隐藏维度。
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        # 后续隐藏层保持同宽，结构尽量简单。
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        # 最后一层映射到监督目标维度。
        layers.append(nn.Linear(hidden_dim, out_dim))
        # 把线性层栈打包成一个顺序网络。
        self.net = nn.Sequential(*layers)

    def forward(self, data):
        # 点级基线不使用图结构，专门用来评估“图建模”本身的收益。
        # 先拼接节点特征与广播后的全局条件。
        x = torch.cat([data.x, expand_global_cond(data)], dim=-1)
        # 再送入 MLP 得到逐节点预测。
        return self.net(x)


class FieldGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, out_dim: int):
        super().__init__()
        # 先把输入统一投影到隐藏维度。
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        # 这里固定用同宽隐藏层，目的是让 GraphSAGE 和 Transformer 的参数量级别接近。
        self.layers = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim) for _ in range(max(1, num_layers))]
        )
        # dropout 概率单独存起来，forward 时反复使用。
        self.dropout = dropout
        # 输出层把隐藏表示映射回目标维度。
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        # 统一输入格式：节点特征 + 图级条件。
        x = torch.cat([data.x, expand_global_cond(data)], dim=-1)
        # 先做线性投影。
        x = self.in_proj(x)
        for conv in self.layers:
            # 保存残差分支。
            residual = x
            # 保留浅残差，降低小样本场景下深层图网络的训练不稳定。
            # GraphSAGE 消息传递。
            x = conv(x, data.edge_index)
            # 非线性激活。
            x = F.relu(x)
            # 训练时做 dropout。
            x = F.dropout(x, p=self.dropout, training=self.training)
            # 残差相加。
            x = x + residual
        # 输出逐节点回归结果。
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
        # 输入先投影到统一隐藏维度。
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
        # 每个 TransformerConv 后面接一个线性层，补偿多头输出后的特征变换。
        self.post_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in self.layers])
        # Pre-Norm：每层消息传递前先 LayerNorm，利于深层稳定（A-Opt-02 / P0-2）。
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.layers])
        # 保存 dropout 概率。
        self.dropout = dropout
        # 最终输出头。
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        # 拼接节点与图级特征。
        x = torch.cat([data.x, expand_global_cond(data)], dim=-1)
        # 输入投影。
        x = self.in_proj(x)
        for conv, linear, norm in zip(self.layers, self.post_layers, self.norms):
            # 残差分支。
            residual = x
            x = norm(x)
            # 图 Transformer 消息传递。
            x = conv(x, data.edge_index)
            # 用 ELU 做激活。
            x = F.elu(x)
            # 后接线性层整理特征。
            x = linear(x)
            # dropout 只在训练时启用。
            x = F.dropout(x, p=self.dropout, training=self.training)
            # 残差回加。
            x = x + residual
        # 输出逐节点预测。
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# Literature baselines
# ---------------------------------------------------------------------------


class _EdgeModel(nn.Module):
    """MeshGraphNet 的边更新 MLP。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        # 边更新 MLP：线性 -> ReLU -> 线性 -> LayerNorm。
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, src, dst, edge_attr):
        # 把源节点、目标节点、边特征拼接后做边更新。
        inp = torch.cat([src, dst, edge_attr], dim=-1)
        return self.net(inp)


class _NodeModel(nn.Module):
    """MeshGraphNet 的节点更新 MLP。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        # 节点更新 MLP：节点自身状态 + 聚合后的边消息作为输入。
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x, agg_edge):
        # 拼接节点当前表示与聚合边消息。
        inp = torch.cat([x, agg_edge], dim=-1)
        return self.net(inp)


class FieldMeshGraphNet(nn.Module):
    """Pfaff et al. (ICLR 2021) 的 Encode-Process-Decode 基线。

    边特征采用相对位移加欧氏距离，保持与物理仿真代理模型常见写法一致。
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, out_dim: int):
        super().__init__()
        # 边特征由三维相对位移和距离共 4 维组成。
        edge_feat_dim = 4  # dx, dy, dz, ||d||
        # 节点编码器。
        self.node_encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        # 边编码器。
        self.edge_encoder = nn.Sequential(nn.Linear(edge_feat_dim, hidden_dim), nn.ReLU())

        # Process 阶段的边更新模块列表。
        self.edge_models = nn.ModuleList()
        # Process 阶段的节点更新模块列表。
        self.node_models = nn.ModuleList()
        for _ in range(max(1, num_layers)):
            self.edge_models.append(
                _EdgeModel(hidden_dim * 3, hidden_dim, hidden_dim)
            )
            self.node_models.append(
                _NodeModel(hidden_dim * 2, hidden_dim, hidden_dim)
            )

        # dropout 概率。
        self.dropout = dropout
        # Decode 阶段输出头。
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _build_edge_attr(self, pos, edge_index):
        # edge_index 的 row / col 分别表示起点和终点。
        row, col = edge_index
        # 构造相对位移向量。
        diff = pos[col] - pos[row]
        # 构造欧氏距离，并截断最小值避免除零等数值问题。
        dist = diff.norm(dim=1, keepdim=True).clamp_min(1e-8)
        # 拼成最终边特征。
        return torch.cat([diff, dist], dim=-1)

    def forward(self, data):
        # 输入仍然统一为节点特征 + 图级条件。
        x_in = torch.cat([data.x, expand_global_cond(data)], dim=-1)
        # 约定 data.x 前三维就是归一化坐标。
        pos = data.x[:, :3]
        # 图拓扑。
        edge_index = data.edge_index
        # 用坐标与边关系构造边特征。
        edge_attr = self._build_edge_attr(pos, edge_index)

        # 编码初始节点状态。
        x = self.node_encoder(x_in)
        # 编码初始边状态。
        e = self.edge_encoder(edge_attr)

        # row / col 会在每一层消息传递里重复使用。
        row, col = edge_index
        for edge_model, node_model in zip(self.edge_models, self.node_models):
            # 节点残差分支。
            x_res = x
            # 先更新边状态。
            e = edge_model(x[row], x[col], e)
            # 为每个节点准备一块全零缓冲区聚合入边消息。
            agg = torch.zeros_like(x)
            # 按 row 把边消息 scatter_add 回对应节点。
            agg.scatter_add_(0, row.unsqueeze(1).expand_as(e), e)
            # 再做节点更新。
            x = node_model(x, agg)
            # dropout。
            x = F.dropout(x, p=self.dropout, training=self.training)
            # 残差相加。
            x = x + x_res

        # Decode 成最终逐节点预测。
        return self.decoder(x)


class FieldPointNetPP(nn.Module):
    """简化版 PointNet++ 风格基线。

    这里不复现完整的 set abstraction / feature propagation，而是用图邻域
    定义局部点集，再做 max-pooling 聚合，用来回答“简单邻域聚合是否足以
    接近显式消息传递”的问题。
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, out_dim: int):
        super().__init__()
        # 输入投影层。
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        # 每一层局部邻域聚合后接一个小 MLP。
        self.local_mlps = nn.ModuleList()
        for _ in range(max(1, num_layers)):
            self.local_mlps.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )

        # dropout 概率。
        self.dropout = dropout
        # 输出头。
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        # 拼接统一输入。
        x_in = torch.cat([data.x, expand_global_cond(data)], dim=-1)
        # 先投影到隐藏空间。
        x = self.in_proj(x_in)
        # 图边索引。
        edge_index = data.edge_index
        row, col = edge_index

        for mlp in self.local_mlps:
            # 残差分支。
            x_res = x
            # 收集每条边终点节点的特征。
            neighbour_feats = x[col]
            # 初始化 max-pooling 缓冲区。
            max_pool = torch.full_like(x, float("-inf"))
            # 对每个节点的邻居特征做逐维最大池化。
            max_pool.scatter_reduce_(0, row.unsqueeze(1).expand_as(neighbour_feats), neighbour_feats, reduce="amax")
            # 把没有邻居的位置从 -inf 截成 0，避免数值污染。
            max_pool = max_pool.clamp_min(0)
            # 拼接“节点自身 + 邻域池化”后过 MLP。
            x = mlp(torch.cat([x, max_pool], dim=-1))
            # dropout。
            x = F.dropout(x, p=self.dropout, training=self.training)
            # 残差。
            x = x + x_res

        # 输出逐节点回归结果。
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# Model registry & factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, type] = {
    "mlp": FieldMLP,
    "graphsage": FieldGraphSAGE,
    "transformer": FieldTransformer,
    "meshgraphnet": FieldMeshGraphNet,
    "pointnetpp": FieldPointNetPP,
}


def build_model(model_name: str, hidden_dim: int, num_layers: int, dropout: float, heads: int):
    # 先检查模型名是否合法。
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {model_name}, 可选: {list(MODEL_REGISTRY)}")
    # 取出对应模型类。
    cls = MODEL_REGISTRY[model_name]
    # 先构造所有模型都共用的参数。
    kwargs = dict(
        in_dim=MODEL_INPUT_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        out_dim=TARGET_DIM,
    )
    # 只有 transformer 额外需要 heads 参数。
    if model_name == "transformer":
        kwargs["heads"] = heads
    # 实例化具体模型并返回。
    return cls(**kwargs)
