from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, TransformerConv, knn_graph

from pipeline.config import GLOBAL_COND_DIM, MODEL_INPUT_DIM, NODE_FEATURE_NAMES, TARGET_DIM

from .losses import infer_wss_from_vel_diff, wall_pressure_grad_mag

_IS_WALL_IDX = NODE_FEATURE_NAMES.index("is_wall")

ModelOutput = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def expand_global_cond(data) -> torch.Tensor:
    if not hasattr(data, "global_cond") or data.global_cond is None:
        return torch.zeros(
            data.x.size(0),
            GLOBAL_COND_DIM,
            dtype=data.x.dtype,
            device=data.x.device,
        )

    if hasattr(data, "batch") and data.batch is not None:
        return data.global_cond[data.batch]
    return data.global_cond.expand(data.x.size(0), -1)


def split_model_output(output: ModelOutput) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(output, tuple):
        field_pred, wss_pred = output
        return field_pred, wss_pred
    return output, None


def _pack_model_output(
    field_pred: torch.Tensor,
    wss_pred: Optional[torch.Tensor],
) -> ModelOutput:
    if wss_pred is None:
        return field_pred
    return field_pred, wss_pred


class FieldMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, out_dim: int, wss_dim: int = 0):
        super().__init__()
        self.backbone = nn.ModuleList()
        self.backbone.append(nn.Linear(in_dim, hidden_dim))
        for _ in range(max(0, num_layers - 1)):
            self.backbone.append(nn.Linear(hidden_dim, hidden_dim))
        self.dropout = dropout
        self.field_head = nn.Linear(hidden_dim, out_dim)
        self.wss_head = nn.Linear(hidden_dim, wss_dim) if wss_dim > 0 else None

    def forward(self, data) -> ModelOutput:
        x = torch.cat([data.x, expand_global_cond(data)], dim=-1)
        for linear in self.backbone:
            x = F.dropout(F.relu(linear(x)), p=self.dropout, training=self.training)
        field_pred = self.field_head(x)
        wss_pred = self.wss_head(x) if self.wss_head is not None else None
        return _pack_model_output(field_pred, wss_pred)


class FieldGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, out_dim: int, wss_dim: int = 0):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim) for _ in range(max(1, num_layers))]
        )
        self.dropout = dropout
        self.field_head = nn.Linear(hidden_dim, out_dim)
        self.wss_head = nn.Linear(hidden_dim, wss_dim) if wss_dim > 0 else None

    def forward(self, data) -> ModelOutput:
        x = torch.cat([data.x, expand_global_cond(data)], dim=-1)
        x = self.in_proj(x)
        for conv in self.layers:
            residual = x
            x = conv(x, data.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual
        field_pred = self.field_head(x)
        wss_pred = self.wss_head(x) if self.wss_head is not None else None
        return _pack_model_output(field_pred, wss_pred)


class FieldTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        out_dim: int,
        heads: int,
        use_prenorm: bool = True,
        wss_dim: int = 0,
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
        self.use_prenorm = use_prenorm
        self.norms: Optional[nn.ModuleList]
        if use_prenorm:
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.layers])
        else:
            self.norms = None
        self.dropout = dropout
        self.field_head = nn.Linear(hidden_dim, out_dim)
        self.wss_head = nn.Linear(hidden_dim, wss_dim) if wss_dim > 0 else None

    def forward(self, data) -> ModelOutput:
        x = torch.cat([data.x, expand_global_cond(data)], dim=-1)
        x = self.in_proj(x)
        if self.use_prenorm and self.norms is not None:
            for conv, linear, norm in zip(self.layers, self.post_layers, self.norms):
                residual = x
                x = norm(x)
                x = conv(x, data.edge_index)
                x = F.elu(x)
                x = linear(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x + residual
        else:
            for conv, linear in zip(self.layers, self.post_layers):
                residual = x
                x = conv(x, data.edge_index)
                x = F.elu(x)
                x = linear(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = x + residual
        field_pred = self.field_head(x)
        wss_pred = self.wss_head(x) if self.wss_head is not None else None
        return _pack_model_output(field_pred, wss_pred)


# ---------------------------------------------------------------------------
# Literature baselines
# ---------------------------------------------------------------------------


class _EdgeModel(nn.Module):
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
    """Pfaff et al. (ICLR 2021) Encode-Process-Decode baseline."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, out_dim: int, wss_dim: int = 0):
        super().__init__()
        edge_feat_dim = 4
        self.node_encoder = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.edge_encoder = nn.Sequential(nn.Linear(edge_feat_dim, hidden_dim), nn.ReLU())

        self.edge_models = nn.ModuleList()
        self.node_models = nn.ModuleList()
        for _ in range(max(1, num_layers)):
            self.edge_models.append(_EdgeModel(hidden_dim * 3, hidden_dim, hidden_dim))
            self.node_models.append(_NodeModel(hidden_dim * 2, hidden_dim, hidden_dim))

        self.dropout = dropout
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.field_head = nn.Linear(hidden_dim, out_dim)
        self.wss_head = nn.Linear(hidden_dim, wss_dim) if wss_dim > 0 else None

    def _build_edge_attr(self, pos, edge_index):
        row, col = edge_index
        diff = pos[col] - pos[row]
        dist = diff.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.cat([diff, dist], dim=-1)

    def forward(self, data) -> ModelOutput:
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

        h = self.decoder(x)
        field_pred = self.field_head(h)
        wss_pred = self.wss_head(h) if self.wss_head is not None else None
        return _pack_model_output(field_pred, wss_pred)


class FieldPointNetPP(nn.Module):
    """Simplified PointNet++ style baseline."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, out_dim: int, wss_dim: int = 0):
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
        self.shared_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.field_head = nn.Linear(hidden_dim, out_dim)
        self.wss_head = nn.Linear(hidden_dim, wss_dim) if wss_dim > 0 else None

    def forward(self, data) -> ModelOutput:
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

        h = self.shared_decoder(x)
        field_pred = self.field_head(h)
        wss_pred = self.wss_head(h) if self.wss_head is not None else None
        return _pack_model_output(field_pred, wss_pred)


def _local_pool_mean_max(
    x_norm: torch.Tensor,
    edge_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """沿 edge_index 对邻居做 mean / max pool（row 聚合 col）。"""
    row, col = edge_index
    neighbour_feats = x_norm[col]

    max_pool = torch.full_like(x_norm, float("-inf"))
    max_pool.scatter_reduce_(
        0,
        row.unsqueeze(1).expand_as(neighbour_feats),
        neighbour_feats,
        reduce="amax",
    )
    max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

    mean_pool = torch.zeros_like(x_norm)
    mean_pool.scatter_add_(0, row.unsqueeze(1).expand_as(neighbour_feats), neighbour_feats)
    degree = torch.zeros(x_norm.size(0), 1, device=x_norm.device, dtype=x_norm.dtype)
    degree.scatter_add_(0, row.unsqueeze(1), torch.ones_like(row, dtype=x_norm.dtype).unsqueeze(1))
    mean_pool = mean_pool / degree.clamp_min(1.0)
    return mean_pool, max_pool


def _build_pool_edge_indices(
    data,
    pool_k_tiers: List[int],
) -> List[torch.Tensor]:
    """多档 k：首档 k=6 复用预构建 edge_index，其余档运行时 knn_graph。"""
    pos = data.x[:, :3]
    batch = getattr(data, "batch", None)
    edge_list: List[torch.Tensor] = []
    for k in pool_k_tiers:
        if k == 6:
            edge_list.append(data.edge_index)
        else:
            effective_k = min(k, max(1, pos.size(0) - 1))
            edge_list.append(
                knn_graph(pos, k=effective_k, batch=batch, loop=False, flow="source_to_target")
            )
    return edge_list


def compute_wall_vel_grad_context(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    vel: torch.Tensor,
    context_dim: int = 4,
) -> torch.Tensor:
    """近壁速度梯度上下文（TODO-17）：沿 edge_index 壁面→内部邻点估计 dv/dn。"""
    n_nodes = vel.size(0)
    ctx = torch.zeros(n_nodes, context_dim, device=vel.device, dtype=vel.dtype)
    wall_mask = x[:, _IS_WALL_IDX].bool()
    if not wall_mask.any():
        return ctx

    interior_mask = ~wall_mask
    row, col = edge_index
    wall_to_int = wall_mask[row] & interior_mask[col]
    if not wall_to_int.any():
        return ctx

    r = row[wall_to_int]
    c = col[wall_to_int]
    diff = x[c, :3] - x[r, :3]
    dist = diff.norm(dim=-1).clamp_min(1e-8)
    dv = vel[c] - vel[r]
    grad_vec = dv / dist.unsqueeze(-1)
    grad_mag = dv.norm(dim=-1) / dist

    n_feat = min(3, context_dim)
    for k in range(n_feat):
        ctx[:, k].index_add_(0, r, grad_vec[:, k])
    if context_dim > 3:
        ctx[:, 3].index_add_(0, r, grad_mag)

    counts = torch.zeros(n_nodes, device=vel.device, dtype=vel.dtype)
    ones = torch.ones(r.size(0), device=vel.device, dtype=vel.dtype)
    counts.index_add_(0, r, ones)
    valid = counts > 0
    ctx[valid] = ctx[valid] / counts[valid].unsqueeze(-1)
    return ctx * wall_mask.unsqueeze(-1).to(ctx.dtype)


def _build_head(
    hidden_dim: int,
    out_dim: int,
    layout: str = "single_linear",
    head_dropout: float = 0.0,
) -> nn.Module:
    """构造输出头。single_linear 返回 nn.Linear（与旧 checkpoint 参数名兼容），
    mlp2 返回 2 层 MLP（参数名带 .0. / .2. 前缀，新 checkpoint 命名空间）。"""
    if layout == "mlp2":
        layers: list[nn.Module] = [
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        if head_dropout > 0:
            layers.append(nn.Dropout(p=head_dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)
    return nn.Linear(hidden_dim, out_dim)


class FieldPointNeXt(nn.Module):
    """PointNeXt-style residual point backbone built on local neighbourhood pooling."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        out_dim: int,
        wss_dim: int = 0,
        head_layout: str = "single_linear",
        wss_head_dropout: float = 0.0,
        wss_vel_context: bool = False,
        wss_vel_context_dim: int = 4,
        wss_pgrad_context: bool = False,
        wss_output_mode: str = "head",
        wss_metric_dim: int = 1,
        pool_k_tiers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.wss_vel_context = wss_vel_context
        self.wss_vel_context_dim = wss_vel_context_dim if wss_vel_context else 0
        self.wss_pgrad_context = wss_pgrad_context
        self.wss_pgrad_context_dim = 1 if wss_pgrad_context else 0
        self.wss_output_mode = wss_output_mode
        self.wss_metric_dim = wss_metric_dim
        self.pool_k_tiers: List[int] = list(pool_k_tiers or [])
        self.num_pool_tiers = len(self.pool_k_tiers) if self.pool_k_tiers else 1
        block_in_dim = hidden_dim * (1 + 2 * self.num_pool_tiers)
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList()
        for _ in range(max(1, num_layers)):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(hidden_dim),
                        "mlp": nn.Sequential(
                            nn.Linear(block_in_dim, hidden_dim * 2),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim * 2, hidden_dim),
                        ),
                    }
                )
            )

        self.dropout = dropout
        self.shared_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.field_head = _build_head(hidden_dim, out_dim, head_layout)
        wss_in_dim = hidden_dim + self.wss_vel_context_dim + self.wss_pgrad_context_dim
        use_wss_head = wss_dim > 0 and wss_output_mode == "head"
        self.wss_head = (
            _build_head(wss_in_dim, wss_dim, head_layout, head_dropout=wss_head_dropout)
            if use_wss_head
            else None
        )

    def _wss_head_input(self, h: torch.Tensor, data, field_pred: torch.Tensor) -> torch.Tensor:
        if self.wss_head is None:
            return h
        extras: List[torch.Tensor] = []
        if self.wss_vel_context:
            vel = data.y[:, :3] if self.training else field_pred[:, :3]
            ctx = compute_wall_vel_grad_context(
                data.x, data.edge_index, vel, context_dim=self.wss_vel_context_dim
            )
            extras.append(ctx)
        if self.wss_pgrad_context:
            wall_mask = data.x[:, _IS_WALL_IDX].bool()
            pgrad = wall_pressure_grad_mag(
                field_pred, data.x[:, :3], wall_mask, data.edge_index
            )
            extras.append(pgrad.unsqueeze(-1))
        if not extras:
            return h
        return torch.cat([h, *extras], dim=-1)

    def _infer_wss_from_field(self, field_pred: torch.Tensor, data) -> torch.Tensor:
        wall_mask = data.x[:, _IS_WALL_IDX].bool()
        batch_index = getattr(data, "batch", None)
        if batch_index is None:
            batch_index = torch.zeros(field_pred.size(0), dtype=torch.long, device=field_pred.device)
        return infer_wss_from_vel_diff(
            field_pred,
            data.x[:, :3],
            wall_mask,
            batch_index,
            out_dim=self.wss_metric_dim,
        )

    def forward(self, data) -> ModelOutput:
        x_in = torch.cat([data.x, expand_global_cond(data)], dim=-1)
        x = self.in_proj(x_in)
        if self.pool_k_tiers:
            pool_edges = _build_pool_edge_indices(data, self.pool_k_tiers)
        else:
            pool_edges = [data.edge_index]

        for block in self.blocks:
            residual = x
            x_norm = block["norm"](x)
            pool_feats = [x_norm]
            for edge_index in pool_edges:
                mean_pool, max_pool = _local_pool_mean_max(x_norm, edge_index)
                pool_feats.extend([mean_pool, max_pool])
            x = block["mlp"](torch.cat(pool_feats, dim=-1))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        h = self.shared_decoder(x)
        field_pred = self.field_head(h)
        if self.wss_output_mode == "vel_diff":
            wss_pred = self._infer_wss_from_field(field_pred, data)
        else:
            h_wss = self._wss_head_input(h, data, field_pred)
            wss_pred = self.wss_head(h_wss) if self.wss_head is not None else None
        return _pack_model_output(field_pred, wss_pred)


# ---------------------------------------------------------------------------
# Model registry & factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, type] = {
    "mlp": FieldMLP,
    "graphsage": FieldGraphSAGE,
    "transformer": FieldTransformer,
    "meshgraphnet": FieldMeshGraphNet,
    "pointnetpp": FieldPointNetPP,
    "pointnext": FieldPointNeXt,
}


def build_model(
    model_name: str,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    heads: int,
    *,
    use_transformer_prenorm: bool = False,
    wss_dim: int = 0,
    head_layout: str = "single_linear",
    wss_head_dropout: float = 0.0,
    wss_vel_context: bool = False,
    wss_vel_context_dim: int = 4,
    wss_pgrad_context: bool = False,
    wss_output_mode: str = "head",
    wss_metric_dim: int = 1,
    pool_k_tiers: Optional[List[int]] = None,
):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {model_name}, 可选: {list(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[model_name]
    kwargs = dict(
        in_dim=MODEL_INPUT_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        out_dim=TARGET_DIM,
        wss_dim=wss_dim,
    )
    if model_name == "transformer":
        kwargs["heads"] = heads
        kwargs["use_prenorm"] = use_transformer_prenorm
    if model_name == "pointnext":
        kwargs["head_layout"] = head_layout
        kwargs["wss_head_dropout"] = wss_head_dropout
        kwargs["wss_vel_context"] = wss_vel_context
        kwargs["wss_vel_context_dim"] = wss_vel_context_dim
        kwargs["wss_pgrad_context"] = wss_pgrad_context
        kwargs["wss_output_mode"] = wss_output_mode
        kwargs["wss_metric_dim"] = wss_metric_dim
        kwargs["pool_k_tiers"] = pool_k_tiers
    return cls(**kwargs)


def build_field_model_from_config(config) -> nn.Module:
    """从 ExperimentConfig 构造场重建模型（统一传递 WSS 可选特性，默认与旧实验兼容）。"""
    m = config.model
    return build_model(
        model_name=m.name,
        hidden_dim=m.hidden_dim,
        num_layers=m.num_layers,
        dropout=m.dropout,
        heads=m.heads,
        use_transformer_prenorm=m.use_transformer_prenorm,
        wss_dim=m.wss_dim,
        head_layout=m.head_layout,
        wss_head_dropout=m.wss_head_dropout,
        wss_vel_context=m.wss_vel_context,
        wss_vel_context_dim=m.wss_vel_context_dim,
        wss_pgrad_context=m.wss_pgrad_context,
        wss_output_mode=m.wss_output_mode,
        wss_metric_dim=m.wss_metric_dim,
        pool_k_tiers=m.pool_k_tiers or None,
    )
