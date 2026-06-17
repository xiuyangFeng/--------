"""G2-b · 壁面 query × 近壁内部 key/value 的 kernel-attention（压力→WSS 结构耦合）。"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .losses import wall_pressure_grad_features

# 每条壁面→内部边的 KV：[p_j, Δp/dist, 1/r, |v_j|, |Δv|/dist]
KV_DIM = 5


def _segment_softmax(scores: torch.Tensor, group_idx: torch.Tensor, n_groups: int) -> torch.Tensor:
    """按 group_idx 分组 softmax（scores [E], group_idx [E]）。"""
    if scores.numel() == 0:
        return scores
    max_s = torch.full((n_groups,), float("-inf"), device=scores.device, dtype=scores.dtype)
    max_s.scatter_reduce_(0, group_idx, scores, reduce="amax", include_self=True)
    max_s = torch.where(torch.isfinite(max_s), max_s, torch.zeros_like(max_s))
    exp_s = torch.exp(scores - max_s[group_idx])
    denom = torch.zeros(n_groups, device=scores.device, dtype=scores.dtype)
    denom.index_add_(0, group_idx, exp_s)
    return exp_s / denom[group_idx].clamp_min(1e-12)


def build_wall_interior_kv(
    field_pred: torch.Tensor,
    coords: torch.Tensor,
    wall_mask: torch.Tensor,
    edge_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造壁面→内部边的 KV 特征与索引。

    Returns:
        w_idx, kv_feats [E, KV_DIM], dist [E], n_wall_nodes
    """
    row, col = edge_index
    interior = ~wall_mask
    mask = wall_mask[row] & interior[col]
    n_nodes = wall_mask.numel()
    if not mask.any():
        empty_f = field_pred.new_zeros((0, KV_DIM))
        empty_l = torch.empty(0, dtype=torch.long, device=field_pred.device)
        return empty_l, empty_f, empty_l, n_nodes

    w_idx = row[mask]
    j_idx = col[mask]
    diff = coords[j_idx] - coords[w_idx]
    dist = diff.norm(dim=-1).clamp_min(1e-8)

    p_w = field_pred[w_idx, 3]
    p_j = field_pred[j_idx, 3]
    dp_over_dist = (p_j - p_w) / dist

    vel_j = field_pred[j_idx, :3]
    vel_w = field_pred[w_idx, :3]
    vel_j_mag = vel_j.norm(dim=-1)
    dv_over_dist = (vel_j - vel_w).norm(dim=-1) / dist

    inv_r = 1.0 / dist
    kv = torch.stack([p_j, dp_over_dist, inv_r, vel_j_mag, dv_over_dist], dim=-1)
    return w_idx, kv, dist, n_nodes


class WallPressureKernelAttention(nn.Module):
    """壁面 hidden query + 近壁 (p, Δp, vel) key/value；score 含 log(1/r) Green 偏置。"""

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int = 64,
        n_heads: int = 4,
        kv_dim: int = KV_DIM,
        dropout: float = 0.0,
        green_bias: bool = True,
    ):
        super().__init__()
        if out_dim % n_heads != 0:
            raise ValueError("out_dim 须能被 n_heads 整除")
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.kv_dim = kv_dim
        self.green_bias = green_bias

        self.q_proj = nn.Linear(hidden_dim, out_dim)
        self.k_proj = nn.Linear(kv_dim, out_dim)
        self.v_proj = nn.Linear(kv_dim, out_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if green_bias:
            self.green_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_parameter("green_scale", None)

    def forward(
        self,
        h: torch.Tensor,
        field_pred: torch.Tensor,
        coords: torch.Tensor,
        wall_mask: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """返回 [N, out_dim]，仅壁面行非零。"""
        out_dim = self.q_proj.out_features
        ctx = field_pred.new_zeros(field_pred.size(0), out_dim)
        w_idx, kv, dist, n_nodes = build_wall_interior_kv(
            field_pred, coords, wall_mask, edge_index
        )
        if w_idx.numel() == 0:
            return ctx

        q = self.q_proj(h)[w_idx].view(-1, self.n_heads, self.head_dim)
        k = self.k_proj(kv).view(-1, self.n_heads, self.head_dim)
        v = self.v_proj(kv).view(-1, self.n_heads, self.head_dim)

        scores = (q * k).sum(-1) / (self.head_dim ** 0.5)
        scores = scores.mean(dim=-1)
        if self.green_scale is not None:
            scores = scores + self.green_scale * torch.log(1.0 / dist.clamp_min(1e-8))

        attn = _segment_softmax(scores, w_idx, n_nodes)
        v_flat = v.reshape(-1, out_dim)
        weighted = attn.unsqueeze(-1) * v_flat
        ctx.index_add_(0, w_idx, weighted)
        # 壁面点可能有多条入边，index_add 已聚合；对无邻居壁面保持 0
        ctx = self.out_proj(ctx)
        return ctx * wall_mask.unsqueeze(-1).to(ctx.dtype)
