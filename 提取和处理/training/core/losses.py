from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from pipeline.config import GLOBAL_COND_NAMES, NODE_FEATURE_NAMES

from .config import DomainLossConfig, PhysicsConfig


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # 数据监督项始终作为主损失存在；physics 只是在其上叠加约束。
    # 先计算每个输出维度的 MSE。
    mse_per_dim = F.mse_loss(pred, target, reduction="none").mean(dim=0)
    # 再按各目标维度权重加权求和。
    return (mse_per_dim * weights).sum()


def region_weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    is_wall: torch.Tensor,
    interior_boost: float,
) -> torch.Tensor:
    # 与 weighted_mse_loss 在 interior_boost=1 且 is_wall 全 0/1 时数值等价（均匀节点权重）。
    # is_wall: [N, 1]，非壁面节点（内部点）乘以 interior_boost。
    sq = F.mse_loss(pred, target, reduction="none")
    node_w = torch.where(
        is_wall.squeeze(-1).bool(),
        pred.new_ones(pred.size(0)),
        pred.new_full((pred.size(0),), float(interior_boost)),
    )
    return (sq * weights.unsqueeze(0) * node_w.unsqueeze(-1)).mean()


def _data_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_weights: torch.Tensor,
    batch,
    interior_loss_boost: float,
) -> torch.Tensor:
    # interior_loss_boost=1 时全体节点权重为 1，与 weighted_mse_loss 一致。
    idx = NODE_FEATURE_NAMES.index("is_wall")
    is_wall = batch.x[:, idx : idx + 1]
    return region_weighted_mse_loss(pred, target, data_weights, is_wall, interior_loss_boost)


def wss_supervision_loss(
    wss_pred: torch.Tensor,
    wss_target: torch.Tensor,
    is_wall: torch.Tensor,
    wss_weights: torch.Tensor,
    loss_type: str = "mse",
    huber_beta: float = 1.0,
) -> torch.Tensor:
    """仅在壁面节点 (is_wall=1) 上计算 WSS 监督。loss_type='mse' 为逐维 MSE 再加权；'huber'/'smooth_l1' 为 Smooth L1。"""
    wall_mask = is_wall.squeeze(-1).bool()
    finite_rows = torch.isfinite(wss_target).all(dim=1)
    valid = wall_mask & finite_rows
    n_valid = valid.sum()
    if n_valid == 0:
        return wss_pred.new_zeros(())
    pred_wall = wss_pred[valid]
    target_wall = wss_target[valid]
    lt = (loss_type or "mse").lower()
    n_dims = pred_wall.shape[1]
    per_dim = []
    for d in range(n_dims):
        t_d = target_wall[:, d]
        p_d = pred_wall[:, d]
        dim_valid = torch.isfinite(t_d)
        if dim_valid.sum() == 0:
            per_dim.append(wss_pred.new_zeros(()))
            continue
        if lt == "mse":
            per_dim.append((p_d[dim_valid] - t_d[dim_valid]).square().mean())
        elif lt in ("huber", "smooth_l1"):
            per_dim.append(
                F.smooth_l1_loss(
                    p_d[dim_valid], t_d[dim_valid], reduction="mean", beta=float(huber_beta)
                )
            )
        else:
            raise ValueError(f"未知 wss_loss_type: {loss_type}")
    per_dim_t = torch.stack(per_dim)
    return (per_dim_t * wss_weights[:n_dims]).sum()


def wss_topk_ranking_loss(
    wss_pred: torch.Tensor,
    wss_target: torch.Tensor,
    is_wall: torch.Tensor,
    batch_index: Optional[torch.Tensor] = None,
    top_frac: float = 0.10,
    max_nodes: int = 512,
) -> torch.Tensor:
    """ListNet 风格 top-fraction 壁面 |WSS| 排序 loss（TODO-42）。"""
    wall = is_wall.squeeze(-1).bool()
    finite = torch.isfinite(wss_target[:, 0])
    valid = wall & finite
    if valid.sum() < 2:
        return wss_pred.new_zeros(())

    if batch_index is None:
        batch_index = torch.zeros(wss_pred.size(0), dtype=torch.long, device=wss_pred.device)

    total = wss_pred.new_zeros(())
    n_graphs = 0
    eps = 1e-8
    for graph_id in batch_index[valid].unique():
        mask = valid & (batch_index == graph_id)
        pred = wss_pred[mask, 0]
        target = wss_target[mask, 0]
        n = int(pred.numel())
        k = max(2, min(int(round(n * top_frac)), int(max_nodes), n))
        _, idx = torch.topk(target, k, largest=True)
        pred_k = pred[idx]
        target_k = target[idx]
        pl_true = F.softmax(target_k, dim=0)
        pl_pred = F.softmax(pred_k, dim=0)
        total = total - (pl_true * (pl_pred + eps).log()).sum()
        n_graphs += 1
    if n_graphs == 0:
        return wss_pred.new_zeros(())
    return total / n_graphs


def wss_magnitude_consistency_loss(
    wss_pred: torch.Tensor,
    is_wall: torch.Tensor,
) -> torch.Tensor:
    """壁面点：|pred[wss] − √(wss_x²+wss_y²+wss_z²)|² 的均值（TODO-9 / WSS-04）。"""
    if wss_pred.size(1) < 4:
        return wss_pred.new_zeros(())
    wall_mask = is_wall.squeeze(-1).bool()
    if not wall_mask.any():
        return wss_pred.new_zeros(())
    wp = wss_pred[wall_mask]
    mag = wp[:, 0]
    recomputed = torch.sqrt(torch.clamp((wp[:, 1:4] ** 2).sum(dim=1), min=0.0))
    return (mag - recomputed).square().mean()


def _per_graph_wall_zscore(
    values: torch.Tensor,
    wall_mask: torch.Tensor,
    batch_index: torch.Tensor,
    min_nodes: int = 8,
) -> torch.Tensor:
    """逐图壁面 z-score，使 vel_diff 合成 WSS 与 global z-score 目标同量级（TODO-30 量纲修复）。"""
    out = values.clone()
    for gid in batch_index.unique():
        m = wall_mask & (batch_index == gid)
        if int(m.sum()) < min_nodes:
            continue
        v = values[m]
        out[m] = (v - v.mean()) / (v.std() + 1e-6)
    return out


def _estimate_wall_normals_pca(
    wall_coords: torch.Tensor,
    k: int = 16,
) -> torch.Tensor:
    """壁面点子集 PCA 法向（未定向），返回 [n_wall, 3]。"""
    n_pts = wall_coords.size(0)
    k_eff = min(k, max(2, n_pts - 1))
    normals = wall_coords.new_zeros(n_pts, 3)
    if k_eff < 2:
        normals[:, 2] = 1.0
        return normals
    with torch.no_grad():
        d_wall = torch.cdist(wall_coords, wall_coords)
        d_wall.fill_diagonal_(float("inf"))
        _, knn_idx = d_wall.topk(k_eff, dim=1, largest=False)
        for r in range(n_pts):
            nb = wall_coords[knn_idx[r]]
            c = nb - nb.mean(dim=0, keepdim=True)
            cov = c.T @ c
            _, evecs = torch.linalg.eigh(cov)
            nrm = evecs[:, 0]
            normals[r] = nrm / nrm.norm().clamp_min(1e-12)
    return normals


def _nearwall_velocity_diff_naive(
    pred: torch.Tensor,
    coords: torch.Tensor,
    wall_mask: torch.Tensor,
    batch_index: torch.Tensor,
) -> torch.Tensor:
    """naive：|预测速度(最近内部点) − 预测速度(壁面)| / 欧氏距离。"""
    est_full = pred.new_zeros(pred.size(0))
    interior_mask = ~wall_mask
    for gid in batch_index.unique():
        gsel = batch_index == gid
        w = gsel & wall_mask
        it = gsel & interior_mask
        if int(w.sum()) == 0 or int(it.sum()) == 0:
            continue
        cw = coords[w]
        ci = coords[it]
        with torch.no_grad():
            d = torch.cdist(cw, ci)
            dmin, jmin = d.min(dim=1)
        vel_w = pred[w, :3]
        vel_i = pred[it][jmin, :3]
        est_w = (vel_i - vel_w).norm(dim=1) / dmin.clamp_min(1e-6)
        idx = w.nonzero(as_tuple=True)[0]
        est_full = est_full.index_copy(0, idx, est_w)
    return est_full


def _nearwall_velocity_diff_tang_normal(
    pred: torch.Tensor,
    coords: torch.Tensor,
    wall_mask: torch.Tensor,
    batch_index: torch.Tensor,
    k_interior: int = 40,
    cone_ratio: float = 1.0,
    k_normal: int = 16,
) -> torch.Tensor:
    """tang_normal：切向速度模(法向 cone 内最近内部点) / 法向距离（对齐 oracle v2）。"""
    est_full = pred.new_zeros(pred.size(0))
    interior_mask = ~wall_mask
    for gid in batch_index.unique():
        gsel = batch_index == gid
        w = gsel & wall_mask
        it = gsel & interior_mask
        n_w = int(w.sum())
        n_i = int(it.sum())
        if n_w == 0 or n_i == 0:
            continue
        cw = coords[w]
        ci = coords[it]
        int_idx = it.nonzero(as_tuple=True)[0]
        w_idx = w.nonzero(as_tuple=True)[0]
        k_int = min(k_interior, n_i)

        with torch.no_grad():
            n_hat = _estimate_wall_normals_pca(cw, k=k_normal)
            d_wi = torch.cdist(cw, ci)
            _, idx_k = d_wi.topk(k_int, dim=1, largest=False)
            ci_neigh = ci[idx_k]
            offs = ci_neigh - cw.unsqueeze(1)
            nd = (offs * n_hat.unsqueeze(1)).sum(dim=2)
            flip = nd.mean(dim=1) < 0
            n_hat = torch.where(flip.unsqueeze(1), -n_hat, n_hat)
            nd = torch.where(flip.unsqueeze(1), -nd, nd)
            td = (offs - nd.unsqueeze(2) * n_hat.unsqueeze(1)).norm(dim=2)
            cone = (nd > 1e-9) & (td <= cone_ratio * nd.clamp_min(1e-9))
            nd_masked = nd.clone()
            nd_masked[~cone] = float("inf")
            j = nd_masked.argmin(dim=1)
            has_cone = cone.any(dim=1)
            j = torch.where(has_cone, j, torch.zeros_like(j))
            dn = nd[torch.arange(n_w, device=pred.device), j].clamp_min(1e-6)
            pick = idx_k[torch.arange(n_w, device=pred.device), j]

        vel_b = pred[int_idx[pick], :3]
        ut = vel_b - (vel_b * n_hat).sum(dim=1, keepdim=True) * n_hat
        est_w = ut.norm(dim=1) / dn
        est_full = est_full.index_copy(0, w_idx, est_w)
    return est_full


def nearwall_velocity_diff_estimate(
    pred: torch.Tensor,
    coords: torch.Tensor,
    wall_mask: torch.Tensor,
    batch_index: torch.Tensor,
    mode: str = "naive",
    k_interior: int = 40,
    cone_ratio: float = 1.0,
    k_normal: int = 16,
) -> torch.Tensor:
    """每壁面点近壁差分推 |WSS| 代理；mode=naive 或 tang_normal（oracle v2 切向/法向口径）。"""
    if mode == "tang_normal":
        return _nearwall_velocity_diff_tang_normal(
            pred, coords, wall_mask, batch_index, k_interior, cone_ratio, k_normal
        )
    if mode != "naive":
        raise ValueError(f"nearwall_velocity_diff_estimate: 未知 mode={mode}")
    return _nearwall_velocity_diff_naive(pred, coords, wall_mask, batch_index)


def wall_pressure_grad_mag(
    pred: torch.Tensor,
    coords: torch.Tensor,
    wall_mask: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """每壁面点：用图边邻域有限差分估 |∇p| 的方向均方根（思路 2 压力→WSS 耦合）。

    返回 [N]，仅壁面点非零；可微，梯度回流到预测压力 pred[:,3]。
    """
    return wall_pressure_grad_features(pred, coords, wall_mask, edge_index)[:, 3]


def wall_pressure_grad_features(
    pred: torch.Tensor,
    coords: torch.Tensor,
    wall_mask: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """估计壁面压力梯度 rich features: [gx, gy, gz, |grad p|]。

    signed 分量用于输入侧条件化，模长保持与旧 `wall_pressure_grad_mag` 口径一致。
    """
    p = pred[:, 3]
    src = edge_index[0]
    dst = edge_index[1]
    delta = coords[src] - coords[dst]
    dist2 = delta.square().sum(dim=1).clamp_min(1e-12)
    dist = torch.sqrt(dist2)
    grad_edge = ((p[src] - p[dst]) / dist2).unsqueeze(-1) * delta
    grad_vec = pred.new_zeros(pred.size(0), 3).index_add(0, dst, grad_edge)
    diff2 = ((p[src] - p[dst]) / dist) ** 2
    num = pred.new_zeros(pred.size(0)).index_add(0, dst, diff2)
    cnt = pred.new_zeros(pred.size(0)).index_add(0, dst, torch.ones_like(diff2))
    grad_vec = grad_vec / cnt.clamp_min(1.0).unsqueeze(-1)
    gmag = torch.sqrt(num / cnt.clamp_min(1.0) + 1e-12)
    mask = wall_mask.to(gmag.dtype).unsqueeze(-1)
    return torch.cat([grad_vec, gmag.unsqueeze(-1)], dim=-1) * mask


def wall_boundary_layer_proxy(
    field_pred: torch.Tensor,
    coords: torch.Tensor,
    wall_mask: torch.Tensor,
    batch_index: torch.Tensor,
    mode: str = "tang_normal",
) -> torch.Tensor:
    """近壁速度差分代理，作为 direct WSS head 的边界层上下文输入。

    与 vel_diff 输出不同，这里只提供逐图标准化的 1 维侧带特征，让 WSS head 自行融合。
    """
    proxy = nearwall_velocity_diff_estimate(
        field_pred, coords, wall_mask, batch_index, mode=mode
    )
    return _per_graph_wall_zscore(proxy, wall_mask, batch_index) * wall_mask.to(proxy.dtype)


def standardized_pattern_mse(
    a: torch.Tensor,
    b: torch.Tensor,
    wall_mask: torch.Tensor,
    batch_index: torch.Tensor,
    min_nodes: int = 8,
) -> torch.Tensor:
    """逐图在壁面点上把 a、b 各自标准化后取 MSE（尺度无关，聚焦空间模式）。

    a、b 均归一化（z-score）但量纲不同，故对比模式而非绝对值；与 oracle 的相关口径一致。
    """
    total = a.new_zeros(())
    n = 0
    for gid in batch_index.unique():
        m = wall_mask & (batch_index == gid)
        if int(m.sum()) < min_nodes:
            continue
        av = a[m]
        bv = b[m]
        av = (av - av.mean()) / (av.std() + 1e-6)
        bv = (bv - bv.mean()) / (bv.std() + 1e-6)
        total = total + (av - bv).square().mean()
        n += 1
    if n == 0:
        return a.new_zeros(())
    return total / n


def infer_wss_from_vel_diff(
    field_pred: torch.Tensor,
    coords: torch.Tensor,
    wall_mask: torch.Tensor,
    batch_index: torch.Tensor,
    out_dim: int = 1,
    mode: str = "naive",
) -> torch.Tensor:
    """结构版 TODO-30：由预测近壁速度差分合成 WSS 张量（可微，梯度回流到速度）。

    out_dim=1 时仅输出 |WSS| 模长；out_dim=4 时 col0=模长、分量列为 0（兼容 AsymW 四列指标口径）。

    合成后对壁面点做逐图 z-score，与 y_wss 的 global z-score 目标量级对齐（slope loss 仍用 raw
    nearwall_velocity_diff_estimate + standardized_pattern_mse，互不干扰）。
    """
    mag = nearwall_velocity_diff_estimate(
        field_pred, coords, wall_mask, batch_index, mode=mode
    )
    mag = _per_graph_wall_zscore(mag, wall_mask, batch_index)
    if out_dim <= 1:
        return mag.unsqueeze(-1)
    out = field_pred.new_zeros(field_pred.size(0), out_dim)
    out[:, 0] = mag
    return out


@dataclass
class LossBreakdown:
    total_loss: torch.Tensor
    data_loss: torch.Tensor
    wss_loss: torch.Tensor
    continuity_loss: torch.Tensor
    momentum_loss: torch.Tensor
    no_slip_loss: torch.Tensor

    def scalar_dict(self) -> Dict[str, float]:
        return {
            "loss": self.total_loss.detach().item(),
            "data_loss": self.data_loss.detach().item(),
            "wss_loss": self.wss_loss.detach().item(),
            "physics_continuity_loss": self.continuity_loss.detach().item(),
            "physics_momentum_loss": self.momentum_loss.detach().item(),
            "physics_no_slip_loss": self.no_slip_loss.detach().item(),
        }


class NullPhysicsLoss:
    def __init__(
        self,
        interior_loss_boost: float = 1.0,
        wss_loss_weight: float = 0.0,
        wss_weights: Optional[torch.Tensor] = None,
        wss_loss_type: str = "mse",
        wss_huber_beta: float = 1.0,
    ):
        self.interior_loss_boost = float(interior_loss_boost)
        self.wss_loss_weight = float(wss_loss_weight)
        self.wss_weights = wss_weights
        self.wss_loss_type = wss_loss_type
        self.wss_huber_beta = float(wss_huber_beta)

    def is_enabled(self, epoch: int) -> bool:
        return False

    def build_loss(
        self,
        model: torch.nn.Module,
        batch,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_weights: torch.Tensor,
        epoch: int,
        train: bool,
        wss_pred: Optional[torch.Tensor] = None,
    ) -> LossBreakdown:
        data_loss = _data_mse_loss(
            pred, target, data_weights, batch, self.interior_loss_boost
        )
        zero = data_loss.new_zeros(())

        wss_l = zero
        if wss_pred is not None and self.wss_loss_weight > 0 and self.wss_weights is not None:
            idx = NODE_FEATURE_NAMES.index("is_wall")
            is_wall = batch.x[:, idx : idx + 1]
            wss_target = batch.y_wss if hasattr(batch, "y_wss") and batch.y_wss is not None else None
            if wss_target is not None:
                wss_l = wss_supervision_loss(
                    wss_pred,
                    wss_target,
                    is_wall,
                    self.wss_weights.to(pred.device),
                    loss_type=self.wss_loss_type,
                    huber_beta=self.wss_huber_beta,
                )

        total = data_loss + self.wss_loss_weight * wss_l
        return LossBreakdown(
            total_loss=total,
            data_loss=data_loss,
            wss_loss=wss_l,
            continuity_loss=zero,
            momentum_loss=zero,
            no_slip_loss=zero,
        )


class PhysicsConstraintLoss:
    def __init__(
        self,
        config: PhysicsConfig,
        interior_loss_boost: float = 1.0,
        wss_loss_weight: float = 0.0,
        wss_weights: Optional[torch.Tensor] = None,
        wss_loss_type: str = "mse",
        wss_huber_beta: float = 1.0,
    ):
        self.config = config
        self.interior_loss_boost = float(interior_loss_boost)
        self.wss_loss_weight = float(wss_loss_weight)
        self.wss_weights = wss_weights
        self.wss_loss_type = wss_loss_type
        self.wss_huber_beta = float(wss_huber_beta)
        self.coord_indices = [NODE_FEATURE_NAMES.index(name) for name in ("x", "y", "z")]
        self.is_wall_idx = NODE_FEATURE_NAMES.index("is_wall")
        self.time_idx = GLOBAL_COND_NAMES.index("t_norm")

    def is_enabled(self, epoch: int) -> bool:
        # warmup 的目的是先让模型学到基本场分布，再逐步加入 PDE 约束，降低早期发散概率。
        # 总开关没开时直接关闭。
        if not self.config.enabled:
            return False
        # warmup 轮数内不启用 physics。
        if epoch <= self.config.warmup_epochs:
            return False
        # 至少有一个物理项权重大于 0 时，才真正启用 physics。
        return any(
            weight > 0.0
            for weight in (
                self.config.continuity_weight,
                self.config.momentum_weight,
                self.config.no_slip_weight,
            )
        )

    def build_loss(
        self,
        model: torch.nn.Module,
        batch,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_weights: torch.Tensor,
        epoch: int,
        train: bool,
        wss_pred: Optional[torch.Tensor] = None,
    ) -> LossBreakdown:
        data_loss = _data_mse_loss(
            pred, target, data_weights, batch, self.interior_loss_boost
        )
        zero = data_loss.new_zeros(())

        wss_l = zero
        if wss_pred is not None and self.wss_loss_weight > 0 and self.wss_weights is not None:
            is_wall = batch.x[:, self.is_wall_idx : self.is_wall_idx + 1]
            wss_target = batch.y_wss if hasattr(batch, "y_wss") and batch.y_wss is not None else None
            if wss_target is not None:
                wss_l = wss_supervision_loss(
                    wss_pred,
                    wss_target,
                    is_wall,
                    self.wss_weights.to(pred.device),
                    loss_type=self.wss_loss_type,
                    huber_beta=self.wss_huber_beta,
                )

        if not self.is_enabled(epoch):
            total = data_loss + self.wss_loss_weight * wss_l
            return LossBreakdown(
                total_loss=total,
                data_loss=data_loss,
                wss_loss=wss_l,
                continuity_loss=zero,
                momentum_loss=zero,
                no_slip_loss=zero,
            )

        # 为 physics 分支复制一份 batch，避免污染训练主分支里的张量。
        physics_batch = batch.clone()
        # physics 残差需要对输入坐标和时间自动微分，因此这里必须重新构造 requires_grad 图。
        # 节点特征重新 clone 并开启梯度。
        physics_batch.x = batch.x.detach().clone().requires_grad_(True)
        # 全局条件存在时也要重新开启梯度，时间导数会用到。
        if hasattr(batch, "global_cond") and batch.global_cond is not None:
            physics_batch.global_cond = batch.global_cond.detach().clone().requires_grad_(True)

        physics_output = model(physics_batch)
        physics_pred = physics_output[0] if isinstance(physics_output, tuple) else physics_output

        u = physics_pred[:, 0:1]
        v = physics_pred[:, 1:2]
        w = physics_pred[:, 2:3]
        p = physics_pred[:, 3:4]

        # 取出坐标列。
        x = physics_batch.x[:, self.coord_indices[0] : self.coord_indices[0] + 1]
        y = physics_batch.x[:, self.coord_indices[1] : self.coord_indices[1] + 1]
        z = physics_batch.x[:, self.coord_indices[2] : self.coord_indices[2] + 1]
        # 把图级时间条件广播回每个节点。
        t = self._expand_time(physics_batch)

        # coord_scales/time_scale 用于把“对归一化输入求导”的结果映射回物理尺度。
        # 当前默认值是 1.0，后续如果 pipeline 明确输出真实尺度，应优先改这里。
        inv_sx = 1.0 / self.config.coord_scales[0]
        inv_sy = 1.0 / self.config.coord_scales[1]
        inv_sz = 1.0 / self.config.coord_scales[2]
        inv_st = 1.0 / self.config.time_scale

        # 一阶速度导数。
        du_dx = self._grad(u, x) * inv_sx
        du_dy = self._grad(u, y) * inv_sy
        du_dz = self._grad(u, z) * inv_sz
        du_dt = self._grad(u, t) * inv_st

        dv_dx = self._grad(v, x) * inv_sx
        dv_dy = self._grad(v, y) * inv_sy
        dv_dz = self._grad(v, z) * inv_sz
        dv_dt = self._grad(v, t) * inv_st

        dw_dx = self._grad(w, x) * inv_sx
        dw_dy = self._grad(w, y) * inv_sy
        dw_dz = self._grad(w, z) * inv_sz
        dw_dt = self._grad(w, t) * inv_st

        dp_dx = self._grad(p, x) * inv_sx
        dp_dy = self._grad(p, y) * inv_sy
        dp_dz = self._grad(p, z) * inv_sz

        # 连续性方程残差。
        continuity = du_dx + dv_dy + dw_dz
        # 连续性残差平方均值。
        continuity_loss = (continuity.square()).mean()

        # 二阶导对应动量方程里的黏性扩散项。
        d2u_dx2 = self._grad(self._grad(u, x), x) * (inv_sx ** 2)
        d2u_dy2 = self._grad(self._grad(u, y), y) * (inv_sy ** 2)
        d2u_dz2 = self._grad(self._grad(u, z), z) * (inv_sz ** 2)

        d2v_dx2 = self._grad(self._grad(v, x), x) * (inv_sx ** 2)
        d2v_dy2 = self._grad(self._grad(v, y), y) * (inv_sy ** 2)
        d2v_dz2 = self._grad(self._grad(v, z), z) * (inv_sz ** 2)

        d2w_dx2 = self._grad(self._grad(w, x), x) * (inv_sx ** 2)
        d2w_dy2 = self._grad(self._grad(w, y), y) * (inv_sy ** 2)
        d2w_dz2 = self._grad(self._grad(w, z), z) * (inv_sz ** 2)

        lap_u = d2u_dx2 + d2u_dy2 + d2u_dz2
        lap_v = d2v_dx2 + d2v_dy2 + d2v_dz2
        lap_w = d2w_dx2 + d2w_dy2 + d2w_dz2

        # 三个方向的不可压 Navier-Stokes 动量残差。
        mom_x = self.config.density * (du_dt + u * du_dx + v * du_dy + w * du_dz) + dp_dx - self.config.viscosity * lap_u
        mom_y = self.config.density * (dv_dt + u * dv_dx + v * dv_dy + w * dv_dz) + dp_dy - self.config.viscosity * lap_v
        mom_z = self.config.density * (dw_dt + u * dw_dx + v * dw_dy + w * dw_dz) + dp_dz - self.config.viscosity * lap_w
        # 三个方向残差平方和的均值作为动量损失。
        momentum_loss = mom_x.square().mean() + mom_y.square().mean() + mom_z.square().mean()

        # 取出壁面节点掩码。
        wall_mask = physics_batch.x[:, self.is_wall_idx : self.is_wall_idx + 1]
        # no-slip 目前使用显式 is_wall 节点标记。
        # 如果后续要做更精确的边界条件处理，这里会是首要替换点。
        # 壁面节点速度应接近 0。
        no_slip_loss = (wall_mask * (u.square() + v.square() + w.square())).mean()

        total_loss = (
            data_loss
            + self.wss_loss_weight * wss_l
            + self.config.continuity_weight * continuity_loss
            + self.config.momentum_weight * momentum_loss
            + self.config.no_slip_weight * no_slip_loss
        )
        return LossBreakdown(
            total_loss=total_loss,
            data_loss=data_loss,
            wss_loss=wss_l,
            continuity_loss=continuity_loss,
            momentum_loss=momentum_loss,
            no_slip_loss=no_slip_loss,
        )

    def _expand_time(self, batch) -> torch.Tensor:
        # t_norm 来自图级条件，需要按 batch 广播回每个节点后才能参与自动微分。
        # 没有 global_cond 时，返回全零时间列占位。
        if not hasattr(batch, "global_cond") or batch.global_cond is None:
            return batch.x.new_zeros((batch.x.size(0), 1), requires_grad=True)
        # 批量图时按 batch 索引广播时间条件。
        if hasattr(batch, "batch") and batch.batch is not None:
            return batch.global_cond[batch.batch, self.time_idx : self.time_idx + 1]
        # 单图时直接扩展成与节点数匹配的形状。
        return batch.global_cond[:, self.time_idx : self.time_idx + 1].expand(batch.x.size(0), -1)

    @staticmethod
    def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # 用 autograd 计算 dy/dx。
        grad = torch.autograd.grad(
            y,
            x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        # 对某些未参与计算的变量，autograd 会返回 None，这里补零保持形状一致。
        if grad is None:
            return torch.zeros_like(x)
        return grad


@dataclass
class DualDomainLossBreakdown:
    """V3 双域 mask loss 的分项记录。"""
    total_loss: torch.Tensor
    loss_interior_velocity: torch.Tensor
    loss_noslip_velocity: torch.Tensor
    loss_interior_pressure: torch.Tensor
    loss_wall_pressure: torch.Tensor
    loss_wall_wss: torch.Tensor
    loss_wss_mag_consist: torch.Tensor
    loss_wss_rank: torch.Tensor
    loss_wss_vel_consist: torch.Tensor
    loss_wss_slope: torch.Tensor
    loss_wss_pgrad_consist: torch.Tensor
    weighted_loss_interior_velocity: torch.Tensor
    weighted_loss_noslip_velocity: torch.Tensor
    weighted_loss_interior_pressure: torch.Tensor
    weighted_loss_wall_pressure: torch.Tensor
    weighted_loss_wall_wss: torch.Tensor
    weighted_loss_wss_mag_consist: torch.Tensor
    weighted_loss_wss_rank: torch.Tensor
    weighted_loss_wss_vel_consist: torch.Tensor
    weighted_loss_wss_slope: torch.Tensor
    weighted_loss_wss_pgrad_consist: torch.Tensor

    def scalar_dict(self) -> Dict[str, float]:
        return {
            "loss": self.total_loss.detach().item(),
            "loss_interior_velocity": self.loss_interior_velocity.detach().item(),
            "loss_noslip_velocity": self.loss_noslip_velocity.detach().item(),
            "loss_interior_pressure": self.loss_interior_pressure.detach().item(),
            "loss_wall_pressure": self.loss_wall_pressure.detach().item(),
            "loss_wall_wss": self.loss_wall_wss.detach().item(),
            "loss_wss_mag_consist": self.loss_wss_mag_consist.detach().item(),
            "loss_wss_rank": self.loss_wss_rank.detach().item(),
            "loss_wss_vel_consist": self.loss_wss_vel_consist.detach().item(),
            "loss_wss_slope": self.loss_wss_slope.detach().item(),
            "loss_wss_pgrad_consist": self.loss_wss_pgrad_consist.detach().item(),
            "weighted_loss_interior_velocity": self.weighted_loss_interior_velocity.detach().item(),
            "weighted_loss_noslip_velocity": self.weighted_loss_noslip_velocity.detach().item(),
            "weighted_loss_interior_pressure": self.weighted_loss_interior_pressure.detach().item(),
            "weighted_loss_wall_pressure": self.weighted_loss_wall_pressure.detach().item(),
            "weighted_loss_wall_wss": self.weighted_loss_wall_wss.detach().item(),
            "weighted_loss_wss_mag_consist": self.weighted_loss_wss_mag_consist.detach().item(),
            "weighted_loss_wss_rank": self.weighted_loss_wss_rank.detach().item(),
            "weighted_loss_wss_vel_consist": self.weighted_loss_wss_vel_consist.detach().item(),
            "weighted_loss_wss_slope": self.weighted_loss_wss_slope.detach().item(),
            "weighted_loss_wss_pgrad_consist": self.weighted_loss_wss_pgrad_consist.detach().item(),
        }


class DualDomainLoss:
    """V3 双域 mask loss：壁面与内部分开计算，各项独立求均值再加权。

    硬约束：WSS loss 只在 is_wall==1 节点上计算（§8.2 WSS 伪值陷阱）。
    """

    def __init__(
        self,
        domain_loss_config: DomainLossConfig,
        wss_loss_type: str = "mse",
        wss_huber_beta: float = 1.0,
        wss_weights: Optional[torch.Tensor] = None,
        vel_diff_variant: str = "naive",
    ):
        self.cfg = domain_loss_config
        self.wss_loss_type = wss_loss_type
        self.wss_huber_beta = float(wss_huber_beta)
        self.wss_weights = wss_weights
        self.vel_diff_variant = vel_diff_variant
        self._is_wall_idx = NODE_FEATURE_NAMES.index("is_wall")

    def is_enabled(self, epoch: int) -> bool:
        return False

    def build_loss(
        self,
        model: torch.nn.Module,
        batch,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_weights: torch.Tensor,
        epoch: int,
        train: bool,
        wss_pred: Optional[torch.Tensor] = None,
    ) -> DualDomainLossBreakdown:
        is_wall = batch.x[:, self._is_wall_idx]
        wall_mask = is_wall.bool()
        interior_mask = ~wall_mask
        n_wall = wall_mask.sum().clamp_min(1)
        n_interior = interior_mask.sum().clamp_min(1)

        zero = pred.new_zeros(())

        # 内部速度 MSE（均值分母 = 内部点数）
        if self.cfg.lambda_vel_int > 0 and interior_mask.any():
            vel_int = (pred[interior_mask, :3] - target[interior_mask, :3]).square().mean()
        else:
            vel_int = zero

        # 壁面无滑移速度 MSE（均值分母 = 壁面点数，target 取 CFD 真值）
        if self.cfg.lambda_vel_noslip > 0 and wall_mask.any():
            vel_noslip = (pred[wall_mask, :3] - target[wall_mask, :3]).square().mean()
        else:
            vel_noslip = zero

        # 内部压力 MSE
        if self.cfg.lambda_p_int > 0 and interior_mask.any():
            p_int = (pred[interior_mask, 3] - target[interior_mask, 3]).square().mean()
        else:
            p_int = zero

        # 壁面压力 MSE
        if self.cfg.lambda_p_wall > 0 and wall_mask.any():
            p_wall = (pred[wall_mask, 3] - target[wall_mask, 3]).square().mean()
        else:
            p_wall = zero

        # 壁面 WSS MSE（严格 wall_mask，禁止全节点计算）
        if self.cfg.lambda_wss > 0 and wss_pred is not None and self.wss_weights is not None:
            wss_target = getattr(batch, "y_wss", None)
            if wss_target is not None:
                wss_l = wss_supervision_loss(
                    wss_pred, wss_target,
                    is_wall.unsqueeze(-1),
                    self.wss_weights.to(pred.device),
                    loss_type=self.wss_loss_type,
                    huber_beta=self.wss_huber_beta,
                )
            else:
                wss_l = zero
        else:
            wss_l = zero

        if (
            self.cfg.lambda_wss_mag_consist > 0
            and wss_pred is not None
            and wss_pred.size(1) >= 4
        ):
            mag_consist = wss_magnitude_consistency_loss(wss_pred, is_wall.unsqueeze(-1))
        else:
            mag_consist = zero

        wss_target = getattr(batch, "y_wss", None)
        if (
            self.cfg.lambda_wss_rank > 0
            and wss_pred is not None
            and wss_target is not None
            and wss_pred.size(1) >= 1
        ):
            batch_index = getattr(batch, "batch", None)
            rank_l = wss_topk_ranking_loss(
                wss_pred,
                wss_target,
                is_wall.unsqueeze(-1),
                batch_index=batch_index,
                top_frac=self.cfg.wss_rank_top_frac,
                max_nodes=self.cfg.wss_rank_max_nodes,
            )
        else:
            rank_l = zero

        # --- 近壁速度差分 / 压力梯度耦合（TODO-30/33 + 思路2）---
        # 三项默认 lambda=0，旧配置完全不触发；仅在需要时构造几何量。
        vel_consist = zero
        slope_l = zero
        pgrad_consist = zero
        need_vel_diff = self.cfg.lambda_wss_vel_consist > 0 or self.cfg.lambda_wss_slope > 0
        need_pgrad = self.cfg.lambda_wss_pgrad_consist > 0
        if need_vel_diff or need_pgrad:
            coords = batch.x[:, :3]
            batch_index = getattr(batch, "batch", None)
            if batch_index is None:
                batch_index = torch.zeros(pred.size(0), dtype=torch.long, device=pred.device)

        if need_vel_diff:
            vel_diff = nearwall_velocity_diff_estimate(
                pred, coords, wall_mask, batch_index, mode=self.vel_diff_variant
            )
            # TODO-30：预测 WSS 模长 ↔ 差分代理（自洽，双向拉近）
            if self.cfg.lambda_wss_vel_consist > 0 and wss_pred is not None and wss_pred.size(1) >= 1:
                vel_consist = standardized_pattern_mse(
                    wss_pred[:, 0], vel_diff, wall_mask, batch_index
                )
            # TODO-33：差分代理 ↔ GT |WSS|（监督，塑造速度场近壁梯度；GT 侧 detach）
            if self.cfg.lambda_wss_slope > 0 and wss_target is not None:
                slope_l = standardized_pattern_mse(
                    vel_diff, wss_target[:, 0].detach(), wall_mask, batch_index
                )

        if need_pgrad and wss_pred is not None and wss_pred.size(1) >= 1:
            edge_index = getattr(batch, "edge_index", None)
            if edge_index is not None:
                pgrad_mag = wall_pressure_grad_mag(pred, coords, wall_mask, edge_index)
                pgrad_consist = standardized_pattern_mse(
                    wss_pred[:, 0], pgrad_mag, wall_mask, batch_index
                )

        w_vel_int = self.cfg.lambda_vel_int * vel_int
        w_vel_noslip = self.cfg.lambda_vel_noslip * vel_noslip
        w_p_int = self.cfg.lambda_p_int * p_int
        w_p_wall = self.cfg.lambda_p_wall * p_wall
        w_wss = self.cfg.lambda_wss * wss_l
        w_mag_consist = self.cfg.lambda_wss_mag_consist * mag_consist
        w_rank = self.cfg.lambda_wss_rank * rank_l
        w_vel_consist = self.cfg.lambda_wss_vel_consist * vel_consist
        w_slope = self.cfg.lambda_wss_slope * slope_l
        w_pgrad = self.cfg.lambda_wss_pgrad_consist * pgrad_consist

        total = (
            w_vel_int + w_vel_noslip + w_p_int + w_p_wall + w_wss
            + w_mag_consist + w_rank + w_vel_consist + w_slope + w_pgrad
        )

        return DualDomainLossBreakdown(
            total_loss=total,
            loss_interior_velocity=vel_int,
            loss_noslip_velocity=vel_noslip,
            loss_interior_pressure=p_int,
            loss_wall_pressure=p_wall,
            loss_wall_wss=wss_l,
            loss_wss_mag_consist=mag_consist,
            loss_wss_rank=rank_l,
            loss_wss_vel_consist=vel_consist,
            loss_wss_slope=slope_l,
            loss_wss_pgrad_consist=pgrad_consist,
            weighted_loss_interior_velocity=w_vel_int,
            weighted_loss_noslip_velocity=w_vel_noslip,
            weighted_loss_interior_pressure=w_p_int,
            weighted_loss_wall_pressure=w_p_wall,
            weighted_loss_wall_wss=w_wss,
            weighted_loss_wss_mag_consist=w_mag_consist,
            weighted_loss_wss_rank=w_rank,
            weighted_loss_wss_vel_consist=w_vel_consist,
            weighted_loss_wss_slope=w_slope,
            weighted_loss_wss_pgrad_consist=w_pgrad,
        )


def build_loss_plugin(
    config: Optional[PhysicsConfig],
    interior_loss_boost: float = 1.0,
    wss_loss_weight: float = 0.0,
    wss_weights: Optional[torch.Tensor] = None,
    wss_loss_type: str = "mse",
    wss_huber_beta: float = 1.0,
    domain_loss_config: Optional[DomainLossConfig] = None,
    vel_diff_variant: str = "naive",
):
    # V3 双域 mask loss 优先级最高：启用后不走旧路径。
    if domain_loss_config is not None and domain_loss_config.enabled:
        return DualDomainLoss(
            domain_loss_config=domain_loss_config,
            wss_loss_type=wss_loss_type,
            wss_huber_beta=wss_huber_beta,
            wss_weights=wss_weights,
            vel_diff_variant=vel_diff_variant,
        )
    kwargs = dict(
        interior_loss_boost=interior_loss_boost,
        wss_loss_weight=wss_loss_weight,
        wss_weights=wss_weights,
        wss_loss_type=wss_loss_type,
        wss_huber_beta=wss_huber_beta,
    )
    if config is None or not config.enabled:
        return NullPhysicsLoss(**kwargs)
    return PhysicsConstraintLoss(config, **kwargs)
