from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from pipeline.config import GLOBAL_COND_NAMES, NODE_FEATURE_NAMES

from .config import PhysicsConfig


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


def wss_mse_loss(
    wss_pred: torch.Tensor,
    wss_target: torch.Tensor,
    is_wall: torch.Tensor,
    wss_weights: torch.Tensor,
) -> torch.Tensor:
    """仅在壁面节点 (is_wall=1) 上计算 WSS 监督损失。"""
    wall_mask = is_wall.squeeze(-1).bool()
    n_wall = wall_mask.sum()
    if n_wall == 0:
        return wss_pred.new_zeros(())
    pred_wall = wss_pred[wall_mask]
    target_wall = wss_target[wall_mask]
    per_dim = (pred_wall - target_wall).square().mean(dim=0)
    return (per_dim * wss_weights).sum()


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
    def __init__(self, interior_loss_boost: float = 1.0,
                 wss_loss_weight: float = 0.0, wss_weights: Optional[torch.Tensor] = None):
        self.interior_loss_boost = float(interior_loss_boost)
        self.wss_loss_weight = float(wss_loss_weight)
        self.wss_weights = wss_weights

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
                wss_l = wss_mse_loss(wss_pred, wss_target, is_wall, self.wss_weights.to(pred.device))

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
    def __init__(self, config: PhysicsConfig, interior_loss_boost: float = 1.0,
                 wss_loss_weight: float = 0.0, wss_weights: Optional[torch.Tensor] = None):
        self.config = config
        self.interior_loss_boost = float(interior_loss_boost)
        self.wss_loss_weight = float(wss_loss_weight)
        self.wss_weights = wss_weights
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
                wss_l = wss_mse_loss(wss_pred, wss_target, is_wall, self.wss_weights.to(pred.device))

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


def build_loss_plugin(
    config: Optional[PhysicsConfig],
    interior_loss_boost: float = 1.0,
    wss_loss_weight: float = 0.0,
    wss_weights: Optional[torch.Tensor] = None,
):
    kwargs = dict(
        interior_loss_boost=interior_loss_boost,
        wss_loss_weight=wss_loss_weight,
        wss_weights=wss_weights,
    )
    if config is None or not config.enabled:
        return NullPhysicsLoss(**kwargs)
    return PhysicsConstraintLoss(config, **kwargs)
