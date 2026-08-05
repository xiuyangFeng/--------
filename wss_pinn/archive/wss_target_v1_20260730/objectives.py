"""训练目标（loss）组装：按 Gate 阶段开关各损失项。

学习要点
--------
PINN 阶梯实验「一次只开一类信息源」：

- **F0-U**：监督速度 + 直接 WSS（``direct_wss`` / ``velocity_data``）
- **F0-UP**：再加压力监督（``pressure_data``，且用规范不变 MSE）
- **F1**：再加连续性残差 + 壁面无滑移（``continuity`` / ``no_slip``）

权重由配置 ``loss.*_weight`` 控制；权重为 0 的项仍会算出来写进日志，
但 ``total`` 里乘权后为 0，等价于关闭。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .physics.residuals import continuity_residual


def gauge_invariant_pressure_mse(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """压力的规范不变 MSE：两侧各自减去均值后再比。

    不可压缩 NS 里压力只以梯度出现，绝对零点可任意平移（gauge freedom）。
    因此不能直接对原始压力做 MSE，否则网络会浪费容量去拟合一个无物理意义的常数偏置。
    """
    prediction_centered = prediction - prediction.mean()
    target_centered = target - target.mean()
    return F.mse_loss(prediction_centered, target_centered)


def no_slip_loss(velocity_at_wall: torch.Tensor) -> torch.Tensor:
    """刚性壁无滑移：壁面上速度应接近 0，惩罚 |u|² 的均值。"""
    return torch.mean(torch.square(velocity_at_wall))


def stage_losses(
    model,
    descriptor: torch.Tensor,
    wall_coords: torch.Tensor,
    wall_wss: torch.Tensor,
    interior_coords: torch.Tensor,
    velocity_target: torch.Tensor,
    pressure_target: torch.Tensor,
    weights: dict,
) -> dict[str, torch.Tensor]:
    """对一个病例的一个 mini-batch 计算各分项 loss 与加权总和。

    参数
    ----
    model
        ``WSSPINNModel``：输入坐标 + 几何描述子，输出 log_wss / velocity / pressure。
    descriptor
        病例级几何向量（15 维），广播到该病例所有点。
    wall_coords / wall_wss
        壁面采样点坐标与真值标量 WSS（物理单位 Pa；网络学 ``log1p(WSS)``）。
    interior_coords / velocity_target / pressure_target
        近壁+核心体点；速度/压力已按病例特征尺度无量纲化。
    weights
        配置里的 ``loss`` 段字典。

    返回
    ----
    字典，含各分项与 ``total``。调用方可对 ``total`` 做 ``backward()``。
    """
    # 只有连续性权重 > 0 时才需要坐标可微（autograd 求 ∂u_i/∂x_j）
    requires_derivative = float(weights["continuity_weight"]) > 0
    interior_coords = interior_coords.requires_grad_(requires_derivative)

    # 壁面与体域各前向一次：壁面出 WSS / 壁面速度；体域出 (u,v,w,p)
    wall_output = model(wall_coords, descriptor)
    interior_output = model(interior_coords, descriptor)

    losses = {
        # 直接监督 WSS：对 log(1+WSS) 做 MSE，缓解高 WSS 长尾尺度问题
        "direct_wss": F.mse_loss(
            wall_output["log_wss_direct"], torch.log1p(torch.clamp(wall_wss, min=0))
        ),
        # 体域速度数据项（无量纲）
        "velocity_data": F.mse_loss(interior_output["velocity"], velocity_target),
        # 压力数据项（规范不变）
        "pressure_data": gauge_invariant_pressure_mse(
            interior_output["pressure"], pressure_target
        ),
        # 先占位 0；下面按权重决定是否真正计算（避免无谓 autograd）
        "continuity": torch.zeros((), device=wall_coords.device, dtype=wall_coords.dtype),
        "no_slip": torch.zeros((), device=wall_coords.device, dtype=wall_coords.dtype),
    }

    if requires_derivative:
        # ∇·u = 0（不可压缩连续性）；create_graph=True 以便二次反传进网络
        divergence = continuity_residual(
            interior_output["velocity"], interior_coords, create_graph=True
        )
        losses["continuity"] = torch.mean(torch.square(divergence))

    if float(weights["no_slip_weight"]) > 0:
        losses["no_slip"] = no_slip_loss(wall_output["velocity"])

    # 加权求和；权重名约定为 ``{分项名}_weight``
    total = sum(
        losses[name] * float(weights[f"{name}_weight"])
        for name in ("direct_wss", "velocity_data", "pressure_data", "continuity", "no_slip")
    )
    return {**losses, "total": total}
