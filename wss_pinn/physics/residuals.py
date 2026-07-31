"""微分残差：速度雅可比、连续性，以及 CFD 局部线性梯度 Oracle。

学习要点
--------
不可压缩连续性：

    ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z = 0

PINN 用 ``torch.autograd.grad`` 对网络输出求坐标导数（``velocity_jacobian``），
再取迹得到散度（``continuity_residual``）。这是 F1 的核心物理项。

``local_linear_velocity_gradients`` 不依赖网络：在 CFD 点云上做加权局部线性
拟合，得到「真值侧」梯度 Oracle，用于 P0 审计（网络误差 vs 数据/离散误差）。
"""

from __future__ import annotations

import numpy as np
import torch


def velocity_jacobian(
    velocity: torch.Tensor, coords: torch.Tensor, create_graph: bool = True
) -> torch.Tensor:
    """返回 ``∂u_i / ∂x_j``，形状 ``[N, 3, 3]``。

    对每个速度分量 ``velocity[:, i]`` 关于 ``coords`` 求梯度，堆成雅可比。
    ``create_graph=True`` 时二次导数可继续反传到网络参数（训练用）；
    评估残差时可 ``False`` 省显存。
    """
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError("velocity must have shape [N, 3]")
    rows = []
    for component in range(3):
        # sum() 后对标量反传，等价于对每个点的该分量求 ∂/∂x
        grad = torch.autograd.grad(
            velocity[:, component].sum(),
            coords,
            create_graph=create_graph,
            retain_graph=True,  # 后续分量还要用同一张计算图
            allow_unused=False,
        )[0]
        rows.append(grad)
    return torch.stack(rows, dim=1)


def continuity_residual(
    velocity: torch.Tensor, coords: torch.Tensor, create_graph: bool = True
) -> torch.Tensor:
    """连续性残差 ``∇·u``，形状 ``[N]``；理想不可压缩流应接近 0。"""
    jacobian = velocity_jacobian(velocity, coords, create_graph=create_graph)
    return jacobian[:, 0, 0] + jacobian[:, 1, 1] + jacobian[:, 2, 2]


def local_linear_velocity_gradients(
    coords: np.ndarray,
    velocity: np.ndarray,
    query_indices: np.ndarray,
    *,
    neighbors: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """加权局部线性 CFD 梯度 Oracle。

    坐标与速度应已无量纲。返回：

    - ``gradients``：``[Q, 3, 3]``，``∂u_i/∂x_j``；
    - ``conditions``：设计矩阵 Gram 条件数，越大表示局部数值越不可信。

    方法：对每个查询点取 k 近邻，拟合 ``Δv ≈ G · Δx``（带高斯权重）。
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(coords, dtype=np.float64)
    vel = np.asarray(velocity, dtype=np.float64)
    query = np.asarray(query_indices, dtype=np.int64)
    if len(xyz) != len(vel):
        raise ValueError("coordinate/velocity length mismatch")
    k = min(max(int(neighbors), 8), len(xyz))
    tree = cKDTree(xyz)
    distances, indices = tree.query(xyz[query], k=k)
    gradients = np.full((len(query), 3, 3), np.nan, dtype=np.float64)
    conditions = np.full(len(query), np.inf, dtype=np.float64)
    for row, (center_index, dist, neighbor_index) in enumerate(
        zip(query, distances, indices)
    ):
        dx = xyz[neighbor_index] - xyz[center_index]
        dv = vel[neighbor_index] - vel[center_index]
        # 带宽取邻域距离中位数，避免单点极近导致权重退化
        scale = max(float(np.median(dist[1:])), 1e-12)
        weights = np.exp(-np.square(dist / (2.0 * scale)))
        # 设计矩阵 [1, Δx, Δy, Δz]；常数项吸收局部基值
        design = np.column_stack([np.ones(len(dx)), dx])
        weighted = design * np.sqrt(weights[:, None])
        target = dv * np.sqrt(weights[:, None])
        gram = weighted.T @ weighted
        conditions[row] = np.linalg.cond(gram)
        try:
            coefficients, *_ = np.linalg.lstsq(weighted, target, rcond=1e-10)
            # coefficients[0] 是截距；[1:] 才是梯度行 → 转置成 ∂u_i/∂x_j
            gradients[row] = coefficients[1:].T
        except np.linalg.LinAlgError:
            continue
    return gradients, conditions
