"""壁面法向、法向速度梯度拟合，以及由切向梯度得到壁面剪切应力。

学习要点
--------
物理 WSS（向量）近似：

    τ_w ≈ μ(γ̇) · (∂u_τ / ∂n)|_wall

其中 ``u_τ`` 是去掉法向分量后的切向速度，``n`` 为指向流体侧的壁面法向，
``μ`` 由 Carreau–Yasuda 给出。

本模块职责分层：

1. 读 STL 三角面 → 面心与单位法向；
2. 把法向最近邻映射到壁面采样点（``map_stl_normals_to_wall``）；
3. 沿法向距离拟合切向速度斜率（``fit_normal_velocity_gradient``，P0 Oracle）；
4. ``wall_shear_from_normal_gradient``：斜率 × 粘度 → 剪切向量。

注意：训练主路径当前用的是 **直接监督 CFD 标量 WSS**（``direct_wss``）；
本文件主要用于 Oracle / 审计，以及未来 F2 ``WSS_phys`` 一致性项。
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .rheology import carreau_yasuda_numpy


def _binary_stl_triangles(path: Path) -> np.ndarray:
    """解析二进制 STL，返回 ``[F, 3, 3]`` 顶点坐标（跳过文件头里的法向 float）。"""
    size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(80)
        count_raw = handle.read(4)
        if len(count_raw) != 4:
            raise ValueError("truncated STL")
        count = struct.unpack("<I", count_raw)[0]
        # 标准二进制 STL：80 + 4 + 50*三角形
        if 84 + count * 50 != size:
            raise ValueError("not a binary STL")
        triangles = np.empty((count, 3, 3), dtype=np.float64)
        for index in range(count):
            record = handle.read(50)
            # 12 个 float：法向 3 + 顶点 9，再跟 2 字节属性
            values = struct.unpack("<12fH", record)
            triangles[index] = np.asarray(values[3:12]).reshape(3, 3)
    return triangles


def _ascii_stl_triangles(path: Path) -> np.ndarray:
    """解析 ASCII STL：收集所有 ``vertex`` 行，每 3 个组成一个三角面。"""
    vertices: list[list[float]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) == 4 and fields[0].lower() == "vertex":
                vertices.append([float(value) for value in fields[1:]])
    if len(vertices) == 0 or len(vertices) % 3:
        raise ValueError("invalid ASCII STL")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


def read_stl_triangles(path: str | Path) -> np.ndarray:
    """先试二进制，失败再回退 ASCII。"""
    source = Path(path)
    try:
        return _binary_stl_triangles(source)
    except ValueError:
        return _ascii_stl_triangles(source)


def stl_face_geometry(
    path: str | Path, scale_to_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """返回有效三角面的面心与单位法向（坐标乘 ``scale_to_mm`` 到毫米系）。"""
    triangles = read_stl_triangles(path) * float(scale_to_mm)
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge_a, edge_b)
    magnitude = np.linalg.norm(normals, axis=1)
    valid = magnitude > 1e-12  # 丢掉退化面
    centers = triangles[valid].mean(axis=1)
    normals = normals[valid] / magnitude[valid, None]
    if not len(centers):
        raise ValueError("STL contains no non-degenerate faces")
    return centers, normals


def map_stl_normals_to_wall(
    wall_coords_mm: np.ndarray,
    stl_path: str | Path,
    scale_to_mm: float,
    rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """把每个壁面点映射到最近 STL 面法向。

    Returns
    -------
    normals
        ``[N, 3]`` 单位法向；若给了 ``rotation``（与 WSS-min bundle 对齐旋转），先右乘。
    distance
        到最近面心的距离（mm），用于审计映射质量。
    """
    from scipy.spatial import cKDTree

    centers, normals = stl_face_geometry(stl_path, scale_to_mm)
    distance, index = cKDTree(centers).query(np.asarray(wall_coords_mm), k=1)
    mapped = normals[index]
    if rotation is not None:
        mapped = mapped @ np.asarray(rotation, dtype=np.float64)
    mapped /= np.maximum(np.linalg.norm(mapped, axis=1, keepdims=True), 1e-12)
    return mapped.astype(np.float32), np.asarray(distance, dtype=np.float32)


def wall_shear_from_normal_gradient(
    tangential_gradient_s_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """由切向速度对法向距离的梯度（1/s）得到剪切向量与局部粘度。

    ``τ = μ(‖g‖) · g``，其中 ``g = ∂u_τ/∂n``。
    """
    gradient = np.asarray(tangential_gradient_s_inv, dtype=np.float64)
    shear_rate = np.linalg.norm(gradient, axis=-1)
    viscosity = carreau_yasuda_numpy(shear_rate)
    return gradient * viscosity[..., None], viscosity


def fit_normal_velocity_gradient(
    wall_point_mm: np.ndarray,
    wall_normal: np.ndarray,
    interior_coords_mm: np.ndarray,
    interior_velocity_m_s: np.ndarray,
    *,
    neighbors: int = 64,
    degree: int = 1,
) -> tuple[np.ndarray, dict[str, float]]:
    """在壁面一点拟合无滑移切向速度相对法向距离的斜率（P0 velocity→WSS Oracle）。

    步骤简述：
    1. 取壁面点的 k 近邻体点；
    2. 用符号法向距离判断流体侧（点更多的一侧）；
    3. 去掉速度的法向分量得切向速度；
    4. 加权多项式拟合 ``u_τ(d) ≈ a1·d + a2·d² + ...``（默认一次项斜率即 ∂u_τ/∂n）。
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(interior_coords_mm, dtype=np.float64)
    velocity = np.asarray(interior_velocity_m_s, dtype=np.float64)
    normal = np.asarray(wall_normal, dtype=np.float64)
    normal /= max(np.linalg.norm(normal), 1e-12)
    k = min(max(int(neighbors), 12), len(xyz))
    distance, index = cKDTree(xyz).query(np.asarray(wall_point_mm), k=k)
    delta = xyz[index] - wall_point_mm
    signed = delta @ normal
    positive = signed > 0
    negative = signed < 0
    # 选点更多的一侧作为「流体侧」；两侧都太少则退回全用
    use_positive = positive.sum() >= negative.sum()
    mask = positive if use_positive else negative
    if mask.sum() < max(8, degree + 3):
        mask = np.ones_like(signed, dtype=bool)
    d_mm = np.abs(signed[mask])
    # 投影掉法向速度分量 → 切向速度
    tangent = velocity[index][mask] - np.outer(velocity[index][mask] @ normal, normal)
    keep = d_mm > 1e-5  # 丢掉几乎贴在壁上的数值噪声点
    d_m = d_mm[keep] * 1e-3  # mm → m，使斜率单位为 1/s
    tangent = tangent[keep]
    if len(d_m) < max(6, degree + 2):
        return np.full(3, np.nan), {"n_used": float(len(d_m)), "condition": np.inf}
    columns = [d_m**power for power in range(1, int(degree) + 1)]
    design = np.column_stack(columns)
    bandwidth = max(float(np.median(d_m)), 1e-8)
    weights = np.exp(-np.square(d_m / (2.0 * bandwidth)))
    weighted = design * np.sqrt(weights[:, None])
    target = tangent * np.sqrt(weights[:, None])
    coefficients, *_ = np.linalg.lstsq(weighted, target, rcond=1e-10)
    # 一次项系数即 ∂u_τ/∂n（三方向）
    return coefficients[0], {
        "n_used": float(len(d_m)),
        "condition": float(np.linalg.cond(weighted.T @ weighted)),
        "distance_p50_mm": float(np.median(d_mm[keep])),
        "distance_p95_mm": float(np.quantile(d_mm[keep], 0.95)),
    }
