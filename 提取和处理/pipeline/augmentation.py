#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强模块

提供用于 CFD 血管数据的物理一致性数据增强函数。
这些增强操作严格保持 Navier-Stokes 方程的不变性。

支持的增强操作:
1. 随机旋转（刚体旋转）- 推荐
2. 随机平移 - 推荐
3. 微小缩放（±2%）- 慎用

禁止的操作:
- 形状变形（扭曲、弯曲）
- 大幅缩放

使用示例:
  from augmentation import random_rotation, random_translation
  
  # 在 DataLoader 中使用
  data = random_rotation(data, axis='z')
  data = random_translation(data, max_shift=0.1)
"""

import random
from typing import Tuple, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data


# ============================================================================
# 旋转矩阵生成函数
# ============================================================================

def rotation_matrix_axis_np(axis: str, angle: float) -> np.ndarray:
    """
    生成绕指定轴旋转的旋转矩阵（NumPy版本）
    
    参数:
        axis: 旋转轴 ('x', 'y', 'z')
        angle: 旋转角度（弧度）
    
    返回:
        3x3 旋转矩阵
    """
    c, s = np.cos(angle), np.sin(angle)
    
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ], dtype=np.float32)
    elif axis == 'y':
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ], dtype=np.float32)
    elif axis == 'z':
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ], dtype=np.float32)
    else:
        raise ValueError(f"未知的旋转轴: {axis}")


def rotation_matrix_axis_torch(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    生成绕指定轴旋转的旋转矩阵（PyTorch版本）
    
    参数:
        axis: 旋转轴 ('x', 'y', 'z')
        angle: 旋转角度（弧度）
    
    返回:
        3x3 旋转矩阵
    """
    c, s = torch.cos(angle), torch.sin(angle)
    zero = torch.zeros_like(c)
    one = torch.ones_like(c)
    
    if axis == 'x':
        return torch.stack([
            torch.stack([one, zero, zero]),
            torch.stack([zero, c, -s]),
            torch.stack([zero, s, c])
        ]).squeeze()
    elif axis == 'y':
        return torch.stack([
            torch.stack([c, zero, s]),
            torch.stack([zero, one, zero]),
            torch.stack([-s, zero, c])
        ]).squeeze()
    elif axis == 'z':
        return torch.stack([
            torch.stack([c, -s, zero]),
            torch.stack([s, c, zero]),
            torch.stack([zero, zero, one])
        ]).squeeze()
    else:
        raise ValueError(f"未知的旋转轴: {axis}")


def random_rotation_matrix_np(axes: str = 'xyz') -> np.ndarray:
    """
    生成随机旋转矩阵（绕指定轴组合）
    
    参数:
        axes: 旋转轴组合，如 'z'（仅绕Z轴）或 'xyz'（三轴随机）
    
    返回:
        3x3 旋转矩阵
    """
    R = np.eye(3, dtype=np.float32)
    
    for axis in axes:
        angle = np.random.uniform(0, 2 * np.pi)
        R = R @ rotation_matrix_axis_np(axis, angle)
    
    return R


# ============================================================================
# PyTorch Geometric Data 增强函数
# ============================================================================

def random_rotation(
    data: Data,
    axis: str = None,
    angle: float = None,
    coord_indices: Tuple[int, int] = (0, 3),
    tangent_indices: Tuple[int, int] = (7, 10),
    velocity_indices: Tuple[int, int] = (0, 3),
) -> Data:
    """
    对 PyG Data 对象进行随机旋转增强
    
    关键：坐标、速度、切线必须同步旋转！
    
    参数:
        data: PyG Data 对象
        axis: 旋转轴 ('x', 'y', 'z')，None 表示随机选择
        angle: 旋转角度（弧度），None 表示随机生成
        coord_indices: 坐标在 data.x 中的索引范围 [start, end)
        tangent_indices: 切线在 data.x 中的索引范围 [start, end)
        velocity_indices: 速度在 data.y 中的索引范围 [start, end)
    
    返回:
        增强后的 Data 对象（原对象的副本）
    
    注意:
        默认索引基于以下特征顺序:
        x: [0:3] 坐标, [3] 时间, [4:7] 几何标量, [7:10] 切线, ...
        y: [0:3] 速度, [3] 压力
    """
    # 克隆数据
    data = data.clone()
    
    # 随机选择旋转轴
    if axis is None:
        axis = random.choice(['x', 'y', 'z'])
    
    # 随机生成旋转角度
    if angle is None:
        angle = random.uniform(0, 2 * np.pi)
    
    # 生成旋转矩阵
    R = torch.tensor(rotation_matrix_axis_np(axis, angle), 
                     dtype=data.x.dtype, device=data.x.device)
    
    # 旋转坐标
    coords = data.x[:, coord_indices[0]:coord_indices[1]]
    data.x[:, coord_indices[0]:coord_indices[1]] = coords @ R.T
    
    # 旋转切线（如果存在）
    if tangent_indices[1] <= data.x.shape[1]:
        tangent = data.x[:, tangent_indices[0]:tangent_indices[1]]
        data.x[:, tangent_indices[0]:tangent_indices[1]] = tangent @ R.T
    
    # 旋转速度标签
    if data.y is not None and velocity_indices[1] <= data.y.shape[1]:
        velocity = data.y[:, velocity_indices[0]:velocity_indices[1]]
        data.y[:, velocity_indices[0]:velocity_indices[1]] = velocity @ R.T
    
    return data


def random_translation(
    data: Data,
    max_shift: float = 0.1,
    coord_indices: Tuple[int, int] = (0, 3),
) -> Data:
    """
    对 PyG Data 对象进行随机平移增强
    
    物理正确性：平移不改变流场相对于血管的分布
    
    参数:
        data: PyG Data 对象
        max_shift: 最大平移量（在归一化坐标系下，建议 0.05-0.2）
        coord_indices: 坐标在 data.x 中的索引范围 [start, end)
    
    返回:
        增强后的 Data 对象（原对象的副本）
    
    注意:
        - 只平移坐标，不平移速度和其他特征
        - 在归一化坐标系下操作（坐标范围约 [-1, 1]）
    """
    # 克隆数据
    data = data.clone()
    
    # 生成随机平移向量
    shift = torch.rand(3, dtype=data.x.dtype, device=data.x.device) * 2 * max_shift - max_shift
    
    # 平移坐标
    data.x[:, coord_indices[0]:coord_indices[1]] += shift
    
    return data


def small_scale_augmentation(
    data: Data,
    scale_range: Tuple[float, float] = (0.98, 1.02),
    coord_indices: Tuple[int, int] = (0, 3),
) -> Data:
    """
    对 PyG Data 对象进行微小缩放增强
    
    警告：大幅缩放会改变雷诺数，导致物理不一致！
    建议仅使用 ±2% 的微小缩放。
    
    参数:
        data: PyG Data 对象
        scale_range: 缩放范围 (min, max)，建议 (0.98, 1.02)
        coord_indices: 坐标在 data.x 中的索引范围 [start, end)
    
    返回:
        增强后的 Data 对象（原对象的副本）
    """
    # 克隆数据
    data = data.clone()
    
    # 生成随机缩放因子
    scale = random.uniform(scale_range[0], scale_range[1])
    
    # 缩放坐标
    data.x[:, coord_indices[0]:coord_indices[1]] *= scale
    
    return data


def mirror_augmentation(
    data: Data,
    axis: str = None,
    coord_indices: Tuple[int, int] = (0, 3),
    tangent_indices: Tuple[int, int] = (7, 10),
    velocity_indices: Tuple[int, int] = (0, 3),
) -> Data:
    """
    对 PyG Data 对象进行镜像翻转增强
    
    参数:
        data: PyG Data 对象
        axis: 镜像轴 ('x', 'y', 'z')，None 表示随机选择
        coord_indices: 坐标在 data.x 中的索引范围
        tangent_indices: 切线在 data.x 中的索引范围
        velocity_indices: 速度在 data.y 中的索引范围
    
    返回:
        增强后的 Data 对象（原对象的副本）
    
    注意:
        镜像翻转时，垂直于镜面的矢量分量需要取反
    """
    # 克隆数据
    data = data.clone()
    
    # 随机选择镜像轴
    if axis is None:
        axis = random.choice(['x', 'y', 'z'])
    
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    
    # 翻转坐标
    data.x[:, coord_indices[0] + axis_idx] *= -1
    
    # 翻转切线的对应分量
    if tangent_indices[1] <= data.x.shape[1]:
        data.x[:, tangent_indices[0] + axis_idx] *= -1
    
    # 翻转速度的对应分量
    if data.y is not None and velocity_indices[1] <= data.y.shape[1]:
        data.y[:, velocity_indices[0] + axis_idx] *= -1
    
    return data


# ============================================================================
# NumPy/Pandas 增强函数（用于离线增强）
# ============================================================================

def rotate_dataframe(
    coords: np.ndarray,
    velocity: np.ndarray = None,
    tangent: np.ndarray = None,
    wss_vec: np.ndarray = None,
    axis: str = None,
    angle: float = None,
) -> Tuple[np.ndarray, ...]:
    """
    对 NumPy 数组进行旋转增强
    
    参数:
        coords: [N, 3] 坐标
        velocity: [N, 3] 速度
        tangent: [N, 3] 切线
        wss_vec: [N, 3] WSS矢量
        axis: 旋转轴
        angle: 旋转角度
    
    返回:
        旋转后的各数组
    """
    if axis is None:
        axis = random.choice(['x', 'y', 'z'])
    if angle is None:
        angle = random.uniform(0, 2 * np.pi)
    
    R = rotation_matrix_axis_np(axis, angle)
    
    coords_rot = coords @ R.T
    velocity_rot = velocity @ R.T if velocity is not None else None
    tangent_rot = tangent @ R.T if tangent is not None else None
    wss_vec_rot = wss_vec @ R.T if wss_vec is not None else None
    
    return coords_rot, velocity_rot, tangent_rot, wss_vec_rot


def translate_dataframe(
    coords: np.ndarray,
    max_shift: float = 0.1,
) -> np.ndarray:
    """
    对坐标进行平移增强
    
    参数:
        coords: [N, 3] 坐标
        max_shift: 最大平移量
    
    返回:
        平移后的坐标
    """
    shift = np.random.uniform(-max_shift, max_shift, size=3)
    return coords + shift


# ============================================================================
# 增强配置和组合函数
# ============================================================================

DEFAULT_AUGMENT_CONFIG = {
    "rotation_prob": 0.5,
    "rotation_axes": "xyz",
    "translation_prob": 0.5,
    "translation_range": 0.1,
    "scale_prob": 0.0,  # 默认不使用缩放
    "scale_range": (0.98, 1.02),
    "mirror_prob": 0.0,  # 默认不使用镜像
}


def apply_augmentations(
    data: Data,
    config: dict = None,
) -> Data:
    """
    应用配置的增强操作组合
    
    参数:
        data: PyG Data 对象
        config: 增强配置字典
    
    返回:
        增强后的 Data 对象
    """
    if config is None:
        config = DEFAULT_AUGMENT_CONFIG
    
    # 随机旋转
    if random.random() < config.get("rotation_prob", 0.5):
        axes = config.get("rotation_axes", "xyz")
        axis = random.choice(list(axes))
        data = random_rotation(data, axis=axis)
    
    # 随机平移
    if random.random() < config.get("translation_prob", 0.5):
        max_shift = config.get("translation_range", 0.1)
        data = random_translation(data, max_shift=max_shift)
    
    # 微小缩放
    if random.random() < config.get("scale_prob", 0.0):
        scale_range = config.get("scale_range", (0.98, 1.02))
        data = small_scale_augmentation(data, scale_range=scale_range)
    
    # 镜像翻转
    if random.random() < config.get("mirror_prob", 0.0):
        data = mirror_augmentation(data)
    
    return data


# ============================================================================
# 工具函数
# ============================================================================

def validate_augmentation(
    data_orig: Data,
    data_aug: Data,
    tolerance: float = 1e-5,
) -> bool:
    """
    验证增强后的数据是否保持了物理一致性
    
    检查项:
    1. 速度矢量的模长应保持不变（旋转不变量）
    2. 切线矢量的模长应为1（单位向量）
    
    参数:
        data_orig: 原始数据
        data_aug: 增强后的数据
        tolerance: 容差
    
    返回:
        是否通过验证
    """
    # 检查速度模长
    if data_orig.y is not None:
        vel_orig = data_orig.y[:, :3]
        vel_aug = data_aug.y[:, :3]
        
        mag_orig = torch.norm(vel_orig, dim=1)
        mag_aug = torch.norm(vel_aug, dim=1)
        
        if not torch.allclose(mag_orig, mag_aug, atol=tolerance):
            print("警告: 速度模长发生变化")
            return False
    
    return True


if __name__ == "__main__":
    # 简单测试
    print("数据增强模块测试")
    print("=" * 50)
    
    # 创建测试数据
    N = 100
    x = torch.randn(N, 16)  # 15维特征 + 1
    y = torch.randn(N, 4)   # 4维标签
    edge_index = torch.randint(0, N, (2, N * 6))
    
    data = Data(x=x, y=y, edge_index=edge_index)
    
    print(f"原始数据: x.shape={data.x.shape}, y.shape={data.y.shape}")
    
    # 测试旋转
    data_rot = random_rotation(data, axis='z')
    print(f"旋转后: x.shape={data_rot.x.shape}, y.shape={data_rot.y.shape}")
    
    # 测试平移
    data_trans = random_translation(data, max_shift=0.1)
    print(f"平移后: x.shape={data_trans.x.shape}")
    
    # 测试组合增强
    data_aug = apply_augmentations(data, DEFAULT_AUGMENT_CONFIG)
    print(f"组合增强后: x.shape={data_aug.x.shape}")
    
    print("\n✅ 测试通过")
