#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
几何计算工具模块

提供点云几何特征计算的工具函数。
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np


def compute_centroid(points: np.ndarray) -> np.ndarray:
    """
    计算点云质心
    
    参数:
        points: 点云坐标，形状 (N, 3)
    
    返回:
        质心坐标，形状 (3,)
    """
    return np.mean(points, axis=0)


def compute_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算点云的轴对齐包围盒
    
    参数:
        points: 点云坐标，形状 (N, 3)
    
    返回:
        (min_coords, max_coords): 包围盒的最小和最大坐标
    """
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    return min_coords, max_coords


def normalize_coordinates(
    points: np.ndarray, 
    method: str = "center_scale",
    centroid: Optional[np.ndarray] = None,
    scale_factor: Optional[float] = None
) -> Tuple[np.ndarray, dict]:
    """
    归一化点云坐标
    
    参数:
        points: 点云坐标，形状 (N, 3)
        method: 归一化方法
            - "center_scale": 中心化并缩放到单位球
            - "min_max": 归一化到 [0, 1] 范围
        centroid: 预设质心（如果为 None，则计算）
        scale_factor: 预设缩放因子（如果为 None，则计算）
    
    返回:
        (normalized_points, params): 归一化后的点云和归一化参数
    """
    if method == "center_scale":
        # 计算或使用预设质心
        if centroid is None:
            centroid = compute_centroid(points)
        
        # 中心化
        centered = points - centroid
        
        # 计算或使用预设缩放因子
        if scale_factor is None:
            # 使用最大距离作为缩放因子
            distances = np.linalg.norm(centered, axis=1)
            scale_factor = np.max(distances)
            if scale_factor < 1e-10:
                scale_factor = 1.0
        
        # 缩放
        normalized = centered / scale_factor
        
        params = {
            "method": "center_scale",
            "centroid": centroid.tolist(),
            "scale_factor": float(scale_factor),
        }
        
    elif method == "min_max":
        min_coords, max_coords = compute_bounding_box(points)
        ranges = max_coords - min_coords
        ranges[ranges < 1e-10] = 1.0  # 防止除以零
        
        normalized = (points - min_coords) / ranges
        
        params = {
            "method": "min_max",
            "min_coords": min_coords.tolist(),
            "max_coords": max_coords.tolist(),
        }
    else:
        raise ValueError(f"不支持的归一化方法: {method}")
    
    return normalized, params


def denormalize_coordinates(
    points: np.ndarray,
    params: dict
) -> np.ndarray:
    """
    反归一化点云坐标
    
    参数:
        points: 归一化后的点云坐标，形状 (N, 3)
        params: 归一化参数
    
    返回:
        原始坐标
    """
    method = params.get("method")
    
    if method == "center_scale":
        centroid = np.array(params["centroid"])
        scale_factor = params["scale_factor"]
        return points * scale_factor + centroid
        
    elif method == "min_max":
        min_coords = np.array(params["min_coords"])
        max_coords = np.array(params["max_coords"])
        ranges = max_coords - min_coords
        return points * ranges + min_coords
    else:
        raise ValueError(f"不支持的归一化方法: {method}")


def compute_point_normals(
    points: np.ndarray, 
    k: int = 10
) -> np.ndarray:
    """
    使用 PCA 估计点云的法向量
    
    参数:
        points: 点云坐标，形状 (N, 3)
        k: 用于估计法向量的邻居数
    
    返回:
        法向量数组，形状 (N, 3)
    """
    from scipy.spatial import cKDTree
    
    tree = cKDTree(points)
    _, indices = tree.query(points, k=k)
    
    normals = np.zeros_like(points)
    
    for i, neighbors in enumerate(indices):
        neighbor_points = points[neighbors]
        centered = neighbor_points - neighbor_points.mean(axis=0)
        
        # SVD 分解
        _, _, vh = np.linalg.svd(centered)
        
        # 最小奇异值对应的向量即为法向量
        normals[i] = vh[-1]
    
    return normals


if __name__ == "__main__":
    # 测试
    print("几何工具模块测试")
    print("=" * 50)
    
    # 生成测试数据
    np.random.seed(42)
    test_points = np.random.randn(100, 3) * 50 + np.array([100, 200, 300])
    
    print(f"\n原始点云范围:")
    min_c, max_c = compute_bounding_box(test_points)
    print(f"  min: {min_c}")
    print(f"  max: {max_c}")
    
    # 测试归一化
    print("\ncenter_scale 归一化:")
    norm_points, params = normalize_coordinates(test_points, method="center_scale")
    print(f"  质心: {params['centroid']}")
    print(f"  缩放因子: {params['scale_factor']:.4f}")
    
    min_c, max_c = compute_bounding_box(norm_points)
    print(f"  归一化后范围: [{min_c.min():.4f}, {max_c.max():.4f}]")
    
    # 测试反归一化
    restored = denormalize_coordinates(norm_points, params)
    error = np.max(np.abs(restored - test_points))
    print(f"  反归一化误差: {error:.2e}")
