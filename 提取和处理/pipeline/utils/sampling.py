#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
降采样工具模块

提供多种降采样算法，用于处理大规模点云数据。
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def farthest_point_sampling(
    points: np.ndarray, 
    n_samples: int, 
    seed: Optional[int] = None
) -> np.ndarray:
    """
    最远点采样（Farthest Point Sampling, FPS）算法。
    确保采样点在空间上均匀分布，防止血管几何结构被截断。
    
    优点：空间分布均匀，防止几何截断
    缺点：计算较慢（O(n*m)）
    
    参数:
        points: 点云坐标，形状 (N, 3)
        n_samples: 需要采样的点数
        seed: 随机种子，用于初始点选择
    
    返回:
        采样点的索引数组
    """
    n_points = len(points)
    
    # 如果需要采样的点数大于等于总点数，返回所有索引
    if n_samples >= n_points:
        return np.arange(n_points)
    
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
    
    # 初始化：随机选择第一个点
    sampled_indices = [np.random.randint(n_points)]
    distances = np.full(n_points, np.inf)
    
    # 迭代选择最远点
    for i in range(1, n_samples):
        # 计算所有点到最新采样点的距离
        last_point = points[sampled_indices[-1]]
        dists_to_last = np.sum((points - last_point) ** 2, axis=1)
        
        # 更新每个点到已采样点集的最小距离
        distances = np.minimum(distances, dists_to_last)
        
        # 选择距离最远的点
        farthest_idx = np.argmax(distances)
        sampled_indices.append(farthest_idx)
    
    return np.array(sampled_indices)


def random_sampling(
    points: np.ndarray, 
    n_samples: int, 
    seed: Optional[int] = None
) -> np.ndarray:
    """
    随机采样算法。
    快速简单，但可能导致空间分布不均。
    
    优点：速度快（O(n)）
    缺点：可能空间分布不均，有截断风险
    
    参数:
        points: 点云坐标，形状 (N, 3)
        n_samples: 需要采样的点数
        seed: 随机种子
    
    返回:
        采样点的索引数组
    """
    n_points = len(points)
    
    # 如果需要采样的点数大于等于总点数，返回所有索引
    if n_samples >= n_points:
        return np.arange(n_points)
    
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
    
    # 随机采样
    sampled_indices = np.random.choice(n_points, size=n_samples, replace=False)
    
    return sampled_indices


def stratified_sampling_by_distance(
    surface_df: pd.DataFrame,
    inner_df: pd.DataFrame,
    boundary_threshold: float = 2.0,
    boundary_core_ratio: Tuple[float, float] = (0.7, 0.3),
    target_total: int = 40000,
    sampling_method: str = "fps",
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于距离的分层降采样合并。
    
    采样策略：
    1. 优先级1：无条件保留所有壁面点
    2. 优先级2：内部点分层
       - 近壁层 (< boundary_threshold mm)：按指定比例优先分配预算
       - 核心层 (>= boundary_threshold mm)：次优先
    3. 动态补位：如果某层点数不足配额，将多余配额转给另一层
    4. 采样方法：
       - FPS：最远点采样，空间均匀，防止截断（推荐，但较慢）
       - Random：随机采样，速度快，但可能分布不均
    
    参数:
        surface_df: 壁面点数据
        inner_df: 内部点数据
        boundary_threshold: 近壁区阈值（mm）
        boundary_core_ratio: (近壁层比例, 核心层比例)，默认 (0.7, 0.3)
        target_total: 目标总点数
        sampling_method: 采样方法，"fps" 或 "random"
        seed: 随机种子
    
    返回:
        (merged_df, sampled_inner_df): 合并后的完整数据 和 降采样后的内部点数据
    """
    coord_cols = ['x', 'y', 'z']
    
    surface_coords = surface_df[coord_cols].values
    inner_coords = inner_df[coord_cols].values
    
    print(f"  壁面点数: {len(surface_df)}, 内部点数: {len(inner_df)}")
    
    # 优先级1：无条件保留所有壁面点
    n_surface = len(surface_df)
    remaining_budget = target_total - n_surface
    
    if remaining_budget <= 0:
        print(f"  ⚠️ 壁面点已达 {n_surface}，超过目标点数 {target_total}，仅保留壁面点")
        return surface_df.copy(), pd.DataFrame()
    
    print(f"  剩余预算: {remaining_budget} 个点")
    
    # 优先级2：使用 KDTree 计算内部点到壁面的距离
    tree = cKDTree(surface_coords)
    distances, _ = tree.query(inner_coords, k=1)
    
    # 分层：近壁层 vs 核心层
    boundary_mask = distances < boundary_threshold
    boundary_indices = np.where(boundary_mask)[0]
    core_indices = np.where(~boundary_mask)[0]
    
    n_boundary_available = len(boundary_indices)
    n_core_available = len(core_indices)
    
    print(f"  近壁层点数: {n_boundary_available}, 核心层点数: {n_core_available}")
    
    # 动态配额分配
    boundary_ratio, core_ratio = boundary_core_ratio
    n_boundary_quota = int(remaining_budget * boundary_ratio)
    n_core_quota = int(remaining_budget * core_ratio)
    
    # 动态补位逻辑
    if n_boundary_available < n_boundary_quota:
        # 近壁层不足，将多余配额转给核心层
        surplus = n_boundary_quota - n_boundary_available
        n_boundary_final = n_boundary_available
        n_core_final = min(n_core_available, n_core_quota + surplus)
        print(f"  ⚠️ 近壁层不足配额，转移 {surplus} 个配额到核心层")
    elif n_core_available < n_core_quota:
        # 核心层不足，将多余配额转给近壁层
        surplus = n_core_quota - n_core_available
        n_core_final = n_core_available
        n_boundary_final = min(n_boundary_available, n_boundary_quota + surplus)
        print(f"  ⚠️ 核心层不足配额，转移 {surplus} 个配额到近壁层")
    else:
        # 两层都充足
        n_boundary_final = n_boundary_quota
        n_core_final = n_core_quota
    
    # 确保不超过总预算
    total_allocated = n_boundary_final + n_core_final
    if total_allocated > remaining_budget:
        scale = remaining_budget / total_allocated
        n_boundary_final = int(n_boundary_final * scale)
        n_core_final = remaining_budget - n_boundary_final
    
    print(f"  最终分配: 近壁层 {n_boundary_final}/{n_boundary_available}, "
          f"核心层 {n_core_final}/{n_core_available}")
    
    # 选择采样函数
    if sampling_method.lower() == "fps":
        sampling_func = farthest_point_sampling
        method_name = "FPS"
    elif sampling_method.lower() == "random":
        sampling_func = random_sampling
        method_name = "随机"
    else:
        raise ValueError(f"不支持的采样方法: {sampling_method}，请使用 'fps' 或 'random'")
    
    # 进行采样
    sampled_indices = []
    
    # 采样近壁层
    if n_boundary_final > 0 and n_boundary_available > 0:
        boundary_coords = inner_coords[boundary_indices]
        if n_boundary_final < n_boundary_available:
            print(f"  执行近壁层{method_name}采样...")
            sampled_idx = sampling_func(boundary_coords, n_boundary_final, seed)
            sampled_boundary = boundary_indices[sampled_idx]
        else:
            sampled_boundary = boundary_indices
        sampled_indices.append(sampled_boundary)
    
    # 采样核心层
    if n_core_final > 0 and n_core_available > 0:
        core_coords = inner_coords[core_indices]
        if n_core_final < n_core_available:
            print(f"  执行核心层{method_name}采样...")
            sampled_idx = sampling_func(core_coords, n_core_final, seed)
            sampled_core = core_indices[sampled_idx]
        else:
            sampled_core = core_indices
        sampled_indices.append(sampled_core)
    
    # 合并采样的内部点索引
    if sampled_indices:
        sampled_inner_indices = np.concatenate(sampled_indices)
        sampled_inner_df = inner_df.iloc[sampled_inner_indices].copy()
    else:
        sampled_inner_df = pd.DataFrame()
    
    # 对齐列
    all_cols = list(surface_df.columns)
    for col in sampled_inner_df.columns:
        if col not in all_cols:
            all_cols.append(col)
    
    surface_df_aligned = surface_df.reindex(columns=all_cols)
    sampled_inner_df_aligned = sampled_inner_df.reindex(columns=all_cols)
    
    # 合并并打乱
    merged = pd.concat([surface_df_aligned, sampled_inner_df_aligned], ignore_index=True)
    if seed is not None:
        merged = merged.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        merged = merged.sample(frac=1).reset_index(drop=True)
    
    print(f"  ✅ 合并后总点数: {len(merged)} (目标: {target_total})")
    print(f"  预算利用率: {len(merged)/target_total*100:.1f}%")
    
    return merged, sampled_inner_df


if __name__ == "__main__":
    # 测试
    print("采样工具模块测试")
    print("=" * 50)
    
    # 生成测试数据
    np.random.seed(42)
    test_points = np.random.randn(1000, 3)
    
    # 测试 FPS
    print("\nFPS 采样测试:")
    fps_indices = farthest_point_sampling(test_points, 100, seed=42)
    print(f"  采样点数: {len(fps_indices)}")
    
    # 测试随机采样
    print("\n随机采样测试:")
    random_indices = random_sampling(test_points, 100, seed=42)
    print(f"  采样点数: {len(random_indices)}")
