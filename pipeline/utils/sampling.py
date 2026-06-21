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


def hybrid_sampling(
    points: np.ndarray,
    n_samples: int,
    fps_ratio: float = 0.2,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    混合采样：先 FPS 后随机
    
    策略：
    1. 先用 FPS 采样 fps_ratio 比例的点，确保空间覆盖（保护分支血管）
    2. 再从剩余点中随机采样，增加数据多样性
    
    优点：兼顾空间覆盖和数据多样性
    
    参数:
        points: 点云坐标，形状 (N, 3)
        n_samples: 需要采样的点数
        fps_ratio: FPS 采样占比（默认 0.2 = 20%，可调整为 0.3 或 0.4）
        seed: 随机种子
    
    返回:
        采样点的索引数组
    """
    n_points = len(points)
    
    # 如果需要采样的点数大于等于总点数，返回所有索引
    if n_samples >= n_points:
        return np.arange(n_points)
    
    # 计算 FPS 和随机采样的点数
    n_fps = int(n_samples * fps_ratio)
    n_random = n_samples - n_fps
    
    # 确保至少有 1 个 FPS 点（如果 n_samples > 0）
    if n_fps == 0 and n_samples > 0:
        n_fps = 1
        n_random = n_samples - 1
    
    # 第一步：FPS 确保空间覆盖
    fps_indices = farthest_point_sampling(points, n_fps, seed)
    
    # 第二步：从剩余点中随机采样
    remaining_mask = np.ones(n_points, dtype=bool)
    remaining_mask[fps_indices] = False
    remaining_indices = np.where(remaining_mask)[0]
    
    if n_random > 0 and len(remaining_indices) > 0:
        n_random_actual = min(n_random, len(remaining_indices))
        random_indices_local = random_sampling(
            points[remaining_indices], n_random_actual, seed
        )
        random_indices = remaining_indices[random_indices_local]
        return np.concatenate([fps_indices, random_indices])
    else:
        return fps_indices


def _build_sampling_func(sampling_method: str, fps_ratio: float):
    """根据采样方法名构建采样函数和显示名。"""
    method = sampling_method.lower()
    if method == "fps":
        return farthest_point_sampling, "FPS"
    elif method == "random":
        return random_sampling, "随机"
    elif method == "hybrid":
        func = lambda pts, n, s: hybrid_sampling(pts, n, fps_ratio, s)
        return func, f"混合(FPS {fps_ratio*100:.0f}%)"
    else:
        raise ValueError(f"不支持的采样方法: {sampling_method}，请使用 'fps', 'random' 或 'hybrid'")


def stratified_sampling_by_distance(
    surface_df: pd.DataFrame,
    inner_df: pd.DataFrame,
    boundary_threshold: float = 2.0,
    boundary_core_ratio: Tuple[float, float] = (0.7, 0.3),
    target_total: int = 15000,
    wall_max_points: int = 10000,
    sampling_method: str = "hybrid",
    fps_ratio: float = 0.5,
    allow_core_fallback: bool = True,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于距离的分层降采样合并。

    采样策略（三层）：
    1. 壁面层：保留 min(原始壁面点数, wall_max_points) 个点，
       超出上限时用指定采样方法降采样。
    2. 内部-近壁层 (< boundary_threshold mm)：占内部预算的 boundary_core_ratio[0]
    3. 内部-核心层 (>= boundary_threshold mm)：占内部预算的 boundary_core_ratio[1]
    内部预算 = target_total - 实际壁面点数（动态分配）。
    默认各层若点数不足配额，多余配额自动转给其他层。
    allow_core_fallback=False 时，近壁层不足不会用核心层补齐。

    参数:
        surface_df: 壁面点数据
        inner_df: 内部点数据
        boundary_threshold: 近壁区阈值（mm）
        boundary_core_ratio: 内部点 (近壁层比例, 核心层比例)，默认 (0.7, 0.3)
        target_total: 目标总点数
        wall_max_points: 壁面点上限，超出时降采样
        sampling_method: 采样方法，"fps", "random" 或 "hybrid"
        fps_ratio: 混合采样时 FPS 的占比（默认 0.5），仅当 method="hybrid" 时生效
        allow_core_fallback: 近壁层不足时是否允许核心层补齐预算
        seed: 随机种子

    返回:
        (merged_df, sampled_inner_df): 合并后的完整数据 和 降采样后的内部点数据
    """
    coord_cols = ['x', 'y', 'z']

    surface_coords = surface_df[coord_cols].values
    inner_coords = inner_df[coord_cols].values
    n_surface_raw = len(surface_df)

    print(f"  壁面点数: {n_surface_raw}, 内部点数: {len(inner_df)}")

    sampling_func, method_name = _build_sampling_func(sampling_method, fps_ratio)

    # ── 第 1 层：壁面点 ──
    if n_surface_raw > wall_max_points:
        print(f"  壁面点 {n_surface_raw} 超过上限 {wall_max_points}，执行壁面{method_name}采样...")
        sampled_wall_idx = sampling_func(surface_coords, wall_max_points, seed)
        sampled_surface_df = surface_df.iloc[sampled_wall_idx].copy()
        surface_coords_used = sampled_surface_df[coord_cols].values
        n_surface = wall_max_points
    else:
        sampled_surface_df = surface_df.copy()
        surface_coords_used = surface_coords
        n_surface = n_surface_raw

    remaining_budget = target_total - n_surface
    print(f"  实际壁面点: {n_surface}, 内部点预算: {remaining_budget}")

    if remaining_budget <= 0:
        print(f"  ⚠️ 壁面点已占满总预算，无内部点预算")
        merged = sampled_surface_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        return merged, pd.DataFrame()

    # ── 第 2、3 层：内部点分层 ──
    tree = cKDTree(surface_coords_used)
    distances, _ = tree.query(inner_coords, k=1)

    boundary_mask = distances < boundary_threshold
    boundary_indices = np.where(boundary_mask)[0]
    core_indices = np.where(~boundary_mask)[0]

    n_boundary_available = len(boundary_indices)
    n_core_available = len(core_indices)

    print(f"  近壁层点数: {n_boundary_available}, 核心层点数: {n_core_available}")

    boundary_ratio, core_ratio = boundary_core_ratio
    n_boundary_quota = int(remaining_budget * boundary_ratio)
    n_core_quota = remaining_budget - n_boundary_quota

    if n_boundary_available < n_boundary_quota:
        surplus = n_boundary_quota - n_boundary_available
        n_boundary_final = n_boundary_available
        if allow_core_fallback:
            n_core_final = min(n_core_available, n_core_quota + surplus)
            print(f"  ⚠️ 近壁层不足配额，转移 {surplus} 个配额到核心层")
        else:
            n_core_final = min(n_core_available, n_core_quota)
            print(f"  ⚠️ 近壁层不足配额，未启用核心层回填，空余 {surplus} 个配额")
    elif n_core_available < n_core_quota:
        surplus = n_core_quota - n_core_available
        n_core_final = n_core_available
        n_boundary_final = min(n_boundary_available, n_boundary_quota + surplus)
        print(f"  ⚠️ 核心层不足配额，转移 {surplus} 个配额到近壁层")
    else:
        n_boundary_final = n_boundary_quota
        n_core_final = n_core_quota

    total_allocated = n_boundary_final + n_core_final
    if total_allocated > remaining_budget:
        scale = remaining_budget / total_allocated
        n_boundary_final = int(n_boundary_final * scale)
        n_core_final = remaining_budget - n_boundary_final

    print(f"  最终分配: 壁面 {n_surface}, "
          f"近壁层 {n_boundary_final}/{n_boundary_available}, "
          f"核心层 {n_core_final}/{n_core_available}")

    sampled_indices = []

    if n_boundary_final > 0 and n_boundary_available > 0:
        boundary_coords = inner_coords[boundary_indices]
        if n_boundary_final < n_boundary_available:
            print(f"  执行近壁层{method_name}采样...")
            sampled_idx = sampling_func(boundary_coords, n_boundary_final, seed)
            sampled_boundary = boundary_indices[sampled_idx]
        else:
            sampled_boundary = boundary_indices
        sampled_indices.append(sampled_boundary)

    if n_core_final > 0 and n_core_available > 0:
        core_coords = inner_coords[core_indices]
        if n_core_final < n_core_available:
            print(f"  执行核心层{method_name}采样...")
            sampled_idx = sampling_func(core_coords, n_core_final, seed)
            sampled_core = core_indices[sampled_idx]
        else:
            sampled_core = core_indices
        sampled_indices.append(sampled_core)

    if sampled_indices:
        sampled_inner_indices = np.concatenate(sampled_indices)
        sampled_inner_df = inner_df.iloc[sampled_inner_indices].copy()
    else:
        sampled_inner_df = pd.DataFrame()

    all_cols = list(sampled_surface_df.columns)
    for col in sampled_inner_df.columns:
        if col not in all_cols:
            all_cols.append(col)

    surface_df_aligned = sampled_surface_df.reindex(columns=all_cols)
    sampled_inner_df_aligned = sampled_inner_df.reindex(columns=all_cols)

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
    
    # 测试混合采样
    print("\n混合采样测试 (FPS 20%):")
    hybrid_indices_20 = hybrid_sampling(test_points, 100, fps_ratio=0.2, seed=42)
    print(f"  采样点数: {len(hybrid_indices_20)}")
    print(f"  FPS 点数: {int(100 * 0.2)}, 随机点数: {100 - int(100 * 0.2)}")
    
    print("\n混合采样测试 (FPS 30%):")
    hybrid_indices_30 = hybrid_sampling(test_points, 100, fps_ratio=0.3, seed=42)
    print(f"  采样点数: {len(hybrid_indices_30)}")
    print(f"  FPS 点数: {int(100 * 0.3)}, 随机点数: {100 - int(100 * 0.3)}")
    
    # 验证混合采样的空间覆盖
    print("\n空间覆盖分析:")
    from scipy.spatial import cKDTree
    
    # 计算各种采样方法的最近邻距离分布
    def analyze_coverage(indices, name):
        sampled = test_points[indices]
        tree = cKDTree(sampled)
        dists, _ = tree.query(test_points, k=1)
        print(f"  {name}: 最大距离={dists.max():.3f}, 平均距离={dists.mean():.3f}")
    
    analyze_coverage(fps_indices, "FPS")
    analyze_coverage(random_indices, "随机")
    analyze_coverage(hybrid_indices_20, "混合(20%FPS)")
    analyze_coverage(hybrid_indices_30, "混合(30%FPS)")
