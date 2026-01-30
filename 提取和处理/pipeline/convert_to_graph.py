#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图数据转换模块

将归一化后的 CSV 数据转换为 PyTorch Geometric 图数据格式。

功能:
1. 读取归一化后的特征数据
2. 构建 KNN 图结构
3. 组装输入特征和目标输出
4. 保存为 .pt 文件

输入特征 (15维，移除 BC_Flag):
- 坐标: x, y, z (3)
- 时间: t_norm (1)
- 几何特征: Abscissa, NormRadius, Curvature, Tangent_X/Y/Z (6)
- 边界标志: is_wall (1)
- 边界条件: BC_Inlet, BC_O1~O4 (5)

目标输出 (4维):
- 速度: u, v, w (3)
- 压力: p (1)

使用示例:
  # 处理单个病例
  python convert_to_graph.py --case ZHANG_CHUN
  
  # 处理所有病例
  python convert_to_graph.py
"""

import argparse
import re
import time
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# 导入配置
from config import (
    DATA_ROOT,
    NORMALIZED_DIR,
    GRAPHS_DIR,
    GRAPH_CONFIG,
    get_case_dirs,
)


def extract_time_step(filename: str) -> Optional[int]:
    """
    从文件名中提取时间步编号。
    """
    matches = re.findall(r'(\d+)', filename)
    if matches:
        return int(matches[-1])
    return None


def build_graph_from_csv(
    file_path: Path, 
    t_norm: float, 
    k: int = 6
) -> Data:
    """
    读取 CSV 文件并构建图结构。
    
    参数:
        file_path: CSV 文件路径
        t_norm: 归一化的时间值 [0, 1]
        k: KNN 邻居数
    
    返回:
        PyG Data 对象
    """
    df = pd.read_csv(file_path)
    
    # 1. 提取坐标
    coords = df[['x', 'y', 'z']].values
    
    # 2. 构建输入特征 (15维，无 BC_Flag)
    # (x, y, z, t) + 几何特征(6) + is_wall(1) + 边界条件(5)
    
    # 时间特征
    t_feat = np.full((coords.shape[0], 1), t_norm)
    
    # 几何特征
    geom_feats = df[['Abscissa', 'NormRadius', 'Curvature', 
                     'Tangent_X', 'Tangent_Y', 'Tangent_Z']].values
    
    # 边界标志 (仅 is_wall，无 BC_Flag)
    is_wall = df[['is_wall']].values
    
    # 边界条件 (5个值)
    bc_feats = df[['BC_Inlet', 'BC_O1', 'BC_O2', 'BC_O3', 'BC_O4']].values
    
    # 拼接所有特征: [x, y, z, t, Abscissa, NormRadius, Curvature, Tx, Ty, Tz, is_wall, BC_Inlet, BC_O1~O4]
    x = np.hstack([coords, t_feat, geom_feats, is_wall, bc_feats])
    x = torch.from_numpy(x).float()
    
    # 3. 提取目标输出 (4维: u, v, w, p)
    y = df[['u', 'v', 'w', 'p']].values
    y = torch.from_numpy(y).float()
    
    # 4. 构建边索引 (使用 KNN)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    _, indices = nbrs.kneighbors(coords)
    
    # 转换为 PyG 的 edge_index 格式 [2, num_edges]
    # indices 的第一列是节点自身，忽略它
    row = np.repeat(np.arange(coords.shape[0]), k)
    col = indices[:, 1:].flatten()
    
    edge_index = torch.from_numpy(np.stack([row, col])).long()
    
    # 创建 PyG Data 对象
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data


def process_case(
    case_dir: Path,
    input_subdir: str = None,
    output_subdir: str = None,
    k: int = 6,
) -> bool:
    """
    处理单个病例的所有时间步。
    
    参数:
        case_dir: 病例目录
        input_subdir: 输入子目录
        output_subdir: 输出子目录
        k: KNN 邻居数
    
    返回:
        是否成功
    """
    case_dir = Path(case_dir)
    case_name = case_dir.name
    
    if input_subdir is None:
        input_subdir = NORMALIZED_DIR
    if output_subdir is None:
        output_subdir = GRAPHS_DIR
    
    input_dir = case_dir / input_subdir
    output_dir = case_dir / output_subdir
    
    if not input_dir.exists():
        print(f"  ❌ 输入目录不存在: {input_subdir}")
        return False
    
    csv_files = sorted(list(input_dir.glob("result_features_*.csv")))
    if not csv_files:
        print(f"  ❌ 未找到特征文件")
        return False
    
    # 提取时间步信息
    file_step_pairs = []
    steps = []
    for f in csv_files:
        step = extract_time_step(f.name)
        if step is not None:
            steps.append(step)
            file_step_pairs.append((f, step))
    
    if not steps:
        print(f"  ❌ 无法从文件名解析时间步信息")
        return False
    
    # 计算时间归一化范围
    min_step, max_step = min(steps), max(steps)
    total_range = max_step - min_step
    if total_range == 0:
        total_range = 1  # 防止除以0
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  📁 找到 {len(file_step_pairs)} 个时间步")
    print(f"  🔗 KNN 邻居数: {k}")
    
    # 处理每个时间步
    success_count = 0
    for csv_file, step in tqdm(file_step_pairs, desc=f"处理 {case_name}", leave=False):
        try:
            t_norm = (step - min_step) / total_range
            data = build_graph_from_csv(csv_file, t_norm, k=k)
            
            # 保存为 .pt 文件
            out_name = csv_file.stem + ".pt"
            torch.save(data, output_dir / out_name)
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 处理 {csv_file.name} 失败: {e}")
    
    print(f"  ✅ 完成: {success_count}/{len(file_step_pairs)} 个文件")
    return success_count > 0


def process_all_cases(
    data_root: Path = None,
    target_case: Optional[str] = None,
    input_subdir: str = None,
    output_subdir: str = None,
    k: int = None,
) -> None:
    """批量处理所有病例"""
    if data_root is None:
        data_root = DATA_ROOT
    else:
        data_root = Path(data_root)
    
    if input_subdir is None:
        input_subdir = NORMALIZED_DIR
    if output_subdir is None:
        output_subdir = GRAPHS_DIR
    if k is None:
        k = GRAPH_CONFIG["k_neighbors"]
    
    # 获取病例目录
    case_dirs = get_case_dirs(data_root)
    
    # 过滤指定病例
    if target_case:
        target_std = target_case.replace(' ', '_').replace('-', '_').upper()
        case_dirs = [
            d for d in case_dirs 
            if d.name.replace(' ', '_').replace('-', '_').upper() == target_std
        ]
    
    if not case_dirs:
        if target_case:
            print(f"❌ 未找到病例: {target_case}")
        else:
            print(f"❌ 未找到任何病例")
        return
    
    print("🚀 图数据转换")
    print("=" * 50)
    print(f"📁 数据根目录: {data_root}")
    print(f"📂 输入子目录: {input_subdir}")
    print(f"📂 输出子目录: {output_subdir}")
    print(f"🔗 KNN 邻居数: {k}")
    print(f"📊 待处理病例数: {len(case_dirs)}")
    
    total_start = time.time()
    ok = 0
    
    for idx, case_dir in enumerate(case_dirs, 1):
        try:
            rel_path = case_dir.relative_to(data_root)
        except ValueError:
            rel_path = case_dir.name
        
        print(f"\n[{idx}/{len(case_dirs)}] {rel_path}")
        
        if process_case(case_dir, input_subdir, output_subdir, k):
            ok += 1
    
    total_time = time.time() - total_start
    
    print(f"\n{'=' * 50}")
    print("🎉 图数据转换完成!")
    print(f"⏱️  总耗时: {total_time:.1f}s")
    print(f"✅ 成功: {ok}/{len(case_dirs)} 个病例")
    
    # 显示输入特征说明
    print("\n📋 图数据格式说明:")
    print("  输入特征 x (15维):")
    print("    [0:3]   坐标: x, y, z")
    print("    [3]     时间: t_norm")
    print("    [4:10]  几何: Abscissa, NormRadius, Curvature, Tangent_X/Y/Z")
    print("    [10]    边界标志: is_wall")
    print("    [11:16] 边界条件: BC_Inlet, BC_O1~O4")
    print("  目标输出 y (4维):")
    print("    [0:3]   速度: u, v, w")
    print("    [3]     压力: p")


def main():
    parser = argparse.ArgumentParser(
        description="图数据转换",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
图数据格式:
  输入特征 x (15维，无 BC_Flag):
    - 坐标 (3): x, y, z
    - 时间 (1): t_norm
    - 几何 (6): Abscissa, NormRadius, Curvature, Tangent_X/Y/Z
    - 边界标志 (1): is_wall
    - 边界条件 (5): BC_Inlet, BC_O1~O4
    
  目标输出 y (4维):
    - u, v, w, p

示例:
  python convert_to_graph.py --case ZHANG_CHUN
  python convert_to_graph.py --k 8
        """
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据根目录",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="指定处理的病例名称",
    )
    parser.add_argument(
        "--input-subdir",
        type=str,
        default=None,
        help=f"输入子目录，默认 {NORMALIZED_DIR}",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help=f"输出子目录，默认 {GRAPHS_DIR}",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=f"KNN 邻居数，默认 {GRAPH_CONFIG['k_neighbors']}",
    )
    
    args = parser.parse_args()
    
    process_all_cases(
        data_root=args.data_root,
        target_case=args.case,
        input_subdir=args.input_subdir,
        output_subdir=args.output_subdir,
        k=args.k,
    )


if __name__ == "__main__":
    main()
