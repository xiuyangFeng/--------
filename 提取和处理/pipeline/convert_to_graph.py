#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图数据转换模块

将归一化后的 CSV 数据转换为 PyTorch Geometric 图数据格式。

功能:
1. 读取归一化后的特征数据
2. 构建 KNN 图结构
3. 组装输入特征和目标输出
4. 加载全局条件（边界条件 + 时间）作为图级属性
5. 保存为 .pt 文件

节点特征 data.x (10维):
- 坐标: x, y, z (3)
- 几何特征: Abscissa, NormRadius, Curvature, Tangent_X/Y/Z (6)
- 边界标志: is_wall (1)

全局条件 data.global_cond (1x6):
- 时间: t_norm (1)
- 边界条件: BC_Inlet, BC_O1~O4 (5)

目标输出 data.y (4维):
- 速度: u, v, w (3)
- 压力: p (1)

使用示例:
  # 处理单个病例
  python -m pipeline.convert_to_graph --case ZHANG_CHUN
  
  # 处理所有病例
  python -m pipeline.convert_to_graph
"""

import argparse
import json
import re
import shutil
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
try:
    from .config import (
        DATA_ROOT,
        NORMALIZED_DIR,
        COORD_NORMALIZED_DIR,
        GRAPHS_DIR,
        GRAPH_CONFIG,
        get_case_dirs,
    )
    from .utils.progress import case_progress_logging
    from .utils.progress import batch_progress_logging
except ImportError:
    from config import (
        DATA_ROOT,
        NORMALIZED_DIR,
        COORD_NORMALIZED_DIR,
        GRAPHS_DIR,
        GRAPH_CONFIG,
        get_case_dirs,
    )
    from pipeline.utils.progress import case_progress_logging
    from pipeline.utils.progress import batch_progress_logging


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
    bc_values: list = None,
    k: int = 6,
) -> Data:
    """
    读取 CSV 文件并构建图结构。
    
    参数:
        file_path: CSV 文件路径
        t_norm: 归一化的时间值 [0, 1]
        bc_values: 归一化后的边界条件 [BC_Inlet, BC_O1, BC_O2, BC_O3, BC_O4]
        k: KNN 邻居数
    
    返回:
        PyG Data 对象，包含:
        - x: [N, 10] 节点特征 (坐标3 + 几何6 + is_wall1)
        - edge_index: [2, E] 边索引
        - y: [N, 4] 目标输出 (u, v, w, p)
        - global_cond: [1, 6] 全局条件 (t_norm + BC_Inlet + BC_O1~O4)
    """
    df = pd.read_csv(file_path)
    
    # 1. 提取坐标
    coords = df[['x', 'y', 'z']].values
    
    # 2. 构建节点特征 (10维): 坐标(3) + 几何(6) + is_wall(1)
    geom_feats = df[['Abscissa', 'NormRadius', 'Curvature', 
                     'Tangent_X', 'Tangent_Y', 'Tangent_Z']].values
    is_wall = df[['is_wall']].values
    
    # 拼接节点特征: [x, y, z, Abscissa, NormRadius, Curvature, Tx, Ty, Tz, is_wall]
    x = np.hstack([coords, geom_feats, is_wall])
    x = torch.from_numpy(x).float()
    
    # 3. 构建全局条件 (6维): t_norm(1) + BC(5)
    if bc_values is not None and len(bc_values) == 5:
        global_cond = torch.tensor([[t_norm] + bc_values], dtype=torch.float32)  # [1, 6]
    else:
        # 如果没有 BC 数据，用 0 填充
        global_cond = torch.tensor([[t_norm, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    
    # 4. 提取目标输出 (4维: u, v, w, p)
    y = df[['u', 'v', 'w', 'p']].values
    y = torch.from_numpy(y).float()
    
    # 5. 构建边索引 (使用 KNN)
    effective_k = min(k, max(1, coords.shape[0] - 1))
    nbrs = NearestNeighbors(n_neighbors=effective_k + 1, algorithm='ball_tree').fit(coords)
    _, indices = nbrs.kneighbors(coords)
    
    # 转换为 PyG 的 edge_index 格式 [2, num_edges]
    # indices 的第一列是节点自身，忽略它
    row = np.repeat(np.arange(coords.shape[0]), effective_k)
    col = indices[:, 1:].flatten()
    
    edges = np.stack([row, col])
    reverse_edges = np.stack([col, row])
    edge_index = torch.from_numpy(np.concatenate([edges, reverse_edges], axis=1)).long()
    
    # 创建 PyG Data 对象（global_cond 作为图级属性）
    data = Data(x=x, edge_index=edge_index, y=y, global_cond=global_cond)
    
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
    with case_progress_logging(case_dir, "step5_convert_to_graph") as log_path:
        print(f"📝 进度日志: {log_path}")

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

        min_step, max_step = min(steps), max(steps)
        total_range = max_step - min_step
        if total_range == 0:
            total_range = 1

        output_dir.mkdir(parents=True, exist_ok=True)

        coord_norm_dir = case_dir / COORD_NORMALIZED_DIR
        transform_params_src = coord_norm_dir / "transform_params.json"
        if transform_params_src.exists():
            transform_params_dst = output_dir / "transform_params.json"
            shutil.copy2(transform_params_src, transform_params_dst)
            print(f"  📋 已复制坐标系变换参数")
        else:
            print(f"  ⚠️ 未找到坐标系变换参数: {transform_params_src}")

        bc_meta_path = input_dir / "bc_metadata_normalized.json"
        bc_data_map = {}
        if bc_meta_path.exists():
            try:
                with open(bc_meta_path, 'r', encoding='utf-8') as f:
                    bc_meta = json.load(f)
                bc_data_map = bc_meta.get("data", {})
                print(f"  📋 已加载边界条件: {len(bc_data_map)} 个时间步")
            except Exception as e:
                print(f"  ⚠️ 读取边界条件失败: {e}")
        else:
            print(f"  ⚠️ 未找到边界条件元数据: {bc_meta_path}")

        if bc_meta_path.exists():
            shutil.copy2(bc_meta_path, output_dir / "bc_metadata_normalized.json")

        print(f"  📁 找到 {len(file_step_pairs)} 个时间步")
        print(f"  🔗 KNN 邻居数: {k}")

        success_count = 0
        missing_bc_count = 0
        for i, (csv_file, step) in enumerate(tqdm(file_step_pairs, desc=f"处理 {case_name}", leave=False), 1):
            try:
                print(f"  🔄 [{i}/{len(file_step_pairs)}] 文件: {csv_file.name} (step={step})")
                t_norm = (step - min_step) / total_range
                print(f"    时间归一化: t_norm={t_norm:.4f}")

                bc_values = None
                csv_stem = csv_file.stem
                original_stem = csv_stem.replace("result_features_", "")
                if original_stem in bc_data_map:
                    bc_values = bc_data_map[original_stem]
                    print("    已加载对应 BC 元数据")
                else:
                    missing_bc_count += 1
                    print("    ⚠️ 未找到对应 BC 元数据，将使用 0 填充")

                print("    构建图结构...")
                data = build_graph_from_csv(csv_file, t_norm, bc_values=bc_values, k=k)

                out_name = csv_file.stem + ".pt"
                print(f"    保存图文件: {out_name}")
                torch.save(data, output_dir / out_name)
                success_count += 1
                print(f"  ✅ 文件完成: {out_name}")
            except Exception as e:
                print(f"  ❌ 处理 {csv_file.name} 失败: {e}")

        report_path = output_dir / "graph_conversion_report.json"
        report = {
            "case_name": case_name,
            "time_step_count": len(file_step_pairs),
            "success_count": success_count,
            "missing_bc_count": missing_bc_count,
            "k_neighbors": k,
            "edge_direction": "bidirectional_knn",
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"  ✅ 完成: {success_count}/{len(file_step_pairs)} 个文件")
        if missing_bc_count:
            print(f"  ⚠️ 缺少 BC 的时间步: {missing_bc_count}")
        print(f"  📄 报告: {report_path.name}")
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
    
    with batch_progress_logging(data_root, "step5_convert_to_graph_batch.log", "step5_convert_to_graph_batch") as log_path:
        print(f"📝 批量日志: {log_path}")
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
        
        print("\n📋 图数据格式说明:")
        print("  节点特征 x (10维):")
        print("    [0:3]   坐标: x, y, z")
        print("    [3:9]   几何: Abscissa, NormRadius, Curvature, Tangent_X/Y/Z")
        print("    [9]     边界标志: is_wall")
        print("  全局条件 global_cond (1x6):")
        print("    [0]     时间: t_norm")
        print("    [1:6]   边界条件: BC_Inlet, BC_O1~O4")
        print("  目标输出 y (4维):")
        print("    [0:3]   速度: u, v, w")
        print("    [3]     压力: p")


def main():
    parser = argparse.ArgumentParser(
        description="图数据转换",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
图数据格式:
  节点特征 x (10维):
    - 坐标 (3): x, y, z
    - 几何 (6): Abscissa, NormRadius, Curvature, Tangent_X/Y/Z
    - 边界标志 (1): is_wall
    
  全局条件 global_cond (1x6):
    - 时间 (1): t_norm
    - 边界条件 (5): BC_Inlet, BC_O1~O4
    
  目标输出 y (4维):
    - u, v, w, p

示例:
  python -m pipeline.convert_to_graph --case ZHANG_CHUN
  python -m pipeline.convert_to_graph --k 8
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
