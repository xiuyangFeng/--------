#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline 一键运行脚本

按顺序执行完整的数据处理流程:
1. preprocess.py: 数据清洗 + 合并 + 降采样
2. extract_features.py: 几何特征提取 + 边界条件
3. normalize.py: 特征归一化
4. convert_to_graph.py: 转换为图数据

使用示例:
  # 处理单个病例
  python run_all.py --case ZHANG_CHUN
  
  # 处理所有病例
  python run_all.py
  
  # 跳过已完成的步骤
  python run_all.py --start-step 2
  
  # 使用随机采样（速度快）
  python run_all.py --sampling-method random
"""

import argparse
import time
from pathlib import Path
from typing import Optional

# 导入各处理模块
from config import (
    DATA_ROOT,
    MERGED_DIR,
    FEATURES_DIR,
    NORMALIZED_DIR,
    GRAPHS_DIR,
    SAMPLING_CONFIG,
    GRAPH_CONFIG,
    MODE,
    get_case_dirs,
)
from preprocess import process_all_cases as preprocess
from extract_features import process_all_cases as extract_features
from normalize import process_all_cases as normalize
from convert_to_graph import process_all_cases as convert_to_graph


def run_pipeline(
    data_root: Path = None,
    target_case: Optional[str] = None,
    start_step: int = 1,
    end_step: int = 4,
    target_points: int = None,
    sampling_method: str = None,
    mode: str = None,
    k_neighbors: int = None,
) -> None:
    """
    运行完整的数据处理流程。
    
    参数:
        data_root: 数据根目录
        target_case: 指定处理的病例名称
        start_step: 开始步骤 (1-4)
        end_step: 结束步骤 (1-4)
        target_points: 目标点数
        sampling_method: 采样方法
        mode: 处理模式
        k_neighbors: KNN 邻居数
    """
    if data_root is None:
        data_root = DATA_ROOT
    else:
        data_root = Path(data_root)
    
    if target_points is None:
        target_points = SAMPLING_CONFIG["target_total_points"]
    if sampling_method is None:
        sampling_method = SAMPLING_CONFIG["sampling_method"]
    if mode is None:
        mode = MODE
    if k_neighbors is None:
        k_neighbors = GRAPH_CONFIG["k_neighbors"]
    
    print("=" * 60)
    print("🚀 Pipeline - 新数据格式完整处理流程")
    print("=" * 60)
    print(f"📁 数据根目录: {data_root}")
    print(f"📊 目标点数: {target_points}")
    print(f"📊 采样方法: {sampling_method}")
    print(f"📊 处理模式: {mode}")
    print(f"📊 KNN 邻居数: {k_neighbors}")
    
    if target_case:
        print(f"🎯 指定病例: {target_case}")
    else:
        case_dirs = get_case_dirs(data_root)
        print(f"📊 待处理病例数: {len(case_dirs)}")
    
    print(f"📋 执行步骤: {start_step} → {end_step}")
    print()
    
    total_start = time.time()
    
    # 步骤1: 数据预处理
    if start_step <= 1 <= end_step:
        print("\n" + "=" * 60)
        print("📌 步骤1/4: 数据预处理（清洗 + 合并 + 降采样）")
        print("=" * 60)
        
        preprocess(
            data_root=data_root,
            target_case=target_case,
            target_total=target_points,
            sampling_method=sampling_method,
            boundary_threshold=SAMPLING_CONFIG["boundary_threshold"],
            boundary_core_ratio=SAMPLING_CONFIG["boundary_core_ratio"],
            seed=SAMPLING_CONFIG["seed"],
            mode=mode,
        )
    
    # 步骤2: 几何特征提取
    if start_step <= 2 <= end_step:
        print("\n" + "=" * 60)
        print("📌 步骤2/4: 几何特征提取 + 边界条件")
        print("=" * 60)
        
        extract_features(
            data_root=data_root,
            target_case=target_case,
            input_subdir=MERGED_DIR,
            output_subdir=FEATURES_DIR,
        )
    
    # 步骤3: 特征归一化
    if start_step <= 3 <= end_step:
        print("\n" + "=" * 60)
        print("📌 步骤3/4: 特征归一化")
        print("=" * 60)
        
        normalize(
            data_root=data_root,
            target_case=target_case,
            input_subdir=FEATURES_DIR,
            output_subdir=NORMALIZED_DIR,
        )
    
    # 步骤4: 图数据转换
    if start_step <= 4 <= end_step:
        print("\n" + "=" * 60)
        print("📌 步骤4/4: 图数据转换")
        print("=" * 60)
        
        convert_to_graph(
            data_root=data_root,
            target_case=target_case,
            input_subdir=NORMALIZED_DIR,
            output_subdir=GRAPHS_DIR,
            k=k_neighbors,
        )
    
    total_time = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("🎉 Pipeline 执行完成!")
    print("=" * 60)
    print(f"⏱️  总耗时: {total_time:.1f}s ({total_time/60:.1f} 分钟)")
    
    print("\n📂 输出目录结构:")
    print(f"  病例目录/")
    print(f"  └── processed/")
    print(f"      ├── merged/      # 步骤1: 合并降采样后的数据")
    print(f"      ├── features/    # 步骤2: 添加几何特征和边界条件")
    print(f"      ├── normalized/  # 步骤3: 归一化后的数据")
    print(f"      └── graphs/      # 步骤4: PyG 图数据 (.pt)")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline - 一键运行完整处理流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
处理流程:
  步骤1: preprocess.py      - 数据清洗 + 合并 + 降采样
  步骤2: extract_features.py - 几何特征提取 + 边界条件
  步骤3: normalize.py        - 特征归一化
  步骤4: convert_to_graph.py - 转换为图数据

示例:
  # 处理单个病例（完整流程）
  python run_all.py --case ZHANG_CHUN
  
  # 处理所有病例
  python run_all.py
  
  # 从步骤2开始（跳过预处理）
  python run_all.py --start-step 2
  
  # 只执行步骤1和2
  python run_all.py --end-step 2
  
  # 使用随机采样（速度快）
  python run_all.py --sampling-method random
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
        "--start-step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="开始步骤 (1-4)，默认 1",
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="结束步骤 (1-4)，默认 4",
    )
    parser.add_argument(
        "--target-points",
        type=int,
        default=None,
        help=f"目标总点数，默认 {SAMPLING_CONFIG['target_total_points']}",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        choices=["fps", "random"],
        default=None,
        help=f"采样方法，默认 {SAMPLING_CONFIG['sampling_method']}",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["debug", "production"],
        default=None,
        help=f"处理模式，默认 {MODE}",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=f"KNN 邻居数，默认 {GRAPH_CONFIG['k_neighbors']}",
    )
    
    args = parser.parse_args()
    
    if args.start_step > args.end_step:
        print("❌ 错误: start-step 不能大于 end-step")
        return
    
    run_pipeline(
        data_root=args.data_root,
        target_case=args.case,
        start_step=args.start_step,
        end_step=args.end_step,
        target_points=args.target_points,
        sampling_method=args.sampling_method,
        mode=args.mode,
        k_neighbors=args.k,
    )


if __name__ == "__main__":
    main()
