#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块

整合数据清洗、合并、降采样功能。
将原始的 ascii (壁面) 和 ascii_in (内部) 数据处理为统一格式。

功能:
1. 读取原始 Fluent 输出数据
2. 清洗数据（列名标准化、单位转换等）
3. 分层降采样合并（保留壁面点，内部点分层采样）
4. 输出到 processed/merged/

使用示例:
  # 处理单个病例（调试模式）
  python preprocess.py --case ZHANG_CHUN --mode debug
  
  # 处理所有病例（生产模式）
  python preprocess.py --mode production
  
  # 自定义目标点数
  python preprocess.py --target-points 50000
"""

import argparse
import time
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

# 导入配置和工具
from config import (
    DATA_ROOT, 
    SURFACE_DIR, 
    INNER_DIR, 
    MERGED_DIR,
    SAMPLING_CONFIG,
    MODE,
    get_case_dirs,
)
from utils.io import load_ascii_df, clean_cfd_data, save_csv
from utils.sampling import stratified_sampling_by_distance


def find_matching_files(case_dir: Path) -> dict:
    """
    查找壁面和内部点数据文件，按编号匹配。
    
    参数:
        case_dir: 病例目录
    
    返回:
        匹配的文件对字典 {编号: (壁面文件, 内部文件)}
    """
    surface_path = case_dir / SURFACE_DIR
    inner_path = case_dir / INNER_DIR
    
    if not surface_path.is_dir() or not inner_path.is_dir():
        return {}
    
    # 查找壁面文件
    surface_files = {}
    for p in surface_path.iterdir():
        if not p.is_file():
            continue
        stem = p.stem
        if '-' in stem:
            number_part = stem.split('-')[-1]
            if number_part.isdigit():
                surface_files[number_part] = p
    
    # 查找内部文件
    inner_files = {}
    for p in inner_path.iterdir():
        if not p.is_file():
            continue
        stem = p.stem
        if '-' in stem:
            number_part = stem.split('-')[-1]
            if number_part.isdigit():
                inner_files[number_part] = p
    
    # 匹配
    common_keys = sorted(set(surface_files) & set(inner_files))
    matched = {k: (surface_files[k], inner_files[k]) for k in common_keys}
    
    return matched


def process_single_frame(
    surface_file: Path,
    inner_file: Path,
    output_path: Path,
    target_total: int = 40000,
    boundary_threshold: float = 2.0,
    boundary_core_ratio: tuple = (0.7, 0.3),
    sampling_method: str = "fps",
    seed: Optional[int] = 1234,
    convert_to_mm: bool = True,
) -> bool:
    """
    处理单个时间帧的数据。
    
    参数:
        surface_file: 壁面数据文件
        inner_file: 内部数据文件
        output_path: 输出文件路径
        target_total: 目标总点数
        boundary_threshold: 近壁区阈值（mm）
        boundary_core_ratio: 预算分配比例
        sampling_method: 采样方法
        seed: 随机种子
        convert_to_mm: 是否转换坐标单位
    
    返回:
        是否成功
    """
    try:
        # 1. 读取数据
        surface_raw_df = load_ascii_df(surface_file)
        inner_raw_df = load_ascii_df(inner_file)
        
        # 2. 清洗数据
        surface_df = clean_cfd_data(surface_raw_df, convert_to_mm=convert_to_mm)
        inner_df = clean_cfd_data(inner_raw_df, convert_to_mm=convert_to_mm)
        
        # 3. 分层降采样合并
        merged_df, _ = stratified_sampling_by_distance(
            surface_df,
            inner_df,
            boundary_threshold=boundary_threshold,
            boundary_core_ratio=boundary_core_ratio,
            target_total=target_total,
            sampling_method=sampling_method,
            seed=seed,
        )
        
        # 4. 保存结果
        save_csv(merged_df, output_path)
        
        return True
        
    except Exception as e:
        print(f"  ❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_single_case(
    case_dir: Path,
    output_subdir: str = None,
    target_total: int = 40000,
    boundary_threshold: float = 2.0,
    boundary_core_ratio: tuple = (0.7, 0.3),
    sampling_method: str = "fps",
    seed: Optional[int] = 1234,
    mode: str = "debug",
) -> bool:
    """
    处理单个病例的所有时间帧。
    
    参数:
        case_dir: 病例目录
        output_subdir: 输出子目录（默认使用配置）
        target_total: 目标总点数
        boundary_threshold: 近壁区阈值
        boundary_core_ratio: 预算分配比例
        sampling_method: 采样方法
        seed: 随机种子
        mode: 处理模式 (debug/production)
    
    返回:
        是否成功
    """
    case_dir = Path(case_dir)
    case_name = case_dir.name
    
    # 查找匹配文件
    matched_files = find_matching_files(case_dir)
    
    if not matched_files:
        print(f"  ⚠️ 跳过: 未找到匹配的壁面与内部点文件")
        return False
    
    # 设置输出目录
    if output_subdir is None:
        output_subdir = MERGED_DIR
    output_dir = case_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 处理病例: {case_name}")
    print(f"   找到 {len(matched_files)} 个时间帧")
    print(f"   目标点数: {target_total}")
    print(f"   采样方法: {sampling_method}")
    print(f"   处理模式: {mode}")
    
    # 处理每个时间帧
    success_count = 0
    start_time = time.time()
    
    for i, (key, (surface_file, inner_file)) in enumerate(matched_files.items(), 1):
        # 构造输出文件名
        output_name = f"merged-{key}.csv"
        output_path = output_dir / output_name
        
        print(f"\n🔄 [{i}/{len(matched_files)}] 处理编号 {key}...")
        
        if process_single_frame(
            surface_file,
            inner_file,
            output_path,
            target_total=target_total,
            boundary_threshold=boundary_threshold,
            boundary_core_ratio=boundary_core_ratio,
            sampling_method=sampling_method,
            seed=seed,
        ):
            success_count += 1
            print(f"   ✅ 已保存: {output_path.name}")
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 {case_name} 处理完成!")
    print(f"   成功: {success_count}/{len(matched_files)} 个时间帧")
    print(f"   耗时: {total_time:.1f}s")
    print(f"   输出: {output_dir}")
    
    return success_count > 0


def process_all_cases(
    data_root: Path = None,
    target_case: Optional[str] = None,
    target_total: int = 40000,
    boundary_threshold: float = 2.0,
    boundary_core_ratio: tuple = (0.7, 0.3),
    sampling_method: str = "fps",
    seed: Optional[int] = 1234,
    mode: str = "debug",
) -> None:
    """
    批量处理所有病例。
    
    参数:
        data_root: 数据根目录
        target_case: 指定处理的病例名称
        target_total: 目标总点数
        boundary_threshold: 近壁区阈值
        boundary_core_ratio: 预算分配比例
        sampling_method: 采样方法
        seed: 随机种子
        mode: 处理模式
    """
    if data_root is None:
        data_root = DATA_ROOT
    else:
        data_root = Path(data_root)
    
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
    
    print("🚀 数据预处理 - 清洗+合并+降采样")
    print("=" * 50)
    print(f"📁 数据根目录: {data_root}")
    print(f"📊 目标点数: {target_total}")
    print(f"📊 采样方法: {sampling_method}")
    print(f"📊 近壁区阈值: {boundary_threshold}mm")
    print(f"📊 预算分配: 近壁层 {boundary_core_ratio[0]*100:.0f}% : 核心层 {boundary_core_ratio[1]*100:.0f}%")
    print(f"📊 处理模式: {mode}")
    print(f"📊 待处理病例数: {len(case_dirs)}")
    
    total_start = time.time()
    ok = 0
    
    for idx, case_dir in enumerate(case_dirs, 1):
        try:
            rel_path = case_dir.relative_to(data_root)
        except ValueError:
            rel_path = case_dir.name
        
        print(f"\n\n{'=' * 50}")
        print(f"[{idx}/{len(case_dirs)}] {rel_path}")
        print("=" * 50)
        
        if process_single_case(
            case_dir,
            target_total=target_total,
            boundary_threshold=boundary_threshold,
            boundary_core_ratio=boundary_core_ratio,
            sampling_method=sampling_method,
            seed=seed,
            mode=mode,
        ):
            ok += 1
    
    total_time = time.time() - total_start
    
    print(f"\n\n{'=' * 50}")
    print("🎉 批量预处理完成!")
    print(f"⏱️  总耗时: {total_time:.1f}s")
    print(f"✅ 成功: {ok}/{len(case_dirs)} 个病例")


def main():
    parser = argparse.ArgumentParser(
        description="数据预处理：清洗 + 合并 + 降采样",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
处理流程:
  1. 读取 ascii/ (壁面) 和 ascii_in/ (内部) 原始数据
  2. 清洗数据（列名标准化、单位转换）
  3. 分层降采样合并（优先保留壁面点）
  4. 输出到 processed/merged/

示例:
  # 处理指定病例
  python preprocess.py --case ZHANG_CHUN
  
  # 使用随机采样（速度快）
  python preprocess.py --sampling-method random
  
  # 自定义目标点数
  python preprocess.py --target-points 50000
  
  # 生产模式（不保留中间文件）
  python preprocess.py --mode production
        """
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="数据根目录，默认使用配置文件中的 DATA_ROOT",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="指定处理的病例名称（可选）",
    )
    parser.add_argument(
        "--target-points",
        type=int,
        default=SAMPLING_CONFIG["target_total_points"],
        help=f"目标总点数，默认 {SAMPLING_CONFIG['target_total_points']}",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        choices=["fps", "random"],
        default=SAMPLING_CONFIG["sampling_method"],
        help=f"采样方法，默认 {SAMPLING_CONFIG['sampling_method']}",
    )
    parser.add_argument(
        "--boundary-threshold",
        type=float,
        default=SAMPLING_CONFIG["boundary_threshold"],
        help=f"近壁区阈值（mm），默认 {SAMPLING_CONFIG['boundary_threshold']}",
    )
    parser.add_argument(
        "--boundary-ratio",
        type=float,
        default=SAMPLING_CONFIG["boundary_core_ratio"][0],
        help=f"近壁层预算比例，默认 {SAMPLING_CONFIG['boundary_core_ratio'][0]}",
    )
    parser.add_argument(
        "--core-ratio",
        type=float,
        default=SAMPLING_CONFIG["boundary_core_ratio"][1],
        help=f"核心层预算比例，默认 {SAMPLING_CONFIG['boundary_core_ratio'][1]}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SAMPLING_CONFIG["seed"],
        help=f"随机种子，默认 {SAMPLING_CONFIG['seed']}",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["debug", "production"],
        default=MODE,
        help=f"处理模式，默认 {MODE}",
    )
    
    args = parser.parse_args()
    
    # 验证比例之和
    if abs(args.boundary_ratio + args.core_ratio - 1.0) > 0.01:
        print(f"⚠️  警告：近壁层比例 ({args.boundary_ratio}) + 核心层比例 ({args.core_ratio}) != 1.0")
        total = args.boundary_ratio + args.core_ratio
        args.boundary_ratio /= total
        args.core_ratio /= total
        print(f"   已自动归一化为: {args.boundary_ratio:.2f} : {args.core_ratio:.2f}")
    
    process_all_cases(
        data_root=args.data_root,
        target_case=args.case,
        target_total=args.target_points,
        boundary_threshold=args.boundary_threshold,
        boundary_core_ratio=(args.boundary_ratio, args.core_ratio),
        sampling_method=args.sampling_method,
        seed=args.seed,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
