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
  python -m pipeline.preprocess --case ZHANG_CHUN --mode debug
  
  # 处理所有病例（生产模式）
  python -m pipeline.preprocess --mode production
  
  # 自定义目标点数
  python -m pipeline.preprocess --target-points 50000
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

# 导入配置和工具
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.config import (
        DATA_ROOT,
        SURFACE_DIR,
        INNER_DIR,
        MERGED_DIR,
        SAMPLING_CONFIG,
        MODE,
        get_case_dirs,
    )
    from pipeline.utils.io import load_ascii_df, clean_cfd_data, save_csv
    from pipeline.utils.progress import batch_progress_logging
    from pipeline.utils.progress import case_progress_logging
    from pipeline.validation import build_batch_issue_report, inspect_case_inputs, save_batch_issue_report
else:
    from .config import (
        DATA_ROOT,
        SURFACE_DIR,
        INNER_DIR,
        MERGED_DIR,
        SAMPLING_CONFIG,
        MODE,
        get_case_dirs,
    )
    from .utils.io import load_ascii_df, clean_cfd_data, save_csv
    from .utils.progress import batch_progress_logging
    from .utils.progress import case_progress_logging
    from .validation import build_batch_issue_report, inspect_case_inputs, save_batch_issue_report


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


def summarize_frame_alignment(case_dir: Path) -> Dict[str, object]:
    """统计 ascii 与 ascii_in 的时间步对齐情况。"""
    surface_path = case_dir / SURFACE_DIR
    inner_path = case_dir / INNER_DIR

    def collect_steps(folder: Path) -> set:
        steps = set()
        if not folder.is_dir():
            return steps
        for p in folder.iterdir():
            if not p.is_file():
                continue
            stem = p.stem
            if '-' not in stem:
                continue
            number_part = stem.split('-')[-1]
            if number_part.isdigit():
                steps.add(int(number_part))
        return steps

    surface_steps = collect_steps(surface_path)
    inner_steps = collect_steps(inner_path)
    matched_steps = sorted(surface_steps & inner_steps)
    surface_only = sorted(surface_steps - inner_steps)
    inner_only = sorted(inner_steps - surface_steps)

    return {
        "surface_count": len(surface_steps),
        "inner_count": len(inner_steps),
        "matched_count": len(matched_steps),
        "surface_only_count": len(surface_only),
        "inner_only_count": len(inner_only),
        "matched_steps": matched_steps,
        "surface_only_steps": surface_only,
        "inner_only_steps": inner_only,
    }


def process_single_frame(
    surface_file: Path,
    inner_file: Path,
    output_path: Path,
    frame_label: str = "",
    target_total: int = 15000,
    wall_max_points: int = 10000,
    boundary_threshold: float = 2.0,
    boundary_core_ratio: tuple = (0.7, 0.3),
    sampling_method: str = "hybrid",
    fps_ratio: float = 0.5,
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
        wall_max_points: 壁面点上限，超出时降采样
        boundary_threshold: 近壁区阈值（mm）
        boundary_core_ratio: 预算分配比例
        sampling_method: 采样方法，"fps", "random" 或 "hybrid"
        fps_ratio: 混合采样时 FPS 的占比（默认 0.5），仅当 method="hybrid" 时生效
        seed: 随机种子
        convert_to_mm: 是否转换坐标单位
    
    返回:
        是否成功
    """
    if __package__ in {None, ""}:
        from pipeline.utils.sampling import stratified_sampling_by_distance
    else:
        from .utils.sampling import stratified_sampling_by_distance

    try:
        prefix = f"   {frame_label} " if frame_label else "   "

        # 1. 读取数据
        print(f"{prefix}读取壁面文件: {surface_file.name}")
        surface_raw_df = load_ascii_df(surface_file)
        print(f"{prefix}读取内部文件: {inner_file.name}")
        inner_raw_df = load_ascii_df(inner_file)
        
        # 2. 清洗数据
        print(f"{prefix}清洗壁面数据...")
        surface_df = clean_cfd_data(surface_raw_df, convert_to_mm=convert_to_mm, is_wall=True)
        print(f"{prefix}清洗内部数据...")
        inner_df = clean_cfd_data(inner_raw_df, convert_to_mm=convert_to_mm, is_wall=False)
        
        # 3. 分层降采样合并
        print(f"{prefix}分层降采样与合并...")
        merged_df, _ = stratified_sampling_by_distance(
            surface_df,
            inner_df,
            boundary_threshold=boundary_threshold,
            boundary_core_ratio=boundary_core_ratio,
            target_total=target_total,
            wall_max_points=wall_max_points,
            sampling_method=sampling_method,
            fps_ratio=fps_ratio,
            seed=seed,
        )
        
        # 4. 保存结果
        print(f"{prefix}保存输出: {output_path.name}")
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
    target_total: int = 15000,
    wall_max_points: int = 10000,
    boundary_threshold: float = 2.0,
    boundary_core_ratio: tuple = (0.7, 0.3),
    sampling_method: str = "hybrid",
    fps_ratio: float = 0.5,
    seed: Optional[int] = 1234,
    mode: str = "debug",
) -> bool:
    """
    处理单个病例的所有时间帧。
    
    参数:
        case_dir: 病例目录
        output_subdir: 输出子目录（默认使用配置）
        target_total: 目标总点数
        wall_max_points: 壁面点上限，超出时降采样
        boundary_threshold: 近壁区阈值
        boundary_core_ratio: 预算分配比例
        sampling_method: 采样方法，"fps", "random" 或 "hybrid"
        fps_ratio: 混合采样时 FPS 的占比（默认 0.5），仅当 method="hybrid" 时生效
        seed: 随机种子
        mode: 处理模式 (debug/production)
    
    返回:
        是否成功
    """
    case_dir = Path(case_dir)
    case_name = case_dir.name
    with case_progress_logging(case_dir, "step1_preprocess") as log_path:
        print(f"📝 进度日志: {log_path}")
        input_check = inspect_case_inputs(case_dir)

        alignment = summarize_frame_alignment(case_dir)
        matched_files = find_matching_files(case_dir)

        if input_check["issues"]:
            print(f"\n📂 处理病例: {case_name}")
            print("   🔎 输入检查:")
            for issue in input_check["issues"]:
                icon = "⚠️" if issue["severity"] == "warning" else "❌"
                print(f"   {icon} {issue['message']}")
        
        if not matched_files:
            print(f"   ⚠️ 跳过: 未找到匹配的壁面与内部点文件")
            return False
        
        # 设置输出目录
        if output_subdir is None:
            output_subdir = MERGED_DIR
        output_dir = case_dir / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📂 处理病例: {case_name}")
        print(
            f"   时间帧: 壁面 {alignment['surface_count']} / 内部 {alignment['inner_count']} / "
            f"匹配 {alignment['matched_count']}"
        )
        if alignment["surface_only_count"] or alignment["inner_only_count"]:
            print(
                f"   ⚠️ 未配对帧: 仅壁面 {alignment['surface_only_count']} / "
                f"仅内部 {alignment['inner_only_count']}"
            )
        print(f"   目标点数: {target_total} (壁面上限: {wall_max_points})")
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
                frame_label=f"[{i}/{len(matched_files)}|{key}]",
                target_total=target_total,
                wall_max_points=wall_max_points,
                boundary_threshold=boundary_threshold,
                boundary_core_ratio=boundary_core_ratio,
                sampling_method=sampling_method,
                fps_ratio=fps_ratio,
                seed=seed,
            ):
                success_count += 1
                print(f"   ✅ 已保存: {output_path.name}")
        
        total_time = time.time() - start_time

        report_path = output_dir / "preprocess_report.json"
        report = {
            "case_name": case_name,
            "surface_count": alignment["surface_count"],
            "inner_count": alignment["inner_count"],
            "matched_count": alignment["matched_count"],
            "surface_only_count": alignment["surface_only_count"],
            "inner_only_count": alignment["inner_only_count"],
            "surface_only_steps": alignment["surface_only_steps"],
            "inner_only_steps": alignment["inner_only_steps"],
            "target_total": target_total,
            "wall_max_points": wall_max_points,
            "sampling_method": sampling_method,
            "fps_ratio": fps_ratio if sampling_method == "hybrid" else None,
            "success_count": success_count,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n🎉 {case_name} 处理完成!")
        print(f"   成功: {success_count}/{len(matched_files)} 个时间帧")
        print(f"   耗时: {total_time:.1f}s")
        print(f"   输出: {output_dir}")
        print(f"   报告: {report_path.name}")
        
        return success_count > 0


def process_all_cases(
    data_root: Path = None,
    target_case: Optional[str] = None,
    target_total: int = 15000,
    wall_max_points: int = 10000,
    boundary_threshold: float = 2.0,
    boundary_core_ratio: tuple = (0.7, 0.3),
    sampling_method: str = "hybrid",
    fps_ratio: float = 0.5,
    seed: Optional[int] = 1234,
    mode: str = "debug",
) -> None:
    """
    批量处理所有病例。
    
    参数:
        data_root: 数据根目录
        target_case: 指定处理的病例名称
        target_total: 目标总点数
        wall_max_points: 壁面点上限，超出时降采样
        boundary_threshold: 近壁区阈值
        boundary_core_ratio: 预算分配比例
        sampling_method: 采样方法，"fps", "random" 或 "hybrid"
        fps_ratio: 混合采样时 FPS 的占比（默认 0.5），仅当 method="hybrid" 时生效
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
    
    with batch_progress_logging(data_root, "step1_preprocess_batch.log", "step1_preprocess_batch") as log_path:
        print(f"📝 批量日志: {log_path}")
        print("🚀 数据预处理 - 清洗+合并+降采样")
        print("=" * 50)
        print(f"📁 数据根目录: {data_root}")
        print(f"📊 目标点数: {target_total} (壁面上限: {wall_max_points})")
        if sampling_method.lower() == "hybrid":
            print(f"📊 采样方法: {sampling_method} (FPS {fps_ratio*100:.0f}%)")
        else:
            print(f"📊 采样方法: {sampling_method}")
        print(f"📊 近壁区阈值: {boundary_threshold}mm")
        print(f"📊 内部点分配: 近壁层 {boundary_core_ratio[0]*100:.0f}% : 核心层 {boundary_core_ratio[1]*100:.0f}%")
        print(f"📊 处理模式: {mode}")
        print(f"📊 待处理病例数: {len(case_dirs)}")

        audit_report = build_batch_issue_report(case_dirs)
        audit_dir = data_root / "pipeline_reports"
        report_name = "preprocess_input_audit" if target_case is None else f"preprocess_input_audit_{case_dirs[0].name}"
        audit_json, audit_csv = save_batch_issue_report(audit_report, audit_dir, report_name)
        print(f"📝 输入检查报告: {audit_json}")
        print(f"📝 输入检查表格: {audit_csv}")
        if audit_report["issue_case_count"]:
            print(f"⚠️ 存在输入问题的病例: {audit_report['issue_case_count']} 个")
            print(f"⚠️ 可直接跑步骤1的病例: {audit_report['preprocess_ready_count']} 个")
        
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
                wall_max_points=wall_max_points,
                boundary_threshold=boundary_threshold,
                boundary_core_ratio=boundary_core_ratio,
                sampling_method=sampling_method,
                fps_ratio=fps_ratio,
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
  python -m pipeline.preprocess --case ZHANG_CHUN
  
  # 使用随机采样（速度快）
  python -m pipeline.preprocess --sampling-method random
  
  # 使用混合采样（推荐，兼顾覆盖和多样性）
  python -m pipeline.preprocess --sampling-method hybrid --fps-ratio 0.3
  
  # 自定义目标点数
  python -m pipeline.preprocess --target-points 50000
  
  # 生产模式（不保留中间文件）
  python -m pipeline.preprocess --mode production
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
        "--wall-max-points",
        type=int,
        default=SAMPLING_CONFIG["wall_max_points"],
        help=f"壁面点上限，超出时降采样，默认 {SAMPLING_CONFIG['wall_max_points']}",
    )
    parser.add_argument(
        "--sampling-method",
        type=str,
        choices=["fps", "random", "hybrid"],
        default=SAMPLING_CONFIG["sampling_method"],
        help=f"采样方法：fps/random/hybrid，默认 {SAMPLING_CONFIG['sampling_method']}",
    )
    parser.add_argument(
        "--fps-ratio",
        type=float,
        default=SAMPLING_CONFIG.get("hybrid_fps_ratio", 0.2),
        help=f"混合采样时 FPS 占比，默认 {SAMPLING_CONFIG.get('hybrid_fps_ratio', 0.2)}（仅当 method=hybrid 时生效）",
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
        wall_max_points=args.wall_max_points,
        boundary_threshold=args.boundary_threshold,
        boundary_core_ratio=(args.boundary_ratio, args.core_ratio),
        sampling_method=args.sampling_method,
        fps_ratio=args.fps_ratio,
        seed=args.seed,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
