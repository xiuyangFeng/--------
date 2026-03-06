#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量几何特征提取工具（适用于 data 文件夹结构）

用于处理 data 目录下各病例的插值后数据，提取血管几何特征。
默认使用 ascii_merged_128 作为点云输入，out_interpolated_128 作为边界条件来源。

使用示例:
  # 处理所有病例
  python batch_process_min.py
  
  # 处理指定病例
  python batch_process_min.py --case FENG_LI_XIN
  
  # 指定输出目录（默认输出到病例目录下）
  python batch_process_min.py --output-dir ../outdata
  
  # 自定义点云和边界条件目录
  python batch_process_min.py --cloud-subdir ascii_merged --bc-subdir .
"""

import os
import argparse
import glob
import time
import traceback
import re
from pathlib import Path
from typing import Optional, List, Dict

import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from legacy.preprocess.Script_Scenario_B_Volumetric import (
    prepare_geometry_data,
    process_single_cloud,
    load_normalization_params,
)


# 数据源路径配置
DATA_SOURCES = [
    "AAA/rupture",
    "AAA/unrupture",
    "fast",
    "slow",
    "ILO/sq",
    "ILO/sh",
]


def find_surface_file(case_dir: Path) -> Optional[Path]:
    """
    在病例目录中查找 STL 或 VTP 表面文件。
    优先匹配与目录同名的文件。
    """
    case_name = case_dir.name
    
    # 尝试多种可能的命名格式
    possible_names = [
        f"{case_name}.stl",
        f"{case_name.replace('_', ' ')}.stl",
        f"{case_name.replace(' ', '_')}.stl",
    ]
    
    for name in possible_names:
        prefer_path = case_dir / name
        if prefer_path.exists():
            return prefer_path
    
    # 查找所有 stl 和 vtp 文件
    surface_files = list(case_dir.glob("*.stl")) + list(case_dir.glob("*.vtp"))
    
    if len(surface_files) == 0:
        return None
    elif len(surface_files) > 1:
        print(f"  ⚠️  找到多个表面文件，使用第一个: {surface_files[0].name}")
    
    return surface_files[0]


def find_cloud_files(case_dir: Path, cloud_subdir: str = "ascii_merged_128") -> List[Path]:
    """
    在病例目录中查找所有点云文件。
    排除以 'result_' 开头的文件，避免重复处理输出文件。
    """
    search_dir = case_dir / cloud_subdir
    if not search_dir.is_dir():
        # 回退到病例根目录
        search_dir = case_dir
    
    all_files = list(search_dir.glob("*.csv")) + list(search_dir.glob("*.npy"))
    
    # 过滤掉输出结果文件和 json 文件
    cloud_files = [f for f in all_files 
                   if not f.name.startswith("result_") 
                   and not f.name.endswith(".json")]
    
    return sorted(cloud_files)


def detect_boundary_condition_type(bc_dir: Path) -> Optional[int]:
    """
    检测边界条件类型。
    
    基于出口文件类型来区分：
    - 压力边界：有 p-out*.out 文件（出口为压力）
    - 流量边界：有 flow-out*.out 文件（出口为流量）
    
    返回: 
        1 = 压力边界
        0 = 流量边界
        None = 无法检测
    """
    inlet_file = bc_dir / "vf-in-rfile.out"
    pressure_outlet = bc_dir / "p-outle-rfile.out"
    flow_outlet = bc_dir / "flow-outle-rfile.out"
    
    if inlet_file.exists() and pressure_outlet.exists():
        return 1  # 压力边界
    elif inlet_file.exists() and flow_outlet.exists():
        return 0  # 流量边界
    else:
        # 回退检测
        original_flow_inlet = bc_dir / "report-def-2-rfile.out"
        if original_flow_inlet.exists() and flow_outlet.exists():
            return 0  # 流量边界
        return None


def read_bc_file(file_path: Path) -> Dict[int, float]:
    """
    读取单个边界条件文件，返回 step -> value 的字典。
    """
    data = {}
    if not file_path.exists():
        return data
        
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # 跳过头部，找到数据开始的地方
        start_idx = 0
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                start_idx = i
                break
        
        # 读取数据
        for line in lines[start_idx:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            try:
                step = int(parts[0])
                val = float(parts[1])
                data[step] = val
            except ValueError:
                continue
                
    except Exception as e:
        print(f"  ❌ 读取 {file_path.name} 失败: {e}")
        
    return data


def load_boundary_conditions(bc_dir: Path) -> Dict[int, List[float]]:
    """
    加载边界条件文件。
    
    自动检测边界条件类型：
    - BC_Flag=1: 压力边界 (vf-in + p-out*)
    - BC_Flag=0: 流量边界 (vf-in + flow-out*)
    
    返回一个字典: time_step -> [BC_Flag, in, O1, O2, O3, O4]
    """
    bc_type = detect_boundary_condition_type(bc_dir)
    
    if bc_type is None:
        print(f"  ⚠️  无法检测边界条件类型")
        return {}
    
    bc_type_name = "压力边界" if bc_type == 1 else "流量边界"
    print(f"  📊 边界条件类型: {bc_type_name} (BC_Flag={bc_type})")
    
    # 根据边界条件类型定义文件映射
    if bc_type == 1:  # 压力边界
        file_mapping = {
            "inlet": "vf-in-rfile.out",
            "O1": "p-outle-rfile.out",
            "O2": "p-outli-rfile.out",
            "O3": "p-outre-rfile.out",
            "O4": "p-outri-rfile.out"
        }
    else:  # 流量边界
        file_mapping = {
            "inlet": "vf-in-rfile.out",
            "O1": "flow-outle-rfile.out",
            "O2": "flow-outli-rfile.out",
            "O3": "flow-outre-rfile.out",
            "O4": "flow-outri-rfile.out"
        }
    
    # 读取各个文件的数据
    bc_raw_data = {}
    for key, filename in file_mapping.items():
        file_path = bc_dir / filename
        if not file_path.exists():
            print(f"  ⚠️  未找到: {filename}")
            bc_raw_data[key] = {}
        else:
            bc_raw_data[key] = read_bc_file(file_path)
    
    # 获取所有时间步的并集
    all_steps = set()
    for key in bc_raw_data:
        all_steps.update(bc_raw_data[key].keys())
    
    # 构建最终的边界条件数据
    bc_data = {}
    for step in all_steps:
        inlet_val = bc_raw_data["inlet"].get(step, 0.0)
        o1_val = bc_raw_data["O1"].get(step, 0.0)
        o2_val = bc_raw_data["O2"].get(step, 0.0)
        o3_val = bc_raw_data["O3"].get(step, 0.0)
        o4_val = bc_raw_data["O4"].get(step, 0.0)
        
        bc_data[step] = [float(bc_type), inlet_val, o1_val, o2_val, o3_val, o4_val]
    
    if bc_data:
        print(f"  ✅ 加载 {len(bc_data)} 个时间步的边界条件")
    
    return bc_data


def process_case(
    case_dir: Path,
    output_dir: Optional[Path],
    cloud_subdir: str,
    bc_subdir: str,
    output_subdir: str,
) -> bool:
    """
    处理单个病例目录。
    
    参数:
        case_dir: 病例目录
        output_dir: 输出根目录（None 则输出到病例目录下）
        cloud_subdir: 点云所在子目录
        bc_subdir: 边界条件文件所在子目录
        output_subdir: 输出特征文件的子目录名
    """
    case_name = case_dir.name
    
    # 1. 查找表面模型
    surface_path = find_surface_file(case_dir)
    if not surface_path:
        print(f"  ❌ 跳过: 缺少表面模型")
        return False

    # 2. 查找点云文件
    cloud_files = find_cloud_files(case_dir, cloud_subdir)
    if not cloud_files:
        print(f"  ❌ 跳过: 缺少点云数据 ({cloud_subdir})")
        return False
    
    print(f"  📁 找到 {len(cloud_files)} 个点云文件")
        
    # 3. 准备输出目录
    if output_dir:
        case_output_dir = output_dir / case_name / output_subdir
    else:
        case_output_dir = case_dir / output_subdir
    case_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. 预处理几何数据 (只做一次)
    try:
        print(f"  🔧 预处理几何数据: {surface_path.name}")
        geo_data = prepare_geometry_data(str(surface_path))
    except Exception as e:
        print(f"  ❌ 几何预处理失败: {e}")
        traceback.print_exc()
        return False

    # 5. 加载边界条件
    bc_dir = case_dir / bc_subdir if bc_subdir else case_dir
    global_bcs_map = load_boundary_conditions(bc_dir)
    
    if not global_bcs_map:
        print(f"  ⚠️  未加载到边界条件数据，将跳过边界条件特征")

    # 5.5 加载归一化参数（如果存在）
    cloud_dir = case_dir / cloud_subdir
    norm_params_path = cloud_dir / "normalization_params.json"
    norm_params = load_normalization_params(str(norm_params_path))
    
    if norm_params:
        print(f"  📐 检测到坐标归一化参数，将在特征映射时还原坐标")
        print(f"      方法: {norm_params.get('method', 'unknown')}")
        if norm_params.get('method') == 'center_scale':
            print(f"      缩放因子: {norm_params.get('scale_factor', 'N/A'):.4f}")
    else:
        print(f"  📐 未检测到归一化参数，使用原始坐标进行特征映射")

    # 6. 遍历处理每个点云文件
    success_count = 0
    start_time = time.time()
    
    for i, cloud_path in enumerate(cloud_files, 1):
        cloud_filename = cloud_path.name
        output_filename = f"result_features_{cloud_path.stem}.csv"
        output_path = case_output_dir / output_filename
        
        try:
            # 提取时间步
            time_step = None
            match = re.search(r'-(\d+)\.', cloud_filename)
            if match:
                time_step = int(match.group(1))
            
            current_bcs = None
            if time_step is not None and time_step in global_bcs_map:
                current_bcs = global_bcs_map[time_step]
            elif time_step is not None and global_bcs_map:
                # 找最近的时间步
                available_steps = list(global_bcs_map.keys())
                closest_step = min(available_steps, key=lambda s: abs(s - time_step))
                current_bcs = global_bcs_map[closest_step]
            
            # 调用核心处理函数（传入归一化参数以正确映射特征）
            process_single_cloud(geo_data, str(cloud_path), str(output_path), 
                               global_bcs=current_bcs, norm_params=norm_params)
            success_count += 1
            
            # 进度显示（每 10 个文件或最后一个文件）
            if i % 10 == 0 or i == len(cloud_files):
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = avg_time * (len(cloud_files) - i)
                print(f"  📈 进度: {i}/{len(cloud_files)} "
                      f"({success_count} 成功, 剩余约 {remaining:.1f}s)")
            
        except Exception as e:
            print(f"  ❌ 处理 {cloud_filename} 失败: {e}")
            
    total_time = time.time() - start_time
    
    if success_count > 0:
        print(f"  ✅ 完成: {success_count}/{len(cloud_files)} 个文件, 耗时 {total_time:.1f}s")
        print(f"  📂 输出目录: {case_output_dir}")
        return True
    else:
        print(f"  ❌ 失败: 没有文件被成功处理")
        return False


def scan_cases(data_root: Path, target_case: Optional[str] = None) -> List[Path]:
    """
    扫描 data 目录下的所有病例目录。
    """
    cases = []
    
    for source in DATA_SOURCES:
        source_path = data_root / source
        if not source_path.exists():
            continue
        
        for case_dir in source_path.iterdir():
            if not case_dir.is_dir() or case_dir.name.startswith('.'):
                continue
            
            if target_case:
                case_std = case_dir.name.replace(' ', '_').replace('-', '_').upper()
                target_std = target_case.replace(' ', '_').replace('-', '_').upper()
                if case_std != target_std:
                    continue
            
            cases.append(case_dir)
    
    cases.sort(key=lambda p: p.name)
    return cases


def main():
    parser = argparse.ArgumentParser(
        description="批量几何特征提取（适用于 data 文件夹的插值后数据）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
数据源路径:
  - data/AAA/rupture/
  - data/AAA/unrupture/
  - data/fast/
  - data/slow/
  - data/ILO/sq/
  - data/ILO/sh/

默认配置:
  - 点云输入: ascii_merged_128/
  - 边界条件: out_interpolated_128/
  - 特征输出: ascii_mapped_128/

示例:
  # 处理所有病例
  python batch_process_min.py
  
  # 处理指定病例
  python batch_process_min.py --case FENG_LI_XIN
  
  # 输出到独立目录
  python batch_process_min.py --output-dir ../outdata
  
  # 使用非插值数据
  python batch_process_min.py --cloud-subdir ascii_merged --bc-subdir . --output-subdir ascii_mapped
        """
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data",
        help="data 根目录路径，默认 ../data",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="指定处理的病例名称（可选）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出根目录（默认输出到各病例目录下）",
    )
    parser.add_argument(
        "--cloud-subdir",
        type=str,
        default="ascii_merged_128",
        help="点云所在子目录，默认 ascii_merged_128",
    )
    parser.add_argument(
        "--bc-subdir",
        type=str,
        default="out_interpolated_128",
        help="边界条件文件所在子目录，默认 out_interpolated_128",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="ascii_mapped_128",
        help="输出特征文件的子目录名，默认 ascii_mapped_128",
    )
    
    args = parser.parse_args()

    # 解析路径
    script_dir = Path(__file__).parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = script_dir / data_root
    data_root = data_root.resolve()
    
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = script_dir / output_dir
        output_dir = output_dir.resolve()

    if not data_root.is_dir():
        print(f"❌ data 目录不存在: {data_root}")
        return

    # 扫描病例
    cases = scan_cases(data_root, args.case)

    if not cases:
        if args.case:
            print(f"❌ 未找到病例: {args.case}")
        else:
            print(f"❌ 未找到任何病例")
        return

    print(f"🚀 批量几何特征提取")
    print(f"📁 数据根目录: {data_root}")
    print(f"📂 点云子目录: {args.cloud_subdir}")
    print(f"📂 边界条件子目录: {args.bc_subdir}")
    print(f"📂 输出子目录: {args.output_subdir}")
    if output_dir:
        print(f"📂 输出根目录: {output_dir}")
    print(f"📊 待处理病例数: {len(cases)}")
    print()

    total_start = time.time()
    ok = 0
    
    for idx, case_dir in enumerate(cases, start=1):
        try:
            rel_path = case_dir.relative_to(data_root)
        except ValueError:
            rel_path = case_dir.name
        
        print(f"\n[{idx}/{len(cases)}] {rel_path}")
        print("=" * 50)
        
        if process_case(
            case_dir,
            output_dir,
            args.cloud_subdir,
            args.bc_subdir,
            args.output_subdir,
        ):
            ok += 1

    total_time = time.time() - total_start
    
    print(f"\n{'=' * 50}")
    print(f"🎉 批量处理完成!")
    print(f"⏱️  总耗时: {total_time:.1f}s")
    print(f"✅ 成功: {ok}/{len(cases)} 个病例")
    print(f"⚠️  失败: {len(cases) - ok}/{len(cases)} 个病例")


if __name__ == "__main__":
    main()
