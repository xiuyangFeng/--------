#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
几何特征提取模块

从合并后的点云数据中提取血管几何特征，并添加边界条件。

功能:
1. 读取 STL 表面模型，提取中心线几何特征
2. 将几何特征映射到点云数据
3. 加载边界条件（固定从 Global_conditions 目录）
4. 输出到 processed/features/

几何特征:
- Abscissa: 弧长坐标（归一化到 [0, 1]）
- NormRadius: 归一化半径距离（距壁面距离/局部半径）
- Curvature: 曲率
- Tangent_X/Y/Z: 切线向量分量

边界条件（移除 BC_Flag，统一使用压力边界）:
- BC_Inlet: 入口体积流量
- BC_O1~O4: 四个髂支出口压力

使用示例:
  # 处理单个病例
  python extract_features.py --case ZHANG_CHUN
  
  # 处理所有病例
  python extract_features.py
"""

import argparse
import re
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

# 导入配置和工具
from config import (
    DATA_ROOT,
    MERGED_DIR,
    FEATURES_DIR,
    BC_DIR,
    get_case_dirs,
)
from utils.io import load_boundary_conditions

# 添加父目录到路径，以便导入 vmtk_core
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import vmtk_core


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
        print(f"  ⚠️ 找到多个表面文件，使用第一个: {surface_files[0].name}")
    
    return surface_files[0]


def find_cloud_files(case_dir: Path, cloud_subdir: str) -> List[Path]:
    """
    在病例目录中查找所有点云文件。
    排除以 'result_' 开头的文件，避免重复处理输出文件。
    """
    search_dir = case_dir / cloud_subdir
    if not search_dir.is_dir():
        return []
    
    all_files = list(search_dir.glob("*.csv"))
    
    # 过滤掉输出结果文件
    cloud_files = [f for f in all_files 
                   if not f.name.startswith("result_")]
    
    return sorted(cloud_files)


def prepare_geometry_data(stl_path: str) -> dict:
    """
    预处理几何数据：读取STL，提取中心线，构建KDTree。
    
    参数:
        stl_path: STL 表面文件路径
    
    返回:
        包含所有必要几何对象的字典
    """
    print(f"  🔧 预处理几何数据: {Path(stl_path).name}")
    
    # 读取表面并提取中心线
    surface = vmtk_core.read_surface(stl_path)
    centerline = vmtk_core.extract_rich_centerline(surface)
    
    # 获取中心线数据
    cl_points = vtk_to_numpy(centerline.GetPoints().GetData())
    cl_pd = centerline.GetPointData()
    
    geo_data = {
        "cl_points": cl_points,
        "arr_abscissa": vtk_to_numpy(cl_pd.GetArray("Abscissas")),
        "arr_radius": vtk_to_numpy(cl_pd.GetArray("MaximumInscribedSphereRadius")),
        "arr_curv": vtk_to_numpy(cl_pd.GetArray("Curvature")),
        "arr_tangent": vtk_to_numpy(cl_pd.GetArray("FrenetTangent")),
        "locator": None,
    }
    
    # 创建 KDTree 定位器
    locator = vtk.vtkKdTreePointLocator()
    locator.SetDataSet(centerline)
    locator.BuildLocator()
    geo_data["locator"] = locator
    
    return geo_data


def process_single_cloud(
    geo_data: dict,
    cloud_path: str,
    output_path: str,
    global_bcs: Optional[list] = None,
) -> None:
    """
    使用预处理好的几何数据处理单个点云文件。
    
    参数:
        geo_data: 预处理的几何数据字典
        cloud_path: 点云文件路径
        output_path: 输出文件路径
        global_bcs: 边界条件 [inlet, O1, O2, O3, O4]（5 个值，无 BC_Flag）
    """
    # 解包几何数据
    cl_points = geo_data["cl_points"]
    arr_abscissa = geo_data["arr_abscissa"]
    arr_radius = geo_data["arr_radius"]
    arr_curv = geo_data["arr_curv"]
    arr_tangent = geo_data["arr_tangent"]
    locator = geo_data["locator"]
    
    # 加载点云数据
    df_in = pd.read_csv(cloud_path)
    cloud_xyz = df_in[['x', 'y', 'z']].values
    cloud_others_df = df_in.drop(columns=['x', 'y', 'z'])
    
    n_pts = cloud_xyz.shape[0]
    
    # 映射几何特征
    geo_abscissa = np.zeros(n_pts)
    geo_norm_dist = np.zeros(n_pts)
    geo_curv = np.zeros(n_pts)
    geo_tangent = np.zeros((n_pts, 3))
    
    for i in range(n_pts):
        pt = cloud_xyz[i]
        closest_id = locator.FindClosestPoint(pt)
        
        dist = np.linalg.norm(pt - cl_points[closest_id])
        local_r = arr_radius[closest_id]
        
        geo_abscissa[i] = arr_abscissa[closest_id]
        geo_curv[i] = arr_curv[closest_id]
        geo_tangent[i] = arr_tangent[closest_id]
        
        safe_r = local_r if local_r > 1e-6 else 1.0
        geo_norm_dist[i] = dist / safe_r
    
    # 归一化 Abscissa 到 [0, 1]
    ab_min, ab_max = np.nanmin(geo_abscissa), np.nanmax(geo_abscissa)
    denom = ab_max - ab_min
    if denom > 1e-8:
        geo_abscissa = (geo_abscissa - ab_min) / denom
    else:
        geo_abscissa[:] = 0.0
    
    # 处理 NaN 值
    geo_abscissa = np.nan_to_num(geo_abscissa)
    geo_norm_dist = np.nan_to_num(geo_norm_dist)
    geo_curv = np.nan_to_num(geo_curv)
    geo_tangent = np.nan_to_num(geo_tangent)
    
    # 构建输出 DataFrame
    df_out = pd.DataFrame({
        "x": cloud_xyz[:, 0],
        "y": cloud_xyz[:, 1],
        "z": cloud_xyz[:, 2],
        "Abscissa": geo_abscissa,
        "NormRadius": geo_norm_dist,
        "Curvature": geo_curv,
        "Tangent_X": geo_tangent[:, 0],
        "Tangent_Y": geo_tangent[:, 1],
        "Tangent_Z": geo_tangent[:, 2],
    })
    
    # 添加原始点云的其他特征
    df_out = pd.concat([df_out, cloud_others_df], axis=1)
    
    # 添加边界条件（新格式：5 个值，无 BC_Flag）
    if global_bcs is not None:
        if len(global_bcs) == 5:
            # 新格式: [Inlet, O1, O2, O3, O4]
            bc_names = ["BC_Inlet", "BC_O1", "BC_O2", "BC_O3", "BC_O4"]
            for bc_name, bc_value in zip(bc_names, global_bcs):
                df_out[bc_name] = bc_value
        elif len(global_bcs) == 6:
            # 兼容旧格式: [BC_Flag, Inlet, O1, O2, O3, O4]
            # 跳过 BC_Flag，只保留后 5 个值
            bc_names = ["BC_Inlet", "BC_O1", "BC_O2", "BC_O3", "BC_O4"]
            for bc_name, bc_value in zip(bc_names, global_bcs[1:]):
                df_out[bc_name] = bc_value
        else:
            print(f"  ⚠️ 边界条件格式错误: 期望 5 或 6 个值，实际 {len(global_bcs)} 个")
    
    # 保存输出
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)


def process_case(
    case_dir: Path,
    input_subdir: str = None,
    output_subdir: str = None,
) -> bool:
    """
    处理单个病例目录。
    
    参数:
        case_dir: 病例目录
        input_subdir: 输入点云子目录（默认使用 MERGED_DIR）
        output_subdir: 输出特征子目录（默认使用 FEATURES_DIR）
    
    返回:
        是否成功
    """
    case_dir = Path(case_dir)
    case_name = case_dir.name
    
    if input_subdir is None:
        input_subdir = MERGED_DIR
    if output_subdir is None:
        output_subdir = FEATURES_DIR
    
    # 1. 查找表面模型
    surface_path = find_surface_file(case_dir)
    if not surface_path:
        print(f"  ❌ 跳过: 缺少表面模型")
        return False
    
    # 2. 查找点云文件
    cloud_files = find_cloud_files(case_dir, input_subdir)
    if not cloud_files:
        print(f"  ❌ 跳过: 缺少点云数据 ({input_subdir})")
        return False
    
    print(f"  📁 找到 {len(cloud_files)} 个点云文件")
    
    # 3. 准备输出目录
    output_dir = case_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. 预处理几何数据（只做一次）
    try:
        geo_data = prepare_geometry_data(str(surface_path))
    except Exception as e:
        print(f"  ❌ 几何预处理失败: {e}")
        traceback.print_exc()
        return False
    
    # 5. 加载边界条件（固定使用 Global_conditions 目录）
    bc_dir = case_dir / BC_DIR
    global_bcs_map = load_boundary_conditions(bc_dir)
    
    if not global_bcs_map:
        print(f"  ⚠️ 未加载到边界条件数据，将跳过边界条件特征")
    
    # 6. 处理每个点云文件
    success_count = 0
    start_time = time.time()
    
    for i, cloud_path in enumerate(cloud_files, 1):
        cloud_filename = cloud_path.name
        output_filename = f"result_features_{cloud_path.stem}.csv"
        output_path = output_dir / output_filename
        
        try:
            # 提取时间步
            time_step = None
            match = re.search(r'-(\d+)\.', cloud_filename)
            if not match:
                match = re.search(r'-(\d+)$', cloud_path.stem)
            if match:
                time_step = int(match.group(1))
            
            # 获取对应的边界条件
            current_bcs = None
            if time_step is not None and time_step in global_bcs_map:
                current_bcs = global_bcs_map[time_step]
            elif time_step is not None and global_bcs_map:
                # 找最近的时间步
                available_steps = list(global_bcs_map.keys())
                closest_step = min(available_steps, key=lambda s: abs(s - time_step))
                current_bcs = global_bcs_map[closest_step]
            
            # 处理点云
            process_single_cloud(geo_data, str(cloud_path), str(output_path), 
                               global_bcs=current_bcs)
            success_count += 1
            
            # 进度显示
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
        print(f"  📂 输出目录: {output_dir}")
        return True
    else:
        print(f"  ❌ 失败: 没有文件被成功处理")
        return False


def process_all_cases(
    data_root: Path = None,
    target_case: Optional[str] = None,
    input_subdir: str = None,
    output_subdir: str = None,
) -> None:
    """
    批量处理所有病例。
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
    
    print("🚀 几何特征提取 + 边界条件")
    print("=" * 50)
    print(f"📁 数据根目录: {data_root}")
    print(f"📂 输入子目录: {input_subdir or MERGED_DIR}")
    print(f"📂 输出子目录: {output_subdir or FEATURES_DIR}")
    print(f"📂 边界条件目录: {BC_DIR}")
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
        
        if process_case(case_dir, input_subdir, output_subdir):
            ok += 1
    
    total_time = time.time() - total_start
    
    print(f"\n\n{'=' * 50}")
    print("🎉 批量特征提取完成!")
    print(f"⏱️  总耗时: {total_time:.1f}s")
    print(f"✅ 成功: {ok}/{len(case_dirs)} 个病例")


def main():
    parser = argparse.ArgumentParser(
        description="几何特征提取 + 边界条件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
处理流程:
  1. 读取 STL 表面模型，提取中心线
  2. 从 processed/merged/ 读取合并后的点云
  3. 映射几何特征（Abscissa, NormRadius, Curvature, Tangent）
  4. 从 Global_conditions/ 加载边界条件
  5. 输出到 processed/features/

边界条件说明:
  新格式统一使用压力边界，不再需要 BC_Flag:
  - BC_Inlet: 入口体积流量 (m³/s)
  - BC_O1~O4: 四个髂支出口压力 (Pa)

示例:
  # 处理指定病例
  python extract_features.py --case ZHANG_CHUN
  
  # 处理所有病例
  python extract_features.py
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
        "--input-subdir",
        type=str,
        default=None,
        help=f"输入点云子目录，默认 {MERGED_DIR}",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help=f"输出特征子目录，默认 {FEATURES_DIR}",
    )
    
    args = parser.parse_args()
    
    process_all_cases(
        data_root=args.data_root,
        target_case=args.case,
        input_subdir=args.input_subdir,
        output_subdir=args.output_subdir,
    )


if __name__ == "__main__":
    main()
