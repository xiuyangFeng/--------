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

几何特征（逐点，写入 CSV）:
- Abscissa: 弧长坐标（归一化到 [0, 1]）
- NormRadius: 归一化半径距离（距壁面距离/局部半径）
- Curvature: 曲率
- Tangent_X/Y/Z: 切线向量分量

边界条件（全局条件，保存为 bc_metadata.json 侧文件，不再逐行写入 CSV）:
- BC_Inlet: 入口体积流量
- BC_O1~O4: 四个髂支出口压力

使用示例:
  # 处理单个病例
  python extract_features.py --case ZHANG_CHUN
  
  # 处理所有病例
  python extract_features.py
"""

import argparse
import json
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


def export_centerline(centerline, vtp_path: Path, csv_path: Optional[Path] = None) -> None:
    """
    保存中心线为 VTP（可视化）和 CSV（点云）格式。
    
    参数:
        centerline: VTK PolyData 中心线对象
        vtp_path: VTP 文件输出路径（可在 ParaView 中打开）
        csv_path: CSV 文件输出路径（点云数据，可选）
    """
    vtp_path = Path(vtp_path)
    vtp_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存 VTP 格式（可视化）
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(vtp_path))
    writer.SetInputData(centerline)
    if writer.Write() == 0:
        raise RuntimeError(f"写入 VTP 失败: {vtp_path}")
    
    # 保存 CSV 格式（点云数据）
    if csv_path is None:
        return
    
    csv_path = Path(csv_path)
    pts = vtk_to_numpy(centerline.GetPoints().GetData())
    pd_obj = centerline.GetPointData()
    
    def get_arr(name: str):
        arr = pd_obj.GetArray(name)
        return vtk_to_numpy(arr) if arr is not None else None
    
    # 基础坐标
    data = {
        "x": pts[:, 0],
        "y": pts[:, 1],
        "z": pts[:, 2],
    }
    
    # 导出几何属性
    for key in ["Abscissas", "MaximumInscribedSphereRadius", "Curvature"]:
        arr = get_arr(key)
        if arr is not None:
            data[key] = arr
    
    # 导出切线向量
    tangent = get_arr("FrenetTangent")
    if tangent is not None and len(tangent.shape) > 1 and tangent.shape[1] == 3:
        data["Tangent_X"] = tangent[:, 0]
        data["Tangent_Y"] = tangent[:, 1]
        data["Tangent_Z"] = tangent[:, 2]
    
    pd.DataFrame(data).to_csv(csv_path, index=False)


def prepare_geometry_data(stl_path: str) -> dict:
    """
    预处理几何数据：读取STL，提取中心线，构建KDTree。
    
    参数:
        stl_path: STL 表面文件路径
    
    返回:
        包含所有必要几何对象的字典，包括原始中心线对象
    """
    print(f"  🔧 预处理几何数据: {Path(stl_path).name}")
    
    # 读取表面并提取中心线
    surface = vmtk_core.read_surface(stl_path)
    centerline = vmtk_core.extract_rich_centerline(surface)
    
    # 获取中心线数据
    cl_points = vtk_to_numpy(centerline.GetPoints().GetData())
    cl_pd = centerline.GetPointData()
    
    geo_data = {
        "centerline": centerline,  # 保留原始中心线对象，用于导出
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
) -> None:
    """
    使用预处理好的几何数据处理单个点云文件。
    
    注意: 边界条件不再写入 CSV，而是由 process_case() 统一保存到 bc_metadata.json。
    
    参数:
        geo_data: 预处理的几何数据字典
        cloud_path: 点云文件路径
        output_path: 输出文件路径
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
    
    # 注意: 边界条件不再逐行写入 CSV，改为统一保存到 bc_metadata.json
    
    # 保存输出
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)


def process_case(
    case_dir: Path,
    input_subdir: str = None,
    output_subdir: str = None,
    save_centerline: bool = True,
) -> bool:
    """
    处理单个病例目录。
    
    参数:
        case_dir: 病例目录
        input_subdir: 输入点云子目录（默认使用 MERGED_DIR）
        output_subdir: 输出特征子目录（默认使用 FEATURES_DIR）
        save_centerline: 是否保存中心线文件（VTP + CSV）
    
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
    
    # 5. 保存中心线文件（VTP 可视化 + CSV 点云数据）
    if save_centerline:
        try:
            centerline_dir = case_dir / "centerline"
            vtp_path = centerline_dir / "centerline.vtp"
            csv_path = centerline_dir / "centerline_points.csv"
            export_centerline(geo_data["centerline"], vtp_path, csv_path)
            print(f"  📍 中心线已保存: {centerline_dir.name}/")
        except Exception as e:
            print(f"  ⚠️ 中心线保存失败: {e}")
    
    # 6. 加载边界条件（固定使用 Global_conditions 目录）
    bc_dir = case_dir / BC_DIR
    global_bcs_map = load_boundary_conditions(bc_dir)
    
    if not global_bcs_map:
        print(f"  ⚠️ 未加载到边界条件数据，将跳过边界条件特征")
    
    # 7. 处理每个点云文件，同时收集边界条件元数据
    success_count = 0
    start_time = time.time()
    bc_metadata = {}  # {文件stem: [Inlet, O1, O2, O3, O4]}
    
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
            
            # 处理点云（不再传入 BC，BC 单独保存）
            process_single_cloud(geo_data, str(cloud_path), str(output_path))
            success_count += 1
            
            # 收集边界条件到元数据（统一格式为 5 个值）
            if current_bcs is not None:
                if len(current_bcs) == 5:
                    bc_metadata[cloud_path.stem] = [float(v) for v in current_bcs]
                elif len(current_bcs) == 6:
                    # 旧格式: 跳过 BC_Flag，只保留后 5 个值
                    bc_metadata[cloud_path.stem] = [float(v) for v in current_bcs[1:]]
                else:
                    print(f"  ⚠️ 边界条件格式错误: 期望 5 或 6 个值，实际 {len(current_bcs)} 个")
            
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
    
    # 8. 保存边界条件元数据（全局条件，不再逐行存入 CSV）
    if bc_metadata:
        bc_meta_path = output_dir / "bc_metadata.json"
        bc_meta_content = {
            "description": "边界条件元数据（全局条件，每个时间步共享）",
            "fields": ["BC_Inlet", "BC_O1", "BC_O2", "BC_O3", "BC_O4"],
            "units": {
                "BC_Inlet": "m³/s (入口体积流量)",
                "BC_O1": "Pa (左外髂支出口压力)",
                "BC_O2": "Pa (左内髂支出口压力)",
                "BC_O3": "Pa (右外髂支出口压力)",
                "BC_O4": "Pa (右内髂支出口压力)",
            },
            "data": bc_metadata,
        }
        with open(bc_meta_path, 'w', encoding='utf-8') as f:
            json.dump(bc_meta_content, f, indent=2, ensure_ascii=False)
        print(f"  📋 边界条件元数据已保存: {bc_meta_path.name} ({len(bc_metadata)} 个时间步)")
    else:
        print(f"  ⚠️ 未保存边界条件元数据（无有效数据）")
    
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
    save_centerline: bool = True,
) -> None:
    """
    批量处理所有病例。
    
    参数:
        data_root: 数据根目录
        target_case: 指定处理的病例名称
        input_subdir: 输入点云子目录
        output_subdir: 输出特征子目录
        save_centerline: 是否保存中心线文件（VTP + CSV）
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
    print(f"📍 保存中心线: {'是' if save_centerline else '否'}")
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
        
        if process_case(case_dir, input_subdir, output_subdir, save_centerline):
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
  4. 保存中心线文件（VTP 可视化 + CSV 点云数据）
  5. 从 Global_conditions/ 加载边界条件
  6. 输出到 processed/features/

中心线输出:
  - centerline/centerline.vtp: 可在 ParaView 中打开的可视化文件
  - centerline/centerline_points.csv: 点云数据，包含几何属性

边界条件说明:
  新格式统一使用压力边界，不再需要 BC_Flag:
  - BC_Inlet: 入口体积流量 (m³/s)
  - BC_O1~O4: 四个髂支出口压力 (Pa)

示例:
  # 处理指定病例
  python extract_features.py --case ZHANG_CHUN
  
  # 处理所有病例
  python extract_features.py
  
  # 不保存中心线文件
  python extract_features.py --no-centerline
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
    parser.add_argument(
        "--no-centerline",
        action="store_true",
        default=False,
        help="不保存中心线文件（VTP + CSV），默认保存",
    )
    
    args = parser.parse_args()
    
    process_all_cases(
        data_root=args.data_root,
        target_case=args.case,
        input_subdir=args.input_subdir,
        output_subdir=args.output_subdir,
        save_centerline=not args.no_centerline,
    )


if __name__ == "__main__":
    main()
