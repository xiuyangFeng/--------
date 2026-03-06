#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
血管数据一体化预处理工具
整合功能：数据准备、中心线提取、出口流量生成、几何特征提取
"""

import os
import sys
import shutil
import glob
import csv
import json
import traceback
import time
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import argparse

import pandas as pd
import numpy as np
# vtk、vmtk_core 和 Script_Scenario_B_Volumetric 库延迟加载
# 仅在需要时导入，以避免 vtk 依赖问题
# import vtk
# from vtkmodules.util.numpy_support import vtk_to_numpy
# import vmtk_core
# from Script_Scenario_B_Volumetric import process_single_cloud, prepare_geometry_data

from io import StringIO

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# 工具函数
# ============================================================================

def print_header(title, width=70):
    """打印标题"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")


def print_separator(char="-", width=60):
    """打印分隔线"""
    print(char * width)


def confirm_action(message):
    """确认操作"""
    response = input(f"{message} (y/n): ").strip().lower()
    return response == 'y'


def pause():
    """暂停，等待用户按回车"""
    input("\n按回车键继续...")


def load_mapping_file():
    """加载最新的映射文件"""
    stl_dir = PROJECT_ROOT / "stl_data"
    
    # 优先使用 JSON 格式的映射文件
    json_files = sorted(stl_dir.glob("new_mapping_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if json_files:
        with open(json_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ 已加载映射文件: {json_files[0].name}")
        return data['forward_mapping']
    
    # 否则使用 CSV 格式
    csv_files = sorted(stl_dir.glob("new_mapping_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if csv_files:
        mapping = {}
        df = pd.read_csv(csv_files[0])
        for _, row in df.iterrows():
            mapping[row['Original_Name']] = row['Anonymous_ID']
        print(f"✓ 已加载映射文件: {csv_files[0].name}")
        return mapping
    
    print("⚠️  警告: 未找到映射文件")
    return {}


def get_case_dirs(base_dir: Path) -> List[Path]:
    """获取所有病例目录（过滤隐藏目录）"""
    if not base_dir.exists():
        return []
    return sorted([d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith('.')],
                  key=lambda p: p.name)


# ============================================================================
# 数据清洗模块 (来自 clean_fluent_data.py)
# ============================================================================

# Fluent 输出列名映射
RENAME_MAP = {
    "x-coordinate": "x",
    "y-coordinate": "y",
    "z-coordinate": "z",
    "pressure": "p",
    "x-velocity": "u",
    "y-velocity": "v",
    "z-velocity": "w",
    "velocity-magnitude": "vel_mag",
    "wall-shear": "wss",
    "x-wall-shear": "wss_x",
    "y-wall-shear": "wss_y",
    "z-wall-shear": "wss_z",
    "face-area-magnitude": "face_area",
    "nodenumber": "node_id",
    # ascii_in 文件的列名
    "Node Number": "node_id",
    "X [ m ]": "x",
    "Y [ m ]": "y",
    "Z [ m ]": "z",
    "Pressure [ Pa ]": "p",
    "Velocity [ m s^-1 ]": "vel_mag",
    "Velocity u [ m s^-1 ]": "u",
    "Velocity v [ m s^-1 ]": "v",
    "Velocity w [ m s^-1 ]": "w",
}


def _load_ascii_df(input_path: Path) -> pd.DataFrame:
    """根据文件内容读取 ascii 或 ascii_in 数据"""
    raw_text = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # 尝试定位 ascii_in 的逗号分隔表头
    header_idx = None
    for idx, line in enumerate(raw_text):
        if "Node Number" in line and "," in line:
            header_idx = idx
            break
        if line.strip().lower() == "[data]" and idx + 1 < len(raw_text):
            header_idx = idx + 1
            break

    try:
        if header_idx is not None:
            sliced = "\n".join(raw_text[header_idx:])
            df = pd.read_csv(StringIO(sliced), sep=",", engine="python")
        else:
            df = pd.read_csv(input_path, sep=r"\s+", engine="python")
    except Exception as e:
        raise ValueError(f"无法读取 {input_path}: {e}") from e

    df.columns = [c.strip() for c in df.columns]
    return df


def convert_fluent_to_csv(input_path: Path, output_path: Path, convert_to_mm: bool = True, verbose: bool = True) -> None:
    """清洗单个 Fluent 导出的 ASCII 数据文件并写出 CSV"""
    if verbose:
        print(f"  🧹 清洗: {input_path.name}")

    try:
        df = _load_ascii_df(input_path)
    except Exception as e:
        print(f"    ❌ 读取失败: {e}")
        return

    # 重命名列
    lower_map = {k.lower(): v for k, v in RENAME_MAP.items()}
    rename_dict = {}
    for col in df.columns:
        key = col.strip()
        mapped = RENAME_MAP.get(key) or lower_map.get(key.lower())
        if mapped:
            rename_dict[col] = mapped
    df = df.rename(columns=rename_dict)

    # 清洗节点编号
    if "node_id" in df.columns:
        df = df.drop(columns=["node_id"])

    # 确保几何坐标存在
    missing_xyz = [c for c in ("x", "y", "z") if c not in df.columns]
    if missing_xyz:
        print(f"    ❌ 缺少坐标列 {missing_xyz}")
        return

    # 列顺序
    preferred_cols = ["x", "y", "z", "u", "v", "w", "p", "vel_mag", "wss", "wss_x", "wss_y", "wss_z"]
    available_cols = [c for c in preferred_cols if c in df.columns]

    # 补齐速度列
    for vel_col in ("u", "v", "w", "vel_mag"):
        if vel_col not in df.columns:
            df[vel_col] = 0.0
            if vel_col not in available_cols:
                available_cols.append(vel_col)

    # 坐标单位转换：米 -> 毫米
    if convert_to_mm:
        df["x"] = df["x"] * 1000.0
        df["y"] = df["y"] * 1000.0
        df["z"] = df["z"] * 1000.0

    # 补齐壁面剪切力
    for shear_col in ("wss", "wss_x", "wss_y", "wss_z"):
        if shear_col not in df.columns:
            df[shear_col] = 0.0

    # is_wall 标记
    if {"u", "v", "w"}.issubset(df.columns):
        speed = np.sqrt(df["u"] ** 2 + df["v"] ** 2 + df["w"] ** 2)
        df["is_wall"] = (speed < 1e-6).astype(int)
    else:
        df["is_wall"] = 1

    # 去除面积字段
    if "face_area" in df.columns:
        df = df.drop(columns=["face_area"])

    # 重新排序列
    ordered_cols = available_cols + [c for c in df.columns if c not in available_cols]
    df = df[ordered_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    if verbose:
        print(f"    ✅ 已保存至: {output_path.name}")


# ============================================================================
# 功能1: 数据准备 - 从 data 文件夹复制/整理数据到点云文件夹
# ============================================================================

def scan_data_folder() -> Dict[str, Dict]:
    """
    扫描 data 文件夹，获取所有病例的文件信息
    返回: {anonymous_id: {'source_folder': path, 'stl': path, 'ascii': path, ...}}
    """
    data_dir = PROJECT_ROOT / "data"
    stl_dir = PROJECT_ROOT / "stl_data" / "anonymized"
    
    if not data_dir.exists():
        print("❌ data 文件夹不存在")
        return {}
    
    # 加载映射
    mapping = load_mapping_file()
    if not mapping:
        print("❌ 无法加载映射文件，请先运行 anonymization_tool.py")
        return {}
    
    # 反向映射: anonymous_id -> original_name
    reverse_mapping = {v: k for k, v in mapping.items()}
    
    case_info = {}
    
    # 扫描各个数据源文件夹
    data_sources = [
        ("fast", data_dir / "fast"),
        ("slow", data_dir / "slow"),
        ("AAA/rupture", data_dir / "AAA" / "rupture"),
        ("AAA/unrupture", data_dir / "AAA" / "unrupture"),
        ("ILO/sq", data_dir / "ILO" / "sq"),
        ("ILO/sh", data_dir / "ILO" / "sh"),
    ]
    
    for source_tag, source_path in data_sources:
        if not source_path.exists():
            continue
        
        for patient_dir in source_path.iterdir():
            if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
                continue
            
            # 标准化名字
            patient_name = patient_dir.name.replace(' ', '_').replace('–', '_').replace('-', '_')
            patient_name = '_'.join(filter(None, patient_name.split('_')))
            
            # 查找对应的匿名ID
            anonymous_id = None
            for orig_name, anon_id in mapping.items():
                orig_std = orig_name.replace(' ', '_').replace('–', '_').replace('-', '_')
                orig_std = '_'.join(filter(None, orig_std.split('_')))
                if patient_name.upper() == orig_std.upper():
                    anonymous_id = anon_id
                    break
            
            if not anonymous_id:
                print(f"⚠️  未找到映射: {patient_name} (来源: {source_tag})")
                continue
            
            # 查找文件
            info = {
                'original_name': patient_name,
                'source_folder': patient_dir,
                'source_tag': source_tag,
            }
            
            # STL 文件（从 anonymized 文件夹）
            stl_file = stl_dir / f"{anonymous_id}.stl"
            if stl_file.exists():
                info['stl'] = stl_file
            
            # ascii 文件夹
            ascii_dir = patient_dir / "ascii"
            if ascii_dir.exists():
                info['ascii'] = ascii_dir
            
            # ascii_in 文件夹（入口点云）
            ascii_in_dir = patient_dir / "ascii_in"
            if ascii_in_dir.exists():
                info['ascii_in'] = ascii_in_dir
            
            # 边界条件文件
            bc_files = {
                'vf-in': patient_dir / "vf-in-rfile.out",
                'p-outle': patient_dir / "p-outle-rfile.out",
                'p-outli': patient_dir / "p-outli-rfile.out",
                'p-outre': patient_dir / "p-outre-rfile.out",
                'p-outri': patient_dir / "p-outri-rfile.out",
                'report-def-2': patient_dir / "report-def-2-rfile.out",
                'outlet-flow-ratio': patient_dir / "outlet-flow-ratio.csv",
            }
            info['bc_files'] = {k: v for k, v in bc_files.items() if v.exists()}
            
            case_info[anonymous_id] = info
    
    return case_info


def prepare_case_data(dry_run=False):
    """准备病例数据：从 data 和 stl_data 文件夹整理到点云文件夹"""
    print_header("数据准备 - 整理病例文件")
    
    case_info = scan_data_folder()
    
    if not case_info:
        print("❌ 未找到任何病例数据")
        pause()
        return
    
    print(f"📊 找到 {len(case_info)} 个病例\n")
    
    point_cloud_dir = PROJECT_ROOT / "点云"
    point_cloud_dir.mkdir(exist_ok=True)
    
    success_count = 0
    
    for anonymous_id, info in case_info.items():
        print(f"\n处理病例: {anonymous_id}")
        print(f"  原始名字: {info['original_name']}")
        print(f"  数据来源: {info['source_tag']}")
        
        # 创建病例目录
        case_dir = point_cloud_dir / anonymous_id
        
        if dry_run:
            print(f"  [预演] 将创建目录: {case_dir}")
        else:
            case_dir.mkdir(exist_ok=True)
        
        # 复制 STL 文件
        if 'stl' in info:
            dest_stl = case_dir / f"{anonymous_id}.stl"
            if dry_run:
                print(f"  [预演] 复制 STL: {info['stl'].name} -> {dest_stl.name}")
            else:
                if not dest_stl.exists():
                    shutil.copy2(info['stl'], dest_stl)
                    print(f"  ✓ 复制 STL: {dest_stl.name}")
                else:
                    print(f"  ⊙ STL 已存在: {dest_stl.name}")
        
        # 复制 ascii 文件夹（CFD 点云数据）
        if 'ascii' in info:
            dest_ascii = case_dir / "ascii"
            if dry_run:
                print(f"  [预演] 复制 ascii 文件夹")
            else:
                if not dest_ascii.exists():
                    shutil.copytree(info['ascii'], dest_ascii)
                    print(f"  ✓ 复制 ascii 文件夹 ({len(list(dest_ascii.iterdir()))} 个文件)")
                else:
                    print(f"  ⊙ ascii 文件夹已存在")
        
        # 复制 ascii_in 文件夹
        if 'ascii_in' in info:
            dest_ascii_in = case_dir / "ascii_in"
            if dry_run:
                print(f"  [预演] 复制 ascii_in 文件夹")
            else:
                if not dest_ascii_in.exists():
                    shutil.copytree(info['ascii_in'], dest_ascii_in)
                    print(f"  ✓ 复制 ascii_in 文件夹 ({len(list(dest_ascii_in.iterdir()))} 个文件)")
                else:
                    print(f"  ⊙ ascii_in 文件夹已存在")
        
        # 复制边界条件文件
        if info['bc_files']:
            copied_bc = 0
            for bc_name, bc_file in info['bc_files'].items():
                dest_bc = case_dir / bc_file.name
                if dry_run:
                    print(f"  [预演] 复制 BC: {bc_file.name}")
                else:
                    if not dest_bc.exists():
                        shutil.copy2(bc_file, dest_bc)
                        copied_bc += 1
            if not dry_run:
                if copied_bc > 0:
                    print(f"  ✓ 复制 {copied_bc} 个边界条件文件")
                else:
                    print(f"  ⊙ 边界条件文件已存在")
        
        if not dry_run:
            success_count += 1
    
    print_separator()
    if dry_run:
        print("\n[预演模式] 未实际执行任何操作")
    else:
        print(f"\n✅ 数据准备完成！成功处理 {success_count}/{len(case_info)} 个病例")
        print(f"📂 病例数据保存在: {point_cloud_dir}")
    
    pause()


# ============================================================================
# 功能2: 中心线提取
# ============================================================================

def find_stl(case_dir: Path) -> Optional[Path]:
    """在病例目录下寻找 STL 文件"""
    case_name = case_dir.name
    prefer_path = case_dir / f"{case_name}.stl"
    if prefer_path.exists():
        return prefer_path
    
    for fname in case_dir.iterdir():
        if fname.suffix.lower() == ".stl":
            return fname
    return None


def export_centerline(centerline, vtp_path: Path, csv_path: Optional[Path]):
    """保存中心线为 VTP 和 CSV 格式"""
    # 延迟加载 VTK 库
    try:
        import vtk
        from vtkmodules.util.numpy_support import vtk_to_numpy
    except ImportError as e:
        raise ImportError(
            "无法导入 vtk 库。请安装依赖: conda install -c conda-forge vtk netcdf4"
        ) from e
    
    vtp_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存 VTP
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(vtp_path))
    writer.SetInputData(centerline)
    if writer.Write() == 0:
        raise RuntimeError(f"写入 VTP 失败: {vtp_path}")
    
    if not csv_path:
        return
    
    pts = vtk_to_numpy(centerline.GetPoints().GetData())
    pd_obj = centerline.GetPointData()
    
    def get_arr(name: str):
        arr = pd_obj.GetArray(name)
        return vtk_to_numpy(arr) if arr is not None else None
    
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
    
    tangent = get_arr("FrenetTangent")
    if tangent is not None and len(tangent.shape) > 1 and tangent.shape[1] == 3:
        data["Tangent_X"] = tangent[:, 0]
        data["Tangent_Y"] = tangent[:, 1]
        data["Tangent_Z"] = tangent[:, 2]
    
    pd.DataFrame(data).to_csv(csv_path, index=False)


def extract_centerlines(overwrite=False, save_csv=True):
    """批量提取中心线"""
    print_header("中心线提取")
    
    # 检查并延迟导入 VTK 和 vmtk_core 库
    try:
        import vtk
        try:
            from legacy.preprocess import vmtk_core
        except ImportError:
            import vmtk_core
    except ImportError as e:
        print(f"❌ 无法导入所需的库: {e}")
        print("   请运行以下命令安装依赖:")
        print("   conda install -c conda-forge vtk netcdf4")
        print("   或者:")
        print("   conda install -c conda-forge netcdf4")
        pause()
        return
    
    point_cloud_dir = PROJECT_ROOT / "点云"
    if not point_cloud_dir.exists():
        print("❌ 点云目录不存在，请先运行数据准备")
        pause()
        return
    
    case_dirs = get_case_dirs(point_cloud_dir)
    
    if not case_dirs:
        print("❌ 未找到任何病例")
        pause()
        return
    
    print(f"📊 找到 {len(case_dirs)} 个病例\n")
    
    success_count = 0
    
    for idx, case_dir in enumerate(case_dirs, 1):
        case_name = case_dir.name
        print(f"[{idx}/{len(case_dirs)}] 处理 {case_name} ...")
        
        stl_path = find_stl(case_dir)
        if not stl_path:
            print(f"  ⚠️  跳过: 未找到 STL 文件")
            continue
        
        centerline_dir = case_dir / "centerline"
        vtp_path = centerline_dir / "centerline.vtp"
        csv_path = centerline_dir / "centerline_points.csv" if save_csv else None
        
        if vtp_path.exists() and not overwrite:
            print(f"  ⊙ 中心线已存在，使用 --overwrite 可重算")
            success_count += 1
            continue
        
        try:
            surface = vmtk_core.read_surface(str(stl_path))
            centerline = vmtk_core.extract_rich_centerline(surface)
            export_centerline(centerline, vtp_path, csv_path)
            print(f"  ✅ 完成，保存至 {centerline_dir.name}/")
            success_count += 1
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            if confirm_action("  是否显示详细错误信息？"):
                traceback.print_exc()
    
    print_separator()
    print(f"\n✅ 中心线提取完成！成功处理 {success_count}/{len(case_dirs)} 个病例")
    pause()


# ============================================================================
# 功能3: 生成出口流量文件
# ============================================================================

def read_fluent_report_file(file_path: Path) -> List[Dict]:
    """读取 FLUENT report 格式的 .out 文件"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 跳过前3行
    data_lines = lines[3:]
    
    data = []
    for line in data_lines:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) == 3:
                data.append({
                    'Time Step': int(parts[0]),
                    'Value': float(parts[1]),
                    'flow-time': float(parts[2])
                })
    
    return data


def read_outlet_flow_ratio(file_path: Path) -> Dict[str, float]:
    """读取出口流量比例文件"""
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        values = next(reader)
    
    ratios = {}
    for col_name, value in zip(headers, values):
        ratios[col_name] = float(value)
    
    return ratios


def write_fluent_report_file(file_path: Path, data: List[Dict], outlet_name: str):
    """写入 FLUENT report 格式的 .out 文件"""
    file_name = file_path.stem
    
    with open(file_path, 'w') as f:
        f.write(f'"{file_name}"\n')
        f.write(f'"Time Step" "{outlet_name} etc.."\n')
        f.write(f'("Time Step" "{outlet_name}" "flow-time")\n')
        
        for row in data:
            f.write(f"{row['Time Step']} {row['Value']:.15g} {row['flow-time']:.15g}\n")


def generate_outlet_flows(overwrite=False, density=1060.0):
    """批量生成出口流量文件"""
    print_header("生成出口流量文件")
    
    point_cloud_dir = PROJECT_ROOT / "点云"
    if not point_cloud_dir.exists():
        print("❌ 点云目录不存在")
        pause()
        return
    
    case_dirs = get_case_dirs(point_cloud_dir)
    
    if not case_dirs:
        print("❌ 未找到任何病例")
        pause()
        return
    
    print(f"📊 找到 {len(case_dirs)} 个病例")
    print(f"⚙️  血液密度: {density} kg/m³\n")
    
    processed_count = 0
    
    for idx, case_dir in enumerate(case_dirs, 1):
        case_name = case_dir.name
        print(f"[{idx}/{len(case_dirs)}] 处理 {case_name} ...")
        
        inlet_path = case_dir / "report-def-2-rfile.out"
        ratio_path = case_dir / "outlet-flow-ratio.csv"
        
        if not inlet_path.exists():
            print(f"  ⚠️  跳过: 未找到入口流量文件")
            continue
        
        if not ratio_path.exists():
            print(f"  ⚠️  跳过: 未找到流量比文件")
            continue
        
        try:
            # 读取数据
            inlet_data = read_fluent_report_file(inlet_path)
            flow_ratios = read_outlet_flow_ratio(ratio_path)
            
            generated = 0
            for outlet_name, ratio in flow_ratios.items():
                output_file = case_dir / f"{outlet_name}.out"
                
                if output_file.exists() and not overwrite:
                    continue
                
                # 计算出口流量
                outlet_data = []
                for row in inlet_data:
                    outlet_data.append({
                        'Time Step': row['Time Step'],
                        'Value': (row['Value'] / density) * ratio,
                        'flow-time': row['flow-time']
                    })
                
                write_fluent_report_file(output_file, outlet_data, outlet_name)
                generated += 1
            
            if generated > 0:
                print(f"  ✅ 生成 {generated} 个出口流量文件")
                processed_count += 1
            else:
                print(f"  ⊙ 出口流量文件已存在")
                processed_count += 1
        
        except Exception as e:
            print(f"  ❌ 失败: {e}")
    
    print_separator()
    print(f"\n✅ 出口流量生成完成！成功处理 {processed_count}/{len(case_dirs)} 个病例")
    pause()


# ============================================================================
# 功能4: 数据清洗 - 清洗 Fluent ASCII 文件
# ============================================================================

def clean_cfd_data(convert_to_mm=True):
    """批量清洗 CFD ASCII 数据文件"""
    print_header("数据清洗 - Fluent ASCII 文件")
    
    point_cloud_dir = PROJECT_ROOT / "点云"
    if not point_cloud_dir.exists():
        print("❌ 点云目录不存在")
        pause()
        return
    
    case_dirs = get_case_dirs(point_cloud_dir)
    
    if not case_dirs:
        print("❌ 未找到任何病例")
        pause()
        return
    
    print(f"📊 找到 {len(case_dirs)} 个病例")
    print(f"⚙️  坐标单位转换: {'米 -> 毫米' if convert_to_mm else '保持原单位'}\n")
    
    # 需要处理的目录对 (输入子目录, 输出子目录)
    dir_pairs = [
        ("ascii", "ascii_clean"),
        ("ascii_in", "ascii_in_clean"),
    ]
    
    processed_count = 0
    total_files = 0
    
    for idx, case_dir in enumerate(case_dirs, 1):
        case_name = case_dir.name
        print(f"[{idx}/{len(case_dirs)}] 处理 {case_name} ...")
        
        case_has_files = False
        
        for in_dir_name, out_dir_name in dir_pairs:
            ascii_dir = case_dir / in_dir_name
            if not ascii_dir.is_dir():
                continue
            
            out_dir = case_dir / out_dir_name
            
            # 获取所有文件
            ascii_files = [f for f in ascii_dir.iterdir() if f.is_file()]
            if not ascii_files:
                continue
            
            print(f"  📁 {in_dir_name} -> {out_dir_name} ({len(ascii_files)} 个文件)")
            case_has_files = True
            
            for ascii_file in sorted(ascii_files):
                output_path = out_dir / f"{ascii_file.stem}.csv"
                
                try:
                    convert_fluent_to_csv(ascii_file, output_path, convert_to_mm=convert_to_mm, verbose=False)
                    total_files += 1
                except Exception as e:
                    print(f"    ❌ {ascii_file.name} 失败: {e}")
        
        if case_has_files:
            print(f"  ✅ 完成")
            processed_count += 1
        else:
            print(f"  ⚠️  未找到 ascii 或 ascii_in 目录")
    
    print_separator()
    print(f"\n✅ 数据清洗完成！")
    print(f"   成功处理 {processed_count}/{len(case_dirs)} 个病例")
    print(f"   清洗文件总数: {total_files}")
    pause()


# ============================================================================
# 功能5: 合并点云数据
# ============================================================================

def merge_point_clouds():
    """合并 ascii_clean 和 ascii_in_clean 的点云数据"""
    print_header("合并点云数据")
    
    point_cloud_dir = PROJECT_ROOT / "点云"
    if not point_cloud_dir.exists():
        print("❌ 点云目录不存在")
        pause()
        return
    
    case_dirs = get_case_dirs(point_cloud_dir)
    
    if not case_dirs:
        print("❌ 未找到任何病例")
        pause()
        return
    
    print(f"📊 找到 {len(case_dirs)} 个病例\n")
    
    processed_count = 0
    
    for idx, case_dir in enumerate(case_dirs, 1):
        case_name = case_dir.name
        print(f"[{idx}/{len(case_dirs)}] 处理 {case_name} ...")
        
        ascii_clean_dir = case_dir / "ascii_clean"
        ascii_in_clean_dir = case_dir / "ascii_in_clean"
        merged_dir = case_dir / "ascii_merged"
        
        if not ascii_clean_dir.exists():
            print(f"  ⚠️  跳过: 未找到 ascii_clean 目录")
            continue
        
        # 获取所有清洗后的文件
        wall_files = sorted(ascii_clean_dir.glob("*.csv"))
        
        if not wall_files:
            print(f"  ⚠️  跳过: ascii_clean 目录为空")
            continue
        
        merged_dir.mkdir(exist_ok=True)
        merged_count = 0
        
        for wall_file in wall_files:
            base_name = wall_file.stem
            merged_file = merged_dir / f"{base_name}.csv"
            
            try:
                # 读取壁面数据
                df_wall = pd.read_csv(wall_file)
                
                # 尝试读取内部数据（如果存在）
                if ascii_in_clean_dir.exists():
                    in_file = ascii_in_clean_dir / wall_file.name
                    if in_file.exists():
                        df_in = pd.read_csv(in_file)
                        # 合并
                        df_merged = pd.concat([df_wall, df_in], ignore_index=True)
                    else:
                        df_merged = df_wall
                else:
                    df_merged = df_wall
                
                # 保存合并后的数据
                df_merged.to_csv(merged_file, index=False)
                merged_count += 1
                
            except Exception as e:
                print(f"    ❌ {wall_file.name} 合并失败: {e}")
        
        if merged_count > 0:
            print(f"  ✅ 合并 {merged_count} 个文件")
            processed_count += 1
        else:
            print(f"  ⚠️  没有文件被合并")
    
    print_separator()
    print(f"\n✅ 点云合并完成！成功处理 {processed_count}/{len(case_dirs)} 个病例")
    pause()


# ============================================================================
# 功能6: 几何特征提取
# ============================================================================

def load_boundary_conditions(case_dir: Path) -> Dict[int, Dict[str, float]]:
    """加载边界条件文件"""
    file_mapping = {
        "vf-in-rfile.out": "Inlet_Velocity",
        "p-outle-rfile.out": "Outlet_Pressure_LE",
        "p-outli-rfile.out": "Outlet_Pressure_LI",
        "p-outre-rfile.out": "Outlet_Pressure_RE",
        "p-outri-rfile.out": "Outlet_Pressure_RI"
    }
    
    bc_data = {}
    
    for filename, feature_name in file_mapping.items():
        file_path = case_dir / filename
        if not file_path.exists():
            continue
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # 跳过头部
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
                    
                    if step not in bc_data:
                        bc_data[step] = {}
                    
                    bc_data[step][feature_name] = val
                except ValueError:
                    continue
        
        except Exception as e:
            print(f"  ⚠️  读取 {filename} 失败: {e}")
    
    return bc_data


def extract_geometric_features(cloud_subdir="ascii_merged", output_subdir="ascii_mapped"):
    """批量提取几何特征"""
    print_header("几何特征提取")
    
    # 延迟导入依赖 vtk 的模块
    try:
        try:
            from legacy.preprocess.Script_Scenario_B_Volumetric import (
                process_single_cloud,
                prepare_geometry_data,
            )
        except ImportError:
            from Script_Scenario_B_Volumetric import process_single_cloud, prepare_geometry_data
    except ImportError as e:
        print(f"❌ 无法导入所需的库: {e}")
        print("   请运行以下命令安装依赖:")
        print("   conda install -c conda-forge vtk netcdf4")
        pause()
        return
    
    point_cloud_dir = PROJECT_ROOT / "点云"
    output_dir = PROJECT_ROOT / "outdata"
    
    if not point_cloud_dir.exists():
        print("❌ 点云目录不存在")
        pause()
        return
    
    case_dirs = get_case_dirs(point_cloud_dir)
    
    if not case_dirs:
        print("❌ 未找到任何病例")
        pause()
        return
    
    print(f"📊 找到 {len(case_dirs)} 个病例")
    print(f"📁 点云子目录: {cloud_subdir}")
    print(f"📁 输出子目录: {output_subdir}\n")
    
    processed_count = 0
    
    for idx, case_dir in enumerate(case_dirs, 1):
        case_name = case_dir.name
        print(f"\n[{idx}/{len(case_dirs)}] 处理 {case_name} ...")
        
        # 查找 STL 文件
        stl_path = find_stl(case_dir)
        if not stl_path:
            print(f"  ⚠️  跳过: 未找到 STL 文件")
            continue
        
        # 查找点云文件
        cloud_dir = case_dir / cloud_subdir
        if not cloud_dir.exists():
            cloud_dir = case_dir / "ascii"  # 回退到 ascii 目录
        
        if not cloud_dir.exists():
            print(f"  ⚠️  跳过: 未找到点云目录")
            continue
        
        cloud_files = list(cloud_dir.glob("*.csv")) + list(cloud_dir.glob("*.npy"))
        cloud_files = [f for f in cloud_files if "result_" not in f.name]
        
        if not cloud_files:
            print(f"  ⚠️  跳过: 未找到点云文件")
            continue
        
        # 准备输出目录
        case_output_dir = output_dir / case_name / output_subdir
        case_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 预处理几何数据
        try:
            geo_data = prepare_geometry_data(str(stl_path))
        except Exception as e:
            print(f"  ❌ 几何预处理失败: {e}")
            continue
        
        # 加载边界条件
        global_bcs = load_boundary_conditions(case_dir)
        
        # 处理每个点云文件
        success = 0
        for cloud_file in cloud_files:
            cloud_filename = cloud_file.name
            output_filename = f"result_features_{cloud_file.stem}.csv"
            output_path = case_output_dir / output_filename
            
            # 提取时间步
            time_step = None
            match = re.search(r'-(\d+)\.', cloud_filename)
            if match:
                time_step = int(match.group(1))
            
            current_bcs = {}
            if time_step is not None and time_step in global_bcs:
                current_bcs = global_bcs[time_step]
            
            try:
                process_single_cloud(geo_data, str(cloud_file), str(output_path), global_bcs=current_bcs)
                success += 1
            except Exception as e:
                print(f"  ❌ 处理 {cloud_filename} 失败: {e}")
        
        if success > 0:
            print(f"  ✅ 成功处理 {success}/{len(cloud_files)} 个点云文件")
            processed_count += 1
        else:
            print(f"  ⚠️  没有文件被成功处理")
    
    print_separator()
    print(f"\n✅ 几何特征提取完成！成功处理 {processed_count}/{len(case_dirs)} 个病例")
    print(f"📂 结果保存在: {output_dir}")
    pause()


# ============================================================================
# 功能7: 完整流程
# ============================================================================

def run_full_pipeline():
    """运行完整的预处理流程"""
    print_header("完整预处理流程")
    print("此流程将依次执行:")
    print("  1. 数据准备 (从 data 和 stl_data 整理到点云文件夹)")
    print("  2. 中心线提取 (为每个病例提取血管中心线)")
    print("  3. 生成出口流量 (根据入口流量和流量比计算)")
    print("  4. 数据清洗 (清洗 Fluent ASCII 文件)")
    print("  5. 合并点云 (合并壁面和内部点云)")
    print("  6. 几何特征提取 (提取血管几何特征)")
    print()
    
    if not confirm_action("是否继续？"):
        return
    
    # 步骤1: 数据准备
    print_header("步骤 1/6: 数据准备")
    if confirm_action("是否执行数据准备？"):
        prepare_case_data(dry_run=False)
    
    # 步骤2: 中心线提取
    print_header("步骤 2/6: 中心线提取")
    if confirm_action("是否执行中心线提取？"):
        extract_centerlines(overwrite=False, save_csv=True)
    
    # 步骤3: 生成出口流量
    print_header("步骤 3/6: 生成出口流量")
    if confirm_action("是否生成出口流量？"):
        generate_outlet_flows(overwrite=False, density=1060.0)
    
    # 步骤4: 数据清洗
    print_header("步骤 4/6: 数据清洗")
    if confirm_action("是否清洗 CFD 数据？"):
        clean_cfd_data(convert_to_mm=True)
    
    # 步骤5: 合并点云
    print_header("步骤 5/6: 合并点云")
    if confirm_action("是否合并点云数据？"):
        merge_point_clouds()
    
    # 步骤6: 几何特征提取
    print_header("步骤 6/6: 几何特征提取")
    if confirm_action("是否提取几何特征？"):
        extract_geometric_features(cloud_subdir="ascii_merged", output_subdir="ascii_mapped")
    
    print_header("完成！")
    print("✅ 完整预处理流程已完成")
    pause()


# ============================================================================
# 主菜单
# ============================================================================

def show_menu():
    """显示主菜单"""
    print_header("血管数据一体化预处理工具", 70)
    print("功能列表:")
    print("  1. 完整流程 (依次执行所有步骤)")
    print("  2. 数据准备 (从 data 和 stl_data 整理到点云文件夹)")
    print("  3. 中心线提取 (批量提取血管中心线)")
    print("  4. 生成出口流量 (根据入口流量和流量比计算)")
    print("  5. 数据清洗 (清洗 Fluent ASCII 文件)")
    print("  6. 合并点云 (合并壁面和内部点云)")
    print("  7. 几何特征提取 (提取血管几何特征)")
    print("  8. 数据准备 [预演模式] (仅显示计划，不实际执行)")
    print("  0. 退出")
    print("=" * 70)


def main():
    """主函数"""
    os.chdir(PROJECT_ROOT)
    
    while True:
        try:
            show_menu()
            choice = input("\n请选择功能 (0-8): ").strip()
            
            if choice == '0':
                print("\n感谢使用！再见！")
                break
            elif choice == '1':
                run_full_pipeline()
            elif choice == '2':
                prepare_case_data(dry_run=False)
            elif choice == '3':
                extract_centerlines(overwrite=False, save_csv=True)
            elif choice == '4':
                generate_outlet_flows(overwrite=False, density=1060.0)
            elif choice == '5':
                clean_cfd_data(convert_to_mm=True)
            elif choice == '6':
                merge_point_clouds()
            elif choice == '7':
                extract_geometric_features(cloud_subdir="ascii_merged", output_subdir="ascii_mapped")
            elif choice == '8':
                prepare_case_data(dry_run=True)
            else:
                print("\n无效选择，请输入 0-8 之间的数字")
                pause()
        
        except KeyboardInterrupt:
            print("\n\n操作已取消")
            if confirm_action("是否退出程序？"):
                print("再见！")
                break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            traceback.print_exc()
            pause()


if __name__ == "__main__":
    main()
