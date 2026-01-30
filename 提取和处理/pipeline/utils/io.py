#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据读写工具模块

处理 Fluent 输出的 ASCII 数据文件和边界条件文件的读写。
"""

from pathlib import Path
from typing import Dict, Optional
from io import StringIO

import pandas as pd
import numpy as np

# 添加父目录到路径，以便导入配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BC_FILE_MAPPING, BC_DIR, BLOOD_DENSITY


# Fluent 输出列名映射到统一字段
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
    "cellnumber": "node_id",
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


def load_ascii_df(input_path: Path) -> pd.DataFrame:
    """
    根据文件内容读取 ascii 或 ascii_in 数据。
    
    支持两种格式：
    1. 逗号分隔的 CSV 格式
    2. 空格分隔的格式
    
    参数:
        input_path: 输入文件路径
    
    返回:
        原始数据 DataFrame
    """
    input_path = Path(input_path)
    raw_text = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # 尝试定位逗号分隔表头
    header_idx = None
    for idx, line in enumerate(raw_text):
        # 检测逗号分隔的表头
        if "Node Number" in line and "," in line:
            header_idx = idx
            break
        # 检测 [data] 标记
        if line.strip().lower() == "[data]" and idx + 1 < len(raw_text):
            header_idx = idx + 1
            break
        # 检测常见的 Fluent 列名
        if any(key in line for key in ["x-coordinate", "nodenumber", "cellnumber"]):
            # 判断是逗号分隔还是空格分隔
            if "," in line:
                header_idx = idx
                break

    try:
        if header_idx is not None:
            sliced = "\n".join(raw_text[header_idx:])
            # 尝试逗号分隔
            if "," in raw_text[header_idx]:
                df = pd.read_csv(StringIO(sliced), sep=",", engine="python")
            else:
                df = pd.read_csv(StringIO(sliced), sep=r"\s+", engine="python")
        else:
            # 默认尝试空格分隔
            df = pd.read_csv(input_path, sep=r"\s+", engine="python")
    except Exception as e:
        raise ValueError(f"无法读取 {input_path}: {e}") from e

    df.columns = [c.strip() for c in df.columns]
    return df


def clean_cfd_data(df: pd.DataFrame, convert_to_mm: bool = True) -> pd.DataFrame:
    """
    清洗单个 CFD 数据的 DataFrame。
    
    参数:
        df: 原始数据 DataFrame
        convert_to_mm: 是否将坐标从米转换为毫米
    
    返回:
        清洗后的 DataFrame
    """
    # 重命名列
    lower_map = {k.lower(): v for k, v in RENAME_MAP.items()}
    rename_dict = {}
    for col in df.columns:
        key = col.strip()
        mapped = RENAME_MAP.get(key) or lower_map.get(key.lower())
        if mapped:
            rename_dict[col] = mapped
    df = df.rename(columns=rename_dict)
    
    # 移除节点编号
    if "node_id" in df.columns:
        df = df.drop(columns=["node_id"])
    
    # 确保几何坐标存在
    missing_xyz = [c for c in ("x", "y", "z") if c not in df.columns]
    if missing_xyz:
        raise ValueError(f"缺少坐标列 {missing_xyz}")
    
    # 补齐速度列
    for vel_col in ("u", "v", "w", "vel_mag"):
        if vel_col not in df.columns:
            df[vel_col] = 0.0
    
    # 坐标单位转换：米 -> 毫米
    if convert_to_mm:
        df["x"] = df["x"] * 1000.0
        df["y"] = df["y"] * 1000.0
        df["z"] = df["z"] * 1000.0
    
    # 补齐壁面剪切力
    for shear_col in ("wss", "wss_x", "wss_y", "wss_z"):
        if shear_col not in df.columns:
            df[shear_col] = 0.0
    
    # is_wall 标记：基于速度判断
    if {"u", "v", "w"}.issubset(df.columns):
        speed = np.sqrt(df["u"] ** 2 + df["v"] ** 2 + df["w"] ** 2)
        df["is_wall"] = (speed < 1e-6).astype(int)
    else:
        df["is_wall"] = 1
    
    # 移除面积字段
    if "face_area" in df.columns:
        df = df.drop(columns=["face_area"])
    
    # 整理列顺序
    preferred_cols = ["x", "y", "z", "u", "v", "w", "p", "vel_mag", 
                      "wss", "wss_x", "wss_y", "wss_z", "is_wall"]
    available_cols = [c for c in preferred_cols if c in df.columns]
    ordered_cols = available_cols + [c for c in df.columns if c not in available_cols]
    df = df[ordered_cols]
    
    return df


def save_csv(df: pd.DataFrame, output_path: Path, **kwargs) -> None:
    """
    保存 DataFrame 到 CSV 文件
    
    参数:
        df: 数据 DataFrame
        output_path: 输出路径
        **kwargs: 传递给 to_csv 的额外参数
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, **kwargs)


def load_bc_file(file_path: Path) -> Dict[int, float]:
    """
    读取单个边界条件文件，返回 step -> value 的字典。
    
    参数:
        file_path: 边界条件文件路径
    
    返回:
        时间步 -> 值的字典
    """
    data = {}
    file_path = Path(file_path)
    
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
        print(f"  ⚠️ 读取 {file_path.name} 失败: {e}")
        
    return data


def load_inlet_flow(bc_dir: Path) -> Dict[int, float]:
    """
    加载入口流量，自动处理不同文件格式。
    
    优先级：
    1. vf-in-rfile.out - 体积流量（直接使用）
    2. report-file-2.out - 质量流量（需转换）
    
    参数:
        bc_dir: 边界条件目录
    
    返回:
        时间步 -> 体积流量的字典
    """
    bc_dir = Path(bc_dir)
    inlet_config = BC_FILE_MAPPING["inlet"]
    
    # 优先使用体积流量文件
    vf_file = bc_dir / inlet_config["primary"]
    if vf_file.exists():
        return load_bc_file(vf_file)
    
    # 回退到质量流量文件（需要转换）
    mf_file = bc_dir / inlet_config["fallback"]
    if mf_file.exists():
        mass_flow = load_bc_file(mf_file)
        # 质量流量 → 体积流量: Q = m / ρ
        return {step: val / BLOOD_DENSITY for step, val in mass_flow.items()}
    
    return {}


def load_boundary_conditions(bc_dir: Path) -> Dict[int, list]:
    """
    加载边界条件文件。
    
    新格式统一为：入口体积流量 + 四个髂支出口压力
    不再需要 BC_Flag 参数。
    
    返回: time_step -> [inlet, O1, O2, O3, O4]
    """
    bc_dir = Path(bc_dir)
    
    if not bc_dir.exists():
        print(f"  ⚠️ 边界条件目录不存在: {bc_dir}")
        return {}
    
    # 加载入口流量（支持兼容性）
    inlet_data = load_inlet_flow(bc_dir)
    
    if not inlet_data:
        print(f"  ⚠️ 未找到入口流量文件")
        return {}
    
    # 加载出口压力
    outlet_data = {}
    for key in ["O1", "O2", "O3", "O4"]:
        filename = BC_FILE_MAPPING[key]
        file_path = bc_dir / filename
        if file_path.exists():
            outlet_data[key] = load_bc_file(file_path)
        else:
            print(f"  ⚠️ 未找到出口文件: {filename}")
            outlet_data[key] = {}
    
    # 获取所有时间步的并集
    all_steps = set(inlet_data.keys())
    for key in outlet_data:
        all_steps.update(outlet_data[key].keys())
    
    # 构建最终的边界条件数据
    bc_data = {}
    for step in sorted(all_steps):
        inlet_val = inlet_data.get(step, 0.0)
        o1_val = outlet_data["O1"].get(step, 0.0)
        o2_val = outlet_data["O2"].get(step, 0.0)
        o3_val = outlet_data["O3"].get(step, 0.0)
        o4_val = outlet_data["O4"].get(step, 0.0)
        
        # 不再包含 BC_Flag，直接返回 5 个值
        bc_data[step] = [inlet_val, o1_val, o2_val, o3_val, o4_val]
    
    if bc_data:
        print(f"  ✅ 加载 {len(bc_data)} 个时间步的边界条件")
    
    return bc_data


if __name__ == "__main__":
    # 测试
    print("IO 工具模块测试")
    print("=" * 50)
    
    # 测试路径
    test_bc_dir = Path(__file__).parent.parent.parent / "data_new/AG/fast/ZHANG_CHUN/Global_conditions"
    
    if test_bc_dir.exists():
        print(f"\n测试边界条件加载: {test_bc_dir}")
        bc_data = load_boundary_conditions(test_bc_dir)
        if bc_data:
            sample_step = list(bc_data.keys())[0]
            print(f"示例时间步 {sample_step}: {bc_data[sample_step]}")
    else:
        print(f"测试目录不存在: {test_bc_dir}")
