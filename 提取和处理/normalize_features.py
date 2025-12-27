#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征归一化脚本

用于对 ascii_mapped_128 中已提取的几何特征数据进行归一化处理。
支持多种归一化方法，并保存归一化参数供后续还原或推理使用。

归一化策略:
- 坐标 (x, y, z): center_scale 或保持已有归一化
- Abscissa: 已在 [0, 1] 范围，保持不变
- NormRadius: min-max 归一化到 [0, 1]
- Curvature: Z-score 标准化
- Tangent_X/Y/Z: 保持不变（单位向量分量）
- u, v, w: Z-score 标准化
- p: Z-score 标准化
- vel_mag: Z-score 标准化
- wss, wss_x/y/z: Z-score 标准化（可选 log 变换）
- is_wall, BC_Flag: 保持不变（离散标记）

边界条件 - 基于物理意义的手动缩放:
- BC_Inlet (入口流量): Q_in × 1e5 → 0.5~5.0
- BC_O1~O4 (出口边界):
  * BC_Flag=0 (流量边界): Q_out × 1e5 → 0.1~2.0
  * BC_Flag=1 (压力边界): (P - 15000) / 1000 → -1.5~+1.5

使用示例:
  # 处理单个病例
  python normalize_features.py --case FENG_LI_XIN
  
  # 处理所有病例
  python normalize_features.py
  
  # 使用 log 变换处理 WSS（如果分布很偏斜）
  python normalize_features.py --log-wss

作者: 自动生成
日期: 2024
"""

import os
import json
import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


# 数据源路径配置
DATA_SOURCES = [
    "AAA/rupture",
    "AAA/unrupture",
    "fast",
    "slow",
    "ILO/sq",
    "ILO/sh",
]

# 特征分组配置
FEATURE_CONFIG = {
    # 保持不变的特征
    "keep_unchanged": [
        "Abscissa",           # 已归一化到 [0, 1]
        "Tangent_X",          # 单位向量分量
        "Tangent_Y",
        "Tangent_Z",
        "is_wall",            # 二值标记
        "BC_Flag",            # 二值标记（0=流量边界, 1=压力边界）
    ],
    
    # 使用 min-max 归一化的特征
    "min_max": [
        "NormRadius",         # 归一化半径距离
    ],
    
    # 使用 Z-score 标准化的特征
    "z_score": [
        "Curvature",          # 曲率
        "u", "v", "w",        # 速度分量
        "p",                  # 压力
        "vel_mag",            # 速度大小
        "wss",                # 壁面剪切应力
        "wss_x", "wss_y", "wss_z",  # WSS 分量
    ],
    
    # 边界条件特征 - 基于物理意义的手动缩放
    # BC_Inlet: 入口流量，乘以 1e5，从 ~1e-5 变成 0.5~5.0
    # BC_O1~O4: 出口边界，根据 BC_Flag 分类处理
    #   - BC_Flag=0 (流量边界): Q_out × 1e5，从 ~1e-6~1e-5 变成 0.1~2.0
    #   - BC_Flag=1 (压力边界): (P - 15000) / 1000，从 ~13000-16000 Pa 变成 -1.5~+1.5
    "bc_inlet": ["BC_Inlet"],  # 入口流量
    "bc_outlets": ["BC_O1", "BC_O2", "BC_O3", "BC_O4"],  # 出口边界
    
    # 坐标特征（可选归一化）
    "coordinates": ["x", "y", "z"],
}

# 边界条件物理缩放参数
BC_SCALING_CONFIG = {
    # 入口流量缩放：Q_in × scale_factor
    "inlet_flow": {
        "scale_factor": 1e5,  # 乘以 100,000
        "description": "入口流量 (m³/s) × 1e5 → 0.5~5.0"
    },
    # 出口流量缩放（BC_Flag=0）：Q_out × scale_factor
    "outlet_flow": {
        "scale_factor": 1e5,  # 乘以 100,000
        "description": "出口流量 (m³/s) × 1e5 → 0.1~2.0"
    },
    # 出口压力缩放（BC_Flag=1）：(P - offset) / scale
    "outlet_pressure": {
        "offset": 15000,      # 中心压力值 (Pa)
        "scale": 1000,        # 缩放因子
        "description": "出口压力 (Pa): (P - 15000) / 1000 → -1.5~+1.5"
    }
}


def compute_statistics(values: np.ndarray) -> Dict:
    """
    计算数组的统计量
    """
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "median": float(np.nanmedian(values)),
    }


def z_score_normalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Z-score 标准化
    """
    if std < 1e-10:
        return np.zeros_like(values)
    return (values - mean) / std


def min_max_normalize(values: np.ndarray, min_val: float, max_val: float, 
                      target_min: float = 0.0, target_max: float = 1.0) -> np.ndarray:
    """
    Min-max 归一化
    """
    range_val = max_val - min_val
    if range_val < 1e-10:
        return np.full_like(values, (target_min + target_max) / 2)
    
    normalized = (values - min_val) / range_val
    return normalized * (target_max - target_min) + target_min


def log_transform(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    对数变换（用于处理偏斜分布）
    """
    return np.log(np.abs(values) + epsilon) * np.sign(values)


def collect_global_statistics(case_dirs: List[Path], 
                               input_subdir: str,
                               use_log_wss: bool = False) -> Dict:
    """
    收集所有病例的全局统计量
    
    参数:
        case_dirs: 病例目录列表
        input_subdir: 输入数据子目录
        use_log_wss: 是否对 WSS 使用 log 变换
    
    返回:
        各特征的全局统计量
    """
    print("\n📊 收集全局统计量...")
    
    # 初始化累积统计量（仅收集需要 Z-score 的特征）
    feature_values = {
        **{feat: [] for feat in FEATURE_CONFIG["z_score"]},
        **{feat: [] for feat in FEATURE_CONFIG["min_max"]},
        **{feat: [] for feat in FEATURE_CONFIG["coordinates"]},
    }
    
    # 边界条件分类收集（用于验证物理缩放参数是否合适）
    bc_inlet_values = []
    bc_outlet_flow = {feat: [] for feat in FEATURE_CONFIG["bc_outlets"]}  # BC_Flag=0
    bc_outlet_pressure = {feat: [] for feat in FEATURE_CONFIG["bc_outlets"]}  # BC_Flag=1
    
    # 遍历所有病例收集数据
    total_files = 0
    flow_bc_count = 0
    pressure_bc_count = 0
    
    for case_dir in tqdm(case_dirs, desc="扫描病例"):
        input_dir = case_dir / input_subdir
        if not input_dir.exists():
            continue
        
        csv_files = list(input_dir.glob("result_features_*.csv"))
        if not csv_files:
            continue
        
        # 采样部分文件（加速统计）
        sample_files = csv_files[::max(1, len(csv_files) // 10)]  # 每10个取1个
        
        for csv_file in sample_files:
            try:
                df = pd.read_csv(csv_file)
                total_files += 1
                
                # 收集常规特征值
                for feat in feature_values.keys():
                    if feat in df.columns:
                        values = df[feat].values
                        # 对 WSS 应用 log 变换（如果需要）
                        if use_log_wss and feat in ["wss", "wss_x", "wss_y", "wss_z"]:
                            values = log_transform(values)
                        feature_values[feat].extend(values.flatten().tolist())
                
                # 收集入口流量
                if "BC_Inlet" in df.columns:
                    bc_inlet_values.extend(df["BC_Inlet"].values.flatten().tolist())
                
                # 分类收集边界条件（用于验证）
                if "BC_Flag" in df.columns:
                    bc_flag = df["BC_Flag"].iloc[0]  # 每个文件内 BC_Flag 应该一致
                    
                    for feat in FEATURE_CONFIG["bc_outlets"]:
                        if feat in df.columns:
                            values = df[feat].values
                            if bc_flag == 0:  # 流量边界
                                bc_outlet_flow[feat].extend(values.flatten().tolist())
                            else:  # 压力边界
                                bc_outlet_pressure[feat].extend(values.flatten().tolist())
                    
                    if bc_flag == 0:
                        flow_bc_count += 1
                    else:
                        pressure_bc_count += 1
                        
            except Exception as e:
                print(f"  ⚠️  读取 {csv_file.name} 失败: {e}")
    
    print(f"  已扫描 {total_files} 个文件")
    print(f"  流量边界文件: {flow_bc_count}, 压力边界文件: {pressure_bc_count}")
    
    # 计算常规特征统计量（用于 Z-score）
    global_stats = {}
    print("\n  === Z-score 标准化特征 ===")
    for feat, values in feature_values.items():
        if values:
            arr = np.array(values)
            global_stats[feat] = compute_statistics(arr)
            print(f"  {feat}: mean={global_stats[feat]['mean']:.6f}, "
                  f"std={global_stats[feat]['std']:.6f}, "
                  f"range=[{global_stats[feat]['min']:.6f}, {global_stats[feat]['max']:.6f}]")
    
    # 显示边界条件原始范围（用于验证物理缩放参数）
    print("\n  === 边界条件原始范围（物理缩放验证）===")
    
    # 入口流量
    if bc_inlet_values:
        arr = np.array(bc_inlet_values)
        stats = compute_statistics(arr)
        inlet_config = BC_SCALING_CONFIG["inlet_flow"]
        scaled_min = stats['min'] * inlet_config['scale_factor']
        scaled_max = stats['max'] * inlet_config['scale_factor']
        print(f"  BC_Inlet (入口流量):")
        print(f"    原始范围: [{stats['min']:.6e}, {stats['max']:.6e}] m³/s")
        print(f"    缩放后 (×{inlet_config['scale_factor']:.0e}): [{scaled_min:.4f}, {scaled_max:.4f}]")
    
    # 出口流量（BC_Flag=0）
    print(f"\n  出口边界 - 流量边界 (BC_Flag=0):")
    for feat, values in bc_outlet_flow.items():
        if values:
            arr = np.array(values)
            stats = compute_statistics(arr)
            outlet_flow_config = BC_SCALING_CONFIG["outlet_flow"]
            scaled_min = stats['min'] * outlet_flow_config['scale_factor']
            scaled_max = stats['max'] * outlet_flow_config['scale_factor']
            print(f"    {feat}: 原始=[{stats['min']:.6e}, {stats['max']:.6e}], "
                  f"缩放后=[{scaled_min:.4f}, {scaled_max:.4f}]")
    
    # 出口压力（BC_Flag=1）
    print(f"\n  出口边界 - 压力边界 (BC_Flag=1):")
    for feat, values in bc_outlet_pressure.items():
        if values:
            arr = np.array(values)
            stats = compute_statistics(arr)
            outlet_pressure_config = BC_SCALING_CONFIG["outlet_pressure"]
            scaled_min = (stats['min'] - outlet_pressure_config['offset']) / outlet_pressure_config['scale']
            scaled_max = (stats['max'] - outlet_pressure_config['offset']) / outlet_pressure_config['scale']
            print(f"    {feat}: 原始=[{stats['min']:.2f}, {stats['max']:.2f}] Pa, "
                  f"缩放后=[{scaled_min:.4f}, {scaled_max:.4f}]")
    
    # 保存边界条件信息（用于还原）
    global_stats["bc_scaling_config"] = BC_SCALING_CONFIG
    
    return global_stats


def normalize_dataframe(df: pd.DataFrame, 
                        global_stats: Dict,
                        use_log_wss: bool = False,
                        normalize_coords: bool = False,
                        coord_params: Optional[Dict] = None) -> pd.DataFrame:
    """
    对单个 DataFrame 进行归一化
    
    参数:
        df: 输入 DataFrame
        global_stats: 全局统计量
        use_log_wss: 是否对 WSS 使用 log 变换
        normalize_coords: 是否归一化坐标
        coord_params: 坐标归一化参数
    
    返回:
        归一化后的 DataFrame
    """
    df_norm = df.copy()
    
    # 1. 保持不变的特征（无需处理）
    
    # 2. Min-max 归一化
    for feat in FEATURE_CONFIG["min_max"]:
        if feat in df_norm.columns and feat in global_stats:
            stats = global_stats[feat]
            df_norm[feat] = min_max_normalize(
                df_norm[feat].values,
                stats["min"], stats["max"]
            )
    
    # 3. Z-score 标准化
    for feat in FEATURE_CONFIG["z_score"]:
        if feat in df_norm.columns and feat in global_stats:
            values = df_norm[feat].values
            
            # 对 WSS 应用 log 变换
            if use_log_wss and feat in ["wss", "wss_x", "wss_y", "wss_z"]:
                values = log_transform(values)
            
            stats = global_stats[feat]
            df_norm[feat] = z_score_normalize(values, stats["mean"], stats["std"])
    
    # 4. 入口流量 - 物理缩放：Q_in × 1e5
    inlet_config = BC_SCALING_CONFIG["inlet_flow"]
    for feat in FEATURE_CONFIG["bc_inlet"]:
        if feat in df_norm.columns:
            df_norm[feat] = df_norm[feat].values * inlet_config["scale_factor"]
    
    # 5. 出口边界条件 - 根据 BC_Flag 分类物理缩放
    if "BC_Flag" in df_norm.columns:
        bc_flag = df_norm["BC_Flag"].iloc[0]  # 获取边界条件类型
        
        for feat in FEATURE_CONFIG["bc_outlets"]:
            if feat in df_norm.columns:
                if bc_flag == 0:  # 流量边界：Q_out × 1e5
                    outlet_flow_config = BC_SCALING_CONFIG["outlet_flow"]
                    df_norm[feat] = df_norm[feat].values * outlet_flow_config["scale_factor"]
                else:  # 压力边界：(P - 15000) / 1000
                    outlet_pressure_config = BC_SCALING_CONFIG["outlet_pressure"]
                    df_norm[feat] = (df_norm[feat].values - outlet_pressure_config["offset"]) / outlet_pressure_config["scale"]
    
    # 6. 坐标归一化（可选）
    if normalize_coords and coord_params:
        for coord in FEATURE_CONFIG["coordinates"]:
            if coord in df_norm.columns:
                if coord_params.get("method") == "center_scale":
                    centroid = coord_params["centroid"]
                    scale = coord_params["scale_factor"]
                    idx = {"x": 0, "y": 1, "z": 2}[coord]
                    df_norm[coord] = (df_norm[coord].values - centroid[idx]) / scale
    
    # 处理 NaN 和 Inf
    df_norm = df_norm.replace([np.inf, -np.inf], np.nan)
    df_norm = df_norm.fillna(0)
    
    return df_norm


def process_case(case_dir: Path,
                 input_subdir: str,
                 output_subdir: str,
                 global_stats: Dict,
                 use_log_wss: bool = False,
                 normalize_coords: bool = False) -> bool:
    """
    处理单个病例的所有文件
    
    参数:
        case_dir: 病例目录
        input_subdir: 输入子目录
        output_subdir: 输出子目录
        global_stats: 全局统计量
        use_log_wss: 是否对 WSS 使用 log 变换
        normalize_coords: 是否归一化坐标
    
    返回:
        是否成功处理
    """
    input_dir = case_dir / input_subdir
    output_dir = case_dir / output_subdir
    
    if not input_dir.exists():
        print(f"  ❌ 输入目录不存在: {input_subdir}")
        return False
    
    csv_files = list(input_dir.glob("result_features_*.csv"))
    if not csv_files:
        print(f"  ❌ 未找到特征文件")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载坐标归一化参数（如果存在）
    coord_params = None
    if normalize_coords:
        # 尝试从 ascii_merged_128 加载
        norm_params_path = case_dir / "ascii_merged_128" / "normalization_params.json"
        if norm_params_path.exists():
            with open(norm_params_path, 'r') as f:
                coord_params = json.load(f)
    
    # 处理每个文件
    success_count = 0
    for csv_file in tqdm(csv_files, desc=f"处理 {case_dir.name}", leave=False):
        try:
            df = pd.read_csv(csv_file)
            df_norm = normalize_dataframe(
                df, global_stats, 
                use_log_wss=use_log_wss,
                normalize_coords=normalize_coords,
                coord_params=coord_params
            )
            
            # 保存到输出目录
            output_path = output_dir / csv_file.name
            df_norm.to_csv(output_path, index=False)
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 处理 {csv_file.name} 失败: {e}")
    
    print(f"  ✅ 完成: {success_count}/{len(csv_files)} 个文件")
    return success_count > 0


def scan_cases(data_root: Path, target_case: Optional[str] = None) -> List[Path]:
    """
    扫描 data 目录下的所有病例目录
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


def save_normalization_params(global_stats: Dict, 
                               output_path: str,
                               use_log_wss: bool = False):
    """
    保存归一化参数到 JSON 文件
    """
    params = {
        "description": "特征归一化参数",
        "methods": {
            "keep_unchanged": FEATURE_CONFIG["keep_unchanged"],
            "min_max": FEATURE_CONFIG["min_max"],
            "z_score": FEATURE_CONFIG["z_score"],
            "bc_physical_scaling": {
                "bc_inlet": FEATURE_CONFIG["bc_inlet"],
                "bc_outlets": FEATURE_CONFIG["bc_outlets"],
            }
        },
        "boundary_condition_scaling": {
            "description": "边界条件使用基于物理意义的手动缩放",
            "inlet_flow": {
                "formula": "scaled = Q_in × 1e5",
                "original_range": "~1e-5 m³/s",
                "scaled_range": "0.5 ~ 5.0",
                "restore": "Q_in = scaled / 1e5"
            },
            "outlet_flow_BC_Flag_0": {
                "formula": "scaled = Q_out × 1e5",
                "original_range": "~1e-6 ~ 1e-5 m³/s",
                "scaled_range": "0.1 ~ 2.0",
                "restore": "Q_out = scaled / 1e5"
            },
            "outlet_pressure_BC_Flag_1": {
                "formula": "scaled = (P - 15000) / 1000",
                "original_range": "~13000 ~ 16000 Pa",
                "scaled_range": "-1.5 ~ +1.5",
                "restore": "P = scaled × 1000 + 15000"
            }
        },
        "bc_scaling_config": BC_SCALING_CONFIG,
        "use_log_wss": use_log_wss,
        "statistics": {k: v for k, v in global_stats.items() if k != "bc_scaling_config"},
        "restore_formulas": {
            "z_score": "original = normalized * std + mean",
            "min_max": "original = normalized * (max - min) + min",
            "log_wss": "若 use_log_wss=True，先逆 log 变换: sign(x) * (exp(|x|) - epsilon)",
            "bc_inlet": "Q_in = scaled / 1e5",
            "bc_outlet_flow": "Q_out = scaled / 1e5",
            "bc_outlet_pressure": "P = scaled × 1000 + 15000",
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 归一化参数已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="特征归一化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
归一化策略:
  - Abscissa, Tangent_X/Y/Z, is_wall, BC_Flag: 保持不变
  - NormRadius: min-max 归一化到 [0, 1]
  - Curvature, u/v/w, p, vel_mag, wss*: Z-score 标准化

边界条件物理缩放:
  - BC_Inlet (入口流量): Q_in × 1e5 → 0.5~5.0
  - BC_O1~O4 (出口):
    * BC_Flag=0 (流量边界): Q_out × 1e5 → 0.1~2.0
    * BC_Flag=1 (压力边界): (P-15000)/1000 → -1.5~+1.5

示例:
  # 处理所有病例
  python normalize_features.py
  
  # 处理指定病例
  python normalize_features.py --case FENG_LI_XIN
  
  # 对 WSS 使用 log 变换（处理长尾分布）
  python normalize_features.py --log-wss
  
  # 指定输入输出目录
  python normalize_features.py --input-subdir ascii_mapped_128 --output-subdir ascii_normalized
        """
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="data 根目录路径，默认 ./data",
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
        default="ascii_mapped_128",
        help="输入数据子目录，默认 ascii_mapped_128",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="ascii_normalized_128",
        help="输出数据子目录，默认 ascii_normalized_128",
    )
    parser.add_argument(
        "--log-wss",
        action="store_true",
        help="对 WSS 特征使用 log 变换后再标准化（处理长尾分布）",
    )
    parser.add_argument(
        "--normalize-coords",
        action="store_true",
        help="同时归一化坐标（如果尚未归一化）",
    )
    parser.add_argument(
        "--params-output",
        type=str,
        default="normalization_params_global.json",
        help="归一化参数输出文件名，默认 normalization_params_global.json",
    )
    
    args = parser.parse_args()
    
    # 解析路径
    script_dir = Path(__file__).parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = script_dir / data_root
    data_root = data_root.resolve()
    
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
    
    print(f"🚀 特征归一化工具")
    print(f"📁 数据根目录: {data_root}")
    print(f"📂 输入子目录: {args.input_subdir}")
    print(f"📂 输出子目录: {args.output_subdir}")
    print(f"📊 WSS log 变换: {'是' if args.log_wss else '否'}")
    print(f"📊 待处理病例数: {len(cases)}")
    
    # 第一阶段：收集全局统计量
    global_stats = collect_global_statistics(
        cases, 
        args.input_subdir,
        use_log_wss=args.log_wss
    )
    
    if not global_stats:
        print("❌ 未能收集到统计量，请检查数据")
        return
    
    # 保存归一化参数
    params_path = data_root / args.params_output
    save_normalization_params(global_stats, str(params_path), args.log_wss)
    
    # 第二阶段：应用归一化
    print("\n🔄 应用归一化...")
    ok = 0
    
    for idx, case_dir in enumerate(cases, start=1):
        try:
            rel_path = case_dir.relative_to(data_root)
        except ValueError:
            rel_path = case_dir.name
        
        print(f"\n[{idx}/{len(cases)}] {rel_path}")
        
        if process_case(
            case_dir,
            args.input_subdir,
            args.output_subdir,
            global_stats,
            use_log_wss=args.log_wss,
            normalize_coords=args.normalize_coords
        ):
            ok += 1
    
    print(f"\n{'=' * 50}")
    print(f"🎉 归一化完成!")
    print(f"✅ 成功: {ok}/{len(cases)} 个病例")
    print(f"📁 归一化参数: {params_path}")
    print(f"\n提示: 归一化后的数据保存在各病例的 {args.output_subdir} 目录")


if __name__ == "__main__":
    main()

