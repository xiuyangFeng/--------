#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征归一化模块

对坐标系归一化后的数据进行特征归一化处理，为训练做准备。

输入: processed/coord_normalized/ (坐标系归一化后的数据)
输出: processed/normalized/ (特征归一化后的数据)

归一化策略:
- 保持不变: Abscissa (已在[0,1]), Tangent_X/Y/Z (单位向量), is_wall (二值)
- 保持不变: x, y, z (坐标已在coord_normalize步骤归一化到[-1,1])
- Min-max: NormRadius → [0, 1]
- Z-score: Curvature, u, v, w, p, vel_mag, wss, wss_x/y/z

边界条件缩放（从 bc_metadata.json 侧文件读取）:
- BC_Inlet (入口流量): 保持物理缩放 Q_in × 1e5
- BC_O1~O4 (出口压力): 默认使用全局统计量标准化，可切换为固定缩放
归一化后保存为 bc_metadata_normalized.json

使用示例:
  # 处理单个病例
  python -m pipeline.normalize --case ZHANG_CHUN
  
  # 处理所有病例
  python -m pipeline.normalize
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# 导入配置
try:
    from .config import (
        DATA_ROOT,
        FEATURES_DIR,
        COORD_NORMALIZED_DIR,
        NORMALIZED_DIR,
        NORMALIZATION_CONFIG,
        get_case_dirs,
    )
    from .utils.progress import case_progress_logging
    from .utils.progress import batch_progress_logging
except ImportError:
    from config import (
        DATA_ROOT,
        FEATURES_DIR,
        COORD_NORMALIZED_DIR,
        NORMALIZED_DIR,
        NORMALIZATION_CONFIG,
        get_case_dirs,
    )
    from pipeline.utils.progress import case_progress_logging
    from pipeline.utils.progress import batch_progress_logging


# 特征分组配置
FEATURE_GROUPS = {
    # 保持不变的特征
    "keep_unchanged": [
        "Abscissa",      # 已归一化到 [0, 1]
        "Tangent_X",     # 单位向量分量
        "Tangent_Y",
        "Tangent_Z",
        "is_wall",       # 二值标记
        "branch_id",     # G01: 分叉拓扑标记（离散 0/1）
    ],
    
    # 使用 min-max 归一化的特征
    "min_max": [
        "NormRadius",
        "dist_to_bifurcation",  # G01: 距分叉点弧长距离
        "dist_to_wall",         # G04: 到最近壁面点距离
    ],
    
    # 使用 Z-score 标准化的特征
    "z_score": [
        "Curvature",
        "dR_ds",         # G02: 半径变化率
        "torsion",       # G03: 扭率
        "d_tangent_ds",  # G05: 切向变化率
        "u", "v", "w",
        "p",
        "vel_mag",
        "wss", "wss_x", "wss_y", "wss_z",
    ],
    
    # 边界条件特征
    "bc_inlet": ["BC_Inlet"],
    "bc_outlets": ["BC_O1", "BC_O2", "BC_O3", "BC_O4"],
    
    # 坐标（可选归一化）
    "coordinates": ["x", "y", "z"],
}

BC_SCALING = NORMALIZATION_CONFIG["bc_scaling"]


def compute_statistics(values: np.ndarray) -> Dict:
    """计算数组的统计量"""
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
    }


def z_score_normalize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Z-score 标准化"""
    if std < 1e-10:
        return np.zeros_like(values)
    return (values - mean) / std


def min_max_normalize(values: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Min-max 归一化到 [0, 1]"""
    range_val = max_val - min_val
    if range_val < 1e-10:
        return np.full_like(values, 0.5)
    return (values - min_val) / range_val


def collect_global_statistics(
    case_dirs: List[Path],
    input_subdir: str,
    train_cases: Optional[List[str]] = None,
) -> Dict:
    """
    收集全局统计量。

    当 ``train_cases`` 非空时，**仅**使用训练集病例计算统计量，
    以避免验证/测试集信息泄漏到归一化参数中。归一化后所有病例
    （包括验证/测试）使用同一套参数变换。

    参数:
        case_dirs: 全部病例目录
        input_subdir: 输入子目录
        train_cases: 训练集病例名列表。为 None 时使用全部病例（向后兼容，
                     但会在日志中打印警告）。
    """
    if train_cases is not None:
        train_set = {name.replace(' ', '_').replace('-', '_').upper() for name in train_cases}
        stats_dirs = [
            d for d in case_dirs
            if d.name.replace(' ', '_').replace('-', '_').upper() in train_set
        ]
        print(f"\n📊 收集全局统计量（仅训练集 {len(stats_dirs)}/{len(case_dirs)} 病例）...")
    else:
        stats_dirs = list(case_dirs)
        print("\n⚠️  未指定 train_cases，将使用全部病例计算统计量（可能导致数据泄漏）")
        print("📊 收集全局统计量...")
    
    # 初始化累积数据
    feature_values = {
        **{feat: [] for feat in FEATURE_GROUPS["z_score"]},
        **{feat: [] for feat in FEATURE_GROUPS["min_max"]},
    }
    
    # 边界条件值从侧文件收集（用于计算和验证缩放范围）
    bc_inlet_values = []
    bc_outlet_values = {"BC_O1": [], "BC_O2": [], "BC_O3": [], "BC_O4": []}
    
    total_files = 0
    
    for case_dir in tqdm(stats_dirs, desc="扫描病例"):
        input_dir = case_dir / input_subdir
        if not input_dir.exists():
            continue
        
        csv_files = list(input_dir.glob("result_features_*.csv"))
        if not csv_files:
            continue
        
        # 采样部分文件（加速统计）
        sample_files = csv_files[::max(1, len(csv_files) // 10)]
        
        for csv_file in sample_files:
            try:
                df = pd.read_csv(csv_file)
                total_files += 1
                
                # WSS 统计量仅使用壁面节点，避免内部点的无效 WSS 值
                # （通常为 0 或占位）拉偏 mean/std。
                _WSS_FEATURES = {"wss", "wss_x", "wss_y", "wss_z"}
                has_wall_col = "is_wall" in df.columns
                if has_wall_col:
                    wall_rows = df["is_wall"] == 1
                
                for feat in feature_values.keys():
                    if feat in df.columns:
                        if feat in _WSS_FEATURES and has_wall_col:
                            feature_values[feat].extend(
                                df.loc[wall_rows, feat].values.tolist()
                            )
                        else:
                            feature_values[feat].extend(df[feat].values.tolist())
                        
            except Exception as e:
                print(f"  ⚠️ 读取 {csv_file.name} 失败: {e}")
        
        # 从侧文件收集边界条件（用于验证缩放范围）
        bc_meta_path = input_dir / "bc_metadata.json"
        if bc_meta_path.exists():
            try:
                with open(bc_meta_path, 'r', encoding='utf-8') as f:
                    bc_meta = json.load(f)
                for stem, bc_vals in bc_meta.get("data", {}).items():
                    if len(bc_vals) == 5:
                        bc_inlet_values.append(bc_vals[0])
                        for i, key in enumerate(["BC_O1", "BC_O2", "BC_O3", "BC_O4"]):
                            bc_outlet_values[key].append(bc_vals[i + 1])
            except Exception as e:
                print(f"  ⚠️ 读取 bc_metadata.json 失败: {e}")
    
    print(f"  已扫描 {total_files} 个文件")
    
    # 计算统计量
    global_stats = {}
    
    print("\n  === Z-score 标准化特征 ===")
    for feat in FEATURE_GROUPS["z_score"]:
        if feat in feature_values and feature_values[feat]:
            arr = np.array(feature_values[feat])
            global_stats[feat] = compute_statistics(arr)
            print(f"  {feat}: mean={global_stats[feat]['mean']:.6f}, "
                  f"std={global_stats[feat]['std']:.6f}")
    
    print("\n  === Min-max 归一化特征 ===")
    for feat in FEATURE_GROUPS["min_max"]:
        if feat in feature_values and feature_values[feat]:
            arr = np.array(feature_values[feat])
            global_stats[feat] = compute_statistics(arr)
            print(f"  {feat}: min={global_stats[feat]['min']:.6f}, "
                  f"max={global_stats[feat]['max']:.6f}")
    
    # 记录 BC 统计量，供全局条件归一化使用
    global_stats["BC_Inlet"] = compute_statistics(np.array(bc_inlet_values)) if bc_inlet_values else None
    for feat, values in bc_outlet_values.items():
        if values:
            global_stats[feat] = compute_statistics(np.array(values))

    # 显示边界条件范围（用于验证缩放参数）
    print("\n  === 边界条件原始范围 ===")
    if bc_inlet_values:
        arr = np.array(bc_inlet_values)
        stats = compute_statistics(arr)
        scaled_min = stats['min'] * BC_SCALING["inlet"]["scale_factor"]
        scaled_max = stats['max'] * BC_SCALING["inlet"]["scale_factor"]
        print(f"  BC_Inlet: 原始=[{stats['min']:.6e}, {stats['max']:.6e}], "
              f"缩放后=[{scaled_min:.4f}, {scaled_max:.4f}]")
    
    outlet_strategy = BC_SCALING["outlet_pressure"].get("strategy", "fixed")
    for feat, values in bc_outlet_values.items():
        if values:
            arr = np.array(values)
            stats = compute_statistics(arr)
            if outlet_strategy == "z_score":
                scaled_min = z_score_normalize(arr, stats["mean"], stats["std"]).min()
                scaled_max = z_score_normalize(arr, stats["mean"], stats["std"]).max()
                print(f"  {feat}: 原始=[{stats['min']:.2f}, {stats['max']:.2f}] Pa, "
                      f"z-score后=[{scaled_min:.4f}, {scaled_max:.4f}]")
            else:
                cfg = BC_SCALING["outlet_pressure"]
                scaled_min = (stats['min'] - cfg['offset']) / cfg['scale']
                scaled_max = (stats['max'] - cfg['offset']) / cfg['scale']
                print(f"  {feat}: 原始=[{stats['min']:.2f}, {stats['max']:.2f}] Pa, "
                      f"固定缩放后=[{scaled_min:.4f}, {scaled_max:.4f}]")
    
    global_stats["bc_scaling"] = BC_SCALING
    
    return global_stats


def normalize_dataframe(df: pd.DataFrame, global_stats: Dict) -> pd.DataFrame:
    """
    对单个 DataFrame 进行归一化
    """
    df_norm = df.copy()
    
    # 1. 保持不变的特征（无需处理）
    
    # 2. Min-max 归一化
    for feat in FEATURE_GROUPS["min_max"]:
        if feat in df_norm.columns and feat in global_stats:
            stats = global_stats[feat]
            df_norm[feat] = min_max_normalize(
                df_norm[feat].values,
                stats["min"], stats["max"]
            )
    
    # 3. Z-score 标准化
    for feat in FEATURE_GROUPS["z_score"]:
        if feat in df_norm.columns and feat in global_stats:
            stats = global_stats[feat]
            df_norm[feat] = z_score_normalize(
                df_norm[feat].values,
                stats["mean"], stats["std"]
            )
    
    # 注意: 边界条件（BC_Inlet, BC_O1~O4）不再在 CSV 中，
    # 改为在 process_case() 中从 bc_metadata.json 侧文件统一归一化
    
    # 处理 NaN 和 Inf
    df_norm = df_norm.replace([np.inf, -np.inf], np.nan)
    df_norm = df_norm.fillna(0)
    
    return df_norm


def normalize_bc_metadata(bc_meta: Dict, global_stats: Dict) -> Dict:
    """
    对边界条件元数据进行归一化。
    
    缩放策略:
    - BC_Inlet: Q_in × 1e5
    - BC_O1~O4: 默认按全局统计量做 z-score，可回退固定缩放
    
    参数:
        bc_meta: 原始 bc_metadata.json 内容
    
    返回:
        归一化后的元数据字典
    """
    inlet_cfg = BC_SCALING["inlet"]
    outlet_cfg = BC_SCALING["outlet_pressure"]
    outlet_strategy = outlet_cfg.get("strategy", "fixed")
    
    normalized_data = {}
    for stem, bc_vals in bc_meta.get("data", {}).items():
        if len(bc_vals) == 5:
            # [Inlet, O1, O2, O3, O4]
            norm_inlet = bc_vals[0] * inlet_cfg["scale_factor"]
            norm_outlets = []
            for i, feat in enumerate(FEATURE_GROUPS["bc_outlets"], start=1):
                raw_val = bc_vals[i]
                if outlet_strategy == "z_score":
                    stats = global_stats.get(feat)
                    if not stats:
                        raise ValueError(f"缺少 {feat} 的全局统计量，无法执行 z-score 归一化")
                    norm_val = float(z_score_normalize(np.array([raw_val]), stats["mean"], stats["std"])[0])
                else:
                    norm_val = (raw_val - outlet_cfg["offset"]) / outlet_cfg["scale"]
                norm_outlets.append(norm_val)
            normalized_data[stem] = [norm_inlet] + norm_outlets
        else:
            print(f"  ⚠️ 跳过 {stem}: 边界条件格式错误 (期望5个值，实际{len(bc_vals)}个)")
    
    return {
        "description": "归一化后的边界条件元数据",
        "fields": ["BC_Inlet", "BC_O1", "BC_O2", "BC_O3", "BC_O4"],
        "strategy": {
            "BC_Inlet": "fixed_scale",
            "BC_O1~O4": outlet_strategy,
        },
        "scaling": {
            "BC_Inlet": "Q_in × 1e5",
            "BC_O1~O4": (
                "z-score(global mean/std)"
                if outlet_strategy == "z_score"
                else f"(P - {outlet_cfg['offset']}) / {outlet_cfg['scale']}"
            ),
        },
        "data": normalized_data,
    }


def process_case(
    case_dir: Path,
    input_subdir: str,
    output_subdir: str,
    global_stats: Dict,
) -> bool:
    """
    处理单个病例的所有文件
    """
    with case_progress_logging(case_dir, "step4_normalize") as log_path:
        print(f"📝 进度日志: {log_path}")
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
        
        # 处理每个 CSV 文件（逐点特征归一化）
        success_count = 0
        for i, csv_file in enumerate(tqdm(csv_files, desc=f"处理 {case_dir.name}", leave=False), 1):
            try:
                print(f"  🔄 [{i}/{len(csv_files)}] 文件: {csv_file.name}")
                df = pd.read_csv(csv_file)
                print("    应用逐点特征归一化...")
                df_norm = normalize_dataframe(df, global_stats)
                
                output_path = output_dir / csv_file.name
                print(f"    保存结果: {output_path.name}")
                df_norm.to_csv(output_path, index=False)
                success_count += 1
                print(f"  ✅ 文件完成: {csv_file.name}")
                
            except Exception as e:
                print(f"  ❌ 处理 {csv_file.name} 失败: {e}")
        
        # 处理边界条件元数据（全局条件归一化）
        bc_meta_path = input_dir / "bc_metadata.json"
        if bc_meta_path.exists():
            try:
                print("  🔄 归一化边界条件元数据...")
                with open(bc_meta_path, 'r', encoding='utf-8') as f:
                    bc_meta = json.load(f)
                
                bc_meta_norm = normalize_bc_metadata(bc_meta, global_stats)
                
                bc_norm_path = output_dir / "bc_metadata_normalized.json"
                with open(bc_norm_path, 'w', encoding='utf-8') as f:
                    json.dump(bc_meta_norm, f, indent=2, ensure_ascii=False)
                
                print(f"  📋 边界条件归一化完成: {len(bc_meta_norm['data'])} 个时间步")
            except Exception as e:
                print(f"  ⚠️ 边界条件归一化失败: {e}")
        else:
            print(f"  ⚠️ 未找到边界条件元数据: {bc_meta_path}")
        
        print(f"  ✅ 完成: {success_count}/{len(csv_files)} 个文件")
        return success_count > 0


def save_normalization_params(global_stats: Dict, output_path: str) -> None:
    """保存归一化参数到 JSON 文件"""
    params = {
        "description": "特征归一化参数",
        "feature_groups": FEATURE_GROUPS,
        "bc_scaling": BC_SCALING,
        "statistics": {k: v for k, v in global_stats.items() if k != "bc_scaling"},
        "restore_formulas": {
            "z_score": "original = normalized * std + mean",
            "min_max": "original = normalized * (max - min) + min",
            "bc_inlet": "Q_in = scaled / 1e5",
            "bc_outlet_pressure": (
                "P = normalized * std + mean"
                if BC_SCALING["outlet_pressure"].get("strategy", "fixed") == "z_score"
                else f"P = scaled × {BC_SCALING['outlet_pressure']['scale']} + {BC_SCALING['outlet_pressure']['offset']}"
            ),
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 归一化参数已保存到: {output_path}")


def process_all_cases(
    data_root: Path = None,
    target_case: Optional[str] = None,
    input_subdir: str = None,
    output_subdir: str = None,
    train_cases: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
) -> None:
    """批量处理所有病例。

    参数:
        train_cases: 训练集病例名列表。传入后归一化统计量仅基于这些病例
                     计算，防止验证/测试集数据泄漏。
    """
    if data_root is None:
        data_root = DATA_ROOT
    else:
        data_root = Path(data_root)
    
    if input_subdir is None:
        input_subdir = COORD_NORMALIZED_DIR
    if output_subdir is None:
        output_subdir = NORMALIZED_DIR
    
    case_dirs = get_case_dirs(data_root, sources=sources)
    
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
    
    with batch_progress_logging(data_root, "step4_normalize_batch.log", "step4_normalize_batch") as log_path:
        print(f"📝 批量日志: {log_path}")
        print("🚀 特征归一化")
        print("=" * 50)
        print(f"📁 数据根目录: {data_root}")
        print(f"📂 输入子目录: {input_subdir}")
        print(f"📂 输出子目录: {output_subdir}")
        print(f"📊 待处理病例数: {len(case_dirs)}")
        
        global_stats = collect_global_statistics(case_dirs, input_subdir, train_cases=train_cases)
        
        if not global_stats:
            print("❌ 未能收集到统计量，请检查数据")
            return
        
        params_path = data_root / "normalization_params_global.json"
        save_normalization_params(global_stats, str(params_path))
        
        print("\n🔄 应用归一化...")
        total_start = time.time()
        ok = 0
        
        for idx, case_dir in enumerate(case_dirs, 1):
            try:
                rel_path = case_dir.relative_to(data_root)
            except ValueError:
                rel_path = case_dir.name
            
            print(f"\n[{idx}/{len(case_dirs)}] {rel_path}")
            
            if process_case(case_dir, input_subdir, output_subdir, global_stats):
                ok += 1
        
        total_time = time.time() - total_start
        
        print(f"\n{'=' * 50}")
        print("🎉 归一化完成!")
        print(f"⏱️  总耗时: {total_time:.1f}s")
        print(f"✅ 成功: {ok}/{len(case_dirs)} 个病例")
        print(f"📁 归一化参数: {params_path}")


def main():
    parser = argparse.ArgumentParser(
        description="特征归一化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
归一化策略:
  - 保持不变: Abscissa, Tangent_X/Y/Z, is_wall
  - Min-max: NormRadius → [0, 1]
  - Z-score: Curvature, u/v/w, p, vel_mag, wss*

边界条件缩放（新格式，无 BC_Flag）:
  - BC_Inlet: Q_in × 1e5
  - BC_O1~O4: 默认 z-score(global mean/std)，可切换为固定缩放

示例:
  python -m pipeline.normalize --case ZHANG_CHUN
  python -m pipeline.normalize
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
        help=f"输入子目录，默认 {COORD_NORMALIZED_DIR}",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help=f"输出子目录，默认 {NORMALIZED_DIR}",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default=None,
        help="训练集 split JSON 文件路径（包含 train 病例列表），"
             "指定后归一化统计量仅基于训练集计算以避免数据泄漏",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        metavar="SOURCE",
        help="数据源子路径（如 AG/fast AG/slow）；默认使用 config 已启用数据源",
    )
    
    args = parser.parse_args()
    
    train_cases = None
    if args.train_split:
        import json as _json
        with open(args.train_split, "r", encoding="utf-8") as _f:
            _split = _json.load(_f)
        train_cases = _split.get("train", _split.get("cases", []))
        if not train_cases:
            print(f"⚠️  split 文件中未找到 'train' 字段: {args.train_split}")
    
    process_all_cases(
        data_root=args.data_root,
        target_case=args.case,
        input_subdir=args.input_subdir,
        output_subdir=args.output_subdir,
        train_cases=train_cases,
        sources=args.sources,
    )


if __name__ == "__main__":
    main()
