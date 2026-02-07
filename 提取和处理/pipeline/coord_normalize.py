#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标系归一化模块

对点云数据进行坐标系统一处理：中心化 + PCA主轴对齐 + 缩放归一化。
这是数据增强的前置必要步骤，确保所有病例在统一的坐标系下。

功能:
1. 中心化：将点云几何中心移到原点
2. PCA对齐：将血管主轴旋转到Z轴方向
3. 缩放：将坐标归一化到 [-1, 1] 范围

关键：矢量特征（速度、切线、WSS矢量）必须同步旋转！

使用示例:
  # 处理单个病例
  python coord_normalize.py --case ZHANG_CHUN
  
  # 处理所有病例
  python coord_normalize.py
"""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

# 导入配置
from config import (
    DATA_ROOT,
    FEATURES_DIR,
    get_case_dirs,
)

# 输出目录（在 features 和 normalized 之间）
COORD_NORMALIZED_DIR = "processed/coord_normalized"


def normalize_coordinate_system(
    coords: np.ndarray,
    velocity: np.ndarray = None,
    tangent: np.ndarray = None,
    wss_vec: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    坐标系归一化：中心化 + PCA对齐 + 缩放
    
    参数:
        coords: [N, 3] 点云坐标 (x, y, z)
        velocity: [N, 3] 速度场 (u, v, w)，可选
        tangent: [N, 3] 切线方向 (Tangent_X, Tangent_Y, Tangent_Z)，可选
        wss_vec: [N, 3] WSS矢量 (wss_x, wss_y, wss_z)，可选
    
    返回:
        coords_norm: 归一化后的坐标
        velocity_rot: 旋转后的速度（如果提供）
        tangent_rot: 旋转后的切线（如果提供）
        wss_vec_rot: 旋转后的WSS矢量（如果提供）
        transform_params: 变换参数（用于逆变换）
    """
    # 步骤1：中心化
    centroid = coords.mean(axis=0)
    coords_centered = coords - centroid
    
    # 步骤2：PCA主轴对齐
    # 将血管主轴（最大方差方向）旋转到 Z 轴
    pca = PCA(n_components=3)
    pca.fit(coords_centered)
    
    # PCA components 的行是主成分方向，按方差从大到小排列
    # 我们要让第一主成分（最大方差）对齐到 Z 轴
    R = pca.components_.T  # 旋转矩阵 [3, 3]
    
    # 确保旋转矩阵是右手系（行列式为正）
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1  # 翻转第三列
    
    coords_aligned = coords_centered @ R
    
    # 步骤3：缩放到 [-1, 1]
    max_abs = np.abs(coords_aligned).max()
    scale_factor = max_abs if max_abs > 1e-6 else 1.0
    coords_scaled = coords_aligned / scale_factor
    
    # 同步旋转矢量特征（注意：只旋转，不平移不缩放）
    velocity_rot = None
    tangent_rot = None
    wss_vec_rot = None
    
    if velocity is not None:
        velocity_rot = velocity @ R
    
    if tangent is not None:
        tangent_rot = tangent @ R
    
    if wss_vec is not None:
        wss_vec_rot = wss_vec @ R
    
    # 保存变换参数（用于推理时逆变换）
    transform_params = {
        "centroid": centroid.tolist(),
        "rotation_matrix": R.tolist(),
        "scale_factor": float(scale_factor),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    
    return coords_scaled, velocity_rot, tangent_rot, wss_vec_rot, transform_params


def inverse_transform(
    coords_norm: np.ndarray,
    velocity_rot: np.ndarray = None,
    transform_params: Dict = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    逆变换：将归一化坐标系还原到原始坐标系
    
    参数:
        coords_norm: 归一化后的坐标
        velocity_rot: 旋转后的速度
        transform_params: 变换参数
    
    返回:
        coords_orig: 原始坐标
        velocity_orig: 原始速度
    """
    scale = transform_params["scale_factor"]
    R = np.array(transform_params["rotation_matrix"])
    centroid = np.array(transform_params["centroid"])
    
    # 逆缩放
    coords = coords_norm * scale
    
    # 逆旋转 (R 是正交矩阵，逆 = 转置)
    R_inv = R.T
    coords = coords @ R_inv
    
    # 逆平移
    coords = coords + centroid
    
    velocity_orig = None
    if velocity_rot is not None:
        velocity_orig = velocity_rot @ R_inv
    
    return coords, velocity_orig


def process_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    对单个 DataFrame 进行坐标系归一化
    
    参数:
        df: 输入 DataFrame，必须包含 x, y, z 列
    
    返回:
        df_norm: 归一化后的 DataFrame
        transform_params: 变换参数
    """
    # 提取坐标
    coords = df[['x', 'y', 'z']].values
    
    # 提取速度（如果存在）
    velocity = None
    if all(col in df.columns for col in ['u', 'v', 'w']):
        velocity = df[['u', 'v', 'w']].values
    
    # 提取切线（如果存在）
    tangent = None
    if all(col in df.columns for col in ['Tangent_X', 'Tangent_Y', 'Tangent_Z']):
        tangent = df[['Tangent_X', 'Tangent_Y', 'Tangent_Z']].values
    
    # 提取 WSS 矢量（如果存在）
    wss_vec = None
    if all(col in df.columns for col in ['wss_x', 'wss_y', 'wss_z']):
        wss_vec = df[['wss_x', 'wss_y', 'wss_z']].values
    
    # 执行坐标系归一化
    coords_norm, velocity_rot, tangent_rot, wss_vec_rot, transform_params = \
        normalize_coordinate_system(coords, velocity, tangent, wss_vec)
    
    # 构建输出 DataFrame
    df_norm = df.copy()
    
    # 更新坐标
    df_norm['x'] = coords_norm[:, 0]
    df_norm['y'] = coords_norm[:, 1]
    df_norm['z'] = coords_norm[:, 2]
    
    # 更新速度（如果存在）
    if velocity_rot is not None:
        df_norm['u'] = velocity_rot[:, 0]
        df_norm['v'] = velocity_rot[:, 1]
        df_norm['w'] = velocity_rot[:, 2]
    
    # 更新切线（如果存在）
    if tangent_rot is not None:
        df_norm['Tangent_X'] = tangent_rot[:, 0]
        df_norm['Tangent_Y'] = tangent_rot[:, 1]
        df_norm['Tangent_Z'] = tangent_rot[:, 2]
    
    # 更新 WSS 矢量（如果存在）
    if wss_vec_rot is not None:
        df_norm['wss_x'] = wss_vec_rot[:, 0]
        df_norm['wss_y'] = wss_vec_rot[:, 1]
        df_norm['wss_z'] = wss_vec_rot[:, 2]
    
    return df_norm, transform_params


def process_case(
    case_dir: Path,
    input_subdir: str = None,
    output_subdir: str = None,
) -> bool:
    """
    处理单个病例的所有文件
    
    对于同一个病例，使用第一个文件计算变换参数，然后应用到所有文件。
    这确保同一病例的所有时间步使用相同的坐标系。
    
    参数:
        case_dir: 病例目录
        input_subdir: 输入子目录
        output_subdir: 输出子目录
    
    返回:
        是否成功
    """
    case_dir = Path(case_dir)
    case_name = case_dir.name
    
    if input_subdir is None:
        input_subdir = FEATURES_DIR
    if output_subdir is None:
        output_subdir = COORD_NORMALIZED_DIR
    
    input_dir = case_dir / input_subdir
    output_dir = case_dir / output_subdir
    
    if not input_dir.exists():
        print(f"  ❌ 输入目录不存在: {input_subdir}")
        return False
    
    csv_files = sorted(list(input_dir.glob("result_features_*.csv")))
    if not csv_files:
        print(f"  ❌ 未找到特征文件")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  📁 找到 {len(csv_files)} 个文件")
    
    # 使用第一个文件计算变换参数（确保同一病例使用相同的坐标系）
    print(f"  🔧 计算坐标系变换参数...")
    first_df = pd.read_csv(csv_files[0])
    
    # 只用坐标计算变换参数
    coords = first_df[['x', 'y', 'z']].values
    _, _, _, _, transform_params = normalize_coordinate_system(coords)
    
    # 保存变换参数
    params_path = output_dir / "transform_params.json"
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(transform_params, f, indent=2, ensure_ascii=False)
    print(f"  💾 变换参数已保存: {params_path.name}")
    
    # 复制边界条件元数据（BC 是标量，不受坐标系变换影响）
    bc_meta_src = input_dir / "bc_metadata.json"
    if bc_meta_src.exists():
        shutil.copy2(bc_meta_src, output_dir / "bc_metadata.json")
        print(f"  📋 已复制边界条件元数据: bc_metadata.json")
    else:
        print(f"  ⚠️ 未找到边界条件元数据: {bc_meta_src}")
    
    # 显示变换信息
    print(f"  📊 质心: [{transform_params['centroid'][0]:.2f}, "
          f"{transform_params['centroid'][1]:.2f}, "
          f"{transform_params['centroid'][2]:.2f}]")
    print(f"  📊 缩放因子: {transform_params['scale_factor']:.4f}")
    print(f"  📊 PCA方差比: {[f'{r:.2%}' for r in transform_params['pca_explained_variance_ratio']]}")
    
    # 提取旋转矩阵用于后续处理
    R = np.array(transform_params["rotation_matrix"])
    centroid = np.array(transform_params["centroid"])
    scale_factor = transform_params["scale_factor"]
    
    # 处理所有文件
    success_count = 0
    for csv_file in tqdm(csv_files, desc=f"处理 {case_name}", leave=False):
        try:
            df = pd.read_csv(csv_file)
            
            # 提取并变换坐标
            coords = df[['x', 'y', 'z']].values
            coords_centered = coords - centroid
            coords_aligned = coords_centered @ R
            coords_scaled = coords_aligned / scale_factor
            
            # 更新坐标
            df['x'] = coords_scaled[:, 0]
            df['y'] = coords_scaled[:, 1]
            df['z'] = coords_scaled[:, 2]
            
            # 变换速度（如果存在）
            if all(col in df.columns for col in ['u', 'v', 'w']):
                velocity = df[['u', 'v', 'w']].values
                velocity_rot = velocity @ R
                df['u'] = velocity_rot[:, 0]
                df['v'] = velocity_rot[:, 1]
                df['w'] = velocity_rot[:, 2]
            
            # 变换切线（如果存在）
            if all(col in df.columns for col in ['Tangent_X', 'Tangent_Y', 'Tangent_Z']):
                tangent = df[['Tangent_X', 'Tangent_Y', 'Tangent_Z']].values
                tangent_rot = tangent @ R
                df['Tangent_X'] = tangent_rot[:, 0]
                df['Tangent_Y'] = tangent_rot[:, 1]
                df['Tangent_Z'] = tangent_rot[:, 2]
            
            # 变换 WSS 矢量（如果存在）
            if all(col in df.columns for col in ['wss_x', 'wss_y', 'wss_z']):
                wss_vec = df[['wss_x', 'wss_y', 'wss_z']].values
                wss_vec_rot = wss_vec @ R
                df['wss_x'] = wss_vec_rot[:, 0]
                df['wss_y'] = wss_vec_rot[:, 1]
                df['wss_z'] = wss_vec_rot[:, 2]
            
            # 保存
            output_path = output_dir / csv_file.name
            df.to_csv(output_path, index=False)
            success_count += 1
            
        except Exception as e:
            print(f"  ❌ 处理 {csv_file.name} 失败: {e}")
    
    print(f"  ✅ 完成: {success_count}/{len(csv_files)} 个文件")
    return success_count > 0


def process_all_cases(
    data_root: Path = None,
    target_case: Optional[str] = None,
    input_subdir: str = None,
    output_subdir: str = None,
) -> None:
    """批量处理所有病例"""
    if data_root is None:
        data_root = DATA_ROOT
    else:
        data_root = Path(data_root)
    
    if input_subdir is None:
        input_subdir = FEATURES_DIR
    if output_subdir is None:
        output_subdir = COORD_NORMALIZED_DIR
    
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
    
    print("🚀 坐标系归一化")
    print("=" * 50)
    print(f"📁 数据根目录: {data_root}")
    print(f"📂 输入子目录: {input_subdir}")
    print(f"📂 输出子目录: {output_subdir}")
    print(f"📊 待处理病例数: {len(case_dirs)}")
    print("\n处理步骤:")
    print("  1. 中心化：点云几何中心移到原点")
    print("  2. PCA对齐：血管主轴旋转到Z轴")
    print("  3. 缩放：坐标归一化到 [-1, 1]")
    print("  4. 同步旋转：速度、切线、WSS矢量")
    
    total_start = time.time()
    ok = 0
    
    for idx, case_dir in enumerate(case_dirs, 1):
        try:
            rel_path = case_dir.relative_to(data_root)
        except ValueError:
            rel_path = case_dir.name
        
        print(f"\n[{idx}/{len(case_dirs)}] {rel_path}")
        
        if process_case(case_dir, input_subdir, output_subdir):
            ok += 1
    
    total_time = time.time() - total_start
    
    print(f"\n{'=' * 50}")
    print("🎉 坐标系归一化完成!")
    print(f"⏱️  总耗时: {total_time:.1f}s")
    print(f"✅ 成功: {ok}/{len(case_dirs)} 个病例")
    print(f"\n📋 变换参数已保存到各病例的 {output_subdir}/transform_params.json")
    print("   推理时可使用 inverse_transform() 函数还原到原始坐标系")


def main():
    parser = argparse.ArgumentParser(
        description="坐标系归一化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
坐标系归一化步骤:
  1. 中心化：将点云几何中心移到原点 (0, 0, 0)
  2. PCA对齐：将血管主轴（最大方差方向）旋转到 Z 轴
  3. 缩放：将坐标归一化到 [-1, 1] 范围

矢量特征同步变换:
  - 坐标 (x, y, z): 中心化 + 旋转 + 缩放
  - 速度 (u, v, w): 仅旋转
  - 切线 (Tangent_X/Y/Z): 仅旋转
  - WSS矢量 (wss_x/y/z): 仅旋转
  - 标量特征（压力、曲率等）: 不变

示例:
  python coord_normalize.py --case ZHANG_CHUN
  python coord_normalize.py
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
        help=f"输入子目录，默认 {FEATURES_DIR}",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help=f"输出子目录，默认 {COORD_NORMALIZED_DIR}",
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
