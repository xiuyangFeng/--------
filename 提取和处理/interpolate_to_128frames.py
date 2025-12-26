#!/usr/bin/env python3
"""
数据插值脚本：将不同帧数（80帧或160帧）的数据插值到统一的128帧

功能：
1. 处理 ascii_merged 文件夹中的 CSV 数据
2. 处理 flow-out* 和 p-out* 文件中的出口流量/压力数据
3. 使用线性插值将数据统一到 128 帧
4. 基于第一帧建立参考坐标系，使用 KDTree 最近邻匹配进行空间对齐
5. 可选的坐标归一化功能（去中心化 + 尺度缩放），并保存归一化参数供还原

作者：自动生成
日期：2024
"""

import os
import re
import glob
import json
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.spatial import cKDTree
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
from tqdm import tqdm


def extract_frame_number(filename: str) -> int:
    """
    从文件名中提取最后四位数字作为帧号
    
    例如:
    - "FENG LI XIN-0161.csv" -> 161
    - "ZHANG_XIU_ZHEN-003-1121.csv" -> 1121
    """
    # 移除扩展名
    basename = os.path.splitext(os.path.basename(filename))[0]
    # 提取最后4位数字
    match = re.search(r'(\d{4})$', basename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"无法从文件名 {filename} 中提取帧号")


def get_csv_files_sorted(ascii_merged_dir: str) -> List[str]:
    """
    获取 ascii_merged 目录中所有 CSV 文件，按帧号排序
    """
    csv_files = glob.glob(os.path.join(ascii_merged_dir, "*.csv"))
    if not csv_files:
        return []
    
    # 按帧号排序
    csv_files_sorted = sorted(csv_files, key=lambda x: extract_frame_number(x))
    return csv_files_sorted


def parse_out_file(out_file_path: str) -> pd.DataFrame:
    """
    解析 flow-out* 或 p-out* 文件
    
    返回 DataFrame，包含 time_step, value, flow_time 三列
    """
    if not os.path.exists(out_file_path):
        return None
    
    data = []
    with open(out_file_path, 'r') as f:
        lines = f.readlines()
    
    # 跳过前3行头部信息
    for line in lines[3:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                time_step = int(parts[0])
                value = float(parts[1])
                flow_time = float(parts[2])
                data.append({
                    'time_step': time_step,
                    'value': value,
                    'flow_time': flow_time
                })
            except ValueError:
                continue
    
    if data:
        return pd.DataFrame(data)
    return None


def normalize_coordinates(dfs: List[pd.DataFrame], 
                          method: str = 'center_scale') -> Tuple[List[pd.DataFrame], Dict]:
    """
    对坐标进行归一化处理
    
    参数:
        dfs: DataFrame 列表（每个 DataFrame 包含 x, y, z 列）
        method: 归一化方法
            - 'center_scale': 去中心化 + 缩放到 [-1, 1] 范围
            - 'min_max': 最小-最大归一化到 [0, 1] 范围
    
    返回:
        归一化后的 DataFrame 列表, 归一化参数字典
    """
    if not dfs:
        return [], {}
    
    # 使用第一帧的坐标计算归一化参数（因为所有帧已经对齐到第一帧的坐标系）
    first_df = dfs[0]
    coords = first_df[['x', 'y', 'z']].values
    
    if method == 'center_scale':
        # 计算中心点（质心）
        centroid = np.mean(coords, axis=0)
        
        # 去中心化后计算到原点的最大距离
        centered_coords = coords - centroid
        max_distance = np.max(np.linalg.norm(centered_coords, axis=1))
        
        # 避免除零
        if max_distance == 0:
            max_distance = 1.0
        
        # 缩放因子（使坐标范围大致在 [-1, 1]）
        scale_factor = max_distance
        
        norm_params = {
            'method': 'center_scale',
            'centroid': centroid.tolist(),  # 平移向量（中心点坐标）
            'scale_factor': float(scale_factor),  # 缩放因子
            'description': '归一化公式: normalized = (original - centroid) / scale_factor',
            'restore_formula': 'original = normalized * scale_factor + centroid'
        }
        
        print(f"\n坐标归一化参数:")
        print(f"  方法: 去中心化 + 尺度缩放")
        print(f"  质心 (centroid): [{centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f}]")
        print(f"  缩放因子 (scale_factor): {scale_factor:.6f}")
        
    elif method == 'min_max':
        # 计算每个坐标轴的最小和最大值
        min_vals = np.min(coords, axis=0)
        max_vals = np.max(coords, axis=0)
        
        # 计算范围
        ranges = max_vals - min_vals
        # 避免除零
        ranges[ranges == 0] = 1.0
        
        norm_params = {
            'method': 'min_max',
            'min_values': min_vals.tolist(),  # 平移向量（最小值）
            'max_values': max_vals.tolist(),
            'ranges': ranges.tolist(),  # 缩放因子（范围）
            'description': '归一化公式: normalized = (original - min) / range',
            'restore_formula': 'original = normalized * range + min'
        }
        
        print(f"\n坐标归一化参数:")
        print(f"  方法: 最小-最大归一化")
        print(f"  最小值: [{min_vals[0]:.6f}, {min_vals[1]:.6f}, {min_vals[2]:.6f}]")
        print(f"  最大值: [{max_vals[0]:.6f}, {max_vals[1]:.6f}, {max_vals[2]:.6f}]")
    else:
        raise ValueError(f"未知的归一化方法: {method}")
    
    # 对所有 DataFrame 应用归一化
    normalized_dfs = []
    for df in dfs:
        df_norm = df.copy()
        
        if method == 'center_scale':
            df_norm['x'] = (df['x'] - centroid[0]) / scale_factor
            df_norm['y'] = (df['y'] - centroid[1]) / scale_factor
            df_norm['z'] = (df['z'] - centroid[2]) / scale_factor
        elif method == 'min_max':
            df_norm['x'] = (df['x'] - min_vals[0]) / ranges[0]
            df_norm['y'] = (df['y'] - min_vals[1]) / ranges[1]
            df_norm['z'] = (df['z'] - min_vals[2]) / ranges[2]
        
        normalized_dfs.append(df_norm)
    
    # 打印归一化后的坐标范围
    norm_coords = normalized_dfs[0][['x', 'y', 'z']].values
    print(f"  归一化后坐标范围:")
    print(f"    X: [{norm_coords[:, 0].min():.4f}, {norm_coords[:, 0].max():.4f}]")
    print(f"    Y: [{norm_coords[:, 1].min():.4f}, {norm_coords[:, 1].max():.4f}]")
    print(f"    Z: [{norm_coords[:, 2].min():.4f}, {norm_coords[:, 2].max():.4f}]")
    
    return normalized_dfs, norm_params


def save_normalization_params(norm_params: Dict, output_path: str):
    """
    保存归一化参数到 JSON 文件
    
    参数:
        norm_params: 归一化参数字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(norm_params, f, indent=2, ensure_ascii=False)
    print(f"  归一化参数已保存到: {output_path}")


def restore_coordinates(df: pd.DataFrame, norm_params: Dict) -> pd.DataFrame:
    """
    根据归一化参数还原坐标（用于可视化）
    
    参数:
        df: 归一化后的 DataFrame
        norm_params: 归一化参数字典
    
    返回:
        还原后的 DataFrame
    """
    df_restored = df.copy()
    method = norm_params.get('method', 'center_scale')
    
    if method == 'center_scale':
        centroid = np.array(norm_params['centroid'])
        scale_factor = norm_params['scale_factor']
        
        df_restored['x'] = df['x'] * scale_factor + centroid[0]
        df_restored['y'] = df['y'] * scale_factor + centroid[1]
        df_restored['z'] = df['z'] * scale_factor + centroid[2]
        
    elif method == 'min_max':
        min_vals = np.array(norm_params['min_values'])
        ranges = np.array(norm_params['ranges'])
        
        df_restored['x'] = df['x'] * ranges[0] + min_vals[0]
        df_restored['y'] = df['y'] * ranges[1] + min_vals[1]
        df_restored['z'] = df['z'] * ranges[2] + min_vals[2]
    
    return df_restored


def interpolate_csv_data(csv_files: List[str], target_frames: int = 128) -> List[pd.DataFrame]:
    """
    对 CSV 文件数据进行插值
    
    使用第一帧作为参考坐标系，对后续所有帧的数据基于最近邻进行空间映射，
    然后在固定的参考坐标系下对时间轴进行插值处理。
    
    参数:
        csv_files: 排序后的 CSV 文件列表
        target_frames: 目标帧数
    
    返回:
        插值后的 DataFrame 列表
    """
    n_source = len(csv_files)
    if n_source == 0:
        return []
    
    print(f"正在读取 {n_source} 个 CSV 文件...")
    
    # 读取所有 CSV 文件
    dfs = []
    point_counts = []
    for csv_file in tqdm(csv_files, desc="读取CSV文件"):
        df = pd.read_csv(csv_file)
        dfs.append(df)
        point_counts.append(len(df))
    
    # 获取第一个文件的结构（作为参考坐标系）
    first_df = dfs[0]
    columns = first_df.columns.tolist()
    
    # 检查点数
    min_points = min(point_counts)
    max_points = max(point_counts)
    
    print(f"数据点数范围: {min_points} - {max_points}")
    
    # 确定需要插值的列（数值列，排除坐标和is_wall）
    numeric_cols = ['u', 'v', 'w', 'p', 'vel_mag', 'wss', 'wss_x', 'wss_y', 'wss_z']
    interp_cols = [col for col in numeric_cols if col in columns]
    
    # 静态列（坐标等）
    static_cols = ['x', 'y', 'z', 'is_wall']
    static_cols = [col for col in static_cols if col in columns]
    
    print(f"需要插值的列: {interp_cols}")
    print(f"静态列: {static_cols}")
    
    # 使用第一帧作为参考坐标系
    print("\n使用第一帧作为参考坐标系...")
    ref_coords = first_df[['x', 'y', 'z']].values
    n_points = len(ref_coords)
    print(f"参考坐标系点数: {n_points}")
    
    # 获取静态数据（使用第一帧的坐标和is_wall）
    static_data = first_df[static_cols].copy() if static_cols else None
    
    # 创建源时间点（归一化到 0-1）
    source_times = np.linspace(0, 1, n_source)
    # 创建目标时间点
    target_times = np.linspace(0, 1, target_frames)
    
    # 准备数据数组，通过空间映射将所有帧的数据对齐到参考坐标系
    print("\n正在进行空间映射（基于最近邻匹配）...")
    data_array = np.zeros((n_source, n_points, len(interp_cols)))
    
    # 第一帧直接使用原始数据
    for j, col in enumerate(interp_cols):
        data_array[0, :, j] = first_df[col].values
    
    # 对后续帧进行空间映射
    for i in tqdm(range(1, n_source), desc="空间映射"):
        df = dfs[i]
        
        # 获取当前帧的坐标
        curr_coords = df[['x', 'y', 'z']].values
        
        # 构建当前帧的 KDTree
        tree = cKDTree(curr_coords)
        
        # 为参考坐标系中的每个点找到当前帧中的最近邻
        distances, indices = tree.query(ref_coords, k=1)
        
        # 统计匹配信息
        if i == 1:  # 只在第二帧时打印统计信息
            mean_dist = np.mean(distances)
            max_dist = np.max(distances)
            print(f"  最近邻匹配统计 - 平均距离: {mean_dist:.6f}, 最大距离: {max_dist:.6f}")
        
        # 使用最近邻的数据
        for j, col in enumerate(interp_cols):
            data_array[i, :, j] = df[col].values[indices]
    
    # 对每个点的每个特征进行时间插值
    print(f"\n正在对时间轴进行插值（{n_source} 帧 -> {target_frames} 帧）...")
    interpolated_data = np.zeros((target_frames, n_points, len(interp_cols)))
    
    for point_idx in tqdm(range(n_points), desc="时间插值"):
        for col_idx in range(len(interp_cols)):
            # 获取该点该特征在所有时间步的值
            values = data_array[:, point_idx, col_idx]
            # 创建插值函数
            f = interpolate.interp1d(source_times, values, kind='linear', 
                                      fill_value='extrapolate')
            # 计算插值结果
            interpolated_data[:, point_idx, col_idx] = f(target_times)
    
    # 构建输出 DataFrame 列表
    print("正在构建输出数据...")
    result_dfs = []
    for frame_idx in tqdm(range(target_frames), desc="构建输出"):
        df_out = pd.DataFrame()
        
        # 添加静态列（使用参考坐标系的坐标）
        if static_data is not None:
            for col in static_cols:
                df_out[col] = static_data[col].values
        
        # 添加插值列
        for col_idx, col in enumerate(interp_cols):
            df_out[col] = interpolated_data[frame_idx, :, col_idx]
        
        # 重新排列列顺序与原始文件一致
        df_out = df_out[columns]
        result_dfs.append(df_out)
    
    return result_dfs


def interpolate_out_file_data(out_df: pd.DataFrame, 
                               frame_numbers: List[int], 
                               target_frames: int = 128) -> pd.DataFrame:
    """
    对 out 文件中的数据进行插值
    
    参数:
        out_df: 原始 out 文件数据
        frame_numbers: 对应 CSV 文件的帧号列表
        target_frames: 目标帧数
    
    返回:
        插值后的 DataFrame
    """
    if out_df is None:
        return None
    
    # 根据帧号筛选数据
    filtered_df = out_df[out_df['time_step'].isin(frame_numbers)].copy()
    filtered_df = filtered_df.sort_values('time_step')
    
    if len(filtered_df) == 0:
        print(f"警告: 没有找到匹配的帧号数据")
        return None
    
    n_source = len(filtered_df)
    
    # 创建源时间点和目标时间点
    source_times = np.linspace(0, 1, n_source)
    target_times = np.linspace(0, 1, target_frames)
    
    # 对 value 和 flow_time 进行插值
    value_interp = interpolate.interp1d(source_times, filtered_df['value'].values, 
                                         kind='linear', fill_value='extrapolate')
    flow_time_interp = interpolate.interp1d(source_times, filtered_df['flow_time'].values, 
                                             kind='linear', fill_value='extrapolate')
    
    # 构建输出 DataFrame
    result = pd.DataFrame({
        'time_step': np.arange(1, target_frames + 1),
        'value': value_interp(target_times),
        'flow_time': flow_time_interp(target_times)
    })
    
    return result


def generate_output_filenames(csv_files: List[str], target_frames: int = 128) -> List[str]:
    """
    生成输出文件名
    
    基于原始文件名格式生成新的文件名
    """
    if not csv_files:
        return []
    
    # 获取文件名前缀（去掉最后的帧号）
    first_file = os.path.basename(csv_files[0])
    prefix = re.sub(r'-?\d{4}\.csv$', '', first_file)
    
    # 生成新的文件名，帧号从 0001 开始
    output_names = []
    for i in range(1, target_frames + 1):
        output_names.append(f"{prefix}-{i:04d}.csv")
    
    return output_names


def process_patient_directory(patient_dir: str, 
                               output_dir: Optional[str] = None,
                               target_frames: int = 128,
                               dry_run: bool = False,
                               normalize: bool = False,
                               norm_method: str = 'center_scale') -> Dict:
    """
    处理单个病人目录
    
    参数:
        patient_dir: 病人目录路径
        output_dir: 输出目录路径（默认在病人目录下创建 ascii_merged_128 和 out_interpolated_128）
        target_frames: 目标帧数
        dry_run: 是否只做检查不实际处理
        normalize: 是否进行坐标归一化
        norm_method: 归一化方法 ('center_scale' 或 'min_max')
    
    返回:
        处理结果信息
    """
    result = {
        'patient_dir': patient_dir,
        'status': 'unknown',
        'ascii_merged_frames': 0,
        'out_files_processed': [],
        'normalized': normalize
    }
    
    ascii_merged_dir = os.path.join(patient_dir, 'ascii_merged')
    
    # 检查 ascii_merged 目录是否存在
    if not os.path.exists(ascii_merged_dir):
        result['status'] = 'no_ascii_merged'
        print(f"警告: {patient_dir} 中没有 ascii_merged 目录")
        return result
    
    # 获取 CSV 文件
    csv_files = get_csv_files_sorted(ascii_merged_dir)
    n_frames = len(csv_files)
    result['ascii_merged_frames'] = n_frames
    
    if n_frames == 0:
        result['status'] = 'no_csv_files'
        print(f"警告: {ascii_merged_dir} 中没有 CSV 文件")
        return result
    
    print(f"\n{'='*60}")
    print(f"处理病人目录: {patient_dir}")
    print(f"原始帧数: {n_frames}")
    print(f"目标帧数: {target_frames}")
    
    # 提取帧号列表
    frame_numbers = [extract_frame_number(f) for f in csv_files]
    print(f"帧号范围: {min(frame_numbers)} - {max(frame_numbers)}")
    
    if dry_run:
        result['status'] = 'dry_run'
        return result
    
    # 设置输出目录
    if output_dir is None:
        csv_output_dir = os.path.join(patient_dir, f'ascii_merged_{target_frames}')
        out_output_dir = os.path.join(patient_dir, f'out_interpolated_{target_frames}')
    else:
        patient_name = os.path.basename(patient_dir)
        csv_output_dir = os.path.join(output_dir, patient_name, f'ascii_merged_{target_frames}')
        out_output_dir = os.path.join(output_dir, patient_name, f'out_interpolated_{target_frames}')
    
    os.makedirs(csv_output_dir, exist_ok=True)
    os.makedirs(out_output_dir, exist_ok=True)
    
    # 处理 CSV 数据
    print("\n--- 处理 CSV 数据 ---")
    interpolated_dfs = interpolate_csv_data(csv_files, target_frames)
    
    # 坐标归一化（如果启用）
    norm_params = None
    if normalize:
        print("\n--- 坐标归一化 ---")
        interpolated_dfs, norm_params = normalize_coordinates(interpolated_dfs, method=norm_method)
        
        # 保存归一化参数
        norm_params_path = os.path.join(csv_output_dir, 'normalization_params.json')
        save_normalization_params(norm_params, norm_params_path)
        result['norm_params_path'] = norm_params_path
    
    # 生成输出文件名
    output_filenames = generate_output_filenames(csv_files, target_frames)
    
    # 保存插值后的 CSV 文件
    print(f"\n正在保存 CSV 文件到 {csv_output_dir}...")
    for df, filename in tqdm(zip(interpolated_dfs, output_filenames), 
                              total=len(interpolated_dfs), desc="保存CSV"):
        output_path = os.path.join(csv_output_dir, filename)
        df.to_csv(output_path, index=False)
    
    # 处理 out 文件（处理所有 .out 文件，包括出口流量、压力、入口信息等）
    print("\n--- 处理 out 文件 ---")
    # 匹配所有 .out 文件
    out_files = glob.glob(os.path.join(patient_dir, "*.out"))
    
    if not out_files:
        print("  未找到任何 .out 文件")
    else:
        print(f"  找到 {len(out_files)} 个 .out 文件")
    
    for out_file in out_files:
        out_filename = os.path.basename(out_file)
        print(f"\n处理: {out_filename}")
        
        # 解析 out 文件
        out_df = parse_out_file(out_file)
        if out_df is None:
            print(f"  警告: 无法解析 {out_filename}")
            continue
        
        # 检查是否有足够的数据点与帧号匹配
        matching_count = out_df['time_step'].isin(frame_numbers).sum()
        print(f"  原始数据点数: {len(out_df)}, 与帧号匹配的数据点数: {matching_count}")
        
        if matching_count == 0:
            print(f"  警告: 没有与帧号 ({min(frame_numbers)}-{max(frame_numbers)}) 匹配的数据，跳过")
            continue
        
        # 插值
        interpolated_out = interpolate_out_file_data(out_df, frame_numbers, target_frames)
        if interpolated_out is None:
            print(f"  警告: 插值失败 {out_filename}")
            continue
        
        # 保存
        output_out_path = os.path.join(out_output_dir, out_filename)
        
        # 保存为与原格式类似的格式
        with open(output_out_path, 'w') as f:
            # 写入头部（简化版）
            name = out_filename.replace('.out', '')
            f.write(f'"{name}"\n')
            f.write(f'"Time Step" "{name} etc.."\n')
            f.write(f'("Time Step" "{name}" "flow-time")\n')
            
            # 写入数据
            for _, row in interpolated_out.iterrows():
                f.write(f"{int(row['time_step'])} {row['value']:.15e} {row['flow_time']:.6f}\n")
        
        print(f"  已保存到: {output_out_path}")
        result['out_files_processed'].append(out_filename)
    
    result['status'] = 'success'
    print(f"\n处理完成!")
    return result


def find_patient_directories(data_dir: str) -> List[str]:
    """
    在数据目录中查找所有病人目录（包含 ascii_merged 子目录的目录）
    """
    patient_dirs = []
    
    for root, dirs, files in os.walk(data_dir):
        if 'ascii_merged' in dirs:
            patient_dirs.append(root)
    
    return patient_dirs


def main():
    parser = argparse.ArgumentParser(
        description='将 CFD 数据从不同帧数（80/160帧）插值到统一的帧数（默认128帧）'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='./data',
        help='数据根目录 (默认: ./data)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='输出目录 (默认: 在原目录下创建)'
    )
    parser.add_argument(
        '--target-frames', '-t',
        type=int,
        default=128,
        help='目标帧数 (默认: 128)'
    )
    parser.add_argument(
        '--patient', '-p',
        type=str,
        default=None,
        help='只处理指定病人目录名'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='只检查不实际处理'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有找到的病人目录'
    )
    parser.add_argument(
        '--normalize', '-n',
        action='store_true',
        help='启用坐标归一化（去中心化 + 尺度缩放）'
    )
    parser.add_argument(
        '--norm-method',
        type=str,
        default='center_scale',
        choices=['center_scale', 'min_max'],
        help='归一化方法: center_scale (去中心化+缩放到[-1,1]) 或 min_max (缩放到[0,1]) (默认: center_scale)'
    )
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    
    print(f"数据目录: {data_dir}")
    
    # 查找所有病人目录
    patient_dirs = find_patient_directories(data_dir)
    
    print(f"\n找到 {len(patient_dirs)} 个病人目录:")
    for pd_path in patient_dirs:
        csv_files = get_csv_files_sorted(os.path.join(pd_path, 'ascii_merged'))
        print(f"  - {pd_path} ({len(csv_files)} 帧)")
    
    if args.list:
        return
    
    # 筛选要处理的目录
    if args.patient:
        patient_dirs = [d for d in patient_dirs if args.patient in d]
        if not patient_dirs:
            print(f"\n错误: 未找到包含 '{args.patient}' 的病人目录")
            return
    
    # 处理每个病人目录
    results = []
    for patient_dir in patient_dirs:
        result = process_patient_directory(
            patient_dir,
            output_dir=args.output_dir,
            target_frames=args.target_frames,
            dry_run=args.dry_run,
            normalize=args.normalize,
            norm_method=args.norm_method
        )
        results.append(result)
    
    # 打印汇总
    print("\n" + "="*60)
    print("处理汇总:")
    print("="*60)
    for r in results:
        print(f"\n{r['patient_dir']}:")
        print(f"  状态: {r['status']}")
        print(f"  原始帧数: {r['ascii_merged_frames']}")
        print(f"  处理的 out 文件: {len(r['out_files_processed'])}")
        print(f"  坐标归一化: {'是' if r.get('normalized', False) else '否'}")
        if r.get('norm_params_path'):
            print(f"  归一化参数文件: {r['norm_params_path']}")


if __name__ == '__main__':
    main()

