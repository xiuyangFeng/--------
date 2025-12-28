import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import re
import argparse
from pathlib import Path

def extract_time_info(filename):
    """
    从文件名中提取时间步编号。
    文件名示例: result_features_ZHANG_XIU_ZHEN-003-0001.csv 
    最后一个数字通常是时间步。
    """
    matches = re.findall(r'(\d+)', filename)
    if matches:
        # 假设文件名中最后的数字序列是时间步
        step_idx = int(matches[-1])
        return step_idx
    return None

def build_graph_from_csv(file_path, t_norm, k=6):
    """
    读取读取CSV文件并构建图结构。
    """
    df = pd.read_csv(file_path)
    
    # 1. 提取坐标 (x, y, z)
    coords = df[['x', 'y', 'z']].values
    
    # 2. 构建输入特征 (16维)
    # (x, y, z, t) + 几何特征(6) + 边界标志(2) + 边界条件(5)
    
    # 时间特征
    t_feat = np.full((coords.shape[0], 1), t_norm)
    
    # 几何特征
    geom_feats = df[['Abscissa', 'NormRadius', 'Curvature', 'Tangent_X', 'Tangent_Y', 'Tangent_Z']].values
    
    # 边界标志
    flag_feats = df[['is_wall', 'BC_Flag']].values
    
    # 边界条件
    bc_feats = df[['BC_Inlet', 'BC_O1', 'BC_O2', 'BC_O3', 'BC_O4']].values
    
    # 拼接所有特征
    x = np.hstack([coords, t_feat, geom_feats, flag_feats, bc_feats])
    x = torch.from_numpy(x).float()
    
    # 3. 提取目标输出 (4维: u, v, w, p)
    y = df[['u', 'v', 'w', 'p']].values
    y = torch.from_numpy(y).float()
    
    # 4. 构建边索引 (使用 KNN, k=6)
    # 基于空间坐标构建边
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # 转换为 PyG 的 edge_index 格式 [2, num_edges]
    # indices 的第一列是节点自身，忽略它
    row = np.repeat(np.arange(coords.shape[0]), k)
    col = indices[:, 1:].flatten()
    
    edge_index = torch.from_numpy(np.stack([row, col])).long()
    
    # 创建 PyG Data 对象
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def process_case(case_dir, output_dir, k=6):
    """
    处理单个病例的所有时间步。
    """
    case_path = Path(case_dir)
    norm_dir = case_path / "ascii_normalized_128"
    
    if not norm_dir.exists():
        print(f"警告: 找不到目录 {norm_dir}, 跳过案例 {case_path.name}")
        return

    csv_files = sorted(list(norm_dir.glob("*.csv")))
    if not csv_files:
        print(f"警告: {norm_dir} 中没有 CSV 文件")
        return

    # 确定时间步范围以进行归一化
    steps = []
    file_step_pairs = []
    for f in csv_files:
        step = extract_time_info(f.name)
        if step is not None:
            steps.append(step)
            file_step_pairs.append((f, step))
    
    if not steps:
        print(f"错误: 无法从文件名解析时间步信息 {case_path.name}")
        return
        
    min_step, max_step = min(steps), max(steps)
    total_range = max_step - min_step
    if total_range == 0: total_range = 1 # 防止除以0
    
    # 创建输出目录
    case_out_dir = Path(output_dir) / case_path.name
    case_out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"正在处理病例: {case_path.name} ({len(file_step_pairs)} 个时间步)...")
    
    for csv_file, step in tqdm(file_step_pairs):
        t_norm = (step - min_step) / total_range
        data = build_graph_from_csv(csv_file, t_norm, k=k)
        
        # 保存为 .pt 文件
        out_name = csv_file.stem + ".pt"
        torch.save(data, case_out_dir / out_name)

def main():
    parser = argparse.ArgumentParser(description="将归一化的CSV数据转换为PyG图数据 (.pt)")
    parser.add_argument("--data-root", type=str, required=True, help="数据集根目录 (包含 AAA, fast 等)")
    parser.add_argument("--output-dir", type=str, default="./processed_data", help="输出目录")
    parser.add_argument("--k", type=int, default=6, help="KNN 的 K 值")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    output_base = Path(args.output_dir)
    
    # 遍历所有子目录寻找包含 ascii_normalized_128 的病例
    # 例如 data/fast/ZHANG_XIU_ZHEN/ascii_normalized_128
    # 或者 data/AAA/rupture/FENG_LI_XIN/ascii_normalized_128
    
    found_cases = []
    for root, dirs, files in os.walk(data_root):
        if "ascii_normalized_128" in dirs:
            found_cases.append(Path(root))
            
    print(f"发现 {len(found_cases)} 个有效病例目录。")
    
    for case_path in found_cases:
        process_case(case_path, output_base, k=args.k)
        
    print("所有转换任务已完成！")

if __name__ == "__main__":
    main()
