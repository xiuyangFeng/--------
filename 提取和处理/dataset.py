import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os
import glob

# ==========================================
# 1. 图构建模块 (k-NN)
# ==========================================
def build_graph_edges(pos_numpy, k=16):
    """
    构建动态图连接 (k-NN)。
    返回: edge_index [2, E]
    """
    # 使用 sklearn 查找最近邻 (CPU并行加速)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(pos_numpy)
    _, indices = nbrs.kneighbors(pos_numpy)
    
    # 1. 源节点 (Neighbors): 剔除第一列(自己)，展平
    source_nodes = indices[:, 1:].flatten()
    
    # 2. 目标节点 (Center): 每个点重复 k 次
    num_nodes = pos_numpy.shape[0]
    target_nodes = np.repeat(np.arange(num_nodes), k)
    
    # 3. 堆叠
    edge_index = np.vstack((source_nodes, target_nodes))
    return torch.from_numpy(edge_index).long()

# ==========================================
# 2. 核心转换逻辑
# ==========================================
class GraphDataset:
    """
    图数据集处理类，用于管理数据的加载和转换
    """
    def __init__(self, k=20):
        self.k = k

    def process_file(self, csv_path, output_path=None):
        """
        将 CSV 转换为包含 Features(x), Labels(y), Masks 的 PyG Data 对象。
        """
        print(f"🔄 正在读取: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ 读取失败: {e}")
            return None
        
        # --- A. 提取坐标 (Pos) ---
        # 原始物理坐标，用于 PINN 物理导数计算 (必须保留真实尺度)
        pos_raw = df[['x', 'y', 'z']].values.astype(np.float32)
        
        # --- B. 构建输入特征 (x) ---
        # 1. 几何特征读取 (确保列存在，如果不存在则用0填充)
        def get_col(name, default=0.0):
            if name in df.columns:
                return df[name].values.astype(np.float32)
            return np.full(len(df), default, dtype=np.float32)

        abscissa = get_col('Abscissa')
        norm_radius = get_col('NormRadius')
        curvature = get_col('Curvature')
        tangent_x = get_col('Tangent_X')
        tangent_y = get_col('Tangent_Y')
        tangent_z = get_col('Tangent_Z')
        tangents = np.stack([tangent_x, tangent_y, tangent_z], axis=1)
        
        # 2. 特征工程
        # (a) 曲率截断与标准化
        curvature = np.clip(curvature, 0.0, 20.0)
        # 简单的标准化，避免依赖外部 scaler 状态
        if np.std(curvature) > 1e-6:
            curvature = (curvature - np.mean(curvature)) / np.std(curvature)
        
        # (b) 坐标归一化 (仅用于输入特征 x，不影响 pos)
        centroid = np.mean(pos_raw, axis=0)
        max_dist = np.max(np.linalg.norm(pos_raw - centroid, axis=1))
        pos_normalized = (pos_raw - centroid) / (max_dist + 1e-8)
        
        # 3. 拼接特征矩阵 X [N, 9]
        # [Abscissa(1), NormRadius(1), Curvature(1), Tangents(3), NormPos(3)]
        x_features = np.hstack([
            abscissa.reshape(-1, 1),
            norm_radius.reshape(-1, 1),
            curvature.reshape(-1, 1),
            tangents,
            pos_normalized
        ])
        x_tensor = torch.from_numpy(x_features).float()
        
        # --- C. 构建标签 (y) ---
        # 检查 CSV 中是否包含 CFD 结果 (u, v, w, p)
        y_tensor = None
        has_labels = False
        potential_labels = ['u', 'v', 'w', 'p']
        
        if all(col in df.columns for col in potential_labels):
            # 提取 u, v, w, p
            labels = df[potential_labels].values.astype(np.float32)
            y_tensor = torch.from_numpy(labels).float()
            has_labels = True
        
        # --- D. 构建边界掩码 (Masks) ---
        # 用于 PINN 区分壁面和内部点
        if 'is_wall' in df.columns:
            is_wall = df['is_wall'].values.astype(bool)
        else:
            # 如果没有标签，根据 NormRadius 推断 (大于 0.95 视为壁面)
            is_wall = norm_radius > 0.95

        # 转换为 boolean tensor
        mask_wall = torch.from_numpy(is_wall).bool()
        mask_inner = ~mask_wall # 内部点就是非壁面点

        # --- E. 构建图连接 (Edge Index) ---
        edge_index = build_graph_edges(pos_raw, k=self.k)
        
        # --- F. 封装 Data 对象 ---
        data = Data(
            x=x_tensor,             # 输入: [N, 9]
            pos=torch.from_numpy(pos_raw).float(), # 物理坐标: [N, 3]
            edge_index=edge_index,  # 边: [2, E]
            
            # 辅助掩码 (用于 Loss 计算)
            mask_wall=mask_wall,    # [N] Bool: 哪些是壁面
            mask_inner=mask_inner   # [N] Bool: 哪些是内部流体
        )
        
        # 如果有标签，加入 y
        if has_labels:
            data.y = y_tensor       # 标签: [N, 4] (u,v,w,p)
            
        # 保存元数据 (用于反归一化)
        data.centroid = torch.from_numpy(centroid).float()
        data.scale = float(max_dist)
        
        if output_path:
            torch.save(data, output_path)
            print(f"💾 PyG 数据已保存至: {output_path}")
            
        return data

# ==========================================
# 主程序测试
# ==========================================
if __name__ == "__main__":
    # 简单的测试逻辑
    dataset = GraphDataset(k=20)
    # 查找当前目录下是否有 csv 文件
    csv_files = glob.glob("*.csv")
    if csv_files:
        print(f"找到 {len(csv_files)} 个 CSV 文件，正在处理第一个: {csv_files[0]}")
        dataset.process_file(csv_files[0], "test_graph.pt")
    else:
        print("当前目录下没有 CSV 文件，请先运行 batch_process.py 生成数据。")
