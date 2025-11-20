import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os

def build_graph_edges(pos_numpy, k=16):
    """
    使用 k-NN 算法构建图的边。
    输入: (N, 3) 坐标数组
    输出: (2, E) PyG 格式的 edge_index
    """
    # 使用 sklearn 查找最近邻 (速度快，CPU并行)
    # n_jobs=-1 表示使用所有 CPU 核心
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto', n_jobs=-1).fit(pos_numpy)
    _, indices = nbrs.kneighbors(pos_numpy)
    
    # indices 形状为 (N, k+1)，第一列是点自己
    # 我们构建有向图：从 Neighbor -> Center (消息传递方向)
    
    # 1. 源节点 (邻居): 剔除第一列(自己)，展平
    source_nodes = indices[:, 1:].flatten()
    
    # 2. 目标节点 (中心点): 每个点重复 k 次
    num_nodes = pos_numpy.shape[0]
    target_nodes = np.repeat(np.arange(num_nodes), k)
    
    # 3. 堆叠为 (2, E)
    edge_index = np.vstack((source_nodes, target_nodes))
    
    return torch.from_numpy(edge_index).long()

def process_single_case(csv_path, output_path=None, k=20):
    """
    读取 CSV 特征文件，转换为 PyG Data 对象 (.pt)
    """
    print(f"🔄 正在处理: {csv_path}")
    
    # 1. 读取数据
    df = pd.read_csv(csv_path)
    num_nodes = len(df)
    
    # ==========================================
    # 2. 提取原始特征
    # ==========================================
    
    # A. 物理坐标 (Physical Coordinates) - 用于 PINN 导数计算
    # 注意：这里保留原始毫米单位，不归一化，否则无法计算真实的 Navier-Stokes 残差
    pos_raw = df[['x', 'y', 'z']].values.astype(np.float32)
    
    # B. 几何特征
    abscissa = df['Abscissa'].values.astype(np.float32)
    norm_radius = df['NormRadius'].values.astype(np.float32)
    curvature = df['Curvature'].values.astype(np.float32)
    tangents = df[['Tangent_X', 'Tangent_Y', 'Tangent_Z']].values.astype(np.float32)
    
    # ==========================================
    # 3. 特征预处理 (Feature Engineering)
    # ==========================================
    
    # --- 3.1 曲率处理 (关键!) ---
    # 处理你发现的极端值 (如 218.12)，将其截断。
    # 物理上血管曲率一般不会超过 10-20 (对应半径非常小的转弯)
    curvature = np.clip(curvature, 0.0, 20.0)
    
    # Z-Score 标准化: (x - mean) / std
    # 神经网络喜欢 0 均值，1 方差的数据
    scaler = StandardScaler()
    curvature = scaler.fit_transform(curvature.reshape(-1, 1)).flatten()
    
    # --- 3.2 坐标归一化 (用于输入特征 x) ---
    # 虽然 pos 保留原始值，但作为 Input Feature 的坐标必须归一化
    centroid = np.mean(pos_raw, axis=0)
    max_dist = np.max(np.linalg.norm(pos_raw - centroid, axis=1))
    pos_normalized = (pos_raw - centroid) / max_dist
    
    # ==========================================
    # 4. 组装节点特征矩阵 (x)
    # ==========================================
    
    # 我们将挑选好的特征拼接到一起作为 GNN 的输入
    # 维度说明:
    # 1 (Abscissa) + 1 (NormRadius) + 1 (Curvature) + 3 (Tangent) + 3 (NormPos) = 9 维特征
    
    feature_list = [
        abscissa.reshape(-1, 1),      # [N, 1]
        norm_radius.reshape(-1, 1),   # [N, 1]
        curvature.reshape(-1, 1),     # [N, 1]
        tangents,                     # [N, 3]
        pos_normalized                # [N, 3]
    ]
    
    x = np.hstack(feature_list) # 形状 [N, 9]
    
    # ==========================================
    # 5. 构建图结构 & 转换为 Tensor
    # ==========================================
    
    x_tensor = torch.from_numpy(x).float()
    pos_tensor = torch.from_numpy(pos_raw).float()
    
    # 构建边 (k-NN)
    edge_index = build_graph_edges(pos_raw, k=k)
    
    # ==========================================
    # 6. 创建 Data 对象
    # ==========================================
    
    data = Data(
        x=x_tensor,            # 输入特征 [N, 9]
        pos=pos_tensor,        # 物理坐标 [N, 3] (用于物理损失)
        edge_index=edge_index  # 图连接 [2, E]
    )
    
    # 保存元数据，方便后续反归一化
    data.centroid = torch.from_numpy(centroid).float()
    data.scale = float(max_dist)
    
    # 打印信息检查
    print("✅ 转换成功:")
    print(f"   - 节点数: {data.num_nodes}")
    print(f"   - 边数:   {data.num_edges} (k={k})")
    print(f"   - 特征维: {data.num_node_features} (Abs, Rad, Curv, Tan*3, Pos*3)")
    print(f"   - Pos范围: x[{pos_raw[:,0].min():.1f}, {pos_raw[:,0].max():.1f}]")
    
    # 保存
    if output_path:
        torch.save(data, output_path)
        print(f"💾 已保存至: {output_path}")
    
    return data

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 配置路径
    input_csv = "result_features.csv"  # 上一步生成的 CSV
    output_pt = "vessel_graph.pt"      # 输出的 PyG 文件
    
    # 邻居数 k (根据显存大小调整，通常 10-30 之间)
    k_neighbors = 20
    
    if os.path.exists(input_csv):
        data = process_single_case(input_csv, output_pt, k=k_neighbors)
        
        # 简单的 Tensor 检查
        print("\n🔍 Tensor 数据预览:")
        print(f"Data Object: {data}")
        print(f"x (前3行):\n{data.x[:3]}")
        print(f"edge_index (前5列):\n{data.edge_index[:, :5]}")
    else:
        print(f"❌ 找不到文件: {input_csv}")