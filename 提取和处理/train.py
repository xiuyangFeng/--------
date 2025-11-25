import torch
import torch.optim as optim
from torch_geometric.utils import subgraph
from model import PI_GNN
from physics import compute_pinn_loss
import time
import numpy as np
import math
import gc
import os
import glob

# ==========================================
# 几何分块加载器
# ==========================================
class GeometricBatchLoader:
    """
    几何分块加载器，按照几何位置对图数据进行分块处理
    """
    def __init__(self, data, batch_size=2048, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_nodes = data.num_nodes
        
        # 计算数据在各轴上的分布范围，选择分布最广的轴作为主轴
        pos = data.pos.numpy()
        spread = pos.max(0) - pos.min(0)
        self.main_axis = np.argmax(spread)
        
        # 按主轴坐标排序节点索引
        self.sorted_indices = torch.argsort(data.pos[:, self.main_axis])
        self.num_batches = math.ceil(self.num_nodes / batch_size)

    def __iter__(self):
        batch_order = torch.randperm(self.num_batches) if self.shuffle else torch.arange(self.num_batches)
        
        for i in batch_order:
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.num_nodes)
            current_node_indices = self.sorted_indices[start_idx:end_idx]
            
            # 提取当前批次的子图
            edge_index, _ = subgraph(
                current_node_indices, 
                self.data.edge_index, 
                relabel_nodes=True, 
                num_nodes=self.num_nodes
            )
            
            # 提取数据
            x = self.data.x[current_node_indices]
            pos = self.data.pos[current_node_indices]
            y = self.data.y[current_node_indices] if hasattr(self.data, 'y') else None
            mask_wall = self.data.mask_wall[current_node_indices]
            mask_inner = self.data.mask_inner[current_node_indices]
            
            # 创建批次对象
            class BatchData: pass
            batch = BatchData()
            batch.x = x
            batch.edge_index = edge_index
            batch.pos = pos
            batch.y = y
            batch.mask_wall = mask_wall
            batch.mask_inner = mask_inner
            batch.num_nodes = len(current_node_indices)
            
            yield batch

    def __len__(self):
        return self.num_batches

def train():
    # 1. 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")
    
    # 2. 查找并加载数据
    # 优先查找 .pt 文件，如果没有则提示用户生成
    pt_files = glob.glob("*.pt")
    if not pt_files:
        print("❌ 未找到 .pt 图数据文件。请先运行 dataset.py 或 batch_process.py 生成数据。")
        return
    
    data_path = pt_files[0] # 默认使用第一个找到的
    print(f"📂 读取数据: {data_path}")
    
    try:
        full_data = torch.load(data_path, weights_only=False)
    except TypeError:
        full_data = torch.load(data_path)

    print(f"   - 总节点数: {full_data.num_nodes}")
    
    # 获取全局归一化参数
    global_centroid = full_data.centroid.to(device)
    global_scale = full_data.scale

    # 3. 初始化加载器
    BATCH_SIZE = 2048 
    loader = GeometricBatchLoader(full_data, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✅ 准备就绪! Batch Size={BATCH_SIZE}, 共 {len(loader)} 个 Batch。")

    # 4. 初始化模型
    model = PI_GNN(in_channels=9, hidden_channels=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. 训练循环
    EPOCHS = 50
    print(f"\n🏃 开始训练 (Epochs: {EPOCHS})...")
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        start_epoch = time.time()
        
        for i, batch in enumerate(loader):
            # 数据移至 GPU
            x_static = batch.x[:, :6].to(device) # 静态特征
            edge_index = batch.edge_index.to(device)
            pos = batch.pos.to(device)
            y = batch.y.to(device) if batch.y is not None else None
            mask_wall = batch.mask_wall.to(device)
            mask_inner = batch.mask_inner.to(device)
            
            # 激活梯度计算 (PINN)
            pos.requires_grad_(True)
            
            # 动态重计算归一化坐标
            pos_normalized = (pos - global_centroid) / (global_scale + 1e-8)
            x_new = torch.cat([x_static, pos_normalized], dim=1)
            
            optimizer.zero_grad()
            
            # 前向传播
            u_pred, p_pred = model(x_new, edge_index)
            
            # Loss 计算
            loss_data = torch.tensor(0.0, device=device)
            if y is not None:
                loss_u = torch.nn.functional.mse_loss(u_pred, y[:, 0:3])
                loss_p = torch.nn.functional.mse_loss(p_pred, y[:, 3:4])
                loss_data = loss_u + loss_p
            
            # Physics Loss
            loss_phy = torch.tensor(0.0, device=device)
            if mask_inner.sum() > 10:
                loss_phy = compute_pinn_loss(u_pred, p_pred, pos, mask_inner)
            
            # Boundary Loss
            loss_bc = torch.tensor(0.0, device=device)
            if mask_wall.sum() > 0:
                loss_bc = torch.mean(u_pred[mask_wall]**2) # No-slip
            
            # Total Loss
            loss = loss_data + 0.1 * loss_phy + 1.0 * loss_bc
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 显存清理
            del u_pred, p_pred, loss, x_new, pos_normalized
            
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.6f} | Time: {time.time()-start_epoch:.1f}s")
        
        # 每 10 轮保存一次
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

    print("\n✅ 训练完成！")
    torch.save(model.state_dict(), "model_final.pth")
    print("💾 模型已保存: model_final.pth")

if __name__ == "__main__":
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train()
