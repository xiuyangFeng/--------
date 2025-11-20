import torch
import torch.optim as optim
from torch_geometric.utils import subgraph
from model import PI_GNN
from physics import compute_pinn_loss
import time
import numpy as np
import math
import gc # 垃圾回收

# ==========================================
# 自定义：几何分块加载器 (逻辑不变)
# ==========================================
class GeometricBatchLoader:
    def __init__(self, data, batch_size=2048, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_nodes = data.num_nodes
        
        pos = data.pos.numpy()
        spread = pos.max(0) - pos.min(0)
        self.main_axis = np.argmax(spread)
        
        self.sorted_indices = torch.argsort(data.pos[:, self.main_axis])
        self.num_batches = math.ceil(self.num_nodes / batch_size)

    def __iter__(self):
        batch_order = torch.randperm(self.num_batches) if self.shuffle else torch.arange(self.num_batches)
        
        for i in batch_order:
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.num_nodes)
            current_node_indices = self.sorted_indices[start_idx:end_idx]
            
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


def train_safe_mode():
    # 1. 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"   显卡型号: {torch.cuda.get_device_name(0)}")
    
    # 2. 加载数据
    data_path = "vessel_graph_training.pt"
    print(f"📂 读取数据: {data_path}")
    
    try:
        full_data = torch.load(data_path, weights_only=False)
    except TypeError:
        full_data = torch.load(data_path)

    print(f"   - 总节点数: {full_data.num_nodes}")
    
    # 获取元数据
    global_centroid = full_data.centroid.to(device)
    global_scale = full_data.scale

    # 3. 初始化加载器 (关键修改：batch_size=2048)
    print("\n🧩 初始化加载器 (安全模式)...")
    # 这里设为 2048，3090 绝对能跑起来
    BATCH_SIZE = 2048 
    loader = GeometricBatchLoader(full_data, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✅ 准备就绪! Batch Size={BATCH_SIZE}, 共 {len(loader)} 个 Batch。")

    # 4. 初始化模型
    model = PI_GNN(in_channels=9, hidden_channels=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. 训练循环 (只跑 2 个 Epoch 用于测试流程)
    print("\n🏃 开始测试训练 (只跑 2 个 Epoch)...")
    model.train()
    
    for epoch in range(1, 3): # 只跑2轮
        epoch_loss = 0.0
        start_epoch = time.time()
        
        for i, batch in enumerate(loader):
            # 清理上一轮的显存
            # torch.cuda.empty_cache() 
            
            # A. 数据移到 GPU
            x_static = batch.x[:, :6].to(device) 
            edge_index = batch.edge_index.to(device)
            pos = batch.pos.to(device)
            y = batch.y.to(device)
            mask_wall = batch.mask_wall.to(device)
            mask_inner = batch.mask_inner.to(device)
            
            # B. 激活梯度
            pos.requires_grad_(True)
            
            # C. 动态重计算归一化坐标
            pos_normalized = (pos - global_centroid) / (global_scale + 1e-8)
            x_new = torch.cat([x_static, pos_normalized], dim=1)
            
            optimizer.zero_grad()
            
            # Forward
            u_pred, p_pred = model(x_new, edge_index)
            
            # Loss 计算
            # Data Loss
            loss_u = torch.nn.functional.mse_loss(u_pred, y[:, 0:3])
            loss_p = torch.nn.functional.mse_loss(p_pred, y[:, 3:4])
            loss_data = loss_u + loss_p
            
            # Physics Loss
            if mask_inner.sum() > 10:
                loss_phy = compute_pinn_loss(u_pred, p_pred, pos, mask_inner)
            else:
                loss_phy = torch.tensor(0.0, device=device)
            
            # Boundary Loss
            if mask_wall.sum() > 0:
                loss_bc = torch.mean(u_pred[mask_wall]**2)
            else:
                loss_bc = torch.tensor(0.0, device=device)
            
            # Total Loss
            loss = loss_data + 0.1 * loss_phy + 1.0 * loss_bc
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 打印第一个 batch，然后每 50 个打印一次
            if i == 0 or (i+1) % 50 == 0:
                 current_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                 print(f"   [Epoch {epoch} Batch {i+1}/{len(loader)}] Loss: {loss.item():.4f} | VRAM: {current_mem:.1f}GB")
                 
            # 手动删除变量以释放显存图
            del u_pred, p_pred, loss, x_new, pos_normalized
            
        # Epoch 总结
        print(f"Epoch {epoch:03d} | Avg Loss: {epoch_loss/len(loader):.6f} | Time: {time.time()-start_epoch:.1f}s")

    print("\n✅ 流程测试成功！未发生死机。")
    torch.save(model.state_dict(), "model_safe_test.pth")
    print("💾 模型已保存: model_safe_test.pth")

if __name__ == "__main__":
    # 强制垃圾回收和显存清理
    gc.collect()
    torch.cuda.empty_cache()
    train_safe_mode()