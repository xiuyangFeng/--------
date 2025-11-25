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
    """
    几何分块加载器，按照几何位置对图数据进行分块处理
    用于处理大规模图数据，避免显存不足的问题
    """
    def __init__(self, data, batch_size=2048, shuffle=True):
        """
        初始化几何分块加载器
        参数:
            data: 图数据对象
            batch_size: 每个批次的节点数量
            shuffle: 是否打乱数据顺序
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_nodes = data.num_nodes  # 节点总数
        
        # 计算数据在各轴上的分布范围，选择分布最广的轴作为主轴
        pos = data.pos.numpy()
        spread = pos.max(0) - pos.min(0)
        self.main_axis = np.argmax(spread)  # 确定主轴（分布最广的轴）
        
        # 按主轴坐标排序节点索引
        self.sorted_indices = torch.argsort(data.pos[:, self.main_axis])
        self.num_batches = math.ceil(self.num_nodes / batch_size)  # 计算批次数

    def __iter__(self):
        """
        迭代器实现，逐批返回数据
        """
        # 确定批次处理顺序
        batch_order = torch.randperm(self.num_batches) if self.shuffle else torch.arange(self.num_batches)
        
        # 按顺序处理每个批次
        for i in batch_order:
            start_idx = i * self.batch_size  # 当前批次起始索引
            end_idx = min((i + 1) * self.batch_size, self.num_nodes)  # 当前批次结束索引
            current_node_indices = self.sorted_indices[start_idx:end_idx]  # 当前批次节点索引
            
            # 提取当前批次的子图
            edge_index, _ = subgraph(
                current_node_indices, 
                self.data.edge_index, 
                relabel_nodes=True,  # 重新标记节点
                num_nodes=self.num_nodes  # 总节点数
            )
            
            # 提取当前批次的数据特征
            x = self.data.x[current_node_indices]  # 节点特征
            pos = self.data.pos[current_node_indices]  # 节点位置
            y = self.data.y[current_node_indices] if hasattr(self.data, 'y') else None  # 标签数据
            mask_wall = self.data.mask_wall[current_node_indices]  # 壁面节点掩码
            mask_inner = self.data.mask_inner[current_node_indices]  # 内部节点掩码
            
            # 创建批次数据对象
            class BatchData: pass
            batch = BatchData()
            batch.x = x  # 节点特征
            batch.edge_index = edge_index  # 边索引
            batch.pos = pos  # 节点位置
            batch.y = y  # 标签
            batch.mask_wall = mask_wall  # 壁面掩码
            batch.mask_inner = mask_inner  # 内部节点掩码
            batch.num_nodes = len(current_node_indices)  # 当前批次节点数
            
            yield batch  # 返回当前批次数据

    def __len__(self):
        """
        返回总批次数
        """
        return self.num_batches


def train_safe_mode():
    """
    安全模式训练函数，使用较小的批次大小避免显存溢出
    """
    # 1. 配置训练设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"   显卡型号: {torch.cuda.get_device_name(0)}")
    
    # 2. 加载训练数据
    data_path = "vessel_graph_training.pt"
    print(f"📂 读取数据: {data_path}")
    
    try:
        full_data = torch.load(data_path, weights_only=False)
    except TypeError:
        full_data = torch.load(data_path)

    print(f"   - 总节点数: {full_data.num_nodes}")
    
    # 获取数据的全局归一化参数
    global_centroid = full_data.centroid.to(device)  # 全局中心点
    global_scale = full_data.scale  # 全局缩放因子

    # 3. 初始化数据加载器 (关键修改：batch_size=2048)
    print("\n🧩 初始化加载器 (安全模式)...")
    # 这里设为 2048，3090 绝对能跑起来
    BATCH_SIZE = 2048 
    loader = GeometricBatchLoader(full_data, batch_size=BATCH_SIZE, shuffle=True)
    print(f"✅ 准备就绪! Batch Size={BATCH_SIZE}, 共 {len(loader)} 个 Batch。")

    # 4. 初始化模型和优化器
    model = PI_GNN(in_channels=9, hidden_channels=64).to(device)  # 创建模型并移至设备
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 创建Adam优化器
    
    # 5. 训练循环 (只跑 2 个 Epoch 用于测试流程)
    print("\n🏃 开始测试训练 (只跑 2 个 Epoch)...")
    model.train()  # 设置模型为训练模式
    
    # 执行两个训练轮次
    for epoch in range(1, 3): # 只跑2轮
        epoch_loss = 0.0  # 累计轮次损失
        start_epoch = time.time()  # 记录轮次开始时间
        
        # 遍历所有批次
        for i, batch in enumerate(loader):
            # 清理上一轮的显存（注释掉以提高效率）
            # torch.cuda.empty_cache() 
            
            # A. 将数据移到 GPU
            x_static = batch.x[:, :6].to(device)  # 静态特征
            edge_index = batch.edge_index.to(device)  # 边索引
            pos = batch.pos.to(device)  # 节点位置
            y = batch.y.to(device)  # 标签数据
            mask_wall = batch.mask_wall.to(device)  # 壁面掩码
            mask_inner = batch.mask_inner.to(device)  # 内部节点掩码
            
            # B. 激活梯度计算（用于物理损失计算）
            pos.requires_grad_(True)
            
            # C. 动态重计算归一化坐标
            pos_normalized = (pos - global_centroid) / (global_scale + 1e-8)  # 归一化位置
            x_new = torch.cat([x_static, pos_normalized], dim=1)  # 合并特征
            
            optimizer.zero_grad()  # 清零梯度
            
            # Forward 前向传播
            u_pred, p_pred = model(x_new, edge_index)  # 模型预测速度和压力
            
            # Loss 计算
            # Data Loss 数据损失（MSE）
            loss_u = torch.nn.functional.mse_loss(u_pred, y[:, 0:3])  # 速度损失
            loss_p = torch.nn.functional.mse_loss(p_pred, y[:, 3:4])  # 压力损失
            loss_data = loss_u + loss_p  # 总数据损失
            
            # Physics Loss 物理损失（PINN损失）
            if mask_inner.sum() > 10:  # 确保有足够的内部节点
                loss_phy = compute_pinn_loss(u_pred, p_pred, pos, mask_inner)  # 计算物理损失
            else:
                loss_phy = torch.tensor(0.0, device=device)  # 物理损失为0
            
            # Boundary Loss 边界条件损失
            if mask_wall.sum() > 0:  # 确保有壁面节点
                loss_bc = torch.mean(u_pred[mask_wall]**2)  # 壁面速度应接近0
            else:
                loss_bc = torch.tensor(0.0, device=device)  # 边界损失为0
            
            # Total Loss 总损失
            loss = loss_data + 0.1 * loss_phy + 1.0 * loss_bc  # 加权总损失
            
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器更新参数
            
            epoch_loss += loss.item()  # 累计损失
            
            # 打印第一个 batch，然后每 50 个打印一次训练状态
            if i == 0 or (i+1) % 50 == 0:
                 current_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                 print(f"   [Epoch {epoch} Batch {i+1}/{len(loader)}] Loss: {loss.item():.4f} | VRAM: {current_mem:.1f}GB")
                 
            # 手动删除变量以释放显存图
            del u_pred, p_pred, loss, x_new, pos_normalized
            
        # Epoch 总结
        print(f"Epoch {epoch:03d} | Avg Loss: {epoch_loss/len(loader):.6f} | Time: {time.time()-start_epoch:.1f}s")

    print("\n✅ 流程测试成功！未发生死机。")
    torch.save(model.state_dict(), "model_safe_test.pth")  # 保存模型
    print("💾 模型已保存: model_safe_test.pth")

if __name__ == "__main__":
    # 强制垃圾回收和显存清理
    gc.collect()  # 清理Python垃圾
    torch.cuda.empty_cache()  # 清理CUDA显存缓存
    train_safe_mode()  # 执行安全模式训练