import torch
import torch.optim as optim
from model import PI_GNN
from physics import compute_pinn_loss
import time

def train():
    # 1. 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")
    
    # 2. 加载数据
    # 注意：这里直接加载你刚才生成的单个 .pt 文件
    # 在实际项目中，这里会用 DataLoader 加载 Dataset
    data_path = "vessel_graph_training.pt"
    print(f"📂 加载数据: {data_path}")
    data = torch.load(data_path)
    data = data.to(device)
    
    # ⚠️ 关键：启用 pos 的梯度记录
    # 因为 PINN 需要对 pos 求导，所以必须设为 True
    data.pos.requires_grad_(True)
    
    # 3. 初始化模型
    model = PI_GNN(in_channels=9, hidden_channels=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 训练循环
    print("\n🏃 开始训练测试 (Overfitting single batch)...")
    model.train()
    
    # 跑 50 个 Epoch 试试看
    for epoch in range(1, 51):
        start_time = time.time()
        optimizer.zero_grad()
        
        # --- Forward ---
        # 预测速度和压力
        u_pred, p_pred = model(data.x, data.edge_index)
        
        # --- Loss 1: Data Loss (数据驱动) ---
        # 将预测值与 CFD 真实标签 (data.y) 对比
        # data.y 的前3列是 u,v,w，第4列是 p
        u_true = data.y[:, 0:3]
        p_true = data.y[:, 3:4]
        
        loss_u = torch.nn.functional.mse_loss(u_pred, u_true)
        loss_p = torch.nn.functional.mse_loss(p_pred, p_true)
        loss_data = loss_u + loss_p
        
        # --- Loss 2: Physics Loss (物理约束) ---
        # 只有在内部点，且为了显存考虑，进行采样计算
        # 权重 0.1 是随便设的，以后要调
        loss_phy = compute_pinn_loss(u_pred, p_pred, data.pos, data.mask_inner, sample_size=1000)
        
        # --- Loss 3: Boundary Loss (可选) ---
        # 壁面速度应该是 0 (No-slip)
        # 虽然 Data Loss 里包含了壁面点的真实值(0)，但单独强调一下也好
        u_wall = u_pred[data.mask_wall]
        loss_bc = torch.mean(u_wall**2) 
        
        # --- 总 Loss ---
        # 这里的权重系数需要后面精细调整
        total_loss = loss_data + 0.1 * loss_phy + 1.0 * loss_bc
        
        # --- Backward ---
        total_loss.backward()
        optimizer.step()
        
        # 打印日志
        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | "
                  f"Total: {total_loss.item():.6f} | "
                  f"Data: {loss_data.item():.6f} | "
                  f"Phy: {loss_phy.item():.6f} | "
                  f"BC: {loss_bc.item():.6f} | "
                  f"Time: {time.time()-start_time:.2f}s")

    print("\n✅ 流程跑通！模型已成功在单个样本上进行了反向传播。")

if __name__ == "__main__":
    # 清理显存 (防止之前的残留)
    torch.cuda.empty_cache()
    train()