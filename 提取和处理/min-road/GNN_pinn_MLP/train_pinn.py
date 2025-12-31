"""
GNN + PINN 混合架构训练脚本
===========================
采用方案A: GNN编码器 + MLP解码器（坐标条件化）

物理约束:
- 连续性方程 (质量守恒): ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
- Navier-Stokes 方程 (动量守恒): 
    ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u
- 边界条件: 壁面无滑移 (u=v=w=0 when is_wall=1)

作者: Auto-generated
日期: 2024
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.nn import TransformerConv
from tqdm import tqdm
import argparse
from pathlib import Path
import random
import datetime

# ==================== 物理常数 ====================
RE = 300  # 雷诺数

# ==================== 损失权重配置 ====================
# 这些权重经过经验调整，可根据实验结果微调
LOSS_WEIGHTS = {
    'data': 1.0,           # 数据监督损失权重
    'continuity': 0.1,     # 连续性方程损失权重
    'momentum': 0.01,      # NS动量方程损失权重
    'boundary': 1.0,       # 边界条件损失权重
}


# ==================== 数据集类 ====================
class VascularDataset(Dataset):
    """加载由 data_converter.py 生成的 .pt 文件"""
    def __init__(self, pt_files):
        super(VascularDataset, self).__init__()
        self.pt_files = pt_files

    def len(self):
        return len(self.pt_files)

    def get(self, idx):
        data = torch.load(self.pt_files[idx], weights_only=False)
        return data


# ==================== 混合架构模型 ====================
class GNN_PINN(nn.Module):
    """
    GNN + PINN 混合架构模型
    
    架构设计:
    1. GNN编码器: 使用 TransformerConv 学习几何和邻域特征
    2. MLP解码器: 显式接收坐标(x,y,z,t)，支持自动微分计算物理损失
    
    输入特征 (17维):
    - coords: x, y, z (3维) - 空间坐标
    - t: (1维) - 归一化时间
    - geom: Abscissa, NormRadius, Curvature, Tangent_X/Y/Z (6维)
    - flags: is_wall, BC_Flag (2维)
    - bc: BC_Inlet, BC_O1~O4 (5维)
    
    输出 (4维): u, v, w, p
    """
    def __init__(self, in_channels=17, hidden_channels=64, out_channels=4, heads=4):
        super(GNN_PINN, self).__init__()
        
        self.hidden_channels = hidden_channels
        
        # ========== GNN 编码器 ==========
        # 输入特征映射 (不包含坐标和时间，因为它们会单独送入解码器)
        # 几何特征(6) + 边界标志(2) + 边界条件(5) = 13维
        gnn_input_dim = in_channels - 4  # 去掉 x, y, z, t
        
        self.lin_in = nn.Linear(gnn_input_dim, hidden_channels)
        
        # Transformer 卷积层
        self.conv1 = TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=0.1)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        
        self.conv2 = TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=0.1)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        
        self.conv3 = TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=0.1)
        self.lin3 = nn.Linear(hidden_channels, hidden_channels)
        
        # ========== MLP 解码器 (坐标条件化) ==========
        # 输入: GNN嵌入(hidden_channels) + 坐标和时间(4)
        decoder_input_dim = hidden_channels + 4
        
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_channels),
            nn.Tanh(),  # 使用Tanh激活，更适合PINN
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.Tanh(),
            nn.Linear(hidden_channels // 2, out_channels)
        )
    
    def forward(self, data, compute_physics=False, x_grad=None, y_grad=None, z_grad=None, t_grad=None):
        """
        前向传播
        
        Args:
            data: PyG Data 对象
            compute_physics: 是否计算物理损失所需的梯度
            x_grad, y_grad, z_grad: 外部传入的带梯度的坐标分量 [N, 1] (可选)
            t_grad: 外部传入的带梯度的时间 [N, 1] (可选)
        
        Returns:
            output: 预测的 (u, v, w, p)
        """
        feat, edge_index = data.x, data.edge_index
        
        # 分离特征
        if compute_physics and x_grad is not None:
            # 使用外部传入的带梯度坐标
            coords_t = torch.cat([x_grad, y_grad, z_grad, t_grad], dim=-1)
        else:
            coords_t = feat[:, :4]  # x, y, z, t
        
        other_feats = feat[:, 4:]  # 其他特征 (13维)
        
        # ========== GNN 编码 ==========
        h = self.lin_in(other_feats)
        
        # Transformer Block 1
        h_conv = self.conv1(h, edge_index)
        h_conv = F.elu(h_conv)
        h_conv = self.lin1(h_conv)
        h = h + h_conv  # 残差连接
        h = F.dropout(h, p=0.1, training=self.training)
        
        # Transformer Block 2
        h_conv = self.conv2(h, edge_index)
        h_conv = F.elu(h_conv)
        h_conv = self.lin2(h_conv)
        h = h + h_conv
        h = F.dropout(h, p=0.1, training=self.training)
        
        # Transformer Block 3
        h_conv = self.conv3(h, edge_index)
        h_conv = F.elu(h_conv)
        h_conv = self.lin3(h_conv)
        h = h + h_conv
        h = F.dropout(h, p=0.1, training=self.training)
        
        # ========== MLP 解码 (坐标条件化) ==========
        decoder_input = torch.cat([h, coords_t], dim=-1)
        output = self.decoder(decoder_input)
        
        return output


# ==================== 物理损失函数 ====================
def compute_physics_loss(model, data, output, x, y, z, t):
    """
    计算物理约束损失
    
    Args:
        model: GNN_PINN 模型
        data: PyG Data 对象
        output: 模型预测输出 [N, 4] (u, v, w, p)
        x, y, z: 坐标张量 [N, 1] (requires_grad=True)
        t: 时间张量 [N, 1] (requires_grad=True)
    
    Returns:
        loss_cont: 连续性方程损失
        loss_mom: 动量方程损失
        loss_bc: 边界条件损失
    """
    # 分离输出变量
    u = output[:, 0:1]
    v = output[:, 1:2]
    w = output[:, 2:3]
    p = output[:, 3:4]
    
    # ========== 计算一阶空间导数 ==========
    # 对 u 的导数
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                              retain_graph=True, create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u),
                              retain_graph=True, create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              retain_graph=True, create_graph=True)[0]
    
    # 对 v 的导数
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                              retain_graph=True, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v),
                              retain_graph=True, create_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v),
                              retain_graph=True, create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v),
                              retain_graph=True, create_graph=True)[0]
    
    # 对 w 的导数
    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w),
                              retain_graph=True, create_graph=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w),
                              retain_graph=True, create_graph=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w),
                              retain_graph=True, create_graph=True)[0]
    w_t = torch.autograd.grad(w, t, grad_outputs=torch.ones_like(w),
                              retain_graph=True, create_graph=True)[0]
    
    # 对 p (压力) 的导数
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p),
                              retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p),
                              retain_graph=True, create_graph=True)[0]
    p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p),
                              retain_graph=True, create_graph=True)[0]
    
    # ========== 计算二阶空间导数 (粘性项) ==========
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                               retain_graph=True, create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z),
                               retain_graph=True, create_graph=True)[0]
    
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x),
                               retain_graph=True, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y),
                               retain_graph=True, create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z),
                               retain_graph=True, create_graph=True)[0]
    
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x),
                               retain_graph=True, create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y),
                               retain_graph=True, create_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(w_z),
                               retain_graph=True, create_graph=True)[0]
    
    # ========== 1. 连续性方程残差 ==========
    # ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
    continuity = u_x + v_y + w_z
    loss_cont = torch.mean(continuity ** 2)
    
    # ========== 2. Navier-Stokes 动量方程残差 ==========
    # x方向: ∂u/∂t + u·∂u/∂x + v·∂u/∂y + w·∂u/∂z = -∂p/∂x + (1/Re)·∇²u
    ns_x = u_t + u * u_x + v * u_y + w * u_z + p_x - (u_xx + u_yy + u_zz) / RE
    
    # y方向
    ns_y = v_t + u * v_x + v * v_y + w * v_z + p_y - (v_xx + v_yy + v_zz) / RE
    
    # z方向
    ns_z = w_t + u * w_x + v * w_y + w * w_z + p_z - (w_xx + w_yy + w_zz) / RE
    
    loss_mom = torch.mean(ns_x ** 2) + torch.mean(ns_y ** 2) + torch.mean(ns_z ** 2)
    
    # ========== 3. 边界条件损失 (壁面无滑移) ==========
    # is_wall 在 data.x 的第 10 列 (索引从0开始: x,y,z,t,6个几何特征后)
    is_wall = data.x[:, 10:11]  # [N, 1]
    
    # 壁面上速度应为零
    wall_loss = is_wall * (u ** 2 + v ** 2 + w ** 2)
    loss_bc = torch.mean(wall_loss)
    
    return loss_cont, loss_mom, loss_bc


def compute_data_loss(pred, target, weights=None):
    """
    计算数据监督损失 (加权MSE)
    
    Args:
        pred: 预测值 [N, 4]
        target: 真实值 [N, 4]
        weights: 各维度权重 [4]
    
    Returns:
        weighted_loss: 加权MSE损失
    """
    if weights is None:
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=pred.device)
    
    mse_per_dim = F.mse_loss(pred, target, reduction='none').mean(dim=0)
    weighted_loss = (mse_per_dim * weights).sum()
    return weighted_loss


# ==================== 训练函数 ====================
def train_epoch(model, loader, optimizer, device, data_weights, loss_weights, use_physics=True):
    """
    训练一个 epoch
    
    Args:
        model: GNN_PINN 模型
        loader: 数据加载器
        optimizer: 优化器
        device: 计算设备
        data_weights: 数据损失的各维度权重
        loss_weights: 各损失项的权重字典
        use_physics: 是否使用物理损失
    
    Returns:
        avg_loss: 平均总损失
        avg_data: 平均数据损失
        avg_cont: 平均连续性损失
        avg_mom: 平均动量损失
        avg_bc: 平均边界条件损失
    """
    model.train()
    total_loss = 0
    total_data = 0
    total_cont = 0
    total_mom = 0
    total_bc = 0
    num_samples = 0
    
    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        
        if use_physics:
            # 分别创建 x, y, z, t 的可求导张量，每个都是独立的叶节点
            x_grad = data.x[:, 0:1].detach().clone().requires_grad_(True)
            y_grad = data.x[:, 1:2].detach().clone().requires_grad_(True)
            z_grad = data.x[:, 2:3].detach().clone().requires_grad_(True)
            t_grad = data.x[:, 3:4].detach().clone().requires_grad_(True)
            
            # 传入分开的带梯度坐标
            output = model(data, compute_physics=True, x_grad=x_grad, y_grad=y_grad, z_grad=z_grad, t_grad=t_grad)
            
            # 数据损失
            loss_data = compute_data_loss(output, data.y, data_weights)
            
            # 物理损失
            loss_cont, loss_mom, loss_bc = compute_physics_loss(model, data, output, x_grad, y_grad, z_grad, t_grad)
            
            # 总损失
            loss = (loss_weights['data'] * loss_data + 
                    loss_weights['continuity'] * loss_cont +
                    loss_weights['momentum'] * loss_mom +
                    loss_weights['boundary'] * loss_bc)
            
            total_cont += loss_cont.item() * data.num_graphs
            total_mom += loss_mom.item() * data.num_graphs
            total_bc += loss_bc.item() * data.num_graphs
        else:
            # 仅数据损失
            output = model(data, compute_physics=False)
            loss_data = compute_data_loss(output, data.y, data_weights)
            loss = loss_data
        
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        total_data += loss_data.item() * data.num_graphs
        num_samples += data.num_graphs
    
    return (total_loss / num_samples, 
            total_data / num_samples,
            total_cont / num_samples if use_physics else 0,
            total_mom / num_samples if use_physics else 0,
            total_bc / num_samples if use_physics else 0)


@torch.no_grad()
def validate_epoch(model, loader, device, data_weights):
    """验证函数 (仅计算数据损失)"""
    model.eval()
    total_loss = 0
    num_samples = 0
    
    for data in loader:
        data = data.to(device)
        output = model(data, compute_physics=False)
        loss = compute_data_loss(output, data.y, data_weights)
        total_loss += loss.item() * data.num_graphs
        num_samples += data.num_graphs
    
    return total_loss / num_samples


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="GNN + PINN 混合架构训练脚本")
    parser.add_argument("--data-dir", type=str, default="../GNN_train/processed_data", help="处理后的 .pt 数据目录")
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=2, help="每批病例数 (PINN计算量大，建议小批量)")
    parser.add_argument("--lr", type=float, default=0.001, help="初始学习率")
    parser.add_argument("--save-dir", type=str, default="./checkpoints_pinn", help="模型保存目录")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    parser.add_argument("--warmup-epochs", type=int, default=50, help="预热轮数 (仅数据损失)")
    parser.add_argument("--hidden-dim", type=int, default=64, help="隐藏层维度")
    
    # 损失权重参数
    parser.add_argument("--w-data", type=float, default=1.0, help="数据损失权重")
    parser.add_argument("--w-cont", type=float, default=0.1, help="连续性方程损失权重")
    parser.add_argument("--w-mom", type=float, default=0.01, help="动量方程损失权重")
    parser.add_argument("--w-bc", type=float, default=1.0, help="边界条件损失权重")
    
    args = parser.parse_args()
    
    # 更新损失权重
    LOSS_WEIGHTS['data'] = args.w_data
    LOSS_WEIGHTS['continuity'] = args.w_cont
    LOSS_WEIGHTS['momentum'] = args.w_mom
    LOSS_WEIGHTS['boundary'] = args.w_bc
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=" * 60)
    print(f"GNN + PINN 混合架构训练")
    print(f"=" * 60)
    print(f"使用设备: {device}")
    print(f"雷诺数 Re: {RE}")
    print(f"损失权重: {LOSS_WEIGHTS}")
    print(f"预热轮数: {args.warmup_epochs}")
    
    # 1. 加载数据
    data_dir = Path(args.data_dir)
    all_pt_files = sorted(list(data_dir.rglob("*.pt")))
    if not all_pt_files:
        print(f"错误: 在 {args.data_dir} 中找不到 .pt 文件")
        return
    
    print(f"找到 {len(all_pt_files)} 个数据样本。")
    
    # 2. 按病例划分数据集
    case_to_files = {}
    for pt_file in all_pt_files:
        case_name = pt_file.parent.name
        if case_name not in case_to_files:
            case_to_files[case_name] = []
        case_to_files[case_name].append(pt_file)
    
    unique_cases = sorted(list(case_to_files.keys()))
    
    if len(unique_cases) < 3:
        random.shuffle(all_pt_files)
        num_samples = len(all_pt_files)
        train_idx = int(num_samples * 0.8)
        val_idx = int(num_samples * 0.9)
        train_files = all_pt_files[:train_idx]
        val_files = all_pt_files[train_idx:val_idx]
        test_files = all_pt_files[val_idx:]
    else:
        random.shuffle(unique_cases)
        test_case_name = unique_cases[-1]
        dev_case_names = unique_cases[:-1]
        
        test_files = case_to_files[test_case_name]
        
        dev_files = []
        for name in dev_case_names:
            dev_files.extend(case_to_files[name])
        
        random.shuffle(dev_files)
        train_idx = int(len(dev_files) * 0.8)
        train_files = dev_files[:train_idx]
        val_files = dev_files[train_idx:]
        
        print(f"病例分配: 开发病例={dev_case_names}, 测试病例=['{test_case_name}']")
    
    print(f"样本分配: 训练={len(train_files)}, 验证={len(val_files)}, 测试={len(test_files)}")
    
    # 保存测试集文件列表
    with open("test_files_pinn.txt", "w") as f:
        for item in test_files:
            f.write(f"{item}\n")
    
    train_loader = DataLoader(VascularDataset(train_files), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(VascularDataset(val_files), batch_size=args.batch_size)
    
    # 3. 初始化模型
    model = GNN_PINN(in_channels=17, hidden_channels=args.hidden_dim, out_channels=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)
    
    # 数据损失的各维度权重
    data_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
    
    # 4. 训练循环
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    history = []
    
    print(f"\n开始训练... {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 100)
    
    for epoch in range(1, args.epochs + 1):
        # 前 warmup_epochs 轮仅使用数据损失
        use_physics = epoch > args.warmup_epochs
        
        # 训练
        train_loss, data_loss, cont_loss, mom_loss, bc_loss = train_epoch(
            model, train_loader, optimizer, device, data_weights, LOSS_WEIGHTS, use_physics
        )
        
        # 验证
        val_loss = validate_epoch(model, val_loader, device, data_weights)
        
        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印日志
        phase = "PINN" if use_physics else "Warmup"
        print(f"[{phase}] Epoch {epoch:03d} | "
              f"Train: {train_loss:.6f} (data={data_loss:.4f}, cont={cont_loss:.4f}, mom={mom_loss:.4f}, bc={bc_loss:.4f}) | "
              f"Val: {val_loss:.6f} | LR: {current_lr:.6f}")
        
        # 记录历史
        history.append(f"{epoch},{train_loss:.8f},{val_loss:.8f},{data_loss:.8f},{cont_loss:.8f},{mom_loss:.8f},{bc_loss:.8f}\n")
        
        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "best_model_pinn.pt")
            print(f" ---> 已保存最优模型 (Val Loss: {val_loss:.6f})")
        
        # 定期保存检查点
        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_dir / f"checkpoint_pinn_epoch_{epoch}.pt")
    
    print("-" * 100)
    print(f"训练完成! 最优模型已保存至: {save_dir / 'best_model_pinn.pt'}")
    
    # 保存训练日志
    log_path = Path("train_log_pinn.txt")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,data_loss,continuity_loss,momentum_loss,boundary_loss\n")
        f.writelines(history)
    print(f"训练日志已保存至: {log_path}")


if __name__ == "__main__":
    main()

