"""
GNN + PINN 混合架构训练脚本
===========================
采用方案A: GNN编码器 + MLP解码器（坐标条件化）

物理约束 (基于真实物理量纲):
- 连续性方程 (质量守恒): ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
- Navier-Stokes 方程 (动量守恒): 
    ρ(∂u/∂t + (u·∇)u) = -∇p + μ∇²u
- 边界条件: 壁面无滑移 (u=v=w=0 when is_wall=1)

注意：
- 神经网络输入/输出: 归一化数据 (Z-score / Center-Scale)
- 物理方程计算: 还原回真实物理量纲 (SI单位: m, s, m/s, Pa)

作者: Auto-generated
日期: 2024
"""

import os
import json
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

# ==================== 物理常数 (血液动力学 SI单位) ====================
RHO = 1060.0      # 密度 kg/m^3
MU = 0.0035       # 动力粘度 Pa·s
# 注意:不再使用无量纲 Re，直接使用物理参数

# ==================== 损失权重配置 ====================
LOSS_WEIGHTS = {
    'data': 1.0,           # 数据监督损失权重
    'continuity': 0.1,     # 连续性方程损失权重
    'momentum': 0.01,      # NS动量方程损失权重
    'boundary': 1.0,       # 边界条件损失权重
}

# ==================== 归一化参数加载 ====================
def load_normalization_stats(json_path):
    """加载全局归一化参数"""
    with open(json_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    stats = params['statistics']
    
    # 构建恢复参数字典
    scaling_params = {
        'u': {'mean': stats['u']['mean'], 'std': stats['u']['std']},
        'v': {'mean': stats['v']['mean'], 'std': stats['v']['std']},
        'w': {'mean': stats['w']['mean'], 'std': stats['w']['std']},
        'p': {'mean': stats['p']['mean'], 'std': stats['p']['std']},
        # 坐标缩放 (Center-Scale)
        # 注意: interpolate_to_128frames.py 中使用的是 center_scale
        # x_norm = (x - mean) / scale_factor
        # 这里我们需要 scale_factor 来恢复真实的导数尺度
        # 由于我们只关心导数，平移量 centroid 不影响导数，只有缩放因子 scale_factor 影响
        # 但是 normalization_params_global.json 里存的是 x,y,z 的 mean/std (Z-score统计)，
        # 而不是 interpolate_to_128frames.py 生成的 center_scale 参数。
        # 这是一个潜在的不一致。
        # 
        # 修正策略:
        # 如果训练数据是来自 `interpolate_to_128frames.py` 处理后的 `ascii_merged_128` -> `data_converter` -> `.pt`
        # 那么坐标其实是经过了 center_scale 的。
        # 我们假设用户使用的是 `interpolate_to_128frames.py` 里的逻辑。
        # 因为无法直接从 global json 获取每个病例特定的 scale_factor，
        # 我们这里暂时使用统计的 std 作为近似的尺度参考，或者需要用户提供真实的特征长度 L。
        # 
        # 更严谨的做法: 在 data_converter.py 里把 scale_factor 存进 .pt 文件。
        # 
        # 妥协方案 (根据现有信息): 
        # 使用 global json 里的 x_std, y_std, z_std 近似代表特征尺度，或者设为 1.0 
        # 如果数据已经被缩放到 [-1, 1]，那么特征长度大约是 (Max - Min)/2
        'x_scale': (stats['x']['max'] - stats['x']['min']) / 2.0,
        'y_scale': (stats['y']['max'] - stats['y']['min']) / 2.0,
        'z_scale': (stats['z']['max'] - stats['z']['min']) / 2.0,
        
        # 时间 t 也是归一化到 [0, 1] 的
        # 我们假设一个心动周期 T ≈ 1.0s (或者根据数据实际情况)
        't_scale': 1.0 
    }
    return scaling_params

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
    1. 全局条件广播: 将 BC 从 data.global_cond 广播到每个节点
    2. GNN编码器: 使用 TransformerConv 学习几何/BC/邻域特征
    3. MLP解码器: 显式接收坐标(x,y,z,t)，支持自动微分计算物理损失
    
    输入格式:
    - data.x: [N, 10] 节点特征 (坐标3 + 几何6 + is_wall1)
    - data.global_cond: [B, 6] 全局条件 (t_norm1 + BC5)
    - data.edge_index: [2, E] 边索引
    - data.batch: [N] 批次索引（DataLoader 自动生成）
    """
    def __init__(self, node_dim=10, global_dim=6, hidden_channels=64, out_channels=4, heads=4):
        super(GNN_PINN, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.node_dim = node_dim
        self.global_dim = global_dim
        
        # ========== GNN 编码器 ==========
        # 非坐标节点特征(7: 几何6 + is_wall1) + BC(5, 从 global_cond 广播) = 12维
        gnn_input_dim = (node_dim - 3) + (global_dim - 1)  # 去掉坐标3维和t_norm1维
        
        self.lin_in = nn.Linear(gnn_input_dim, hidden_channels)
        
        # Transformer 卷积层
        self.conv1 = TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=0.1)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        
        self.conv2 = TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=0.1)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        
        self.conv3 = TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=0.1)
        self.lin3 = nn.Linear(hidden_channels, hidden_channels)
        
        # ========== MLP 解码器 (坐标条件化) ==========
        # 输入: GNN嵌入(hidden_channels) + 坐标和时间(4: x,y,z,t)
        decoder_input_dim = hidden_channels + 4
        
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_channels),
            nn.Tanh(),  # 使用Tanh激活，更适合PINN的高阶导数
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.Tanh(),
            nn.Linear(hidden_channels // 2, out_channels)
        )
    
    def forward(self, data, compute_physics=False, x_grad=None, y_grad=None, z_grad=None, t_grad=None):
        """
        前向传播
        
        参数:
            data: PyG Data 对象
            compute_physics: 是否使用外部带梯度坐标（用于物理损失计算）
            x_grad, y_grad, z_grad: 带梯度的坐标 [N, 1]
            t_grad: 带梯度的时间 [N, 1]（从 global_cond 广播而来）
        """
        node_feat, edge_index = data.x, data.edge_index
        
        # 1. 提取坐标和时间
        if compute_physics and x_grad is not None:
            # 使用外部传入的带梯度坐标（用于自动微分计算物理损失）
            coords_t = torch.cat([x_grad, y_grad, z_grad, t_grad], dim=-1)  # [N, 4]
        else:
            # 坐标来自节点特征，时间从 global_cond 广播
            coords = node_feat[:, :3]  # [N, 3]: x, y, z
            
            # 广播 t_norm 到每个节点
            if hasattr(data, 'global_cond') and data.global_cond is not None:
                if hasattr(data, 'batch') and data.batch is not None:
                    t_val = data.global_cond[data.batch, 0:1]  # [N, 1]
                else:
                    t_val = data.global_cond[:, 0:1].expand(coords.size(0), -1)  # [N, 1]
            else:
                t_val = torch.zeros(coords.size(0), 1, device=coords.device)
            
            coords_t = torch.cat([coords, t_val], dim=-1)  # [N, 4]
        
        # 2. 提取 GNN 编码器的输入特征
        # 非坐标节点特征: geometry(6) + is_wall(1) = 7维
        non_coord_feats = node_feat[:, 3:]  # [N, 7]
        
        # 广播 BC 到每个节点: BC_Inlet + BC_O1~O4 = 5维
        if hasattr(data, 'global_cond') and data.global_cond is not None:
            if hasattr(data, 'batch') and data.batch is not None:
                bc_expanded = data.global_cond[data.batch, 1:]  # [N, 5]
            else:
                bc_expanded = data.global_cond[:, 1:].expand(node_feat.size(0), -1)  # [N, 5]
        else:
            bc_expanded = torch.zeros(node_feat.size(0), 5, device=node_feat.device)
        
        # 拼接 GNN 输入: [N, 7+5=12]
        gnn_input = torch.cat([non_coord_feats, bc_expanded], dim=-1)
        
        # ========== GNN 编码 ==========
        h = self.lin_in(gnn_input)
        
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


# ==================== 物理损失函数 (带单位还原) ====================
def compute_physics_loss(model, data, output, x_norm, y_norm, z_norm, t_norm, stats, device):
    """
    计算物理约束损失 (还原物理量纲后计算)
    """
    # 1. 提取归一化输出
    u_norm = output[:, 0:1]
    v_norm = output[:, 1:2]
    w_norm = output[:, 2:3]
    p_norm = output[:, 3:4]
    
    # 2. 还原回物理量 (Physical Quantities)
    # Variable = Norm * Std + Mean
    # 注意: 这里仍然保留导数图 (Keep Computational Graph)
    u_phys = u_norm * stats['u']['std'] + stats['u']['mean']
    v_phys = v_norm * stats['v']['std'] + stats['v']['mean']
    w_phys = w_norm * stats['w']['std'] + stats['w']['mean']
    p_phys = p_norm * stats['p']['std'] + stats['p']['mean']
    
    # 3. 计算对 归一化坐标 的导数 (Automatic Differentiation)
    # ∂u_phys / ∂x_norm
    # 注意: create_graph=True 对高阶导数(粘性项)是必须的
    
    # 辅助函数: 计算一阶导数
    def gradient(y, x):
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]

    du_dxn = gradient(u_phys, x_norm)
    du_dyn = gradient(u_phys, y_norm)
    du_dzn = gradient(u_phys, z_norm)
    du_dtn = gradient(u_phys, t_norm)
    
    dv_dxn = gradient(v_phys, x_norm)
    dv_dyn = gradient(v_phys, y_norm)
    dv_dzn = gradient(v_phys, z_norm)
    dv_dtn = gradient(v_phys, t_norm)
    
    dw_dxn = gradient(w_phys, x_norm)
    dw_dyn = gradient(w_phys, y_norm)
    dw_dzn = gradient(w_phys, z_norm)
    dw_dtn = gradient(w_phys, t_norm)
    
    dp_dxn = gradient(p_phys, x_norm)
    dp_dyn = gradient(p_phys, y_norm)
    dp_dzn = gradient(p_phys, z_norm)
    
    # 4. 转换导数到 物理坐标系
    # Chain Rule: ∂u/∂x_phys = (∂u/∂x_norm) * (∂x_norm/∂x_phys)
    # x_norm = (x_phys - c) / scale  =>  ∂x_norm/∂x_phys = 1 / scale
    # 所以: ∂u/∂x_phys = ∂u/∂x_norm / scale
    
    inv_sx = 1.0 / stats['x_scale']
    inv_sy = 1.0 / stats['y_scale']
    inv_sz = 1.0 / stats['z_scale']
    inv_st = 1.0 / stats['t_scale']
    
    u_x = du_dxn * inv_sx
    u_y = du_dyn * inv_sy
    u_z = du_dzn * inv_sz
    u_t = du_dtn * inv_st
    
    v_x = dv_dxn * inv_sx
    v_y = dv_dyn * inv_sy
    v_z = dv_dzn * inv_sz
    v_t = dv_dtn * inv_st
    
    w_x = dw_dxn * inv_sx
    w_y = dw_dyn * inv_sy
    w_z = dw_dzn * inv_sz
    w_t = dw_dtn * inv_st
    
    p_x = dp_dxn * inv_sx
    p_y = dp_dyn * inv_sy
    p_z = dp_dzn * inv_sz
    
    # 二阶导数 (粘性项)
    # ∂²u/∂x²_phys = ∂(u_x)/∂x_phys = ∂(u_x)/∂x_norm * (1/scale)
    #              = ∂(du_dxn * inv_sx)/∂x_norm * inv_sx
    #              = (∂²u/∂x²_norm) * (1/scale^2)
    
    u_xx = gradient(du_dxn, x_norm) * (inv_sx**2)
    u_yy = gradient(du_dyn, y_norm) * (inv_sy**2)
    u_zz = gradient(du_dzn, z_norm) * (inv_sz**2)
    
    v_xx = gradient(dv_dxn, x_norm) * (inv_sx**2)
    v_yy = gradient(dv_dyn, y_norm) * (inv_sy**2)
    v_zz = gradient(dv_dzn, z_norm) * (inv_sz**2)
    
    w_xx = gradient(dw_dxn, x_norm) * (inv_sx**2)
    w_yy = gradient(dw_dyn, y_norm) * (inv_sy**2)
    w_zz = gradient(dw_dzn, z_norm) * (inv_sz**2)
    
    # ========== 5. 物理方程残差 (SI单位) ==========
    
    # 5.1 连续性方程 (Mass): ∇·u = 0  (单位: 1/s)
    continuity = u_x + v_y + w_z
    loss_cont = torch.mean(continuity ** 2)
    
    # 5.2 动量方程 (Navier-Stokes): (单位: N/m^3 或 Pa/m)
    # ρ(∂u/∂t + u·∇u) + ∇p - μ∇²u = 0
    
    # 惯性项 (Convective + Unsteady)
    # u_phys, v_phys, w_phys 已经是物理值
    inertial_x = u_t + u_phys * u_x + v_phys * u_y + w_phys * u_z
    inertial_y = v_t + u_phys * v_x + v_phys * v_y + w_phys * v_z
    inertial_z = w_t + u_phys * w_x + v_phys * w_y + w_phys * w_z
    
    # 粘性扩散项 (Viscous)
    laplacian_u = u_xx + u_yy + u_zz
    laplacian_v = v_xx + v_yy + v_zz
    laplacian_w = w_xx + w_yy + w_zz
    
    # 完整方程残差
    ns_x = RHO * inertial_x + p_x - MU * laplacian_u
    ns_y = RHO * inertial_y + p_y - MU * laplacian_v
    ns_z = RHO * inertial_z + p_z - MU * laplacian_w
    
    loss_mom = torch.mean(ns_x ** 2) + torch.mean(ns_y ** 2) + torch.mean(ns_z ** 2)
    
    # ========== 6. 边界条件损失 (归一化/物理皆可) ==========
    # 壁面速度为 0
    is_wall = data.x[:, 9:10]  # is_wall flag (新格式: 索引9)
    # 壁面上物理速度应为0
    wall_loss = is_wall * (u_phys ** 2 + v_phys ** 2 + w_phys ** 2)
    loss_bc = torch.mean(wall_loss)
    
    return loss_cont, loss_mom, loss_bc


def compute_data_loss(pred, target, weights=None):
    """
    计算数据监督损失 (使用归一化数值计算MSE，保持梯度平稳)
    """
    if weights is None:
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=pred.device)
    
    mse_per_dim = F.mse_loss(pred, target, reduction='none').mean(dim=0)
    weighted_loss = (mse_per_dim * weights).sum()
    return weighted_loss


# ==================== 训练函数 ====================
def train_epoch(model, loader, optimizer, device, data_weights, loss_weights, stats, use_physics=True):
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
            # 开启梯度追踪
            # 坐标来自节点特征 data.x[:, 0:3]
            x_norm = data.x[:, 0:1].detach().clone().requires_grad_(True)
            y_norm = data.x[:, 1:2].detach().clone().requires_grad_(True)
            z_norm = data.x[:, 2:3].detach().clone().requires_grad_(True)
            # 时间来自全局条件 data.global_cond[:, 0]，广播到每个节点
            if hasattr(data, 'batch') and data.batch is not None:
                t_broadcast = data.global_cond[data.batch, 0:1]  # [N, 1]
            else:
                t_broadcast = data.global_cond[:, 0:1].expand(data.x.size(0), -1)
            t_norm = t_broadcast.detach().clone().requires_grad_(True)
            
            # 前向传播 (带梯度)
            output = model(data, compute_physics=True, 
                           x_grad=x_norm, y_grad=y_norm, z_grad=z_norm, t_grad=t_norm)
            
            # 1. 数据损失 (Normalized space)
            loss_data = compute_data_loss(output, data.y, data_weights)
            
            # 2. 物理损失 (Physical space, restored inside function)
            loss_cont, loss_mom, loss_bc = compute_physics_loss(
                model, data, output, x_norm, y_norm, z_norm, t_norm, stats, device
            )
            
            # 总损失
            # 注意: 物理损失是在SI单位下计算的，数值可能很大 (比如 Pressure ~ 10^4 Pa)，
            # 需要调整权重使其与 数据损失 (MSE ~ 0.1-1.0) 平衡。
            # 建议: Momentum损失通常除以 (rho * U^2 / L)^2 或者给一个很小的权重如 1e-4 - 1e-6
            loss = (loss_weights['data'] * loss_data + 
                    loss_weights['continuity'] * loss_cont +
                    loss_weights['momentum'] * loss_mom +
                    loss_weights['boundary'] * loss_bc)
            
            total_cont += loss_cont.item() * data.num_graphs
            total_mom += loss_mom.item() * data.num_graphs
            total_bc += loss_bc.item() * data.num_graphs
        else:
            output = model(data, compute_physics=False)
            loss_data = compute_data_loss(output, data.y, data_weights)
            loss = loss_data
        
        loss.backward()
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
    parser = argparse.ArgumentParser(description="GNN + PINN 混合架构训练脚本 (物理还原版)")
    parser.add_argument("--data-dir", type=str, default="../GNN_train/processed_data", help="处理后的 .pt 数据目录")
    parser.add_argument("--norm-json", type=str, default="./normalization_params_global.json", help="归一化参数JSON")
    parser.add_argument("--epochs", type=int, default=500, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=2, help="每批病例数")
    parser.add_argument("--lr", type=float, default=0.001, help="初始学习率")
    parser.add_argument("--save-dir", type=str, default="./checkpoints_pinn_phys", help="模型保存目录")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    parser.add_argument("--warmup-epochs", type=int, default=50, help="预热轮数")
    parser.add_argument("--hidden-dim", type=int, default=64, help="隐藏层维度")
    
    # 损失权重参数 (需要根据SI单位量级仔细调整)
    # 数据损失 MSE ~ 0.5
    # 连续性损失 (du/dx ~ 10^-1), Residuals^2 ~ 10^-2 => 权重 ~ 0.1-1.0
    # 动量损失 (Pressure grad ~ 10^3, Inertial ~ 10^3), Residuals^2 ~ 10^6 - 10^7
    # 因此动量权重应该非常小, 例如 1e-6 到 1e-7
    parser.add_argument("--w-data", type=float, default=1.0, help="数据损失权重")
    parser.add_argument("--w-cont", type=float, default=0.1, help="连续性损失权重")
    parser.add_argument("--w-mom", type=float, default=1e-6, help="动量损失权重 (因物理单位数值大，需调小)")
    parser.add_argument("--w-bc", type=float, default=1.0, help="边界条件损失权重")
    
    args = parser.parse_args()
    
    # 更新配置
    LOSS_WEIGHTS['data'] = args.w_data
    LOSS_WEIGHTS['continuity'] = args.w_cont
    LOSS_WEIGHTS['momentum'] = args.w_mom
    LOSS_WEIGHTS['boundary'] = args.w_bc
    
    # 加载归一化参数
    print(f"正在加载归一化参数: {args.norm_json}")
    if not os.path.exists(args.norm_json):
        print("错误: 找不到归一化参数文件!")
        return
    normalization_stats = load_normalization_stats(args.norm_json)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"损失权重: {LOSS_WEIGHTS}")
    
    # 1. 加载数据
    data_dir = Path(args.data_dir)
    all_pt_files = sorted(list(data_dir.rglob("*.pt")))
    if not all_pt_files:
        print(f"错误: 找不到 .pt 文件")
        return
    
    # 对病例划分 
    random.shuffle(all_pt_files)
    train_idx = int(len(all_pt_files) * 0.8)
    val_idx = int(len(all_pt_files) * 0.9)
    train_files = all_pt_files[:train_idx]
    val_files = all_pt_files[train_idx:val_idx]
    test_files = all_pt_files[val_idx:]
    
    print(f"样本分配: 训练={len(train_files)}, 验证={len(val_files)}, 测试={len(test_files)}")
    
    train_loader = DataLoader(VascularDataset(train_files), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(VascularDataset(val_files), batch_size=args.batch_size)
    
    # 3. 初始化
    model = GNN_PINN(node_dim=10, global_dim=6, hidden_channels=args.hidden_dim, out_channels=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    data_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. 训练
    best_val_loss = float('inf')
    history = []
    
    print(f"\n开始训练...")
    for epoch in range(1, args.epochs + 1):
        use_physics = epoch > args.warmup_epochs
        
        train_res = train_epoch(
            model, train_loader, optimizer, device, data_weights, LOSS_WEIGHTS, normalization_stats, use_physics
        )
        train_loss, data_loss, cont_loss, mom_loss, bc_loss = train_res
        
        val_loss = validate_epoch(model, val_loader, device, data_weights)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        
        phase = "PINN" if use_physics else "Warmup"
        print(f"[{phase}] Epoch {epoch}: Train={train_loss:.6f} (Dat={data_loss:.4f}, Mom={mom_loss:.4f}), Val={val_loss:.6f}, LR={lr:.6f}")
        
        history.append(f"{epoch},{train_loss},{val_loss},{data_loss},{cont_loss},{mom_loss},{bc_loss}\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "best_model_phys.pt")
            print(f" ---> 已保存最佳模型 (Val Loss: {val_loss:.6f})")
    
    # 保存日志
    with open("train_log_phys.txt", "w") as f:
        f.write("epoch,train_loss,val_loss,data_loss,cont_loss,mom_loss,bc_loss\n")
        f.writelines(history)
    print("完成.")

if __name__ == "__main__":
    main()
