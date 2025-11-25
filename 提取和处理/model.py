import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GATv2Conv

class QuadraticResidualBlock(nn.Module):
    """
    二次残差模块 (Quadratic Residual Block)
    y = x + MLP(x) + Quadratic(x)
    增强模型对非线性物理场（如流体动力学）的拟合能力。
    """
    def __init__(self, in_channels, out_channels):
        super(QuadraticResidualBlock, self).__init__()
        self.linear_1 = nn.Linear(in_channels, out_channels)
        self.linear_2 = nn.Linear(in_channels, out_channels)
        self.activation = nn.GELU()
        
        # 如果输入输出维度不同，需要投影 x 以便相加
        self.project = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # 线性路径
        h1 = self.linear_1(x)
        
        # 二次路径 (Quadratic term): x * W * x 
        # 这里使用简化版: (W2 * x) ^ 2
        h2 = self.linear_2(x)
        h2 = h2 * h2 # Element-wise square
        
        # 残差连接
        return self.activation(self.project(x) + h1 + h2)

class PI_GNN(nn.Module):
    def __init__(self, in_channels=9, hidden_channels=64, out_channels_u=3, out_channels_p=1):
        super(PI_GNN, self).__init__()
        
        # 1. 编码器 (Encoder): 将输入特征映射到高维空间
        # 引入 QN 模块
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            QuadraticResidualBlock(hidden_channels, hidden_channels)
        )
        
        # 2. 主干 (Backbone): 图神经网络层
        # 使用 GATv2Conv 提取图结构特征
        self.conv1 = GATv2Conv(hidden_channels, hidden_channels // 4, heads=4, concat=True)
        self.qn1 = QuadraticResidualBlock(hidden_channels, hidden_channels) # 层间增强
        
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels // 4, heads=4, concat=True)
        self.qn2 = QuadraticResidualBlock(hidden_channels, hidden_channels)
        
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels // 4, heads=4, concat=True)
        self.qn3 = QuadraticResidualBlock(hidden_channels, hidden_channels)
        
        # 3. 解码器头 (Decoders)
        # 头 A: 预测速度 (u, v, w)
        self.head_u = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels_u)
        )
        
        # 头 B: 预测压力 (p)
        self.head_p = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels_p)
        )

    def forward(self, x, edge_index):
        # x: [N, 9]
        # edge_index: [2, E]
        
        # --- 编码 ---
        h = self.encoder(x)
        
        # --- 消息传递 (Message Passing) ---
        # Layer 1
        h_graph = self.conv1(h, edge_index)
        h = h + h_graph # 残差连接
        h = self.qn1(h) # 二次残差增强
        
        # Layer 2
        h_graph = self.conv2(h, edge_index)
        h = h + h_graph
        h = self.qn2(h)
        
        # Layer 3
        h_graph = self.conv3(h, edge_index)
        h = h + h_graph
        h = self.qn3(h)
        
        # --- 输出预测 ---
        u_pred = self.head_u(h) # [N, 3]
        p_pred = self.head_p(h) # [N, 1]
        
        return u_pred, p_pred
