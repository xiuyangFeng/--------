import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GATv2Conv

class PI_GNN(nn.Module):
    def __init__(self, in_channels=9, hidden_channels=64, out_channels_u=3, out_channels_p=1):
        super(PI_GNN, self).__init__()
        
        # 1. 编码器 (Encoder): 将输入特征映射到高维空间
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 2. 主干 (Backbone): 图神经网络层 (这里用 GATv2 为例，也可换 TransformerConv)
        # 注意: 128万节点对显存要求很高，这里只堆叠 3 层演示流程
        self.conv1 = GATv2Conv(hidden_channels, hidden_channels // 4, heads=4, concat=True)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels // 4, heads=4, concat=True)
        self.conv3 = GATv2Conv(hidden_channels, hidden_channels // 4, heads=4, concat=True)
        
        # 如果显存不够，可以将上面的 GATv2Conv 换成更轻量的 GraphConv 或 GCNConv
        
        # 3. 解码器头 (Decoders)
        # 头 A: 预测速度 (u, v, w) -> 输出 3 维
        self.head_u = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels_u)
        )
        
        # 头 B: 预测压力 (p) -> 输出 1 维
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
        # 第一层
        h = self.conv1(h, edge_index)
        h = F.gelu(h)
        
        # 第二层
        h = self.conv2(h, edge_index)
        h = F.gelu(h)
        
        # 第三层
        h = self.conv3(h, edge_index)
        h = F.gelu(h) # 此时 h 是 [N, 64]
        
        # --- 输出预测 ---
        u_pred = self.head_u(h) # [N, 3]
        p_pred = self.head_p(h) # [N, 1]
        
        return u_pred, p_pred