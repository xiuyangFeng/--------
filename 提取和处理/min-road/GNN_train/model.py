import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool

class SimpleGNN(torch.nn.Module):
    """
    基于 Graph Transformer 的图神经网络。
    Graph Transformer 能够通过注意力机制更好地捕捉长程依赖和局部特征。
    
    架构设计：
    - Global Condition Broadcast: 将全局条件 (t_norm + BC) 广播到每个节点并拼接
    - Input Embedding: 将拼接后的 16 维特征映射到隐藏空间
    - Transformer Layers: 3 层 GraphTransformerConv
    - Residual Connections: 每层卷积后进行残差连接
    - Multi-head Attention: 使用 4 个注意力头
    
    输入格式:
    - data.x: [N, 10] 节点特征 (坐标3 + 几何6 + is_wall1)
    - data.global_cond: [B, 6] 全局条件 (t_norm1 + BC5)
    - data.edge_index: [2, E] 边索引
    - data.batch: [N] 节点所属图的索引（DataLoader 自动生成）
    """
    def __init__(self, node_dim=10, global_dim=6, hidden_channels=64, out_channels=4, heads=4):
        super(SimpleGNN, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.node_dim = node_dim
        self.global_dim = global_dim
        
        # 初始特征映射（节点特征 + 全局条件拼接后的维度）
        in_channels = node_dim + global_dim  # 10 + 6 = 16
        self.lin_in = torch.nn.Linear(in_channels, hidden_channels)
        
        # 第 1 层 Transformer: 输入 -> 隐藏
        self.conv1 = TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=0.1)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        
        # 第 2 层 Transformer: 隐藏 -> 隐藏
        self.conv2 = TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=0.1)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        
        # 第 3 层 Transformer: 隐藏 -> 隐藏
        self.conv3 = TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=0.1)
        self.lin3 = torch.nn.Linear(hidden_channels, hidden_channels)
        
        # 输出层 (针对每个节点的 4 个物理量预测)
        self.lin_out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 0. 全局条件广播: 将图级全局条件扩展到每个节点
        # data.global_cond: [B, 6]，data.batch: [N_total]
        if hasattr(data, 'global_cond') and data.global_cond is not None:
            if hasattr(data, 'batch') and data.batch is not None:
                # 批量模式: 通过 batch 索引广播
                gc_expanded = data.global_cond[data.batch]  # [N_total, 6]
            else:
                # 单图模式: 直接扩展
                gc_expanded = data.global_cond.expand(x.size(0), -1)  # [N, 6]
            x = torch.cat([x, gc_expanded], dim=-1)  # [N, 10+6=16]

        # 1. 特征编码
        x = self.lin_in(x)
        
        # 2. Transformer Block 1
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = self.lin1(h)
        x = x + h  # 残差连接
        x = F.dropout(x, p=0.1, training=self.training)

        # 3. Transformer Block 2
        h = self.conv2(x, edge_index)
        h = F.elu(h)
        h = self.lin2(h)
        x = x + h  # 残差连接
        x = F.dropout(x, p=0.1, training=self.training)

        # 4. Transformer Block 3
        h = self.conv3(x, edge_index)
        h = F.elu(h)
        h = self.lin3(h)
        x = x + h  # 残差连接
        x = F.dropout(x, p=0.1, training=self.training)

        # 5. 回归输出
        x = self.lin_out(x)

        return x

def get_model(node_dim=10, global_dim=6, hidden_dim=64, out_dim=4):
    """
    快捷获取模型实例
    
    参数:
        node_dim: 节点特征维度（默认 10）
        global_dim: 全局条件维度（默认 6）
        hidden_dim: 隐藏层维度
        out_dim: 输出维度
    """
    return SimpleGNN(node_dim, global_dim, hidden_dim, out_dim)

if __name__ == "__main__":
    # 测试代码: 模拟一个小图输入
    model = SimpleGNN()
    print(model)
    
    # 构建虚拟数据（新格式：节点特征 + 全局条件分离）
    num_nodes = 100
    x = torch.randn((num_nodes, 10))       # 10维节点特征
    global_cond = torch.randn((1, 6))      # 6维全局条件
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, global_cond=global_cond)
    
    output = model(data)
    print(f"输出形状: {output.shape}")  # 预期 [100, 4]
    
    # 测试批量模式
    from torch_geometric.loader import DataLoader
    batch_data = [
        Data(x=torch.randn(50, 10), edge_index=torch.randint(0, 50, (2, 200)),
             y=torch.randn(50, 4), global_cond=torch.randn(1, 6))
        for _ in range(4)
    ]
    loader = DataLoader(batch_data, batch_size=2)
    batch = next(iter(loader))
    output_batch = model(batch)
    print(f"批量输出形状: {output_batch.shape}")  # 预期 [100, 4]
    print(f"global_cond 批量形状: {batch.global_cond.shape}")  # 预期 [2, 6]
