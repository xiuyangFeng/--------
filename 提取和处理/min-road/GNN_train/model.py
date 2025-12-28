import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool

class SimpleGNN(torch.nn.Module):
    """
    基于 Graph Transformer 的图神经网络。
    Graph Transformer 能够通过注意力机制更好地捕捉长程依赖和局部特征。
    
    架构设计：
    - Input Embedding: 将原始 17 维特征映射到隐藏空间
    - Transformer Layers: 3 层 GraphTransformerConv
    - Residual Connections: 每层卷积后进行残差连接
    - Multi-head Attention: 使用 4 个注意力头
    """
    def __init__(self, in_channels=17, hidden_channels=64, out_channels=4, heads=4):
        super(SimpleGNN, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.heads = heads
        
        # 初始特征映射
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

def get_model(in_dim=17, hidden_dim=64, out_dim=4):
    """
    快捷获取模型实例
    """
    return SimpleGNN(in_dim, hidden_dim, out_dim)

if __name__ == "__main__":
    # 测试代码: 模拟一个小图输入
    model = SimpleGNN()
    print(model)
    
    # 构建虚拟数据
    num_nodes = 100
    x = torch.randn((num_nodes, 17))
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index)
    
    output = model(data)
    print(f"输出形状: {output.shape}") # 预期应该是 [100, 4]
