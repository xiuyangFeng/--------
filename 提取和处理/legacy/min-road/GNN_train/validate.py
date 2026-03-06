import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from model import SimpleGNN
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

class VascularDataset(Dataset):
    def __init__(self, pt_files):
        super(VascularDataset, self).__init__()
        self.pt_files = pt_files
    def len(self): return len(self.pt_files)
    def get(self, idx): return torch.load(self.pt_files[idx], weights_only=False)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            all_preds.append(out.cpu().numpy())
            all_targets.append(data.y.cpu().numpy())
            
    # 合并所有预测和真实值
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return all_preds, all_targets

def plot_scatter(preds, targets, target_names):
    """绘制预测值 vs 真实值的散点图"""
    num_vars = len(target_names)
    fig, axes = plt.subplots(1, num_vars, figsize=(5 * num_vars, 5))
    
    for i in range(num_vars):
        ax = axes[i]
        ax.scatter(targets[:, i], preds[:, i], alpha=0.3, s=1, color='#2ca02c')
        
        # 绘制 y=x 参考线
        min_val = min(targets[:, i].min(), preds[:, i].min())
        max_val = max(targets[:, i].max(), preds[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_title(target_names[i])
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_path = "prediction_scatter.png"
    plt.savefig(output_path, dpi=300)
    print(f"预测散点图已保存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="GNN 模型验证评估脚本")
    parser.add_argument("--model-path", type=str, required=True, help="模型权重文件 (.pt)")
    parser.add_argument("--data-list", type=str, default="test_files.txt", help="要评估的文件列表")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载测试集文件列表
    if not Path(args.data_list).exists():
        print(f"错误: 找不到测试集列表文件 {args.data_list}。请确保已运行过 train.py")
        return
        
    with open(args.data_list, "r") as f:
        test_files = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"测试样本数量: {len(test_files)}")
    
    # 2. 初始化并加载模型 (节点特征10维 + 全局条件6维)
    model = SimpleGNN(node_dim=10, global_dim=6, hidden_channels=64, out_channels=4).to(device)
    
    # 手动指定 weights_only=False 以兼容 PyTorch 2.6+
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    
    # 3. 执行评估
    loader = DataLoader(VascularDataset(test_files), batch_size=1)
    preds, targets = evaluate(model, loader, device)
    
    # 4. 计算指标并分变量报告
    target_names = ['u (Velocity X)', 'v (Velocity Y)', 'w (Velocity Z)', 'p (Pressure)']
    
    print("\n" + "="*50)
    print(f"{'变量':<20} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10}")
    print("-" * 50)
    
    for i, name in enumerate(target_names):
        mae = mean_absolute_error(targets[:, i], preds[:, i])
        rmse = np.sqrt(mean_squared_error(targets[:, i], preds[:, i]))
        r2 = r2_score(targets[:, i], preds[:, i])
        print(f"{name:<20} | {mae:<10.6f} | {rmse:<10.6f} | {r2:<10.6f}")
    print("="*50)

    # 绘制预测散点图
    plot_scatter(preds, targets, target_names)

if __name__ == "__main__":
    main()
