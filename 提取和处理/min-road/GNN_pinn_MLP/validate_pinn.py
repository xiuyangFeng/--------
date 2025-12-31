"""
GNN + PINN 模型验证脚本
======================
计算测试集上的各项指标，并与普通GNN模型对比
"""

import sys
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from train_pinn import GNN_PINN, compute_physics_loss, RE
from pathlib import Path
import argparse
import numpy as np

# 添加 GNN_train 目录到路径，以便导入 SimpleGNN
sys.path.insert(0, str(Path(__file__).parent.parent / "GNN_train"))
from model import SimpleGNN  # type: ignore


class VascularDataset(Dataset):
    """加载 .pt 文件"""
    def __init__(self, pt_files):
        super(VascularDataset, self).__init__()
        self.pt_files = pt_files

    def len(self):
        return len(self.pt_files)

    def get(self, idx):
        data = torch.load(self.pt_files[idx], weights_only=False)
        return data


def compute_metrics(pred, target):
    """计算各种评估指标"""
    # MSE
    mse = F.mse_loss(pred, target, reduction='none').mean(dim=0)
    
    # MAE
    mae = torch.abs(pred - target).mean(dim=0)
    
    # 相对误差 (避免除零)
    rel_error = torch.abs(pred - target) / (torch.abs(target) + 1e-8)
    rel_error_mean = rel_error.mean(dim=0)
    
    # R² 分数
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0)) ** 2).sum(dim=0)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    return {
        'mse': mse.cpu().numpy(),
        'mae': mae.cpu().numpy(),
        'rel_error': rel_error_mean.cpu().numpy(),
        'r2': r2.cpu().numpy()
    }


@torch.no_grad()
def evaluate_model(model, loader, device, model_type='pinn'):
    """评估模型"""
    model.eval()
    
    all_preds = []
    all_targets = []
    total_cont_loss = 0
    total_mom_loss = 0
    total_bc_loss = 0
    num_samples = 0
    
    for data in loader:
        data = data.to(device)
        
        if model_type == 'pinn':
            output = model(data, compute_physics=False)
        else:
            output = model(data)
        
        all_preds.append(output.cpu())
        all_targets.append(data.y.cpu())
        num_samples += data.num_graphs
    
    # 合并所有预测和真实值
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 计算指标
    metrics = compute_metrics(all_preds, all_targets)
    
    return metrics, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description="GNN + PINN 模型验证")
    parser.add_argument("--test-file", type=str, default="./test_files_pinn.txt", help="测试文件列表")
    parser.add_argument("--pinn-model", type=str, default="./checkpoints_pinn/best_model_pinn.pt", help="PINN模型路径")
    parser.add_argument("--gnn-model", type=str, default="../GNN_train/checkpoints/best_model.pt", help="普通GNN模型路径")
    parser.add_argument("--batch-size", type=int, default=4, help="批大小")
    parser.add_argument("--hidden-dim", type=int, default=64, help="隐藏层维度")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载测试文件列表
    test_file_path = Path(args.test_file)
    if not test_file_path.exists():
        print(f"错误: 找不到测试文件列表 {args.test_file}")
        return
    
    with open(test_file_path, 'r') as f:
        test_files = [Path(line.strip()) for line in f if line.strip()]
    
    print(f"加载 {len(test_files)} 个测试样本")
    
    test_loader = DataLoader(VascularDataset(test_files), batch_size=args.batch_size)
    
    # 变量名称
    var_names = ['u', 'v', 'w', 'p']
    
    print("\n" + "=" * 80)
    print("模型评估结果对比")
    print("=" * 80)
    
    # 评估 PINN 模型
    pinn_model_path = Path(args.pinn_model)
    if pinn_model_path.exists():
        print(f"\n[1] GNN + PINN 模型 ({args.pinn_model})")
        print("-" * 60)
        
        pinn_model = GNN_PINN(in_channels=17, hidden_channels=args.hidden_dim, out_channels=4).to(device)
        pinn_model.load_state_dict(torch.load(pinn_model_path, map_location=device, weights_only=True))
        
        pinn_metrics, pinn_preds, pinn_targets = evaluate_model(pinn_model, test_loader, device, 'pinn')
        
        print(f"{'变量':<10} {'MSE':<15} {'MAE':<15} {'相对误差':<15} {'R²':<15}")
        print("-" * 60)
        for i, name in enumerate(var_names):
            print(f"{name:<10} {pinn_metrics['mse'][i]:<15.6f} {pinn_metrics['mae'][i]:<15.6f} "
                  f"{pinn_metrics['rel_error'][i]:<15.6f} {pinn_metrics['r2'][i]:<15.4f}")
        
        print("-" * 60)
        print(f"{'平均':<10} {pinn_metrics['mse'].mean():<15.6f} {pinn_metrics['mae'].mean():<15.6f} "
              f"{pinn_metrics['rel_error'].mean():<15.6f} {pinn_metrics['r2'].mean():<15.4f}")
    else:
        print(f"\n[1] GNN + PINN 模型: 未找到 ({args.pinn_model})")
        pinn_metrics = None
    
    # 评估普通 GNN 模型
    gnn_model_path = Path(args.gnn_model)
    if gnn_model_path.exists():
        print(f"\n[2] 普通 GNN 模型 ({args.gnn_model})")
        print("-" * 60)
        
        gnn_model = SimpleGNN(in_channels=17, hidden_channels=args.hidden_dim, out_channels=4).to(device)
        gnn_model.load_state_dict(torch.load(gnn_model_path, map_location=device, weights_only=True))
        
        gnn_metrics, gnn_preds, gnn_targets = evaluate_model(gnn_model, test_loader, device, 'gnn')
        
        print(f"{'变量':<10} {'MSE':<15} {'MAE':<15} {'相对误差':<15} {'R²':<15}")
        print("-" * 60)
        for i, name in enumerate(var_names):
            print(f"{name:<10} {gnn_metrics['mse'][i]:<15.6f} {gnn_metrics['mae'][i]:<15.6f} "
                  f"{gnn_metrics['rel_error'][i]:<15.6f} {gnn_metrics['r2'][i]:<15.4f}")
        
        print("-" * 60)
        print(f"{'平均':<10} {gnn_metrics['mse'].mean():<15.6f} {gnn_metrics['mae'].mean():<15.6f} "
              f"{gnn_metrics['rel_error'].mean():<15.6f} {gnn_metrics['r2'].mean():<15.4f}")
    else:
        print(f"\n[2] 普通 GNN 模型: 未找到 ({args.gnn_model})")
        gnn_metrics = None
    
    # 对比分析
    if pinn_metrics is not None and gnn_metrics is not None:
        print("\n" + "=" * 80)
        print("改进幅度 (PINN vs GNN)")
        print("=" * 80)
        print(f"{'变量':<10} {'MSE改进%':<15} {'MAE改进%':<15} {'R²改进':<15}")
        print("-" * 60)
        
        for i, name in enumerate(var_names):
            mse_improve = (gnn_metrics['mse'][i] - pinn_metrics['mse'][i]) / gnn_metrics['mse'][i] * 100
            mae_improve = (gnn_metrics['mae'][i] - pinn_metrics['mae'][i]) / gnn_metrics['mae'][i] * 100
            r2_improve = pinn_metrics['r2'][i] - gnn_metrics['r2'][i]
            
            print(f"{name:<10} {mse_improve:<15.2f} {mae_improve:<15.2f} {r2_improve:<+15.4f}")
        
        # 平均改进
        avg_mse_improve = (gnn_metrics['mse'].mean() - pinn_metrics['mse'].mean()) / gnn_metrics['mse'].mean() * 100
        avg_mae_improve = (gnn_metrics['mae'].mean() - pinn_metrics['mae'].mean()) / gnn_metrics['mae'].mean() * 100
        avg_r2_improve = pinn_metrics['r2'].mean() - gnn_metrics['r2'].mean()
        
        print("-" * 60)
        print(f"{'平均':<10} {avg_mse_improve:<15.2f} {avg_mae_improve:<15.2f} {avg_r2_improve:<+15.4f}")
        
        # 简要结论
        print("\n" + "=" * 80)
        print("结论")
        print("=" * 80)
        if avg_mse_improve > 0:
            print(f"✓ PINN 模型在测试集上 MSE 降低了 {avg_mse_improve:.2f}%")
        else:
            print(f"✗ PINN 模型在测试集上 MSE 增加了 {-avg_mse_improve:.2f}%")
        
        if avg_r2_improve > 0:
            print(f"✓ PINN 模型 R² 提升了 {avg_r2_improve:.4f}")
        else:
            print(f"✗ PINN 模型 R² 下降了 {-avg_r2_improve:.4f}")


if __name__ == "__main__":
    main()

