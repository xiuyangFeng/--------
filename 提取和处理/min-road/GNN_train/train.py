import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from model import SimpleGNN
from tqdm import tqdm
import argparse
from pathlib import Path
import random

class VascularDataset(Dataset):
    """
    加载由 data_converter.py 生成的 .pt 文件的简单数据集类。
    """
    def __init__(self, pt_files):
        super(VascularDataset, self).__init__()
        self.pt_files = pt_files

    def len(self):
        return len(self.pt_files)

    def get(self, idx):
        # PyTorch 2.6+ 默认开启 weights_only=True，加载自定义的 Data 对象需要显式关闭
        data = torch.load(self.pt_files[idx], weights_only=False)
        return data

def weighted_mse_loss(pred, target, weights=None):
    """
    加权 MSE 损失函数。
    pred, target: [N, 4] -> u, v, w, p
    weights: [4] 对应每个维度的权重
    """
    if weights is None:
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=pred.device)
    
    # 计算每个维度的 MSE
    mse_per_dim = F.mse_loss(pred, target, reduction='none').mean(dim=0)
    
    # 加权求和
    weighted_loss = (mse_per_dim * weights).sum()
    return weighted_loss

def train(model, loader, optimizer, device, weights):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        
        # 损失计算
        loss = weighted_mse_loss(out, data.y, weights)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, device, weights):
    model.eval()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = weighted_mse_loss(out, data.y, weights)
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser(description="GNN 模型训练脚本")
    parser.add_argument("--data-dir", type=str, default="./processed_data", help="处理后的 .pt 数据目录")
    parser.add_argument("--epochs", type=int, default=1000, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="每批病例数")
    parser.add_argument("--lr", type=float, default=0.0005, help="初始学习率")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="模型保存目录")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 搜根目录下所有的 .pt 文件
    data_dir = Path(args.data_dir)
    all_pt_files = sorted(list(data_dir.rglob("*.pt")))
    if not all_pt_files:
        print(f"错误: 在 {args.data_dir} 中找不到 .pt 文件")
        return
    
    print(f"找到 {len(all_pt_files)} 个数据样本。")
    
    # 2. 按照病例（目录）划分数据集，防止数据泄漏
    # 每个 .pt 文件的路径结构通常是: processed_data/CASE_NAME/filename.pt
    # 我们根据病例名来分组文件
    case_to_files = {}
    for pt_file in all_pt_files:
        case_name = pt_file.parent.name
        if case_name not in case_to_files:
            case_to_files[case_name] = []
        case_to_files[case_name].append(pt_file)
    
    unique_cases = sorted(list(case_to_files.keys()))
    if len(unique_cases) < 3:
        print(f"警告: 发现 {len(unique_cases)} 个病例，少于推荐的 3 个。将使用默认的 8:1:1 比例或按比例退化。")
        # 如果病例太少，退回到简单的随机划分，或者这里可以根据实际需求调整
        random.shuffle(all_pt_files)
        num_samples = len(all_pt_files)
        train_idx = int(num_samples * 0.8)
        val_idx = int(num_samples * 0.9)
        train_files = all_pt_files[:train_idx]
        val_files = all_pt_files[train_idx:val_idx]
        test_files = all_pt_files[val_idx:]
    else:
        # 用户要求：任意选取 2 个病例作为开发集（其中 80% 训练, 20% 验证），第 3 个病例作为测试集
        random.shuffle(unique_cases)
        test_case_name = unique_cases[-1]  # 选最后一个作为测试病例
        dev_case_names = unique_cases[:-1] # 选剩下的作为开发病例
        
        test_files = case_to_files[test_case_name]
        
        # 合并所有开发病例的文件
        dev_files = []
        for name in dev_case_names:
            dev_files.extend(case_to_files[name])
        
        # 将开发集按 8:2 重新随机划分
        random.shuffle(dev_files)
        train_idx = int(len(dev_files) * 0.8)
        train_files = dev_files[:train_idx]
        val_files = dev_files[train_idx:]
        
        print(f"病例分配结果:")
        print(f" - 开发病例 (训练+验证): {dev_case_names}")
        print(f" - 测试病例: ['{test_case_name}']")

    print(f"样本分配结果: 训练集={len(train_files)}, 验证集={len(val_files)}, 测试集={len(test_files)}")
    
    # 保存测试集列表方便验证
    with open("test_files.txt", "w") as f:
        for item in test_files:
            f.write(f"{item}\n")

    train_loader = DataLoader(VascularDataset(train_files), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(VascularDataset(val_files), batch_size=args.batch_size)
    
    # 3. 初始化模型 (输入维度 17，输出维度 4)
    model = SimpleGNN(in_channels=17, hidden_channels=64, out_channels=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器：当验证集损失在 20 轮内不下降时，将学习率乘以 0.5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    # 权重设定 (可以根据需要在命令行提取此参数)
    loss_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
    
    # 4. 训练循环
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    # 用于记录损失
    history = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, device, loss_weights)
        val_loss = validate(model, val_loader, device, loss_weights)
        
        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
        
        # 记录到 history
        history.append(f"{epoch},{train_loss:.8f},{val_loss:.8f}\n")
        
        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            print(f" ---> 已保存当前最优模型")
            
        # 每10轮保存一个检查点
        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_dir / f"checkpoint_epoch_{epoch}.pt")

    print("\n训练完成! 最优模型已保存至: ", save_dir / "best_model.pt")

    # 保存损失历史记录
    log_path = Path("train_log.txt")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss\n")
        f.writelines(history)
    print(f"训练日志已保存至: {log_path}")

if __name__ == "__main__":
    main()
