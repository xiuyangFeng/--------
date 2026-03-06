import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

def plot_loss(log_file):
    """可视化训练和验证损失曲线"""
    if not os.path.exists(log_file):
        print(f"错误: 找不到日志文件 {log_file}")
        return

    # 读取数据
    df = pd.read_csv(log_file)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2)
    
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 使用对数坐标轴，方便查看后期收敛情况
    plt.yscale('log')
    
    output_path = "loss_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"损失曲线已保存至: {output_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="训练损失可视化脚本")
    parser.add_argument("--log-file", type=str, default="train_log.txt", help="训练日志文件路径")
    args = parser.parse_args()
    
    plot_loss(args.log_file)

if __name__ == "__main__":
    main()
