# 血管血流 GNN 训练流程 (Demo 版)

本目录包含了将处理后的血管几何/物理特征数据转化为图神经网络 (GNN) 训练数据，并进行训练与验证的完整流程脚本。

## 📊 流程概述

1.  **数据转换**：将 `ascii_normalized_128` 文件夹中的 CSV 文件转换为 PyTorch Geometric 的 `.pt` 格式。
2.  **模型训练**：基于 **Graph Transformer** 模型（带残差连接与多头注意力）预测 `u, v, w, p`。
3.  **验证评估**：在测试集上计算 MAE, RMSE 和 R² 等指标。

## 🛠️ 环境依赖 (针对服务器 A100 集群部署)

对于配置了 A100 显卡和 CUDA 12.8 驱动的服务器环境，建议使用以下步骤创建虚拟环境并进行安装。

### 1. 创建虚拟环境
```bash
conda create -n gnn_vascular python=3.10 -y
conda activate gnn_vascular
```

### 2. 安装 PyTorch (针对 CUDA 12.x)
虽然驱动为 12.8，但建议安装成熟的 CUDA 12.4 编译版本：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. 安装 PyTorch Geometric (PyG)
PyG 现在推荐通过以下方式一键安装其所有依赖项（包括 `pyg-lib`, `torch-scatter`, `torch-sparse` 等）：
```bash
pip install torch_geometric
# 安装针对 CUDA 的额外优化组件
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```
> [!NOTE]
> 请根据您安装的实际 torch 版本号修改上面链接中的 `2.4.0`。

### 4. 安装其他数据科学库
```bash
pip install pandas numpy tqdm scikit-learn pynvml
```

## 🚀 使用指南 (集群运行建议)

### 1. 数据转换

将所有病人的归一化 CSV 数据转换为图数据：

```bash
python data_converter.py --data-root ../data --output-dir ./processed_data --k 6
```

- `--data-root`: 指向包含 `fast`, `AAA` 等病历文件夹的根目录。
- `--output-dir`: 转换后的 `.pt` 文件保存位置。
- `--k`: 每个节点的邻居数量（默认 6）。

### 2. 模型训练

使用转换后的数据训练 **Graph Transformer** 模型：

```bash
python train.py --data-dir ./processed_data --epochs 100 --batch-size 4 --lr 0.001
```

- 训练过程中会自动划分 80% 训练集，10% 验证集，10% 测试集。
- 测试集的文件路径会保存到 `test_files.txt`。
- 最优模型权重保存在 `checkpoints/best_model.pt`。
- 训练日志保存在 `train_log.txt`。

### 3. (新增) 训练过程可视化

查看损失函数下降曲线：

```bash
python visualize_loss.py --log-file train_log.txt
```

### 4. 模型验证与可视化评估

在测试集上评估模型预测效果：

```bash
python validate.py --model-path ./checkpoints/best_model.pt --data-list test_files.txt
```

- 该脚本会输出各项误差指标（MAE, RMSE, R²）。
- 同时会在当前目录下生成 `prediction_scatter.png`，直观展示预测值与真实值的偏差。

## 🧠 特征设计说明

### 输入特征 (16维)
1.  **坐标 (3)**: `x, y, z`
2.  **时间 (1)**: `t` (从文件名提取并归一化到 0-1s 范围)
3.  **几何特征 (6)**: `Abscissa, NormRadius, Curvature, Tangent_X/Y/Z`
4.  **边界标志 (2)**: `is_wall, BC_Flag`
5.  **边界条件 (4)**: `BC_Inlet, BC_O1, BC_O2, BC_O3, BC_O4`

### 预测目标 (5维)
- `u, v, w` (速度三分量)
- `p` (压力)
- `wss` (壁面剪切应力)

## 💡 进阶修改建议

- **损失函数**：当前使用的是加权 MSE 损失函数。若需引入物理约束 (PINN)，可修改 `train.py` 中的 `weighted_mse_loss` 函数。
- **模型架构**：可在 `model.py` 中增加层数或尝试更复杂的算子（如 `GATConv`, `GraphConv`）。
