# Pipeline - 血管 CFD 数据处理流程

> 将 Fluent 仿真输出数据转换为 GNN 训练所需的图数据格式

## 📋 目录

- [概述](#概述)
- [目录结构](#目录结构)
- [环境准备](#环境准备)
- [数据要求](#数据要求)
- [快速开始](#快速开始)
- [分步使用](#分步使用)
- [集群运行](#集群运行)
- [配置说明](#配置说明)
- [输出说明](#输出说明)
- [常见问题](#常见问题)

---

## 概述

本 Pipeline 将 Fluent CFD 仿真的原始输出数据，经过 **4 个步骤** 处理为 PyTorch Geometric 图数据：

```
原始数据                    处理流程                         最终输出
┌─────────────┐     ┌────────────────────────┐     ┌─────────────┐
│ ascii/      │     │ 步骤1: preprocess      │     │ graphs/     │
│ (壁面数据)   │ ──▶ │   清洗+合并+降采样      │     │ *.pt        │
├─────────────┤     ├────────────────────────┤     │             │
│ ascii_in/   │     │ 步骤2: extract_features│ ──▶ │ PyG Data:   │
│ (内部数据)   │ ──▶ │   几何特征+边界条件     │     │ - x: 15维   │
├─────────────┤     ├────────────────────────┤     │ - y: 4维    │
│ *.stl       │     │ 步骤3: normalize       │     │ - edge_idx  │
│ (表面模型)   │ ──▶ │   特征归一化            │     └─────────────┘
├─────────────┤     ├────────────────────────┤
│ Global_     │     │ 步骤4: convert_to_graph│
│ conditions/ │ ──▶ │   转换为图数据          │
└─────────────┘     └────────────────────────┘
```

---

## 目录结构

```
pipeline/
├── __init__.py             # 模块初始化
├── config.py               # 配置文件
├── preprocess.py           # 步骤1: 数据预处理
├── extract_features.py     # 步骤2: 几何特征提取
├── normalize.py            # 步骤3: 特征归一化
├── convert_to_graph.py     # 步骤4: 图数据转换
├── run_all.py              # 一键运行全流程
├── utils/                  # 工具模块
│   ├── __init__.py
│   ├── io.py               # 数据读写
│   ├── sampling.py         # 降采样算法
│   └── geometry.py         # 几何计算
├── cluster/                # 集群运行脚本
│   ├── run_pipeline.slurm
│   ├── run_single_step.slurm
│   ├── batch_submit.sh
│   └── README.md
└── README.md               # 本文件
```

---

## 环境准备

### 1. 创建/激活 conda 环境

```bash
conda activate rag_venv
```

### 2. 安装依赖

```bash
pip install numpy pandas scipy scikit-learn tqdm
pip install torch torch_geometric
pip install vtk vmtk
```

### 3. 验证安装

```bash
cd pipeline
python config.py  # 测试配置文件
```

---

## 数据要求

### 输入数据目录结构

```
data_new/
└── AG/
    └── fast/
        └── 病例名/                    # 如 ZHANG_CHUN
            ├── 病例名.stl             # STL 表面模型（必需）
            ├── ascii/                 # 壁面数据目录（必需）
            │   ├── 病例名-1120        # 时间步 1120
            │   ├── 病例名-1121        # 时间步 1121
            │   └── ...
            ├── ascii_in/              # 内部数据目录（必需）
            │   ├── 病例名-1120
            │   ├── 病例名-1122        # 可以隔帧
            │   └── ...
            └── Global_conditions/     # 边界条件目录（必需）
                ├── vf-in-rfile.out    # 入口体积流量
                ├── p-outle-rfile.out  # 左外髂支出口压力
                ├── p-outli-rfile.out  # 左内髂支出口压力
                ├── p-outre-rfile.out  # 右外髂支出口压力
                └── p-outri-rfile.out  # 右内髂支出口压力
```

### 数据格式说明

**壁面数据 (ascii/)**: 逗号分隔 CSV
```
nodenumber, x-coordinate, y-coordinate, z-coordinate, pressure, wall-shear, x-wall-shear, y-wall-shear, z-wall-shear
```

**内部数据 (ascii_in/)**: 逗号分隔 CSV
```
cellnumber, x-coordinate, y-coordinate, z-coordinate, pressure, velocity-magnitude, x-velocity, y-velocity, z-velocity
```

---

## 快速开始

### 方式一：一键处理（推荐）

```bash
cd pipeline

# 处理单个病例（推荐先测试）
python run_all.py --case ZHANG_CHUN

# 处理所有启用的病例
python run_all.py
```

### 方式二：分步处理

```bash
cd pipeline

# 步骤1: 数据预处理
python preprocess.py --case ZHANG_CHUN

# 步骤2: 几何特征提取
python extract_features.py --case ZHANG_CHUN

# 步骤3: 特征归一化
python normalize.py --case ZHANG_CHUN

# 步骤4: 图数据转换
python convert_to_graph.py --case ZHANG_CHUN
```

---

## 分步使用

### 步骤1: preprocess.py - 数据预处理

**功能**：
- 读取原始 Fluent 输出数据
- 清洗数据（列名标准化、单位转换 m→mm）
- 分层降采样合并（保留所有壁面点，内部点智能采样）

**输入**：`ascii/` + `ascii_in/`  
**输出**：`processed/merged/`

```bash
# 基本用法
python preprocess.py --case ZHANG_CHUN

# 使用随机采样（速度快，但质量稍差）
python preprocess.py --case ZHANG_CHUN --sampling-method random

# 自定义目标点数（默认 40000）
python preprocess.py --case ZHANG_CHUN --target-points 50000

# 处理所有病例
python preprocess.py
```

**参数说明**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--case` | 无 | 指定病例名称，不指定则处理所有 |
| `--target-points` | 40000 | 目标总点数 |
| `--sampling-method` | fps | 采样方法：fps(质量高)/random(速度快) |
| `--boundary-threshold` | 2.0 | 近壁区阈值 (mm) |
| `--boundary-ratio` | 0.7 | 近壁层预算比例 |
| `--mode` | debug | 处理模式：debug/production |

---

### 步骤2: extract_features.py - 几何特征提取

**功能**：
- 读取 STL 表面模型，提取血管中心线
- 计算几何特征（Abscissa, NormRadius, Curvature, Tangent）
- 从 `Global_conditions/` 加载边界条件

**输入**：`processed/merged/` + `*.stl` + `Global_conditions/`  
**输出**：`processed/features/`

```bash
# 基本用法
python extract_features.py --case ZHANG_CHUN

# 处理所有病例
python extract_features.py
```

**输出特征**：
| 特征 | 说明 |
|------|------|
| x, y, z | 空间坐标 (mm) |
| Abscissa | 沿中心线的弧长坐标 [0,1] |
| NormRadius | 归一化半径距离 (到壁面距离/局部半径) |
| Curvature | 局部曲率 |
| Tangent_X/Y/Z | 中心线切线方向 |
| u, v, w | 速度分量 |
| p | 压力 |
| wss, wss_x/y/z | 壁面剪切力 |
| is_wall | 壁面标记 (0/1) |
| BC_Inlet | 入口体积流量 |
| BC_O1~O4 | 四个髂支出口压力 |

---

### 步骤3: normalize.py - 特征归一化

**功能**：
- 收集全局统计量
- 对各特征进行归一化/标准化
- 保存归一化参数（用于推理时还原）

**输入**：`processed/features/`  
**输出**：`processed/normalized/` + `normalization_params_global.json`

```bash
# 基本用法
python normalize.py --case ZHANG_CHUN

# 处理所有病例
python normalize.py
```

**归一化策略**：
| 特征类型 | 方法 | 说明 |
|----------|------|------|
| Abscissa, Tangent_X/Y/Z, is_wall | 保持不变 | 已归一化或二值 |
| NormRadius | Min-Max | → [0, 1] |
| Curvature, u/v/w, p, wss* | Z-score | (x - μ) / σ |
| BC_Inlet | × 1e5 | 入口流量缩放 |
| BC_O1~O4 | (x - 15000) / 1000 | 出口压力缩放 |

---

### 步骤4: convert_to_graph.py - 图数据转换

**功能**：
- 构建 KNN 图结构
- 组装输入特征和目标输出
- 保存为 PyTorch Geometric `.pt` 文件

**输入**：`processed/normalized/`  
**输出**：`processed/graphs/`

```bash
# 基本用法
python convert_to_graph.py --case ZHANG_CHUN

# 自定义邻居数（默认 6）
python convert_to_graph.py --case ZHANG_CHUN --k 8

# 处理所有病例
python convert_to_graph.py
```

**输出图数据格式**：
```python
Data(
    x,           # 输入特征 [N, 15]
    edge_index,  # 边索引 [2, E]
    y            # 目标输出 [N, 4]
)
```

**输入特征 x (15维)**：
```
[0:3]   坐标: x, y, z
[3]     时间: t_norm (归一化到 [0,1])
[4:10]  几何: Abscissa, NormRadius, Curvature, Tangent_X/Y/Z
[10]    边界标志: is_wall
[11:16] 边界条件: BC_Inlet, BC_O1~O4
```

**目标输出 y (4维)**：
```
[0:3]   速度: u, v, w
[3]     压力: p
```

---

## 集群运行

如果数据处理在集群上进行：

```bash
cd pipeline/cluster

# 提交单个病例
sbatch run_pipeline.slurm ZHANG_CHUN

# 批量提交所有病例
chmod +x batch_submit.sh
./batch_submit.sh

# 查看作业状态
squeue -u $USER
```

详见 `cluster/README.md`

---

## 配置说明

### config.py 主要配置

```python
# 数据源配置
DATA_SOURCES = {
    "AG/fast": {"enabled": True},   # 当前启用
    "AG/slow": {"enabled": False},  # 待放入数据后启用
    ...
}

# 降采样配置
SAMPLING_CONFIG = {
    "target_total_points": 40000,   # 目标点数
    "sampling_method": "fps",       # fps 或 random
    "boundary_threshold": 2.0,      # 近壁区阈值 (mm)
    ...
}

# 处理模式
MODE = "debug"  # debug: 保留中间文件 / production: 只保留最终输出
```

### 环境变量（集群使用）

```bash
export PIPELINE_DATA_ROOT="/path/to/data_new"  # 覆盖数据目录
export PIPELINE_MODE="production"              # 覆盖处理模式
```

---

## 输出说明

处理完成后，每个病例目录下会生成：

```
病例目录/
├── ascii/                  # 原始数据（保留）
├── ascii_in/               # 原始数据（保留）
├── Global_conditions/      # 边界条件（保留）
├── *.stl                   # 表面模型（保留）
└── processed/              # 处理输出
    ├── merged/             # 步骤1: 合并降采样
    │   ├── merged-1120.csv
    │   ├── merged-1122.csv
    │   └── ...
    ├── features/           # 步骤2: 添加几何特征
    │   ├── result_features_merged-1120.csv
    │   └── ...
    ├── normalized/         # 步骤3: 归一化
    │   ├── result_features_merged-1120.csv
    │   └── ...
    └── graphs/             # 步骤4: 图数据
        ├── result_features_merged-1120.pt
        └── ...
```

全局归一化参数保存在：
```
data_new/normalization_params_global.json
```

---

## 常见问题

### 1. 找不到 vmtk 模块

```bash
pip install vmtk
# 或
conda install -c vmtk vmtk
```

### 2. 内存不足 (FPS 采样)

使用随机采样替代：
```bash
python preprocess.py --sampling-method random
```

### 3. 处理中断，如何继续？

使用 `--start-step` 跳过已完成的步骤：
```bash
python run_all.py --case ZHANG_CHUN --start-step 2  # 从步骤2开始
```

### 4. 如何添加新的数据源？

1. 在 `config.py` 中添加数据源：
```python
DATA_SOURCES = {
    "AG/fast": {"enabled": True},
    "新路径/子目录": {"enabled": True},  # 添加这行
}
```

2. 将数据放入 `data_new/新路径/子目录/病例名/`

### 5. 如何调整目标点数？

```bash
# 命令行方式
python run_all.py --target-points 50000

# 或修改 config.py
SAMPLING_CONFIG = {
    "target_total_points": 50000,
    ...
}
```

### 6. 查看处理进度

每个脚本都会打印详细的进度信息，包括：
- 当前处理的病例和文件
- 成功/失败数量
- 预估剩余时间
- 总耗时

---

## 完整示例

```bash
# 1. 进入 pipeline 目录
cd pipeline

# 2. 先测试单个病例
python run_all.py --case ZHANG_CHUN --sampling-method random

# 3. 检查输出
ls ../data_new/AG/fast/ZHANG_CHUN/processed/graphs/

# 4. 确认无误后，处理所有病例
python run_all.py

# 5. 完成后，图数据可用于 GNN 训练
# 图数据位于: data_new/AG/fast/*/processed/graphs/*.pt
```
