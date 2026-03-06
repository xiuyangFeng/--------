# Pipeline - 血管 CFD 数据处理流程

> 将 Fluent 仿真输出数据转换为 GNN 训练所需的图数据格式，支持在线数据增强

## 📋 目录

- [概述](#概述)
- [目录结构](#目录结构)
- [环境准备](#环境准备)
- [数据要求](#数据要求)
- [快速开始](#快速开始)
- [分步使用](#分步使用)
- [数据增强](#数据增强)
- [集群运行](#集群运行)
- [配置说明](#配置说明)
- [输出说明](#输出说明)
- [常见问题](#常见问题)

---

## 概述

本 Pipeline 将 Fluent CFD 仿真的原始输出数据，经过 **5 个步骤** 处理为 PyTorch Geometric 图数据：

```
原始数据                    处理流程                         最终输出
┌─────────────┐     ┌────────────────────────┐     ┌──────────────────┐
│ ascii/      │     │ 步骤1: preprocess      │     │ graphs/          │
│ (壁面数据)   │ ──▶ │   清洗+合并+降采样      │     │ *.pt             │
├─────────────┤     ├────────────────────────┤     │                  │
│ ascii_in/   │     │ 步骤2: extract_features│     │ PyG Data:        │
│ (内部数据)   │ ──▶ │   几何特征+边界条件     │     │ - x: [N,10] 节点 │
├─────────────┤     ├────────────────────────┤     │ - global_cond:   │
│ *.stl       │     │ 步骤3: coord_normalize │ ──▶ │     [1,6] 全局   │
│ (表面模型)   │ ──▶ │   坐标系归一化          │     │ - y: [N,4] 标签  │
├─────────────┤     ├────────────────────────┤     │ - edge_index     │
│ Global_     │     │ 步骤4: normalize       │     │                  │
│ conditions/ │ ──▶ │   特征归一化            │     │ + 在线增强       │
└─────────────┘     ├────────────────────────┤     │   (旋转/平移)    │
                    │ 步骤5: convert_to_graph│     └──────────────────┘
                    │   转换为图数据          │
                    └────────────────────────┘
```

### 新增功能 (v2.0)

- **坐标系归一化**：中心化 + PCA主轴对齐 + 缩放，消除病例间位置/朝向差异
- **在线数据增强**：训练时动态应用旋转、平移等增强，提升模型泛化能力
- **矢量同步变换**：速度、切线等矢量特征在坐标变换时自动同步旋转

### 新增功能 (v2.1) - 全局条件分离

- **全局条件分离存储**：边界条件(BC)和时间(t_norm)不再逐行复制到每个点，改为图级属性 `data.global_cond`
- **BC 侧文件**：边界条件保存为 `bc_metadata.json` 独立文件，避免 CSV 中大量冗余列
- **节点特征精简**：`data.x` 从 15 维降为 10 维，减少约 40% 存储和内存占用
- **模型全局条件注入**：模型 forward 时通过 `data.batch` 索引广播全局条件到节点，再拼接输入

---

## 目录结构

```
pipeline/
├── __init__.py             # 模块初始化
├── config.py               # 配置文件
├── preprocess.py           # 步骤1: 数据预处理
├── extract_features.py     # 步骤2: 几何特征提取
├── coord_normalize.py      # 步骤3: 坐标系归一化【新增】
├── normalize.py            # 步骤4: 特征归一化
├── convert_to_graph.py     # 步骤5: 图数据转换
├── run_all.py              # 一键运行全流程
├── augmentation.py         # 数据增强函数【新增】
├── dataset.py              # 数据集类（含在线增强）【新增】
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
cd <repo-root>
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
            │   ├── 病例名-1122        # 时间步 1122（必须与 ascii_in 对齐）
            │   └── ...
            ├── ascii_in/              # 内部数据目录（必需）
            │   ├── 病例名-1120        # 时间步 1120
            │   ├── 病例名-1122        # 时间步 1122
            │   └── ...
            └── Global_conditions/     # 边界条件目录（必需）
                ├── vf-in-rfile.out    # 入口体积流量
                ├── p-outle-rfile.out  # 左外髂支出口压力
                ├── p-outli-rfile.out  # 左内髂支出口压力
                ├── p-outre-rfile.out  # 右外髂支出口压力
                └── p-outri-rfile.out  # 右内髂支出口压力
```

> **重要**: `ascii/` 和 `ascii_in/` 目录中的文件必须**一一对应**。Pipeline 只会处理两个目录中**同时存在**的时间步文件。如果文件不对齐，多余的文件会被跳过。建议在处理前先清理数据，确保两个目录的文件编号完全一致。

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
cd <repo-root>

# 处理单个病例（推荐先测试）
python -m pipeline.run_all --case ZHANG_CHUN

# 处理所有启用的病例
python -m pipeline.run_all
```

### 方式二：分步处理

```bash
cd <repo-root>

# 步骤1: 数据预处理
python -m pipeline.preprocess --case ZHANG_CHUN

# 步骤2: 几何特征提取
python -m pipeline.extract_features --case ZHANG_CHUN

# 步骤3: 坐标系归一化【新增】
python -m pipeline.coord_normalize --case ZHANG_CHUN

# 步骤4: 特征归一化
python -m pipeline.normalize --case ZHANG_CHUN

# 步骤5: 图数据转换
python -m pipeline.convert_to_graph --case ZHANG_CHUN
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
# 基本用法（默认使用混合采样）
python -m pipeline.preprocess --case ZHANG_CHUN

# 使用混合采样，调整 FPS 比例（更好的空间覆盖）
python -m pipeline.preprocess --case ZHANG_CHUN --sampling-method hybrid --fps-ratio 0.3

# 使用纯 FPS（最佳空间覆盖，但较慢）
python -m pipeline.preprocess --case ZHANG_CHUN --sampling-method fps

# 使用随机采样（速度快，但可能丢失分支血管）
python -m pipeline.preprocess --case ZHANG_CHUN --sampling-method random

# 自定义目标点数（默认 40000）
python -m pipeline.preprocess --case ZHANG_CHUN --target-points 50000

# 处理所有病例
python -m pipeline.preprocess
```

**采样方法对比**：
| 方法 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| `hybrid` | 混合采样（默认）| 兼顾空间覆盖与数据多样性 | 需要调参 |
| `fps` | 最远点采样 | 空间分布均匀，防止截断 | 计算较慢 |
| `random` | 随机采样 | 速度快 | 可能丢失分支血管数据 |

> **混合采样策略**：先用 FPS 采样 20%（可配置）的点确保空间覆盖（保护分支血管），再从剩余点中随机采样 80% 增加数据多样性。

**参数说明**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--case` | 无 | 指定病例名称，不指定则处理所有 |
| `--target-points` | 40000 | 目标总点数 |
| `--sampling-method` | hybrid | 采样方法：hybrid/fps/random |
| `--fps-ratio` | 0.2 | 混合采样时 FPS 占比（仅 hybrid 模式生效） |
| `--boundary-threshold` | 2.0 | 近壁区阈值 (mm) |
| `--boundary-ratio` | 0.7 | 近壁层预算比例 |
| `--mode` | debug | 处理模式：debug/production |

---

### 步骤2: extract_features.py - 几何特征提取

**功能**：
- 读取 STL 表面模型，提取血管中心线
- 计算几何特征（Abscissa, NormRadius, Curvature, Tangent）
- 从 `Global_conditions/` 加载边界条件，保存为 `bc_metadata.json` 侧文件

**输入**：`processed/merged/` + `*.stl` + `Global_conditions/`  
**输出**：`processed/features/`（CSV + `bc_metadata.json`）

```bash
# 基本用法
python -m pipeline.extract_features --case ZHANG_CHUN

# 处理所有病例
python -m pipeline.extract_features
```

**CSV 输出特征（逐点）**：
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

**bc_metadata.json（全局条件，每个时间步一组值）**：
| 字段 | 说明 |
|------|------|
| BC_Inlet | 入口体积流量 (m³/s) |
| BC_O1~O4 | 四个髂支出口压力 (Pa) |

> **v2.1 变更**：边界条件不再逐行写入 CSV（避免每个点重复存储相同的全局值），改为以 JSON 侧文件统一存储。

---

### 步骤3: coord_normalize.py - 坐标系归一化【新增】

**功能**：
- 中心化：将点云几何中心移到原点 (0, 0, 0)
- PCA对齐：将血管主轴旋转到 Z 轴方向
- 缩放：将坐标归一化到 [-1, 1] 范围
- 同步旋转：速度、切线等矢量特征自动同步旋转

**输入**：`processed/features/`（CSV + `bc_metadata.json`）  
**输出**：`processed/coord_normalized/`（CSV + `transform_params.json` + `bc_metadata.json` 副本）

```bash
# 基本用法
python -m pipeline.coord_normalize --case ZHANG_CHUN

# 处理所有病例
python -m pipeline.coord_normalize
```

**为什么需要坐标系归一化？**

1. **消除无关变异**：不同病例血管位置/朝向差异与流体力学规律无关
2. **数值稳定性**：原始坐标（毫米）数值较大，归一化后更适合神经网络
3. **增强一致性**：统一坐标系后，在线增强（旋转、平移）才有意义

**矢量特征同步变换**：

| 特征类型 | 变换方式 | 说明 |
|----------|----------|------|
| 坐标 (x, y, z) | 中心化 + 旋转 + 缩放 | 核心变换目标 |
| 速度 (u, v, w) | 仅旋转 | 矢量，必须同步旋转 |
| 切线 (Tangent_X/Y/Z) | 仅旋转 | 矢量，必须同步旋转 |
| WSS矢量 (wss_x/y/z) | 仅旋转 | 矢量，必须同步旋转 |
| 压力、曲率、半径等 | 不变 | 标量/内蕴属性 |

**输出的 transform_params.json**：
```json
{
  "centroid": [x, y, z],       // 原始质心
  "rotation_matrix": [[...]],  // PCA旋转矩阵
  "scale_factor": 123.45       // 缩放因子
}
```
> 推理时可使用这些参数将预测结果逆变换回原始坐标系

---

### 步骤4: normalize.py - 特征归一化

**功能**：
- 收集全局统计量
- 对各特征进行归一化/标准化
- 保存归一化参数（用于推理时还原）

**输入**：`processed/coord_normalized/`（CSV + `bc_metadata.json`）  
**输出**：`processed/normalized/`（CSV + `bc_metadata_normalized.json` + `normalization_params_global.json`）

```bash
# 基本用法
python -m pipeline.normalize --case ZHANG_CHUN

# 处理所有病例
python -m pipeline.normalize
```

**CSV 归一化策略（逐点特征）**：
| 特征类型 | 方法 | 说明 |
|----------|------|------|
| x, y, z | 保持不变 | 已在 coord_normalize 归一化到 [-1, 1] |
| Abscissa, Tangent_X/Y/Z, is_wall | 保持不变 | 已归一化或二值 |
| NormRadius | Min-Max | → [0, 1] |
| Curvature, u/v/w, p, wss* | Z-score | (x - μ) / σ |

**BC 归一化策略（全局条件，从 bc_metadata.json 读取，输出到 bc_metadata_normalized.json）**：
| 特征类型 | 方法 | 说明 |
|----------|------|------|
| BC_Inlet | × 1e5 | 入口流量缩放 → 0.5~5.0 |
| BC_O1~O4 | (x - 15000) / 1000 | 出口压力缩放 → -1.5~+1.5 |

---

### 步骤5: convert_to_graph.py - 图数据转换

**功能**：
- 构建 KNN 图结构
- 组装节点特征和目标输出
- 从 `bc_metadata_normalized.json` 加载全局条件，存为图级属性
- 保存为 PyTorch Geometric `.pt` 文件

**输入**：`processed/normalized/`（CSV + `bc_metadata_normalized.json`）  
**输出**：`processed/graphs/`

```bash
# 基本用法
python -m pipeline.convert_to_graph --case ZHANG_CHUN

# 自定义邻居数（默认 6）
python -m pipeline.convert_to_graph --case ZHANG_CHUN --k 8

# 处理所有病例
python -m pipeline.convert_to_graph
```

**输出图数据格式**：
```python
Data(
    x,            # 节点特征 [N, 10]
    edge_index,   # 边索引 [2, E]
    y,            # 目标输出 [N, 4]
    global_cond,  # 全局条件 [1, 6]（图级属性）
)
```

**节点特征 x (10维)**：
```
[0:3]  坐标: x, y, z
[3:9]  几何: Abscissa, NormRadius, Curvature, Tangent_X/Y/Z
[9]    边界标志: is_wall
```

**全局条件 global_cond (1x6)**：
```
[0]    时间: t_norm (归一化到 [0,1])
[1:6]  边界条件: BC_Inlet, BC_O1~O4
```

> **v2.1 变更**：t_norm 和 BC 不再作为节点特征逐点存储，改为图级属性 `global_cond`。  
> DataLoader batch 时自动沿 dim=0 拼接：单图 `[1, 6]` → batch 后 `[B, 6]`。  
> 模型中通过 `data.global_cond[data.batch]` 广播到每个节点后拼接。

**目标输出 y (4维)**：
```
[0:3]   速度: u, v, w
[3]     压力: p
```

---

## 数据增强

Pipeline 提供在线数据增强功能，在训练时动态应用，提升模型泛化能力。

### 使用 CFDAugmentedDataset

```python
from pipeline.dataset import CFDAugmentedDataset, CFDDataModule

# 方式1：直接使用数据集
train_dataset = CFDAugmentedDataset(
    root='data_new/AG/fast',
    case_names=['ZHANG_CHUN', 'LI_SI'],
    augment=True,  # 启用增强
    augment_config={
        "rotation_prob": 0.5,      # 旋转概率
        "translation_prob": 0.5,   # 平移概率
        "translation_range": 0.1,  # 平移范围
    }
)

val_dataset = CFDAugmentedDataset(
    root='data_new/AG/fast',
    case_names=['WANG_WU'],
    augment=False,  # 验证集不增强
)

# 方式2：使用 DataModule
dm = CFDDataModule(
    root='data_new/AG/fast',
    train_cases=['ZHANG_CHUN', 'LI_SI'],
    val_cases=['WANG_WU'],
    test_cases=['ZHAO_LIU'],
)
train_loader = dm.train_dataloader(batch_size=32)
val_loader = dm.val_dataloader(batch_size=32)
```

### 支持的增强操作

| 操作 | 函数 | 说明 | 推荐 |
|------|------|------|------|
| 随机旋转 | `random_rotation()` | 绕 x/y/z 轴随机旋转，矢量同步变换 | ✅ 推荐 |
| 随机平移 | `random_translation()` | 在归一化坐标系下平移 | ✅ 推荐 |
| 微小缩放 | `small_scale_augmentation()` | ±2% 缩放 | ⚠️ 慎用 |
| 镜像翻转 | `mirror_augmentation()` | 沿指定轴镜像 | ⚠️ 视情况 |

### 增强配置

```python
DEFAULT_AUGMENT_CONFIG = {
    "rotation_prob": 0.5,       # 旋转概率
    "rotation_axes": "xyz",     # 可旋转的轴
    "translation_prob": 0.5,    # 平移概率
    "translation_range": 0.1,   # 平移范围 [-0.1, 0.1]
    "scale_prob": 0.0,          # 缩放概率（默认关闭）
    "scale_range": (0.98, 1.02),
    "mirror_prob": 0.0,         # 镜像概率（默认关闭）
}
```

### 重要提醒

1. **矢量同步变换**：旋转时，坐标、速度(u,v,w)、切线(Tangent)、标签(y) 会自动同步旋转
2. **验证集不增强**：验证/测试集应设置 `augment=False`
3. **禁止大幅缩放**：会改变雷诺数，导致物理不一致
4. **禁止形状变形**：血管变形后流场完全改变，旧标签不适用

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
    "sampling_method": "hybrid",    # hybrid / fps / random
    "hybrid_fps_ratio": 0.2,        # 混合采样时 FPS 占比
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
    │   └── ...
    ├── features/           # 步骤2: 添加几何特征
    │   ├── result_features_merged-1120.csv
    │   ├── ...
    │   └── bc_metadata.json         # 边界条件元数据（全局条件）
    ├── coord_normalized/   # 步骤3: 坐标系归一化
    │   ├── result_features_merged-1120.csv
    │   ├── ...
    │   ├── transform_params.json    # 坐标变换参数
    │   └── bc_metadata.json         # 边界条件元数据（副本）
    ├── normalized/         # 步骤4: 特征归一化
    │   ├── result_features_merged-1120.csv
    │   ├── ...
    │   └── bc_metadata_normalized.json  # 归一化后的边界条件
    └── graphs/             # 步骤5: 图数据
        ├── result_features_merged-1120.pt
        ├── ...
        ├── transform_params.json         # 变换参数副本
        └── bc_metadata_normalized.json   # BC元数据副本
```

**参数文件说明**：

| 文件 | 位置 | 用途 |
|------|------|------|
| `transform_params.json` | coord_normalized/, graphs/ | 坐标系变换参数，推理时用于逆变换 |
| `normalization_params_global.json` | data_new/ | 全局特征归一化参数 |
| `bc_metadata.json` | features/, coord_normalized/ | 原始边界条件（每时间步 5 个值） |
| `bc_metadata_normalized.json` | normalized/, graphs/ | 归一化后的边界条件 |

---

## 常见问题

### 1. 找不到 vmtk 模块

```bash
pip install vmtk
# 或
conda install -c vmtk vmtk
```

### 2. 内存不足 (FPS 采样)

使用混合采样或随机采样替代：
```bash
# 推荐：混合采样（保留 20% FPS 确保覆盖）
python -m pipeline.preprocess --sampling-method hybrid

# 或：纯随机采样（最快，但可能丢失分支血管）
python -m pipeline.preprocess --sampling-method random
```

### 3. 处理中断，如何继续？

使用 `--start-step` 跳过已完成的步骤：
```bash
python -m pipeline.run_all --case ZHANG_CHUN --start-step 3  # 从步骤3（坐标系归一化）开始
```

### 3.1 如何跳过坐标系归一化？

如果不需要坐标系归一化（不推荐），可以直接从 features 目录读取：
```bash
python -m pipeline.normalize --input-subdir processed/features
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
python -m pipeline.run_all --target-points 50000

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

### 7. ascii 和 ascii_in 文件不对齐怎么办？

Pipeline 只会处理两个目录中**同时存在**的时间步。如果需要清理多余文件，可以使用以下脚本：

```python
import os

case_dir = "data_new/AG/fast/病例名"
ascii_dir = f"{case_dir}/ascii"
ascii_in_dir = f"{case_dir}/ascii_in"

# 获取文件编号
ascii_files = {f.split('-')[-1]: f for f in os.listdir(ascii_dir) if '-' in f}
ascii_in_numbers = set(f.split('-')[-1] for f in os.listdir(ascii_in_dir) if '-' in f)

# 删除 ascii 中多余的文件（不在 ascii_in 中）
for number, filename in ascii_files.items():
    if number not in ascii_in_numbers:
        os.remove(os.path.join(ascii_dir, filename))
        print(f"已删除: {filename}")
```

> **注意**: 通常 `ascii_in/` 的采样频率较低（如每隔一帧），需要以 `ascii_in/` 为基准，删除 `ascii/` 中多余的文件。

---

## 完整示例

### 数据处理

```bash
# 1. 进入 pipeline 目录
cd <repo-root>

# 2. 先测试单个病例（完整 5 步流程）
python -m pipeline.run_all --case ZHANG_CHUN

# 3. 检查输出
ls ../data_new/AG/fast/ZHANG_CHUN/processed/graphs/
ls ../data_new/AG/fast/ZHANG_CHUN/processed/coord_normalized/transform_params.json

# 4. 确认无误后，处理所有病例
python -m pipeline.run_all

# 5. 完成后，图数据可用于 GNN 训练
# 图数据位于: data_new/AG/fast/*/processed/graphs/*.pt
```

### 训练时使用在线增强

```python
from pipeline.dataset import CFDAugmentedDataset
from torch_geometric.loader import DataLoader

# 加载数据（训练集启用增强）
train_dataset = CFDAugmentedDataset(
    root='data_new/AG/fast',
    case_names=['ZHANG_CHUN', 'LI_SI', 'WANG_WU'],
    augment=True,
)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 训练循环
for batch in train_loader:
    # batch.x: [B*N, 10] 节点特征
    # batch.global_cond: [B, 6] 全局条件 (t_norm + BC)
    # batch.y: [B*N, 4] 目标输出
    # batch.edge_index: [2, E] 边索引
    # batch.batch: [B*N] 节点所属图索引
    #
    # 模型 forward 中广播全局条件:
    #   gc = batch.global_cond[batch.batch]  # [B*N, 6]
    #   x_in = torch.cat([batch.x, gc], dim=-1)  # [B*N, 16]
    ...
```

### 推理时还原坐标系

```python
import json
import numpy as np
from pipeline.coord_normalize import inverse_transform

# 加载变换参数
with open('transform_params.json') as f:
    params = json.load(f)

# 逆变换预测结果
coords_orig, velocity_orig = inverse_transform(
    coords_pred,    # 模型预测的坐标
    velocity_pred,  # 模型预测的速度
    params
)
```
