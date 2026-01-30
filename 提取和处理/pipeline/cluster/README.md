# 集群运行指南

## 目录结构

```
cluster/
├── run_pipeline.slurm      # 完整流程 SLURM 脚本
├── run_single_step.slurm   # 单步骤 SLURM 脚本
├── batch_submit.sh         # 批量提交脚本
├── logs/                   # 日志目录（自动创建）
└── README.md               # 本文件
```

## 快速开始

### 1. 配置环境

编辑 SLURM 脚本中的环境配置部分：

```bash
# 根据你的集群修改
#SBATCH --partition=compute    # 分区名称
source ~/.bashrc
conda activate rag_venv        # conda 环境名称
```

### 2. 提交作业

```bash
cd pipeline/cluster

# 处理单个病例（推荐先测试）
sbatch run_pipeline.slurm ZHANG_CHUN

# 处理所有病例
sbatch run_pipeline.slurm

# 批量提交多个病例（每个病例一个作业）
chmod +x batch_submit.sh
./batch_submit.sh

# 从步骤2开始（跳过预处理）
sbatch run_pipeline.slurm ZHANG_CHUN 2

# 只运行步骤1（预处理）
sbatch run_pipeline.slurm ZHANG_CHUN 1 1
```

### 3. 单步骤运行

如果需要分步骤运行：

```bash
# 步骤1: 数据预处理
sbatch run_single_step.slurm preprocess ZHANG_CHUN

# 步骤2: 几何特征提取
sbatch run_single_step.slurm extract_features ZHANG_CHUN

# 步骤3: 归一化
sbatch run_single_step.slurm normalize ZHANG_CHUN

# 步骤4: 图数据转换
sbatch run_single_step.slurm convert_to_graph ZHANG_CHUN
```

## 监控作业

```bash
# 查看作业队列
squeue -u $USER

# 实时查看输出
tail -f logs/pipeline_<JOB_ID>.out

# 取消作业
scancel <JOB_ID>

# 查看作业详情
scontrol show job <JOB_ID>
```

## 资源配置建议

| 步骤 | CPU | 内存 | 预估时间（单病例） |
|------|-----|------|-------------------|
| preprocess | 4-8 | 16G | 10-30 分钟 |
| extract_features | 2-4 | 8G | 30-60 分钟 |
| normalize | 2-4 | 8G | 5-10 分钟 |
| convert_to_graph | 4-8 | 16G | 10-20 分钟 |

完整流程建议：8 CPU，32G 内存，2-4 小时

## 常见问题

### 1. 模块加载失败

在 SLURM 脚本中添加：
```bash
module load anaconda3
module load vtk  # 如果需要
```

### 2. 内存不足

增加 `#SBATCH --mem=64G`

### 3. VMTK 报错

确保 conda 环境中安装了 vmtk：
```bash
conda activate rag_venv
pip install vmtk
```

### 4. 查看详细日志

```bash
# 查看标准输出
cat logs/pipeline_<JOB_ID>.out

# 查看错误输出
cat logs/pipeline_<JOB_ID>.err
```
