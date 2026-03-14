# 集群运行指南

## 目录结构

```
cluster/
├── run_pipeline.slurm      # 完整流程 SLURM 脚本
├── run_single_step.slurm   # 单步骤 SLURM 脚本
├── run_array.slurm         # Array Job 脚本（并行处理多病例）
├── batch_submit.sh         # 批量提交脚本
├── generate_case_list.sh   # 生成病例列表
├── logs/                   # 日志目录（自动创建）
└── README.md               # 本文件
```

## 集群配置

当前配置基于以下集群环境：
- **分区**: CPU
- **每节点核心数**: 192
- **主 Conda 环境**: `GNN`
- **几何 Conda 环境**: `GNN_vmtk`

## 快速开始

### 1. 配置权限

```bash
cd pipeline/cluster
chmod +x *.sh
```

### 2. 提交作业

正式提交前，建议先在登录节点做一次输入审计，避免把缺 `stl` / `ascii` / `ascii_in` / `Global_conditions` 的病例直接送进队列：

```bash
cd <repo-root>
conda activate GNN
python -m pipeline.audit_inputs --groups AAA AG ILO
```

#### 方式一：Array Job（推荐，并行处理多个病例）

```bash
# 自动扫描所有病例并提交 Array Job
./batch_submit.sh --array

# 指定病例
./batch_submit.sh --array ZHANG_CHUN LI_MING WANG_WEI

# 只从步骤 2 跑到步骤 5，并行度改成 4
./batch_submit.sh --array --start-step 2 --max-parallel 4

# Array Job 下切换采样策略
./batch_submit.sh --array --sampling-method fps

# 显式指定双环境
PIPELINE_ENV=GNN GEOMETRY_ENV=GNN_vmtk ./batch_submit.sh --array
```

**Array Job 优势**：
- 单节点 192 核可同时处理 ~6 个病例（每个32核）
- 统一管理，方便监控和取消
- 资源利用率高

#### 方式二：独立作业模式

```bash
# 处理单个病例（推荐先测试）
sbatch run_pipeline.slurm ZHANG_CHUN

# 批量提交（每个病例一个独立作业）
./batch_submit.sh
./batch_submit.sh ZHANG_CHUN LI_MING

# 独立作业模式同样支持步骤范围和采样参数
./batch_submit.sh --start-step 2 --end-step 5 --sampling-method hybrid --fps-ratio 0.3 ZHANG_CHUN

# 显式指定 geometry-python
GEOMETRY_PYTHON=/public/newhome/cy/.conda/envs/GNN_vmtk/bin/python \
./batch_submit.sh ZHANG_CHUN
```

默认行为：
- `run_pipeline.slurm` / `run_array.slurm` 在 `GNN` 环境中启动
- 若检测到 `$HOME/.conda/envs/GNN_vmtk/bin/python`，会自动把步骤2 `extract_features` 切到该解释器执行
- 如需覆盖，可在提交前设置 `PIPELINE_ENV`、`GEOMETRY_ENV` 或 `GEOMETRY_PYTHON`
- `batch_submit.sh` 会把 `--start-step`、`--end-step`、`--sampling-method`、`--fps-ratio`、`--allow-nearest-bc` 同时透传给独立作业和 Array Job

示例：

```bash
GEOMETRY_PYTHON=/public/newhome/cy/.conda/envs/GNN_vmtk/bin/python \
sbatch run_pipeline.slurm ZHANG_CHUN
```

### 3. 单步骤运行

```bash
# 步骤1: 数据预处理（清洗+合并+降采样）
sbatch run_single_step.slurm preprocess ZHANG_CHUN

# 步骤2: 几何特征提取（中心线+边界条件）
sbatch run_single_step.slurm extract_features ZHANG_CHUN

# 步骤3: 坐标系归一化
sbatch run_single_step.slurm coord_normalize ZHANG_CHUN

# 步骤4: 特征归一化
sbatch run_single_step.slurm normalize ZHANG_CHUN

# 步骤5: 图数据转换
sbatch run_single_step.slurm convert_to_graph ZHANG_CHUN
```

`run_single_step.slurm` 会自动选择环境：
- `extract_features` 使用 `GNN_vmtk`
- 其他步骤使用 `GNN`

### 4. 从指定步骤开始

```bash
# 从步骤2开始（跳过预处理）
sbatch run_pipeline.slurm ZHANG_CHUN 2

# 只运行步骤1-2
sbatch run_pipeline.slurm ZHANG_CHUN 1 2

# 允许 extract_features 在 BC 缺失时使用最近时间步兜底
ALLOW_NEAREST_BC=1 sbatch run_pipeline.slurm ZHANG_CHUN
```

批量入口也支持同样的控制项：

```bash
./batch_submit.sh --array --start-step 2 --allow-nearest-bc
./batch_submit.sh --end-step 3 --sampling-method random ZHANG_CHUN
```

## 监控作业

```bash
# 查看作业队列
squeue -u $USER

# 实时查看输出
tail -f logs/gnn_pipeline_<JOB_ID>.out
tail -f logs/gnn_array_<JOB_ID>_<TASK_ID>.out  # Array Job

# 查看 pipeline 自己写入的数据目录日志
tail -f ../data_new/pipeline_reports/logs/run_all.log
tail -f ../data_new/AG/fast/ZHANG_CHUN/processed/logs/progress.log

# 取消作业
scancel <JOB_ID>

# 取消所有 Array Job 任务
scancel <ARRAY_JOB_ID>

# 查看作业详情
scontrol show job <JOB_ID>
```

说明：

- `logs/gnn_pipeline_<JOB_ID>.out` / `logs/gnn_array_<JOB_ID>_<TASK_ID>.out` 是 SLURM stdout/stderr
- `data_new/pipeline_reports/logs/*.log` 是 pipeline 的批量/总流程日志
- `<case_dir>/processed/logs/progress.log` 是病例级步骤日志，适合定位某个病例卡在哪个文件、哪个子步骤

## 时间估算（单病例，约80帧数据）

| 步骤 | 描述 | CPU | 预估时间 |
|------|------|-----|----------|
| preprocess | 清洗+合并+降采样 | 16-32 | 15-30 分钟 |
| extract_features | 中心线提取(VMTK)+特征映射 | 16-32 | 30-60 分钟 |
| normalize | 特征归一化 | 8-16 | 5-10 分钟 |
| convert_to_graph | KNN图构建 | 16-32 | 10-20 分钟 |
| **总计** | 完整流程 | 32 | **1-2 小时** |

### 多病例并行估算

| 病例数 | 模式 | 并行度 | 总耗时 |
|--------|------|--------|--------|
| 1 | 单作业 | 1 | 1-2 小时 |
| 6 | Array Job | 6 | 1-2 小时 |
| 10 | Array Job | 6 | 2-4 小时 |
| 20 | Array Job | 6 | 4-8 小时 |

## 资源配置说明

当前配置（192核节点）：
- **run_pipeline.slurm**: 32核/任务，单病例完整流程
- **run_single_step.slurm**: 16核/任务，单步骤运行
- **run_array.slurm**: 32核/任务，同时最多6个任务并行

如需调整，编辑对应脚本中的 `--ntasks-per-node` 参数。

## 常见问题

### 1. 模块加载失败

在 SLURM 脚本中添加：
```bash
module load anaconda3
module load vtk  # 如果需要
```

### 2. 内存不足

脚本未设置内存限制，如遇内存问题可添加：
```bash
#SBATCH --mem=64G
```

### 3. VMTK 报错

确保 `GNN_vmtk` 环境已创建且可导入 `vmtk`：
```bash
conda activate GNN_vmtk
python -c "from vmtk import vmtkscripts; print('vmtk ok')"
```

### 4. Array Job 任务失败

```bash
# 查看失败任务的错误日志
cat logs/gnn_array_<JOB_ID>_<TASK_ID>.err

# 重新提交单个失败的病例
sbatch run_pipeline.slurm <CASE_NAME>
```

### 5. 指定特定节点运行

取消脚本中的注释：
```bash
#SBATCH -w node01
```
