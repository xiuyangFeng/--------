# Training 集群运行指南

## 目录结构

```text
cluster/
├── run_train_field.slurm    # 单配置训练
├── run_plan.slurm           # 顺序执行 manifest
├── run_array.slurm          # Array Job 并行执行多个配置
├── batch_submit.sh          # 批量提交脚本
├── generate_manifest_list.sh # 从 manifest 生成配置列表
├── logs/                    # 日志目录（自动创建）
└── README.md
```

## 适用入口

- 单个配置训练：`python -m training.train_field`
- 按 manifest 顺序批量执行：`python -m training.run_field_plan`

这套集群脚本只负责调度，不复制训练逻辑。

## 默认集群配置

当前脚本默认沿用 `pipeline/cluster` 的风格：

- 分区：`CPU`
- 默认 Conda 环境：`GNN`
- 单实验资源：`8` 核

如果你的训练要跑 GPU，建议直接在提交时覆盖资源，而不是改 Python 代码：

```bash
sbatch -p GPU --gres=gpu:1 run_train_field.slurm training/configs/field/transformer_geometry.json
```

只要 `config.system.device=auto`，拿到 GPU 资源后训练会自动选 CUDA。

## 快速开始

### 1. 配置权限

```bash
cd training/cluster
chmod +x *.sh
```

### 2. 运行单个实验

```bash
cd training/cluster
sbatch run_train_field.slurm training/configs/field/transformer_geometry.json
```

如需显式指定环境或 Python：

```bash
TRAINING_ENV=GNN sbatch run_train_field.slurm training/configs/field/mlp_baseline.json

TRAINING_PYTHON=/public/newhome/cy/.conda/envs/GNN/bin/python \
sbatch run_train_field.slurm training/configs/field/transformer_geometry.json
```

### 3. 顺序跑一整份 manifest

```bash
sbatch run_plan.slurm training/configs/field/generated/manifest.json

# 只跑 baseline 组
sbatch run_plan.slurm training/configs/field/generated/manifest.json baseline

# 只跑某个实验，且最多执行 4 个配置
sbatch run_plan.slurm training/configs/field/generated/manifest.json baseline A-Main-01 "" 4
```

如果只想检查命令，不真正执行：

```bash
DRY_RUN=1 sbatch run_plan.slurm training/configs/field/generated/manifest.json baseline
```

### 4. Array Job 并行跑多个实验

先生成配置列表：

```bash
./generate_manifest_list.sh

# 只导出 baseline 组
./generate_manifest_list.sh training/configs/field/generated/manifest.json baseline
```

再提交 Array Job：

```bash
sbatch --array=0-11%4 run_array.slurm
```

说明：

- `0-11` 表示共 12 个任务
- `%4` 表示最多同时跑 4 个任务
- 配置列表默认写到 `training/cluster/manifest_list.tsv`

### 5. 一键批量提交

独立作业模式：

```bash
./batch_submit.sh
./batch_submit.sh --study-group baseline --limit 4
./batch_submit.sh --study-group baseline --exp-id A-Main-01 --seed 2
```

Array Job 模式：

```bash
./batch_submit.sh --array
MAX_PARALLEL=6 ./batch_submit.sh --array --study-group baseline
```

## 常用环境变量

- `TRAINING_ENV`：默认 `GNN`
- `TRAINING_PYTHON`：直接指定 Python 解释器，优先级高于 `TRAINING_ENV`
- `MANIFEST_LIST_FILE`：覆盖 Array Job 使用的配置列表路径
- `MAX_PARALLEL`：`batch_submit.sh --array` 的最大并发数，默认 `4`
- `DRY_RUN`：给 `run_plan.slurm` 传 `1` 时只打印命令

## 监控作业

```bash
squeue -u $USER

tail -f logs/field_train_<JOB_ID>.out
tail -f logs/field_plan_<JOB_ID>.out
tail -f logs/field_array_<JOB_ID>_<TASK_ID>.out

scancel <JOB_ID>
scancel <ARRAY_JOB_ID>
```

## 配置列表格式

`generate_manifest_list.sh` 输出的是一个四列 TSV：

```text
exp_id    study_group    seed    config_path
```

`run_array.slurm` 逐行读取它，并对每一行执行：

```bash
python -m training.train_field --config <config_path>
```

## 常见问题

### 1. manifest 不存在

先在仓库根目录生成：

```bash
python -m training.make_field_plan \
  --data-root /path/to/data_root \
  --split-file training/splits/split_v1.json \
  --output-dir training/configs/field/generated
```

### 2. 节点上没有 conda 命令

在脚本里按集群实际情况补：

```bash
module load anaconda3
source ~/.bashrc
```

### 3. 训练想走 GPU

提交时显式覆盖：

```bash
sbatch -p GPU --gres=gpu:1 run_train_field.slurm training/configs/field/transformer_geometry.json
```

### 4. Array Job 某个任务失败

```bash
cat logs/field_array_<JOB_ID>_<TASK_ID>.err
```

然后把对应的 `config_path` 单独拿出来重新提交：

```bash
sbatch run_train_field.slurm training/configs/field/generated/baseline/A-Main-01_seed1.json
```
