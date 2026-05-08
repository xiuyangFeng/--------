# Training 集群运行指南

## 目录结构

```text
cluster/
├── run_train_field.slurm       # 单配置训练
├── run_v3_diag00.slurm         # V3P-Diag-00：run_v3_diag00.py（诊断产出 §9.1.1）
├── run_v3_diag00_train_chain.slurm  # 诊断脚本 → train_field 同一 JSON（推荐首开链路）
├── run_plan.slurm              # 顺序执行 manifest
├── run_array.slurm             # Array Job 并行执行多个配置
├── run_compare_hemo_wss.slurm   # 多 run WSS 对比（compare_hemo_wss_runs）
├── run_wss_multitask_predict_figs_array.slurm  # WSS 多任务 15 run：删旧 predictions_test → predict → A3/A4/误差/A5
├── manifest_list_wss_multitask_predict.tsv     # 上述 array 的 run 目录清单（每行一条，相对仓库根）
├── hemo_wss_runs_A_Opt03_04_05_seed1.tsv  # 示例：A-Opt-03/04/05 seed1 的 manifest 列表
├── batch_submit.sh             # 批量提交脚本
├── generate_manifest_list.sh   # 从 manifest 生成配置列表
├── logs/                       # 日志目录（自动创建）
└── README.md
```

## 适用入口

- 单个配置训练：`python -m training.scripts.train_field`
- 按 manifest 顺序批量执行：`python -m training.scripts.run_field_plan`
- 多 run 壁面 WSS 对比（需已跑过 `predict_field`）：`python -m training.scripts.compare_hemo_wss_runs`

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

### 2b. V3P-Diag-00（诊断脚本 + 可选紧随 train）

先在 GPU 节点产出 `outputs/field/diagnostics/v3p_diag00_seed*/`，再跑与诊断相同的 JSON（默认 Diag-00 为 1 epoch）：

```bash
cd /path/to/GNN

# 推荐：诊断（含前向）→ train_field 一单串联
sbatch training/cluster/run_v3_diag00_train_chain.slurm \
  training/configs/field/generated/v3_pointcloud/V3P-Diag-00_seed1.json

# 只要诊断、不训练
RUN_TRAIN=0 sbatch training/cluster/run_v3_diag00_train_chain.slurm \
  training/configs/field/generated/v3_pointcloud/V3P-Diag-00_seed1.json

# 只要诊断脚本（与 chain 的第一步等价）
sbatch training/cluster/run_v3_diag00.slurm \
  training/configs/field/generated/v3_pointcloud/V3P-Diag-00_seed1.json

# 诊断阶段仅 CPU 统计、跳过模型前向
SKIP_DIAG_FORWARD=1 sbatch training/cluster/run_v3_diag00_train_chain.slurm ...
SKIP_FORWARD=1 sbatch training/cluster/run_v3_diag00.slurm ...
```

日志：`logs/v3_diag00_<JOBID>.out` / `logs/v3_diag00_train_<JOBID>.out`（若在仓库根提交）。

如需显式指定环境或 Python：

```bash
TRAINING_ENV=GNN sbatch run_train_field.slurm training/configs/field/mlp_baseline.json

TRAINING_PYTHON=/public/newhome/cy/.conda/envs/GNN/bin/python \
sbatch run_train_field.slurm training/configs/field/transformer_geometry.json
```

`generate_manifest_list.sh` 和 `batch_submit.sh` 现在也会复用同一套 `TRAINING_ENV` / `TRAINING_PYTHON`，避免登录节点和计算节点解释器不一致。

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

### 5. 多 run WSS 对比（任务 A / Line W）

在 **计算节点** 上跑 `compare_hemo_wss_runs`，避免登录节点长时间占核。依赖各实验目录下已有 `predictions_test/manifest.json`（或你改为 `predictions_val` 则在 TSV 里写对应路径）。

**Run 列表**：TSV 两列，制表符分隔——**第 1 列**为对比用标签（如 `A-Opt-03`），**第 2 列**为 `manifest.json` 的**相对仓库根目录**路径。以 `#` 开头的行视为注释。可参考 `training/cluster/hemo_wss_runs_A_Opt03_04_05_seed1.tsv` 复制修改为其它实验。

```bash
cd /path/to/GNN

# 默认使用示例 TSV（A-Opt-03/04/05 seed1）
sbatch training/cluster/run_compare_hemo_wss.slurm

# 指定自有 TSV
sbatch training/cluster/run_compare_hemo_wss.slurm training/cluster/my_wss_runs.tsv
```

日志路径与 **提交时当前目录** 有关：`#SBATCH --output=logs/%x_%j.out` 写在仓库根执行时，一般为 **`GNN/logs/field_hemo_wss_<JOBID>.out`**；若在 `training/cluster` 下执行 `sbatch run_compare_hemo_wss.slurm`，则落在 **`training/cluster/logs/`**。

**环境变量**（`VAR=value sbatch ...` 或在脚本内 export）：

| 变量 | 含义 |
|------|------|
| `HEMO_OUTPUT_DIR` | 输出目录（相对仓库根）；默认 `outputs/field/plots/optimization/wss_compare_<JOBID>` |
| `HEMO_MAX_ITEMS` | 只处理每个 manifest 前 N 条（试跑）；不设或 `0` 为全量 |
| `HEMO_NO_EXPORT=1` | 不执行 `export_hemo`（仅用已有 `hemo_ai`/`hemo_cfd` 做汇总） |
| `TRAINING_ENV` / `TRAINING_PYTHON` | 与训练脚本相同 |

**示例**（试跑 20 条 + 固定输出目录名）：

```bash
HEMO_MAX_ITEMS=20 HEMO_OUTPUT_DIR=outputs/field/plots/optimization/wss_A_Opt03_04_05_seed1_n20 \
  sbatch training/cluster/run_compare_hemo_wss.slurm
```

**查看日志与结果**：

```bash
squeue -u $USER
# 若在仓库根 sbatch：
tail -f logs/field_hemo_wss_<JOBID>.out
# 完成后（HEMO_OUTPUT_DIR 未改时，默认带作业号子目录）
cat outputs/field/plots/optimization/wss_compare_<JOBID>/wss_compare_summary.json
```

说明：作业默认 `#SBATCH -w node03`、`--partition=CPU`、`--cpus-per-task=8`、 walltime `12:00:00`。**全量 test + export_hemo 可能极慢**，可加大 `--time` 或先用 `HEMO_MAX_ITEMS` 试通；正式论文表建议全量且 `--export` 重导 hemo。`export_hemo` 已带 **tqdm** 进度（加载 `.pt`、病例循环、时间步）；纯日志环境可加 `--no-progress`。

### 5b. WSS 多任务 15 run：`predict_field` + A3/A4/误差分析/Fig A5

针对 `experiment_index` 中 **A-Base-01/02/03、A-Main-01、A-Opt-05** 的 **wss-multi** 共 **15** 个 `run_dir`：**先删除** 各 run 下已有 `predictions_test`（避免不完整导出残留），再导出测试集预测，并对该 `manifest.json` 串联 `plot_taskA_scatter`、`plot_taskA_per_case_boxplot`、`plot_error_analysis`、`plot_taskA_regional_bar`；脚本内校验 `.pt` 数量与 `num_predictions` 一致。

```bash
cd /path/to/GNN
sbatch training/cluster/run_wss_multitask_predict_figs_array.slurm
# 自定义清单：MANIFEST_LIST_FILE=training/cluster/my_runs.tsv sbatch --array=0-9%4 training/cluster/run_wss_multitask_predict_figs_array.slurm
```

默认 `#SBATCH --array=0-14%4`、**GPU 分区**、`logs/wss_mt_pred_fig_<ARRAY_JOB_ID>_<TASK_ID>.out`。全仓库多模型汇总图未包含在内；需要时可另交 `run_taskA_interior_figs.slurm`。

### 6. 一键批量提交

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

# 让 manifest 生成和训练作业都使用同一个解释器
TRAINING_PYTHON=/public/newhome/cy/.conda/envs/GNN/bin/python \
./batch_submit.sh --array --study-group baseline
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
tail -f logs/field_hemo_wss_<JOB_ID>.out

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
python -m training.scripts.train_field --config <config_path>
```

## 常见问题

### 1. manifest 不存在

先在仓库根目录生成：

```bash
python -m training.scripts.make_field_plan \
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
