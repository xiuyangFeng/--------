# 集群运行指南（Pipeline · 通用入口）

本目录仅保留 **五步预处理主链路** 的通用 SLURM / 批提交脚本。历史 campaign、V3 数据批处理、Fluent 等见 **`../archive/README.md`**。

## 活跃目录结构

```text
cluster/
├── run_pipeline.slurm       # 单病例完整流程
├── run_single_step.slurm    # 单步骤
├── run_array.slurm          # Array Job（多病例并行）
├── run_renorm_regraph.slurm # 按 train split 重算归一化 + 转图
├── batch_submit.sh          # 批量提交
├── generate_case_list.sh    # 生成病例列表
├── logs/                    # SLURM 日志（.gitignore，勿提交）
└── README.md
```

## 归档入口（`pipeline/archive/`）

| 脚本 | 用途 |
| --- | --- |
| `onetime_batch_jobs/batch_preprocess_gap.sh` | 缺口队列步骤 1 预处理 |
| `onetime_batch_jobs/batch_aaa_ilo_steps_2_5.sh` | AAA+ILO 步骤 2–5 |
| `onetime_batch_jobs/run_v3_prepool_audit.slurm` | 刷新 `cases_data_new_v3_candidate_pool.txt` |
| `onetime_batch_jobs/run_v3_split_data_new_renorm_regraph.slurm` | 旧版 v3 单次 renorm 模板（宜用主目录 `run_renorm_regraph.slurm`） |
| `cluster_legacy/fluent.slurm` | Fluent CFD（与 Python 流水线无集成） |

归档批处理脚本会调用本目录的 `run_array.slurm`；请在 **`pipeline/archive/onetime_batch_jobs/`** 下执行对应 `.sh`。

## 集群配置

- **分区**: CPU（默认）
- **Conda**: 主环境 `GNN`；步骤 2 几何 `GNN_vmtk`
- **CPU 重任务**: 规范要求指定 **node03**（在脚本中 `#SBATCH -w node03`）

## 快速开始

### 审计（提交前）

```bash
cd <repo-root>
conda activate GNN
python -m pipeline.audit_inputs --groups AAA AG ILO
```

### Array Job（推荐）

```bash
cd pipeline/cluster
./batch_submit.sh --array
./batch_submit.sh --array --start-step 2 --max-parallel 4
```

### 单病例 / 单步骤

```bash
sbatch run_pipeline.slurm ZHANG_CHUN
sbatch run_single_step.slurm preprocess ZHANG_CHUN
```

### 按 split 重归一化 + 转图

```bash
cd <repo-root>
TRAIN_SPLIT=training/splits/split_data_new_v3.json \
  sbatch pipeline/cluster/run_renorm_regraph.slurm
```

### V3 候选池终审（归档脚本）

```bash
cd pipeline/archive/onetime_batch_jobs
sbatch run_v3_prepool_audit.slurm
```

## 监控

```bash
squeue -u $USER
tail -f pipeline/cluster/logs/gnn_array_<JOB_ID>_<TASK>.out
```

作业日志目录已加入 `.gitignore`（`pipeline/cluster/logs/`），勿将 `*.out` / `*.err` 提交进 Git。

## 时间估算（单病例）

| 步骤 | 预估 |
| --- | --- |
| preprocess | 15–30 min |
| extract_features | 30–60 min |
| normalize + convert_to_graph | 15–30 min |
| **完整流程** | **约 1–2 h** |

更细的 Array 并行估算见历史版 README 或 `../archive/README.md`。
