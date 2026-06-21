# Pipeline 归档区

与 **`pipeline/cluster/` 主链路**（`run_pipeline` / `run_array` / `batch_submit` / `run_renorm_regraph`）无关，或已为**单次批处理 / 历史 CFD** 使用过的脚本，集中在此，降低 `pipeline/cluster` 噪音。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `cluster_legacy/fluent.slurm` | Ansys Fluent + 磁盘清理（与 Python GNN 流水线无集成） |
| `workspace_misc/` | 旧 VS Code 工作区占位 |
| `cluster_case_snapshots/` | 单病例调试用小清单（`*_before_only.txt`） |
| `onetime_batch_jobs/` | V3 数据 campaign：缺口预处理、AAA/ILO 2–5、候选池终审、旧版 v3 renorm 模板 |

## `onetime_batch_jobs/` 说明

| 脚本 | 用途 | 提交位置 |
| --- | --- | --- |
| `batch_preprocess_gap.sh` | `export_gap_preprocess_queue` → 步骤 1 阵列 | 在本目录执行；内部 `sbatch` 指向 `pipeline/cluster/run_array.slurm` |
| `batch_aaa_ilo_steps_2_5.sh` | `export_post_preprocess_queue` → 步骤 2–5 | 同上 |
| `run_v3_prepool_audit.slurm` | 刷新 `training/splits/cases_data_new_v3_candidate_pool.txt` | `sbatch run_v3_prepool_audit.slurm`（**node03** / CPU） |
| `run_v3_split_data_new_renorm_regraph.slurm` | 旧版固定 v3 split 的 renorm 模板 | 推荐改用 **`pipeline/cluster/run_renorm_regraph.slurm`** + `TRAIN_SPLIT` |

运行时生成的 `case_list*.txt`、`case_list.txt` **不要**长期留在 Git；日志在 `onetime_batch_jobs/logs/`（已 `.gitignore`）。

## 仍留在 `pipeline/cluster/` 的通用入口

- `run_pipeline.slurm` / `run_single_step.slurm` / `run_array.slurm`
- `batch_submit.sh` / `generate_case_list.sh`
- `run_renorm_regraph.slurm`（任意 train split + `DATA_ROOT`）

## 日志

- `pipeline/cluster/logs/` — 主链路 SLURM 日志
- `pipeline/archive/onetime_batch_jobs/logs/` — 归档批处理日志  

均已通过根 `.gitignore` 忽略，勿提交 `*.out` / `*.err`。
