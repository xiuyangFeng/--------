# 集群运行指南（Training · 通用入口）

本目录仅保留 **训练与通用评估** 的 SLURM / 批提交脚本。历史预测阵列、WSS 对比、V3 诊断链、论文 Fig 批跑等见 **`archive/README.md`**。

## 活跃目录结构

```text
cluster/
├── run_train_field.slurm          # 单配置训练
├── run_plan.slurm                 # manifest 顺序训练
├── run_array.slurm                # 多配置 Array Job
├── run_evaluate_field_run_full.slurm  # 单 run 完整后处理
├── run_eval_field_by_domain.slurm     # AAA/AG/ILO 分域 test
├── submit_eval_by_domain.sh       # 分域评估提交包装（推荐）
├── batch_submit.sh
├── generate_manifest_list.sh
├── logs/                          # .gitignore
└── README.md
```

## 归档（`training/cluster/archive/`）

| 类型 | 示例 |
| --- | --- |
| `onetime_slurm/` | `run_wss_multitask_predict_figs_array.slurm`、`run_field_predict_test_array.slurm`、`run_v3_diag00*.slurm`、`submit_v3d_probe_eval_by_domain.sh` |
| `manifests/` | `manifest_list_v3_mainline_seed1_predict.tsv` 等 |
| `lists/` | `run_list_figA6_geometry_opt05_mean3seed.txt` |

## 默认配置

- 分区：脚本默认 **CPU**；GPU 训练提交时覆盖：  
  `sbatch -p GPU --gres=gpu:1 run_train_field.slurm <config.json>`
- Conda：**GNN**（`TRAINING_ENV` / `TRAINING_PYTHON` 可覆盖）

## 快速开始

### 单实验训练

```bash
cd <repo-root>
sbatch training/cluster/run_train_field.slurm \
  training/configs/field/generated/v3_pointcloud/V3P-Main-01_seed1.json
```

### V1 PINN 阶梯消融（2026-06）

```bash
cd <repo-root>
bash training/cluster/submit_v1_pinn_ladder.sh   # seed1：cont / cont+noslip / full 三作业
squeue -u $USER
tail -f logs/v1pinn_cont_<JOB_ID>.out            # 日志在仓库根 logs/
```

### manifest 顺序 / Array

```bash
cd training/cluster
sbatch run_plan.slurm training/configs/field/generated/manifest.json
./generate_manifest_list.sh
sbatch --array=0-11%4 run_array.slurm
```

### 分域 test 评估（V3D 门禁等）

```bash
cd <repo-root>
RUN_DIR_REL=outputs/field/<run_dir> CHECKPOINT=best_model.pt \
  bash training/cluster/submit_eval_by_domain.sh
```

产出：`<run_dir>/eval_by_domain_test/metrics_by_domain.json`

### 单 run 完整评估（predict + A3/A4 + WSS 可信度等）

```bash
RUN_DIR_REL=outputs/field/<run_dir> \
  sbatch training/cluster/run_evaluate_field_run_full.slurm
```

## 监控

```bash
squeue -u $USER
tail -f logs/field_train_<JOB_ID>.out    # 仓库根提交时
tail -f training/cluster/logs/field_train_<JOB_ID>.out
```

## 常见问题

1. **manifest 不存在**：在仓库根生成 field plan / manifest 后再 `generate_manifest_list.sh`。
2. **GPU**：配置里 `device=auto` 且提交时申请 `--gres=gpu:1` 即可。
3. **WSS 批量对比 / hemo**：归档脚本 `archive/onetime_slurm/run_compare_hemo_wss.slurm`；**须明确要求后再跑**。

详细历史用法（V3 Diag 链、15-run 图件阵列等）见 **`archive/README.md`** 与各归档脚本头部注释。
