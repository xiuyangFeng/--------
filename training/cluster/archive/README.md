# Training 集群脚本归档区

与 **`training/cluster/` 主目录通用入口**（训练 / plan / array / 分域评估）无关，或已为**单次实验 / 历史 campaign** 使用过的 SLURM、清单、提交包装脚本，集中放在此目录，避免主目录噪音。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `onetime_slurm/` | 一次性或历史阵列作业：预测导出、WSS 多任务图件、hemo 对比、V3 诊断链、双 checkpoint 重算 test 等 |
| `manifests/` | 上述阵列作业用的 **run 目录清单**（`.tsv`，每行相对仓库根） |
| `lists/` | 其他 run 列表（如 Fig A6 消融 `--run-list`） |

## 仍留在 `training/cluster/` 的通用入口

| 脚本 | 用途 |
| --- | --- |
| `run_train_field.slurm` | 单配置训练 |
| `run_plan.slurm` | 按 manifest 顺序训练 |
| `run_array.slurm` | 多配置 Array Job |
| `batch_submit.sh` / `generate_manifest_list.sh` | 批量提交与清单生成 |
| `run_evaluate_field_run_full.slurm` | 单 run 完整后处理评估 |
| `run_eval_field_by_domain.slurm` | 按 AAA/AG/ILO 分域 test 指标 |
| `submit_eval_by_domain.sh` | 上述分域评估的提交包装（**推荐**） |

## 复用归档脚本时注意

1. **清单路径**：默认改为 `$ARCHIVE_DIR/manifests/...`；也可 `MANIFEST_LIST_FILE=... sbatch ...` 覆盖。
2. **仓库根**：归档 SLURM 内已按 `archive/onetime_slurm` → `training/cluster` → 仓库根解析 `PROJECT_DIR`。
3. **日志**：`#SBATCH --output=logs/...` 在**仓库根** `sbatch` 时落在 `GNN/logs/`；在 `onetime_slurm/` 下提交则落在该子目录 `logs/`。
4. **WSS 对比 / 批量 hemo**：`run_compare_hemo_wss.slurm` 等须用户**明确要求**后再跑（见仓库 WSS 规则）。

## 一次性提交示例（V3D 探针，已跑过）

```bash
bash training/cluster/archive/onetime_slurm/submit_v3d_probe_eval_by_domain.sh
```

等价通用写法：

```bash
RUN_DIR_REL=outputs/field/<run> CHECKPOINT=best_model.pt \
  bash training/cluster/submit_eval_by_domain.sh
```
