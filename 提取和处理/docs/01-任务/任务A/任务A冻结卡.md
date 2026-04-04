# 任务A 实验冻结卡

> 本文件记录任务 A 正式实验阶段的不可变配置。  
> 一旦填写完毕并开始正式训练，以下字段**不得中途修改**。  
> 上位文档：[任务A实验清单](任务A实验清单.md) / [任务A配置与启动说明](任务A配置与启动说明.md)

---

## 1. 核心冻结字段

| 字段 | 当前值 | 状态 |
|---|---|---|
| `data_version` | `AG_v1` | ✅ 已确定 |
| `data_root` | `data_new/AG` | ✅ 已确定 |
| `case_name_format` | `{group}/{patient_id}`，如 `fast/HAN_JIAN_JUN` | ✅ 已确定 |
| `graphs_subdir` | `processed/graphs` | ✅ 已确定 |
| `split_version` | `split_AG_v1` | ✅ 已确定 |
| `split_file` | `training/splits/split_AG_v1.json` | ✅ 已生成并用于全部基线训练 |
| `split_protocol` | `single_split` | ✅ 已确定 |
| `preprocess_version` | pipeline 当前版本（采样混合FPS20%，kNN图） | ✅ 已确定 |
| `normalize_source` | 仅训练集统计量 | ✅ 已确定 |
| `seed_plan` | smoke test: seed=1；正式结果: [1, 2, 3] | ✅ 已确定 |
| `primary_metric` | `interior.RMSE_|v|`（内部节点速度模长误差；`all.RMSE_|v|` 仅作补充） | ✅ 已确定（2026-03-26 更新为 interior 口径） |
| `result_root` | `outputs/field/` | ✅ 已确定 |

---

## 2. 数据范围说明

### 2.1 数据来源

| 组别 | 路径 | 有效病例数 | 状态 |
|---|---|---|---|
| fast | `data_new/AG/fast/` | 24 | ✅ 全部完成 |
| slow | `data_new/AG/slow/` | 43（已完成）+ 19（处理中） | ⏳ 等待集群作业 1177 完成 |

> 合并后预计总病例数：**67 ~ 86 个**（取决于 slow 剩余 19 个最终结果）

### 2.2 已知排除病例

| 病例 | 组别 | 原因 |
|---|---|---|
| ZHANG_XIU_ZHEN | fast | 无 log 无 pt 文件，数据缺失 |

---

## 3. 训练通用设置（已固定）

| 参数 | 值 |
|---|---|
| `optimizer` | Adam |
| `lr` | 0.0005 |
| `scheduler` | ReduceLROnPlateau，factor=0.5，patience=10 |
| `epochs` | 200 |
| `early_stopping_patience` | 30 |
| `batch_size` | 2 |
| `grad_clip_norm` | 1.0 |
| `target_weights` | [1.0, 1.0, 1.0, 1.0]（u/v/w/p 等权） |
| `deterministic` | true |

---

## 4. 特征定义（已固定，来自 pipeline/config.py）

### 节点特征 data.x（10维）

| 索引 | 特征名 | 说明 |
|---|---|---|
| [0:3] | x, y, z | 空间坐标 |
| [3] | Abscissa | 沿轴弧长 |
| [4] | NormRadius | 归一化半径 |
| [5] | Curvature | 曲率 |
| [6:9] | Tangent_X/Y/Z | 切向量 |
| [9] | is_wall | 壁面标签（0/1） |

### 图级条件 data.global_cond（6维）

| 索引 | 特征名 | 说明 |
|---|---|---|
| [0] | t_norm | 归一化时间 |
| [1] | BC_Inlet | 入口边界条件 |
| [2:6] | BC_O1~O4 | 出口边界条件 |

### 预测目标 data.y（4维）

| 索引 | 目标名 | 说明 |
|---|---|---|
| [0:3] | u, v, w | 速度分量 |
| [3] | p | 压力 |

---

## 5. 输出目录结构（已固定）

```
outputs/field/
└── {experiment_name}_seed{seed}/
    ├── config.snapshot.json    # 配置副本
    ├── split.snapshot.json     # split 副本
    ├── history.csv             # 训练曲线
    ├── summary.json            # 最优指标
    ├── best_model.pt           # 最优权重
    ├── last_model.pt           # 末轮权重
    ├── predictions.parquet     # 测试集预测结果
    ├── fig_loss.png            # 损失曲线图
    └── fig_scatter.png         # 散点图
```

---

## 6. 执行检查清单（数据就绪后按序执行）

- [x] slow 组全部处理完成（集群作业 1177 结束）
- [x] 排查 slow 剩余 19 个病例是否全部生成 pt 文件
- [x] 生成病例清单文件 `training/splits/cases_AG_v1.txt`
- [x] 执行 `python -m training.scripts.make_split` 生成 `split_AG_v1.json`
- [x] 核对 split 中 train/val/test 病例数比例（4860 / 648 / 1458 graphs）
- [x] 核对 `data.x`、`data.global_cond`、`data.y` 维度
- [x] 执行 `python -m training.scripts.make_field_plan --groups baseline` 生成配置
- [x] 跑通 A-Base-01 smoke test（1 epoch，seed=1）
- [x] 跑通 A-Main-01 smoke test（1 epoch，seed=1）
- [x] 确认冻结卡 split_version 字段已填写
- [x] **A-Base-01 / A-Base-02 / A-Base-03 / A-Main-01 全部 3 seed 训练完成**（2026-03-22/23）

---

## 7. 后续数据就绪后的操作命令（已准备好，等待执行）

### Step 1：生成完整病例清单

```bash
# 数据就绪后，在项目根目录执行
cd /public/newhome/cy/Digital_twin/GNN
python -c "
import os
from pathlib import Path

cases = []
for group in ['fast', 'slow']:
    group_dir = Path(f'data_new/AG/{group}')
    for case_dir in sorted(group_dir.iterdir()):
        pt_files = list((case_dir / 'processed' / 'graphs').glob('*.pt'))
        if len(pt_files) > 0:
            cases.append(f'{group}/{case_dir.name}')

with open('training/splits/cases_AG_v1.txt', 'w') as f:
    f.write('\n'.join(cases))

print(f'共 {len(cases)} 个有效病例')
for c in cases:
    print(c)
"
```

### Step 2：生成患者级 split 文件

```bash
python -m training.scripts.make_split \
  --cases-file training/splits/cases_AG_v1.txt \
  --output training/splits/split_AG_v1.json \
  --split-version split_AG_v1 \
  --source AG \
  --seed 1 \
  --train-ratio 0.7 \
  --val-ratio 0.1 \
  --test-ratio 0.2
```

### Step 3：生成实验配置

```bash
python -m training.scripts.make_field_plan \
  --data-root data_new/AG \
  --split-file training/splits/split_AG_v1.json \
  --groups baseline \
  --output-dir training/configs/field/generated
```

### Step 4：Dry-run 确认配置无误

```bash
python -m training.scripts.run_field_plan \
  --manifest training/configs/field/generated/manifest.json \
  --study-group baseline \
  --dry-run
```

---

## 8. 当前状态

**2026-03-16**：等待 slow 组集群作业 1177 完成（19 个病例处理中/排队中）。  
冻结卡字段除 `split_version` 和 `split_file` 外均已确定，数据就绪后立即执行第 6 节检查清单。

**2026-03-23（更新）**：所有冻结字段已全部确定，第 6 节执行清单已全部完成。  
第一批基线实验（A-Base-01 ~ A-Main-01）全部 3 seed 训练完成，结果已归档至 `outputs/field/`，实验状态详见 [任务A实验状态表](任务A实验状态表.md)。  
当前数据集规模：训练 4860 graphs / 验证 648 graphs / 测试 1458 graphs。  
测试集预测与后处理图已补齐，当前已生成：

- `fig_A3_scatter.png`
- `fig_A4_per_case_boxplot.png`
- `fig_A5_regional_bar_rmse_vel_mag.png`
- `fig_A5_regional_bar_rmse_p.png`
- `fig_error_distribution.png`
- `fig_error_cdf.png`

**（2026-03-24 更新）分区域评估口径已统一**：`plot_taskA_regional_bar` 在聚合指标时从各样本 `graph_path` 读回**未按训练配置 mask 的完整节点特征**生成区域 mask，因此 `A-Base-01` / `A-Base-02` / `A-Base-03` / `A-Main-01` 在 **`high_curvature / near_wall / bifurcation / trunk` 等全部预定义区域**上均为**同一几何定义**，可与模型是否启用几何输入解耦。各 run 已重算 `predictions_test/regional_eval/fig_A5_regional_metrics.json`，汇总层 `outputs/field/plots/multimodel_baseline/fig_A5_multimodel_regional_bar_*.png` 已按该口径更新（集群批处理见 `training/cluster/run_regional_a5.slurm`）。  
**（2026-03-26 补充）** 各区域的默认名称、区间与易混点（采样 2 mm 阈值 vs 评估 `NormRadius` 等）已整理为 [任务A分区域评估口径](../../00-规范与记录/任务A分区域评估口径.md)。  
效率 benchmark 已补齐，当前 `outputs/field/plots/efficiency/` 下已新增：

- `fig_A7_efficiency_benchmark.json`
- `fig_A7_efficiency_bars.png`
- `fig_A7_pareto_rmse_vel_mag_vs_latency.png`

当前 benchmark 口径已升级为：4 个 baseline 的 **seed 1/2/3 共 12 个 run**，测试病例 `slow/GUO_XI_JIANG`（81 snapshots），`n_warmup=5`，`n_runs=20`。当前效率图既包含 `aggregated` 的 `mean±std`，也包含 `rows_per_seed` 的分 seed 结果。结果显示：

- `A-Base-01` 最快：`0.54 ± 0.27 ms / snapshot`，`127.34 ± 0.00 MB`
- `A-Base-02` 提供较好的折中：`2.35 ± 0.23 ms / snapshot`，`529.69 ± 0.47 MB`
- `A-Base-03` 与 `A-Main-01` 时延和显存几乎相同：`6.95 ± 0.09` vs `6.88 ± 0.02 ms / snapshot`，显存约 `2.18 GB`
- `A-Main-01` 在几乎不增加部署开销的前提下，相比 `A-Base-03` 显著降低了 `RMSE_|v|`

新增的效率图现已不止主柱图和主 Pareto 图，还包括（均在 **`plots/efficiency/`**）：

- `fig_A7_efficiency_bars_mean_std.png`
- `fig_A7_latency_per_seed.png`
- `fig_A7_peak_memory_per_seed.png`
- `fig_A7_fullcase_peak_memory_per_seed.png`
- `fig_A7_pareto_per_seed_points.png`
- `fig_A7_pareto_rmse_vel_mag_vs_latency_mean_std.png`

下一步：**区域标签评估口径已统一**，可启动 **A-Abl-01**（输入特征消融）；效率证据层面当前已经具备 3-seed 版本，后续若还要继续加固，可再补多病例 benchmark 或给出 CFD 时间基线以填写 `speedup_vs_CFD`。

**（2026-03-26 补充）P0-1 / `A-Opt-01`**：`target_weights=[2,2,2,0.5]` 三 seed 已完成训练；测试集预测与 **`predictions_test/regional_eval/`** 已补全，多模型 Fig A5 汇总图已包含该 `exp_id`。模型结构仍与主线 `A-Main-01` 一致（无 Pre-Norm 时的 `FieldTransformer`）；若启用 Pre-Norm（`A-Opt-02`），须使用配置 **`use_transformer_prenorm: true`** 并**从头训练**，不得与旧 `best_model.pt` 混用。  
**（2026-03-27 补充）P0-2 / `A-Opt-02`**：Pre-Norm **`FieldTransformer`** 三 seed 已完成（`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_split_AG_v1_seed{1,2,3}_20260327_*`）；已补 **`predictions_test/`**、**`error_analysis_interior/`**、**`regional_eval/`**；与 **`A-Main-01`** 同 seed 的内部节点 |v|/p 对照散点见 **`outputs/field/plots/optimization/prenorm_A_Opt02_vs_Main01/`**。数值摘要见 [任务A实验状态表](任务A实验状态表.md)「A-Opt-02」与 [任务A优化路径与近期实验建议](任务A优化路径与近期实验建议.md) P0-2 节。  
**（2026-03-27 更新）P0-3 / `A-Opt-02_warmup` 已归档**：三 seed run 位于 `outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_warmup5_split_AG_v1_seed{1,2,3}_20260327_*`；已具备 **`predictions_test/`**、**`error_analysis_interior/`**、**`regional_eval/`**。与 **`A-Main-01` / `A-Opt-02`** 同 seed 的 **内部节点** |v|/p 散点、Fig A5 区域柱、Fig A4 per-case 箱线见 **`outputs/field/plots/optimization/prenorm_Main_P02_P02w/`**（一键重生成：`python -m training.scripts.regenerate_p02_warmup_comparison_figures`）。数值与叙事见 [任务A实验状态表](任务A实验状态表.md)「A-Opt-02_warmup」与 [任务A优化路径与近期实验建议](任务A优化路径与近期实验建议.md) P0-3 节。  
**（2026-03-28 同步）P0 组合线已归档**：**`A-Opt-03`**（`A-Opt-01` 损失权重 + Pre-Norm）与 **`A-Opt-03w`**（再叠 `warmup_epochs=5`）均 **三 seed** 完成（**03w 未更好**）。**`A-Opt-03`** 现为 **h128 轻量对照 / P0-4 锚点**。数值与 trade-off 见 [任务A实验状态表](任务A实验状态表.md)「A-Opt-03 / A-Opt-03w」与 [任务A优化路径与近期实验建议](任务A优化路径与近期实验建议.md) P0-4 节；训练期汇总 CSV：`outputs/field/plots/optimization/prenorm_A_Opt03_vs_Opt03w/best_metrics.csv`。  
**（2026-03-29 同步）P0-5 容量线已归档**：**`A-Opt-04`**（`hidden_dim=256`）与 **`A-Opt-05`**（`num_layers=4`）均 **三 seed** 完成；**`A-Opt-05` 在 `interior` 与 `near_wall` 等略优于 `A-Opt-03`，方差与成本更高**。**（2026-03-31）** **新开跑母版统一为 `A-Opt-05`**（见状态表「战略锚点」）。明细见 [任务A实验状态表](任务A实验状态表.md)「A-Opt-04 / A-Opt-05」。  
**（2026-03-31 同步）`A-Opt-05_tune`**：在 **`A-Opt-05`** 骨架上的 **warmup / lr / wd / scheduler_patience** 等试跑，**已入账部分**已补 **`predictions_test` 全链**；**多模型对照（对 `A-Opt-03`，seed1）** 在 **`outputs/field/plots/optimization/A_Opt05_tune_vs_Opt03_seed1/`**。**`compare_hemo_wss_runs` 全量 WSS 对比暂缓**。清单与 **`outputs/field/`** 目录不一致处见状态表「`A-Opt-05_tune`」。  
**（2026-04-02 同步）P1-2 / `A-Opt-07`**：**`interior_loss_boost=3`** 三 seed 已完成（`*_iboost3_*_20260331_175619`）；已 **`predictions_test`** + **`regional_eval`**；与 **`A-Main-01` / `A-Opt-05`** 对照 **`plots/optimization/A_Opt07_vs_Opt05_Main01/`**（`python -m training.scripts.regenerate_opt07_vs_opt05_main_figures`）。**相对母版 `A-Opt-05`**：内部与全图 **`RMSE_|v|`** 未更好，**壁面/近壁变差**——**负结果**，见 [任务A实验状态表](任务A实验状态表.md)「A-Opt-07」。  
**P0 下一里程碑（叙事收敛）**：优先 **补齐 `A-Abl-02` / Line G 小步验证** 或对 **`A-Opt-05` vs `A-Opt-03` 补效率 benchmark**；**不默认启动 `A-Opt-06`（6L）**，除非明确要写「容量极限」附录。

**（2026-03-26 重要更新）主指标口径统一为 interior-only**：
- 此前主表、消融图、效率 Pareto 图等均使用 `summary.json.test_metrics.rmse_vel_mag`（all 节点），wall 节点的近零误差会系统性拉低 RMSE，导致主结论偏乐观。
- 自本次起，所有出图脚本的默认 `--region` 改为 `interior`：
  - **Fig A1 主表**：主指标列读取 `regional_eval` 的 `interior` 区，同时保留 `all_rmse_vel_mag` 作参考
  - **Fig A3 散点图**、**Fig A4 箱线图**、**误差分析**：默认只聚合内部节点
  - **Fig A6 消融/汇总图**：默认读取 `interior.rmse_vel_mag`
  - **Fig A7 效率 Pareto**：accuracy 轴改为 `interior.rmse_vel_mag`
  - **Fig A5 分区域图**本身已为 region-aware，不受影响
- 论文口径固定为：**主速度指标 = `interior.rmse_vel_mag`**；`all.rmse_vel_mag` 仅作为补充报告。
