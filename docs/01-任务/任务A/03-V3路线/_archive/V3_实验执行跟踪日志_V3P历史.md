# V3 实验执行跟踪日志 · V3P 历史（AG-only）

> **归档说明**：`split_AG_v1` 上 V3P 探针 / 锚点 / 主线 / WSS 扫描等逐实验记录（2026-05-03 ～ 2026-05-08）。
> **活跃日志**（三域 V3D、数据口径修复）见 [../V3_实验执行跟踪日志.md](../01-执行与待办/V3_实验执行跟踪日志.md)。
> **维护**：V3P 新实验若重启，在活跃日志新建块或回迁本节顶部，勿与 V3D 混表。

---
## 数据资产：`split_data_new_v3` 全局重归一化 + 转图（4664，2026-05-17）

> **性质**：**非训练实验**，属 **`data_new` 步骤 4–5** 批量刷新；与 **`split_AG_v1`** 历史流水线指标 **分表**。

### 作业与配置

| 字段 | 值 |
| --- | --- |
| **作业 ID** | **4664**（**4663** 失败：Slurm spool 下 `PROJECT_DIR` 推导错误） |
| **脚本** | `pipeline/archive/onetime_batch_jobs/run_v3_split_data_new_renorm_regraph.slurm` |
| **`TRAIN_SPLIT`** | `training/splits/split_data_new_v3.json` |
| **`--sources`** | `AAA/ruputer` `AAA/unruputer` `AG/fast` `AG/slow` `ILO` |
| **日志** | 仓库根 **`logs/v3_renorm_regraph_4664.out`** / **`.err`** |

### 结果摘要

- **normalize / convert_to_graph**：各 **267 / 277** 成功；**10** 例缺 **`processed/coord_normalized`**（与 **`cases_data_new_v3_candidate_pool.txt`**（**259**）及 **`split_data_new_v3` 并集** **无交集**）。
- **口径警示**：当前 **`pipeline/normalize.py`** 对 **`train_cases`** 的目录匹配在 **ILO** 上存在 **尾名 `after`/`before` 碰撞**，全局统计扫描目录数 **多于 train 名单（约 226 vs 181）**——**修复代码前不宜将本次 JSON 视为严格 train-only**；见 [推进记录](../../../../02-推进与变更/代码修改与实验推进记录.md) **2026-05-17**。

---

## 探针 Probe-WSS-01：修复后重训（3645，2026-05-08）

> **结论**：单目标壁面 WSS 探针在 **trainer / 全局配置修复后**（warmup、patience、EMA、双 checkpoint、`v3_pointcloud` 生成 json 等）重跑，**测试集标量 `wss_r2_wss` 未较历史对照显著提升**，仍在 **Anchor / 主线 ~0.367** 一带；与历史单次探针 **~0.397** 同属 **~0.36–0.40** 量级（差值可归因 checkpoint 选取、训练轨迹与 **单 seed**）。详细判读见 [V3_代码问题诊断与修复计划](../04-代码诊断/V3_代码问题诊断与修复计划.md) §10.6、[推进记录](../../../../02-推进与变更/代码修改与实验推进记录.md)。

### 作业与产出

| 作业 ID | 脚本 / 配置 | 说明 |
| --- | --- | --- |
| **3645** | `training/cluster/run_train_field.slurm` | `training/configs/field/generated/v3_pointcloud/V3P-Probe-WSS-01_seed1.json`（**仅 `lambda_wss`**，速度/压力监督关闭） |

| 条目 | 路径或数值 |
| --- | --- |
| **修复后 run** | `outputs/field/field_v3_pointnext_localpool_probe_wss01_geom_wall13000_near2000_split_AG_v1_seed1_20260508_153506/` |
| **历史探针对照（修复前链路）** | `…20260506_012735/`：`summary.test_metrics.wss_r2_wss`≈**0.397**，`best_epoch=2` |
| **3645 `best_epoch` / `best_wss_epoch`** | **6** / **19**；约 epoch **66** 因 patience 结束 |
| **`test_metrics`（`best_model.pt`）** | `wss_r2_wss`≈**0.364**；（`r2_p`、`r2_vel_*` **不参与判读**：探针未训压力/速度） |
| **`test_metrics_best_wss`** | `wss_r2_wss`≈**0.364**（与上差 **<0.001**） |

### 训练后完整评估（3841 → 绘图 Bug 修复后闭环）

| 条目 | 说明 |
| --- | --- |
| **3841（集群）** | `evaluate_field_run_full` 在 **`plot_taskA_regional_bar --wss`** 处因 `plot_regional_bar` 非法画布崩溃中断；已完成 **predict**、A3/A4、误差分析及部分 regional JSON。 |
| **修复** | `training/analysis/visualization.py`：`plot_regional_bar` **去掉不可靠的 `tight_layout`**，改为 **`subplots_adjust` + 有限值过滤**；详见 [V3_代码问题诊断与修复计划](../04-代码诊断/V3_代码问题诊断与修复计划.md)「变更历史」最新一行。 |
| **补跑（本地，`--skip-predict --force`）** | 复用 `predictions_test_best_wss/`，重写 `evaluation/test_best_wss_model/`，产出 **`evaluation_summary.json`**、`regional_eval/fig_A5_regional_wss_metrics.json`（含 **`wall.r2_wss≈0.364`**）、`wss_direct/wss_credibility_summary.json` 等全集。 |

**`wss_credibility_summary.json` 摘要（direct WSS head，壁面点）**

- **点级（Wall WSS magnitude）**：`wss_mag_r2`≈**0.364**，RMSE≈**0.733**，Pearson≈**0.610**，Spearman≈**0.630**（样本 16238232 wall 点）。
- **病例级相关**：均值 Spearman≈**0.164**，p95 Spearman≈**0.446**（排序一致性病例间差异大）；`max_rmse`/`max_mae` 存在极端离群（单图/分部问题，须在论文中 caveat）。
- **高 WSS 区域重合**：top **10%** 平均 **Dice≈0.192**（top 5% **≈0.086**）——与高剪切区空间对齐仍有明显缺口。
- **`evaluation_summary.json` 的 `wss_quick_view`**：与上行一致，可作汇报速览。

---

## val_score / 早停修复验证：V3P-Main-01-PW 重训 + 单 run 预测图件（2026-05-08）

> **背景**：见 [V3_代码问题诊断与修复计划](../04-代码诊断/V3_代码问题诊断与修复计划.md) 与 [推进记录](../../../../02-推进与变更/代码修改与实验推进记录.md)。代码与 `v3_pointcloud` 配置已更新（warmup=5、patience=60、EMA、双 checkpoint 等）。

### 作业与清单

| 作业 ID | 脚本 / 用途 | 说明 |
| --- | --- | --- |
| **3634** | `training/cluster/run_train_field.slurm` | 配置 `V3P-Main-01-PW_seed1.json`；GPU 单卡训练 |
| **3643** | `training/cluster/run_wss_multitask_predict_figs_array.slurm`，`--array=0-0%1` | `MANIFEST_LIST_FILE=training/cluster/manifest_list_v3_main01_pw_3634_predict_figs.tsv`：`predict_field` + A3/A4 + `plot_error_analysis --wss` + Fig A5 |

### 产出目录

| 阶段 | 路径（节选） |
| --- | --- |
| **修复前**（主线层 3544，仍见旧 `val_score` bug） | `outputs/field/field_v3_pointnext_localpool_main01_geom_pw_wall13000_near2000_split_AG_v1_seed1_20260506_230831/` |
| **修复后训练**（3634） | `outputs/field/field_v3_pointnext_localpool_main01_geom_pw_wall13000_near2000_split_AG_v1_seed1_20260508_001936/` |
| **图件**（3643，基于上目录 `best_model.pt` 的 `predictions_test`） | 同上 run 下 `predictions_test/`、`plots/`、`error_analysis_interior/`（含 `wss/`）、`regional_eval/` |

### 训练与 summary 字段对比（seed=1）

| 条目 | 修复前（230831） | 修复后（001936） |
| --- | --- | --- |
| `best_epoch` | 11 | **104** |
| `summary.best_val_loss`（旧字段含义） | **8.248**（实为复合 `val_score`，误导） | **0.206**（**真实 val loss**；复合分见 `best_val_score`≈0.438） |
| `best_val_score`（新字段） | — | 0.438 |
| `best_wss_epoch` / `best_val_wss_r2` | — | 32 / 0.511（验证集峰值，另存 `best_wss_model.pt`） |
| **`test_metrics`（`best_model.pt`）** |  |  |
| `r2_p` | 0.920 | **0.936** |
| `wss_r2_wss` | **0.367** | 0.344 |

### `best_model.pt` vs `best_wss_model.pt`（测试集，001936 run）

补算脚本 `python -m training.scripts.recompute_dual_test_metrics --run-dir …001936` 已执行；`summary.json` 含 **`test_metrics_best_wss`**。

| 指标 | `test_metrics`（best_model） | `test_metrics_best_wss` |
| --- | --- | --- |
| `r2_p` | **0.936** | 0.935 |
| `rmse_p` | **0.0803** | 0.0810 |
| `wss_r2_wss` | 0.344 | **0.365** |
| `wss_rmse_wss` | 0.745 | **0.733** |

**判读**：早停/选优修复使 **`best_epoch` 合理化**；`best_model` 在压力上略优、标量 WSS 略低于修复前单次 run（0.344 vs 0.367），属配方权衡与单 seed 方差；**主报 WSS 时可并列 `test_metrics_best_wss`**（0.365，更接近旧 0.367）。其他 PW 线建议同样补跑并重看 `best_wss` 列。

---

## 正式主线层 3537–3547 结果汇总（seed=1，2026-05-07）

> **测试集**：`split_AG_v1`，17 test cases（PENG 已移除）。**仅凭 seed=1**；计划中的 seed=2/3 待补种后改报告为三 seed 均值±方差。
>
> **测试集后处理**：各 run 均无 `predictions_test/`（训练结束未自动导出）。已于 **2026-05-07** 提交阵列作业 **3623**（`MANIFEST_LIST_FILE=training/cluster/manifest_list_v3_mainline_seed1_predict.tsv`，`run_wss_multitask_predict_figs_array.slurm`，`--array=0-10%3`）：对下列 11 个目录执行 `predict_field` + A3/A4 + `plot_error_analysis --wss` + Fig A5。

### 主指标对照表（`summary.test_metrics`）

| 作业 | Exp ID | `best_epoch` | `r2_p` | `wss_r2_wss` | `r2_vel_mag` | 产出目录（节选） |
| --- | --- | --- | --- | --- | --- | --- |
| 3537 | V3P-Anchor-01 | 5 | 0.898 | **0.367** | 0.509 | `…/field_v3_transformer_localpool_anchor01_…_20260506_134310/` |
| 3538 | V3P-Base-01 | 46 | 0.936 | 0.331 | 0.483 | `…/base01_nogeom_…_20260506_134310/` |
| 3539 | V3P-Main-01 | 56 | 0.936 | 0.336 | **0.679** | `…/main01_geom_…_20260506_134310/` |
| 3540 | V3P-WSS-01-a | 75 | 0.943 | **0.368** | 0.702 | `…/wss01a_geom_…_20260506_134310/` |
| 3541 | V3P-WSS-01-b | 120 | **0.953** | 0.338 | 0.694 | `…/wss01b_geom_…_20260506_155445/` |
| 3542 | V3P-WSS-01-c | 109 | 0.939 | 0.358 | 0.691 | `…/wss01c_geom_…_20260506_181838/` |
| 3543 | V3P-Base-01-PW | 31 | 0.916 | 0.343 | −0.070 | `…/base01_nogeom_pw_…_20260506_230830/` |
| 3544 | V3P-Main-01-PW | 11 | 0.920 | **0.367** | −0.070 | `…/main01_geom_pw_…_20260506_230831/` |
| 3545 | V3P-WSS-01-a-PW | 9 | 0.931 | **0.395** | −0.070 | `…/wss01a_geom_pw_lambda005_…_20260507_001902/` |
| 3546 | V3P-WSS-01-b-PW | 11 | 0.930 | 0.390 | −0.070 | `…/wss01b_geom_pw_lambda010_…_20260507_015153/` |
| 3547 | V3P-WSS-01-c-PW | 11 | 0.933 | 0.376 | −0.070 | `…/wss01c_geom_pw_lambda020_…_20260507_020230/` |

说明：**`-PW` 组**不显式监督速度，`r2_vel_mag` 为未训练头的参考值，**不参与判读**。含弱速度组中 **`r2_vel_mag` 可与 Anchor/Probe 对照**。

### 判读（seed=1）

1. **同采样锚点（Anchor）**：`wss_r2_wss≈0.367`，`r2_p≈0.90`。仍为后续 Go/No-Go 的 **WSS 对标均值**（正式口径需三 seed）。
2. **含弱速度（vel+P+WSS）**：`V3P-Main-01` 相对 `Base-01` **速度大幅更好**（0.68 vs 0.48）、WSS 略好（0.336 vs 0.331），但 **WSS 仍低于 Anchor**（0.336 vs 0.367）。`lambda_wss` 穷扫中 **0.05 档（WSS-01-a）WSS 最高**（0.368），与锚点基本持平；**0.10 档压力最优**（`r2_p≈0.95`）但 WSS 降至 0.338。
3. **纯 P+WSS（-PW）**：整体 **抬升 `wss_r2_wss`**：`WSS-01-a-PW` 达 **0.395**，为全表最高并 **超过 Anchor**；`Main-01-PW`（0.367）与 Anchor **WSS 持平**，显著优于 `Main-01` 含速度（0.336）。**几何在 PW 配方下仍带来 WSS 增益**（Main-PW vs Base-PW：0.367 vs 0.343）。
4. **与 Probe 一致性**：结果支持 **弃用速度监督、采用 P+WSS** 的主线决策；`-PW` 与 `WSS-01-*-PW` 可作为下一阶段 **补 seed** 与 **区域性 / 壁面子图** 的主力配置。

---

## 全局锁定项（V3 P0 不可变）

| 锁定项 | 取值 | 来源 |
| --- | --- | --- |
| `split` | `split_AG_v1` | §2.1.1 |
| `target_total_points` | 15000 | `pipeline/config.py:91` |
| `wall_max_points` | 13000 | `pipeline/config.py:96` |
| `boundary_threshold` | 2.0 mm | `pipeline/config.py:99` |
| `boundary_core_ratio` | `(1.0, 0.0)` | `pipeline/config.py:102` |
| `allow_core_fallback` | `False` | `pipeline/config.py:106` |
| `hybrid_fps_ratio` | 0.5 | `pipeline/config.py:114` |

---

## Diag-00 校准决策摘要

> 来源：`outputs/field/diagnostics/v3p_diag00_seed1/`（2026-05-03 作业 3415）

| 决策项 | Diag-00 数据依据 | 最终取值 | 适用范围 |
| --- | --- | --- | --- |
| `lambda_vel_noslip` | 壁面 `max\|v\| = 3.34e-02` > 1e-3 | **1.0**（raw_truth） | Base/Main/WSS-* |
| `lambda_vel_int` | `weighted_loss_interior_velocity` 中位数 2.15 >> 其它项 5× | **0.15**（原 0.3 减半） | VP/VWSS/Base/Main/WSS-* |
| `augment.rotation` | rotation on/off L_total 差异 -0.008（可忽略） | **0.0**（关闭） | 全部 V3 |
| Huber 触发 | `>3σ` 占比 ~2%（<5%），`p99/p50=Infinity` 为 z-score 伪迹 | **暂不触发**，V3P-WSS-03 暂不排队 | — |
| Mask 等价 | `interior_max_dist_to_wall = 0.01mm` << 2mm | **通过** | — |
| OOD 警示 | WSS 测/训 q99 比 41×；NormRadius 21.8×；BC_O 33× | **记入论文 limitations，不阻断** | — |
| 归一化统计来源 | `normalization_params_global.json` 使用全库统计（非 train-only） | **沿用**（R² 不受影响，保持 V1/V2 可比） | — |
| **split_data_new_v3 · 步骤 4–5（4664）** | **意图** train-split；当前 **`normalize.py`** ILO 路径匹配存在 **`after`/`before` 碰撞**，全局统计 **≠ 严格 train-only** | **修复 `normalize.py` 后须重做 normalize+转图**；见上文 **4664** 数据资产块与 [推进记录](../../../../02-推进与变更/代码修改与实验推进记录.md) **2026-05-17** | — |

---

## V3P-Diag-00（诊断）

### 实验信息

| 字段 | 值 |
| --- | --- |
| 配置 | `training/configs/field/generated/v3_pointcloud/V3P-Diag-00_seed1.json` |
| 作业 ID | 3415（链式：diag + 1 epoch train） |
| 提交时间 | 2026-05-03 23:47 |
| 完成时间 | 2026-05-03 ~23:52 |
| 产出目录（诊断） | `outputs/field/diagnostics/v3p_diag00_seed1/` |
| 产出目录（训练） | `outputs/field/field_v3_pointnext_localpool_diag00_geom_wall13000_near2000_split_AG_v1_seed1_20260503_235232/` |
| 状态 | ✅ 已完成 |

### 关键指标（1 epoch，仅供 smoke test 参考）

| 指标 | 值 | 说明 |
| --- | --- | --- |
| `r2_p` | -0.039 | 1 epoch 未收敛，预期中 |
| `r2_vel_mag` | 0.486 | — |
| `wss_r2_wss` | 0.007 | 接近 0 |
| `test_loss` | 15.95 | — |

### 诊断产出文件

| 文件 | 主要内容 |
| --- | --- |
| `noslip_decision.txt` | `raw_truth`，`lambda_vel_noslip=1.0` |
| `weight_calibration.txt` | `lambda_vel_int` 需减半 |
| `augment_decision.txt` | rotation 差异可忽略 |
| `interior_dist_to_wall_stats.json` | mask 等价通过（max=0.01mm） |
| `wss_distribution_train.json` | `>3σ` 占比 ~2%，不触发 Huber |
| `wss_magnitude_consistency.json` | 随机初始化基线，p50 相对误差 115% |
| `train_test_distribution_diff.json` | WSS/NormRadius/BC OOD 警示 |
| `weighted_loss_calibration.json` | 5 项 loss 中位数详细数据 |
| `norm_params_consistency.txt` | `normalization_params_global.json` 已通过符号链接修复 |

---

## V3P-Probe-P-01（压力单目标上限）

### 实验信息

| 字段 | 值 |
| --- | --- |
| 配置 | `training/configs/field/generated/v3_pointcloud/V3P-Probe-P-01_seed1.json` |
| 作业 ID | **3472**（重跑，数据预处理后） |
| 提交时间 | 2026-05-06 01:27 |
| 完成时间 | 2026-05-06 |
| 产出目录 | `outputs/field/field_v3_pointnext_localpool_probe_p01_geom_wall13000_near2000_split_AG_v1_seed1_20260506_012735/` |
| 监督目标 | 仅 `p`（`lambda_p_int=1.0, lambda_p_wall=1.0`，其余 0） |
| head_layout | `mlp2` |
| 训练 epoch | 130（best=100，patience=30） |
| 数据来源 | **新预处理数据**（`data_new/AG`，PENG 已从 split 移除，17 test cases） |
| 状态 | ✅ 已完成 |

> **历史 run（旧，已作废）**：作业 3418，`…/20260504_170252/`，当时 test 含 18 cases（含 PENG），best_epoch=81，r2_p=0.046。

### 关键指标（新数据，17 test cases）

| 指标 | 值 | 对比旧排PENG后 | 说明 |
| --- | --- | --- | --- |
| `r2_p` | **0.9621** | +0.003（旧=0.959） | ✅ 与旧排PENG后高度一致，压力可学性确认 |
| `rmse_p` | 0.0618 | — | — |
| `best_epoch` | 100 | 旧=81 | 新数据训练更长才收敛 |
| `best_val_loss` | 8.023 | — | — |

### 判读

> **最终判读**：新预处理数据（PENG 已从 split 移除）下 `r2_p = 0.962` → **压力单目标上限优秀，结论与旧排PENG后完全一致**。新数据验证了数据预处理和 split 修正的正确性。

---

## V3P-Probe-V-01（速度单目标上限）

### 实验信息

| 字段 | 值 |
| --- | --- |
| 配置 | `training/configs/field/generated/v3_pointcloud/V3P-Probe-V-01_seed1.json` |
| 作业 ID | **3473**（重跑，数据预处理后） |
| 提交时间 | 2026-05-06 01:27 |
| 完成时间 | 2026-05-06 |
| 产出目录 | `outputs/field/field_v3_pointnext_localpool_probe_v01_geom_wall13000_near2000_split_AG_v1_seed1_20260506_012735/` |
| 监督目标 | 仅内部速度（`lambda_vel_int=1.0`，其余 0） |
| head_layout | `mlp2` |
| 训练 epoch | 71（best=41，patience=30） |
| 数据来源 | **新预处理数据**（`data_new/AG`，PENG 已从 split 移除，17 test cases） |
| 状态 | ✅ 已完成 |

> **历史 run（旧，已作废）**：作业 3419，`…/20260504_170252/`，best_epoch=41，r2_vel_mag=0.463（含PENG）→0.474（排PENG）。

### 关键指标（新数据，17 test cases）

| 指标 | 新值 | 旧值（排PENG后） | 说明 |
| --- | --- | --- | --- |
| `r2_vel_mag` | **0.294** | 0.474 | ⚠️ 较旧版下降，需关注 |
| `r2_w` | **0.299** | 0.426 | w 分量有下降 |
| `r2_v` | -0.101 | 0.035 | v 分量负值 |
| `r2_u` | -0.324 | -0.057 | u 分量不可学 |
| `best_epoch` | 41 | 41 | — |
| `best_val_mag` | 1.449 | — | — |

### 判读

> **判读**：`r2_vel_mag=0.294` 低于旧排PENG后结果（0.474），**可能与新预处理后内部点归一化参数变化有关**（PENG 移除后全库统计轻微改变）。速度在当前 wall13000+near2000（仅 2000 内部点）下整体偏弱。
>
> **VWSS-01 负结果已确认速度上下文不帮助 WSS**，故速度 R² 下降对主线计划影响有限。速度在 V3 中定位为**诊断/参考指标**，主论文指标为 WSS 和壁面压力。

---

## V3P-Probe-WSS-01（WSS 单目标上限）

### 实验信息

| 字段 | 值 |
| --- | --- |
| 配置 | `training/configs/field/generated/v3_pointcloud/V3P-Probe-WSS-01_seed1.json` |
| 作业 ID | **3474**（重跑，数据预处理后） |
| 提交时间 | 2026-05-06 01:27 |
| 完成时间 | 2026-05-06 |
| 产出目录 | `outputs/field/field_v3_pointnext_localpool_probe_wss01_geom_wall13000_near2000_split_AG_v1_seed1_20260506_012735/` |
| 监督目标 | 仅壁面 WSS（`lambda_wss=1.0`，其余 0） |
| head_layout | `mlp2` |
| 训练 epoch | 32（best=2，patience=30） |
| 数据来源 | **新预处理数据**（`data_new/AG`，PENG 已从 split 移除，17 test cases） |
| 状态 | ✅ 已完成 |

> **历史 run（旧，已作废）**：作业 3420，`…/20260504_170252/`，best_epoch=10，wss_r2_wss=-0.012（含PENG）→0.398（排PENG）。

### 关键指标（新数据，17 test cases）

| 指标 | 新值 | 旧值（排PENG后） | 说明 |
| --- | --- | --- | --- |
| `wss_r2_wss` | **0.397** | 0.398 | ✅ 高度一致，WSS 可学性确认 |
| `wss_rmse_wss` | 0.714 | 0.027 | （归一化单位不同，仅供参考） |
| `wss_r2_wss_z` | **0.364** | 0.348 | z 分量有微小改善 |
| `wss_r2_wss_x` | **0.032** | 0.014 | x 分量有改善 |
| `wss_r2_wss_y` | **0.011** | 0.006 | y 分量有改善 |
| `best_epoch` | **2** | 10 | ⚠️ 极早收敛，可能与新数据信噪比提升有关 |

### 判读

> **最终判读**：新数据下 `wss_r2_wss=0.397`，**与旧排PENG后结果完全一致**。WSS 单目标可学性成立（R²≈0.40）。
>
> `best_epoch=2` 异常早，可能是新预处理数据 WSS 信号更干净（PENG 移除后全库归一化更稳定），导致模型在极早期就找到局部最优后 val loss 不再改善。Probe 性质实验（测可学性上限）结论不受影响。
>
> 分量不均匀现象仍存在（wss_z≈0.36 明显优于 wss_x/y≈0.01~0.03），主线设计需关注 x/y 分量改善。

---

## V3P-Probe-PWSS-01（压力 + WSS 双目标）

### 实验信息

| 字段 | 值 |
| --- | --- |
| 配置 | `training/configs/field/generated/v3_pointcloud/V3P-Probe-PWSS-01_seed1.json` |
| 作业 ID | **3476** |
| 提交时间 | 2026-05-06 03:22 |
| 完成时间 | 2026-05-06 |
| 产出目录 | `outputs/field/field_v3_pointnext_localpool_probe_pwss01_geom_wall13000_near2000_split_AG_v1_seed1_20260506_032254/` |
| 监督目标 | `p + WSS`（`lambda_p_int=0.5, lambda_p_wall=1.0, lambda_wss=0.1`） |
| head_layout | `mlp2` |
| 训练 epoch | 49（best=19，patience=30） |
| 数据来源 | **新预处理数据**（`data_new/AG`，PENG 已从 split 移除，17 test cases） |
| 状态 | ✅ 已完成 |

### 关键指标（17 test cases）

| 指标 | 本实验（P+WSS） | 单目标 P-01 | 单目标 WSS-01 | 说明 |
| --- | --- | --- | --- | --- |
| `r2_p` | **0.929** | 0.962 | — | P 相对单目标轻微下降 -0.033 |
| `rmse_p` | 0.0847 | 0.0618 | — | — |
| `wss_r2_wss` | **0.366** | — | 0.397 | WSS 相对单目标轻微下降 -0.031 |
| `wss_r2_wss_z` | 0.356 | — | 0.364 | z 分量下降轻微 |
| `wss_r2_wss_x` | 0.043 | — | 0.032 | x 分量略有改善 |
| `wss_r2_wss_y` | 0.067 | — | 0.011 | y 分量明显改善 |
| `best_epoch` | 19 | 100 | 2 | — |

### 判读

> **结论：P 与 WSS 双目标可以共存，相互干扰轻微。**
>
> - 压力：R²=0.929，相对 P 单目标（0.962）轻微下降 -3.3%，可接受
> - WSS：R²=0.366，相对 WSS 单目标（0.397）轻微下降 -3.1%，可接受
> - **意外收获**：PWSS 双目标下 wss_y R²=0.067，显著优于 WSS 单目标（0.011），说明压力监督为 y 分量 WSS 学习提供了有益的梯度引导
> - **战略意义**：支持 V3P-Main-01 / V3P-Base-01 采用 P+WSS 联合监督路线（不含速度监督）

---

## V3P-Probe-VP-01（速度 + 压力双目标）

### 实验信息

| 字段 | 值 |
| --- | --- |
| 配置 | `training/configs/field/generated/v3_pointcloud/V3P-Probe-VP-01_seed1.json` |
| 监督目标 | `vel + p`（`lambda_vel_int=0.15, lambda_p_int=0.5, lambda_p_wall=1.0`） |
| 状态 | 🔒 未开始（VWSS-01 负结果已降低本实验优先级） |

> **优先级降低说明**：VWSS-01 结果显示速度监督对壁面量无益甚至有害，VP-01 的诊断价值已大幅降低。Main-01 路线将直接采用 P+WSS（不含速度），VP-01 可根据需要决定是否补跑。

---

## V3P-Probe-VWSS-01（速度 + WSS 双目标）

### 实验信息

| 字段 | 值 |
| --- | --- |
| 配置 | `training/configs/field/generated/v3_pointcloud/V3P-Probe-VWSS-01_seed1.json` |
| 作业 ID | **3475** |
| 提交时间 | 2026-05-06 01:27 |
| 完成时间 | 2026-05-06 |
| 产出目录 | `outputs/field/field_v3_pointnext_localpool_probe_vwss01_geom_wall13000_near2000_split_AG_v1_seed1_20260506_012735/` |
| 监督目标 | `vel + WSS`（`lambda_vel_int=0.15, lambda_wss=0.1`） |
| head_layout | `mlp2` |
| 训练 epoch | 40（best=10，patience=30） |
| 数据来源 | **新预处理数据**（`data_new/AG`，PENG 已从 split 移除，17 test cases） |
| 状态 | ✅ 已完成 |

### 关键指标（17 test cases）

| 指标 | 本实验（V+WSS） | 单目标 V-01 | 单目标 WSS-01 | 说明 |
| --- | --- | --- | --- | --- |
| `wss_r2_wss` | **0.343** | — | 0.397 | ❌ WSS 相对单目标下降 -0.054 |
| `wss_r2_wss_z` | 0.340 | — | 0.364 | z 分量下降 |
| `wss_r2_wss_x` | 0.048 | — | 0.032 | x 分量略优 |
| `wss_r2_wss_y` | 0.054 | — | 0.011 | y 分量有改善 |
| `r2_vel_mag` | **0.074** | 0.294 | — | ❌ 速度相对单目标大幅下降 -0.220 |
| `r2_w` | -0.133 | 0.299 | — | w 分量崩溃 |
| `best_epoch` | 10 | 41 | 2 | — |

### 判读

> **关键负结果：速度上下文并不帮助 WSS，反而使两个目标均恶化。**
>
> - WSS R²=0.343，相对 WSS 单目标（0.397）下降 **-13.6%**
> - 速度 R²=0.074，相对 V 单目标（0.294）下降 **-74.8%**
> - 梯度竞争：`lambda_vel_int=0.15` 的速度 loss 与 `lambda_wss=0.1` 的 WSS loss 存在梯度拉锯，两者均被压低
> - **战略意义（重要）**：V3P-Main-01 和 V3P-Base-01 的监督配方**不应包含速度监督**。最优路线为 **P+WSS 双目标**（由 PWSS-01 支持），放弃弱速度辅助监督假设。V3P-WSS-02（速度上下文增强）条件不再触发，可取消排队。

---

## V3P-Anchor-01（同采样 V1 Transformer 锚点）

### 实验信息

| 字段 | 值 |
| --- | --- |
| 配置 | `training/configs/field/generated/v3_pointcloud/V3P-Anchor-01_seed1.json` |
| 作业 ID | **3537** |
| 提交时间 | 2026-05-06 13:34 |
| backbone | Transformer（A-Opt-05 配方，Pre-Norm） |
| 监督 | 全场 `target_weights=[2,2,2,0.5]` + `wss_loss_weight=0.1`（V1 口径，无 domain_loss） |
| head_layout | `single_linear` |
| 状态 | ✅ 已完成（seed=1） |
| 完成时间 | 2026-05-06 |
| 产出目录 | `outputs/field/field_v3_transformer_localpool_anchor01_geom_wall13000_near2000_split_AG_v1_seed1_20260506_134310/` |
| 测试集摘要 | `r2_p=0.898`，`wss_r2_wss=0.367`，`r2_vel_mag=0.509`，`best_epoch=5` |

---

## V3P-Base-01（无几何 PointNeXt 下限）

### 实验信息

| 字段 | 值 |
| --- | --- |
| 配置 | `training/configs/field/generated/v3_pointcloud/V3P-Base-01_seed1.json` |
| 作业 ID | **3538** |
| 提交时间 | 2026-05-06 13:34 |
| 输入 | `coords + t + BC + is_wall`（无几何） |
| 监督 | 双域 mask loss（含弱速度：`lambda_vel_int=0.15, lambda_vel_noslip=1.0, lambda_p_int=0.5, lambda_p_wall=1.0, lambda_wss=0.1`） |
| 状态 | ✅ 已完成（seed=1，含弱速度） |
| 完成时间 | 2026-05-06 |
| 产出目录 | `outputs/field/field_v3_pointnext_localpool_base01_nogeom_wall13000_near2000_split_AG_v1_seed1_20260506_134310/` |
| 测试集摘要 | `r2_p=0.936`，`wss_r2_wss=0.331`，`r2_vel_mag=0.483`，`best_epoch=46` |

### 对照实验

| 字段 | 值 |
| --- | --- |
| 配置 | `training/configs/field/generated/v3_pointcloud/V3P-Base-01-PW_seed1.json` |
| 作业 ID | **3543** |
| 提交时间 | 2026-05-06 |
| 监督 | 纯 P+WSS（`lambda_vel_int=0.0, lambda_vel_noslip=0.0, lambda_p_int=0.5, lambda_p_wall=1.0, lambda_wss=0.1`） |
| 状态 | ✅ 已完成（seed=1） |
| 产出目录 | `outputs/field/field_v3_pointnext_localpool_base01_nogeom_pw_wall13000_near2000_split_AG_v1_seed1_20260506_230830/` |
| 测试集摘要 | `r2_p=0.916`，`wss_r2_wss=0.343`，`best_epoch=31` |

---

## V3P-Main-01（几何 PointNeXt 主线）

### 实验信息

| 字段 | 值 |
| --- | --- |
| 配置 | `training/configs/field/generated/v3_pointcloud/V3P-Main-01_seed1.json` |
| 作业 ID | **3539** |
| 提交时间 | 2026-05-06 13:34 |
| 输入 | `Base + Abscissa + NormRadius + Curvature + Tangent` |
| 监督 | 含弱速度：`lambda_vel_int=0.15, lambda_vel_noslip=1.0, lambda_p_int=0.5, lambda_p_wall=1.0, lambda_wss=0.1` |
| 状态 | ✅ 已完成（seed=1，含弱速度） |
| 完成时间 | 2026-05-06 |
| 产出目录 | `outputs/field/field_v3_pointnext_localpool_main01_geom_wall13000_near2000_split_AG_v1_seed1_20260506_134310/` |
| 测试集摘要 | `r2_p=0.936`，`wss_r2_wss=0.336`，`r2_vel_mag=0.679`，`best_epoch=56` |

### 对照实验（纯 P+WSS 主路线）

| 字段 | 值 |
| --- | --- |
| 配置 | `training/configs/field/generated/v3_pointcloud/V3P-Main-01-PW_seed1.json` |
| 作业 ID | **3544** |
| 提交时间 | 2026-05-06 |
| 监督 | 纯 P+WSS：`lambda_vel_int=0.0, lambda_vel_noslip=0.0, lambda_p_int=0.5, lambda_p_wall=1.0, lambda_wss=0.1` |
| 状态 | ✅ 已完成（seed=1） |
| 产出目录 | `outputs/field/field_v3_pointnext_localpool_main01_geom_pw_wall13000_near2000_split_AG_v1_seed1_20260506_230831/` |
| 测试集摘要 | `r2_p=0.920`，`wss_r2_wss=0.367`，`best_epoch=11` |
| 说明 | V3 核心实验对；与 Base-01-PW 唯一差异为是否启用几何先验；与 Main-01 唯一差异为是否含速度监督 |

---

## V3P-WSS-01-a/b/c（WSS 权重穷扫）

### 实验信息（含弱速度版，对照组）

| Exp ID | lambda_wss | 作业 ID | 配置 | 状态（seed=1） |
| --- | --- | --- | --- | --- |
| `V3P-WSS-01-a` | 0.05 | **3540** | `V3P-WSS-01-a_seed1.json` | ✅ 已完成：`wss_r2_wss=0.368`，`r2_p=0.943`，`best_epoch=75`；`…/wss01a_geom_…_20260506_134310/` |
| `V3P-WSS-01-b` | 0.10 | **3541** | `V3P-WSS-01-b_seed1.json` | ✅ 已完成：`wss_r2_wss=0.338`，`r2_p=0.953`，`best_epoch=120`；`…/wss01b_geom_…_20260506_155445/` |
| `V3P-WSS-01-c` | 0.20 | **3542** | `V3P-WSS-01-c_seed1.json` | ✅ 已完成：`wss_r2_wss=0.358`，`r2_p=0.939`，`best_epoch=109`；`…/wss01c_geom_…_20260506_181838/` |

### 对照实验（纯 P+WSS 版，主路线配方）

| Exp ID | lambda_wss | 作业 ID | 配置 | 状态（seed=1） |
| --- | --- | --- | --- | --- |
| `V3P-WSS-01-a-PW` | 0.05 | **3545** | `V3P-WSS-01-a-PW_seed1.json` | ✅ 已完成：`wss_r2_wss=0.395`，`r2_p=0.931`，`best_epoch=9`；`…/wss01a_geom_pw_lambda005_…_20260507_001902/` |
| `V3P-WSS-01-b-PW` | 0.10 | **3546** | `V3P-WSS-01-b-PW_seed1.json` | ✅ 已完成：`wss_r2_wss=0.390`，`r2_p=0.930`，`best_epoch=11`；`…/wss01b_geom_pw_lambda010_…_20260507_015153/` |
| `V3P-WSS-01-c-PW` | 0.20 | **3547** | `V3P-WSS-01-c-PW_seed1.json` | ✅ 已完成：`wss_r2_wss=0.376`，`r2_p=0.933`，`best_epoch=11`；`…/wss01c_geom_pw_lambda020_…_20260507_020230/` |

---

## V3P-WSS-02（速度上下文增强）

### 实验信息

| 字段 | 值 |
| --- | --- |
| 配置 | `training/configs/field/generated/v3_pointcloud/V3P-WSS-02_seed1.json` |
| 唯一变化 | `lambda_vel_int=0.25`（相对 Main-01 的 0.15 上调） |
| 触发条件 | WSS-01 至少一档"压力不崩 + WSS 有正信号" |
| 状态 | ❌ 已取消（VWSS-01 负结果：速度监督不帮助WSS，条件不触发） |

---

## 变更历史

| 日期 | 内容 |
| --- | --- |
| 2026-05-07 | 正式主线层 **3537–3547** seed=1 训练结果已自 `summary.json` 归档；**无** `predictions_test` 时补提阵列 **3623**（`manifest_list_v3_mainline_seed1_predict.tsv` + `run_wss_multitask_predict_figs_array.slurm`）。结果汇总见篇首「正式主线层 3537–3547 结果汇总」。 |
| 2026-05-06 | 正式主线层全部提交（11 个作业）：①含弱速度版：Anchor-01(3537), Base-01(3538), Main-01(3539), WSS-01-a/b/c(3540-3542)；②纯P+WSS版：Base-01-PW(3543), Main-01-PW(3544), WSS-01-a/b/c-PW(3545-3547)。4 张 GPU 并行，其余排队。 |
| 2026-05-06 | 数据预处理重跑完成（作业 3470/3471）；在新 split（17 test cases，PENG 已移除）上重跑 P/V/WSS-01（作业 3472-3474）并新增 VWSS-01 + PWSS-01（作业 3475-3476）；关键结论：①P R²=0.962 与排PENG后一致；②WSS R²=0.397 与排PENG后一致；③VWSS-01 **负结果**（速度上下文不帮助WSS，两目标均恶化）；④PWSS-01 P+WSS 可共存（各-3%），y 分量 WSS 在 PWSS 下意外改善；⑤主线确定为 P+WSS 双目标，不含速度监督 |
| 2026-05-05 | OOD 根因定位：`PENG_JI_MING` 唯一异常源（CFD 入口流速~0）；排除后 P R²=0.96, WSS R²=0.40；已从 split 移除并提交图资产重建（作业 3464） |
| 2026-05-05 | Probe-P/V/WSS-01 训练完成；初始 R² 被 PENG 拖低（P=0.046, WSS=-0.012），排除后翻转 |
| 2026-05-04 | 创建本跟踪日志；Diag-00 完成 → 校准决策固化 → Probe-P/V/WSS-01 提交（作业 3418-3420） |
| 2026-05-03 | V3P-Diag-00 作业 3415 提交并完成 |
