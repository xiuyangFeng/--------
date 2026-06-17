# 任务 A 分区域评估口径（复杂区域定义）

> 上位文档：[实验设计总纲](../实验设计总纲.md) | [任务A论文可视化与指标建议](../01-任务/任务A/03-共享执行与状态/任务A论文可视化与指标建议.md)
> 实现代码：`training/analysis/regional_eval.py`（`DEFAULT_REGIONS`、`build_region_masks`、`load_node_features_for_region_masks`）

## 1. 文档定位

任务 A 的 **Fig A5 / `regional_eval`** 按节点几何特征将点云划分为若干区域并计算 `RMSE / MAE / R²` 等。本文档固定**默认**区域名称、语义与数值区间，供论文、汇报与实验记录引用。

**与任务 B 的临床分区（髂动脉分支名等）不同**：本表仅对应场预测管线的 **图节点特征 + `regional_eval`  mask**。

## 2. 所用几何量的数据口径

区域 mask 读取的 `Abscissa`、`NormRadius`、`Curvature`、`is_wall` 来自图数据 `data.x`（与 `pipeline.config.NODE_FEATURE_NAMES` 列顺序一致），且为 **Pipeline 归一化之后**写入图的值：

- **Abscissa**：逐病例缩放到约 `[0, 1]`（见 `pipeline/extract_features.py`）。
- **NormRadius**：**min-max 到 `[0, 1]`**（全数据或分批统计的 min/max，见 `pipeline/normalize.py` 与 `NORMALIZATION_CONFIG`）。
- **Curvature**：**z-score**（同一规范）。
- **高曲率阈值**：在**单张图、全部节点**的 `Curvature` 上取 **P75 分位数**（大图用子采样估计分位数，见 `_curvature_quantile_threshold`），不是全局常数。

评估时优先通过预测 `.pt` 中的 `**graph_path`** 读回**未按训练 `feature_mask` 裁剪的完整 `x`**，使各模型在同一几何定义下可比（见 `load_node_features_for_region_masks`）。

## 3. 默认区域定义表


| 区域 key           | 可称      | 节点范围     | 定义 / 区间                      | 默认参数名（`build_region_masks`）    |
| ---------------- | ------- | -------- | ---------------------------- | ------------------------------ |
| `all`            | 全图      | 全部       | 全部节点                         | —                              |
| `wall`           | 壁面      | 全部       | `is_wall > 0.5`              | —                              |
| `interior`       | 内部      | 全部       | `is_wall ≤ 0.5`              | —                              |
| `high_curvature` | 高曲率     | 全部       | `Curvature >` 本图 **P75**     | `curvature_quantile=0.75`      |
| `low_curvature`  | 低曲率     | 全部       | `Curvature ≤` 本图 **P75**     | 同上                             |
| `near_wall`      | 近壁（内部）  | **仅内部点** | `NormRadius > 0.8`           | `near_wall_threshold=0.8`      |
| `core_flow`      | 核心流（内部） | **仅内部点** | `NormRadius ≤ 0.5`           | `core_flow_threshold=0.5`      |
| `bifurcation`    | 分叉带     | 全部       | `Abscissa ∈ [0.6, 0.9]`（含端点） | `bifurcation_range=(0.6, 0.9)` |
| `trunk`          | 主干      | 全部       | `Abscissa < 0.6`             | 与分叉共用左端点 `0.6`                 |


## 4. 说明与易混点

1. `**near_wall` / `core_flow` 与 `wall`**：`near_wall`、`core_flow` 在实现中额外要求 `**~is_wall**`，只标记**内部节点**上的近壁/核心区；壁面点本身不计入这两个 mask。
2. **Abscissa 空隙**：默认下 `**Abscissa > 0.9`** 的节点既不属于 `trunk` 也不属于 `bifurcation`，仍参与 `all`、`wall`、`interior` 及曲率类等区域。
3. **Pipeline 采样 vs 本表**：`pipeline/config.SAMPLING_CONFIG` 中 `**boundary_threshold = 2.0` mm** 仅用于**降采样时**内部点「近壁层 / 核心层」**预算分配**，**不是**上表中 `near_wall` 的 `NormRadius > 0.8`。
4. **修改阈值**：在调用 `compute_regional_metrics(..., **mask_kwargs)` 时传入 `curvature_quantile`、`near_wall_threshold`、`core_flow_threshold`、`bifurcation_range` 等；若正文或图表换了阈值，须在论文/汇报中**同步写明**。
5. **训练损失中的 `interior_loss_boost`**（如 **`A-Opt-07`**）：仅改变**非壁面节点**上的监督 MSE 权重，**不改变**上表任一区域的 mask 定义或几何阈值；评估仍按 §3 从 `graph_path` 读完整 `x` 后计算。

## 5. 区域级指标字段（RMSE / MAE / R²）

`training/analysis/regional_eval.py` 中 `compute_regional_metrics` 对每个非空区域输出（在样本点集上聚合）：

- 各目标分量：`rmse_u/v/w/p`、`mae_*`、`r2_*`（与 `TARGET_NAMES` 一致）
- 速度模长：`rmse_vel_mag`、`mae_vel_mag`、**`r2_vel_mag`**（|v| 的一元 R²，2026-03-29 起写入 JSON）

旧版仅含 RMSE/MAE 的 `fig_A5_regional_metrics.json` 需重新跑一次 `plot_taskA_regional_bar` 以补齐 `r2_vel_mag`。

## 6. 相关脚本

- 单 run：`training/scripts/plot_taskA_regional_bar.py` → `predictions_test/regional_eval/fig_A5_regional_metrics.json`
- 多模型汇总：`training/scripts/plot_taskA_multimodel_regional_bar.py` → `outputs/field/plots/multimodel_baseline/fig_A5_multimodel_regional_bar_*.png`（或用 `--exp-filter` / `--output-dir` 写入 `plots/optimization/<子课题>/` 或 `plots/ablation/<子目录>/`）
- **跨 seed 均值**：同一脚本在 **不传 `--seed`** 时，对每个 `exp_id` 在 `runs-root` 下扫描到的多个 run（不同 `summary.json` 中的 `seed`）先收集各区域 `rmse_*` / `r2_*`，再 **按区域对数值取算术平均** 后作图；传 `--seed k` 则只纳入该 seed，输出文件名带 `_seed{k}`。**几何消融（`A-Opt-05` + `A-Abl-02-*`）三 seed 均值横比** 的定稿图见 `outputs/field/plots/ablation/geometry_opt05_multimodel_mean3seed/`。
- **Fig A6 消融条带图（单指标汇总 + 可选相对母版的 paired 统计）**：`training/scripts/plot_taskA_ablation_summary.py`；`A-Abl-02` 相对 `A-Opt-05` 的 **interior `rmse_vel_mag` 三 seed 汇总** 见 `outputs/field/plots/ablation/geometry_opt05_mean3seed/`，run 清单示例：`training/cluster/run_list_figA6_geometry_opt05_mean3seed.txt`。
- **P1-2 对照（`A-Main-01` / `A-Opt-05` / `A-Opt-07`）**：`training/scripts/regenerate_opt07_vs_opt05_main_figures.py` → `outputs/field/plots/optimization/A_Opt07_vs_Opt05_Main01/`（Fig A3 / A5 / A4 + `compare_val_loss.png`，依赖各 run 已具备 `predictions_test` 与 `regional_eval`）
- 主结果表 CSV：`training/scripts/plot_taskA_main_table.py` → `plots/summary/fig_A1_main_table.csv`，主区域列可直接导出 **`rmse_u/v/w/p/vel_mag` + `r2_u/v/w/p/vel_mag`**；同时附加 **`all_rmse_vel_mag` / `all_r2_vel_mag`** 参考列，以及 **`near_wall_rmse_*` / `near_wall_r2_*`**（含 `r2_vel_mag`）。若某 run 尚未生成 regional JSON，对应区域列按脚本回退逻辑留空或回退到 `summary.json`。
