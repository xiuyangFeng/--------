# 任务A 实验状态总表

> 本表是任务 A 所有实验的唯一执行状态追踪文件。  
> 每次启动、完成或失败一组实验，**必须更新此表**。  
> 上位文档：[任务A实验清单](任务A实验清单.md) / [任务A冻结卡](任务A冻结卡.md)

---

## 状态说明


| 状态标记            | 含义                      |
| --------------- | ----------------------- |
| 🔒 未开始          | 尚未生成配置或提交               |
| 🔬 待 smoke test | 配置已就绪，等待最小闭环验证          |
| 🚀 进行中          | 至少一个 seed 已启动           |
| 🌱 待补 seed      | seed=1 已通过，等待补 seed=2,3 |
| 📋 待汇总          | 训练完成，等待写入记录表            |
| ✅ 已完成           | 结果、图表、实验记录均已归档          |
| ❌ 失败待重跑         | 出现错误或配置问题，需修复后重跑        |


---

## 第一批：基线实验


| Exp ID    | 研究问题                       | split_version | seeds   | 当前状态  | 输出目录                                                                           | 备注                        |
| --------- | -------------------------- | ------------- | ------- | ----- | ------------------------------------------------------------------------------ | ------------------------- |
| A-Base-01 | 点模型下限（无图结构）                | split_AG_v1   | [1,2,3] | ✅ 已完成 | outputs/field/field_mlp_coord_t_bc_split_AG_v1_seed{seed}_*/                   | 3 seed 均已完成，2026-03-22    |
| A-Base-02 | 图结构是否必要                    | split_AG_v1   | [1,2,3] | ✅ 已完成 | outputs/field/field_graphsage_coord_t_bc_wall_split_AG_v1_seed{seed}_*/        | 3 seed 均已完成，2026-03-22/23 |
| A-Base-03 | Transformer 无 geometry 对照  | split_AG_v1   | [1,2,3] | ✅ 已完成 | outputs/field/field_transformer_coord_t_bc_wall_split_AG_v1_seed{seed}_*/      | 3 seed 均已完成，2026-03-22/23 |
| A-Main-01 | Transformer + geometry 主模型 | split_AG_v1   | [1,2,3] | ✅ 已完成 | outputs/field/field_transformer_coord_t_bc_geom_wall_split_AG_v1_seed{seed}_*/ | 3 seed 均已完成，2026-03-22/23 |


---

## 第二批：必做消融


| Exp ID      | 研究问题    | 唯一变化项              | split_version | seeds | 当前状态   | 备注                     |
| ----------- | ------- | ------------------ | ------------- | ----- | ------ | ---------------------- |
| A-Abl-01-01 | 输入特征消融  | coords + t 仅坐标+时间  | split_AG_v1   | [1]   | 🔒 未开始 | 依赖 A-Main-01 完成 ✅ 已可开始 |
| A-Abl-01-02 | 输入特征消融  | + BC               | split_AG_v1   | [1]   | 🔒 未开始 |                        |
| A-Abl-01-03 | 输入特征消融  | + is_wall          | split_AG_v1   | [1]   | 🔒 未开始 |                        |
| A-Abl-01-04 | 输入特征消融  | + geometry（无 wall） | split_AG_v1   | [1]   | 🔒 未开始 |                        |
| A-Abl-02-01 | 几何分量消融  | 去掉 Abscissa        | split_AG_v1   | [1]   | 🔒 未开始 |                        |
| A-Abl-02-02 | 几何分量消融  | 去掉 NormRadius      | split_AG_v1   | [1]   | 🔒 未开始 |                        |
| A-Abl-02-03 | 几何分量消融  | 去掉 Curvature       | split_AG_v1   | [1]   | 🔒 未开始 |                        |
| A-Abl-02-04 | 几何分量消融  | 去掉 Tangent         | split_AG_v1   | [1]   | 🔒 未开始 |                        |
| A-Abl-03-01 | 坐标归一化消融 | 原始坐标               | split_AG_v1   | [1]   | 🔒 未开始 |                        |
| A-Abl-03-02 | 坐标归一化消融 | 仅中心化               | split_AG_v1   | [1]   | 🔒 未开始 |                        |
| A-Abl-03-03 | 坐标归一化消融 | 中心化+PCA对齐          | split_AG_v1   | [1]   | 🔒 未开始 |                        |
| A-Abl-03-04 | 坐标归一化消融 | 中心化+PCA+缩放（当前版本）   | split_AG_v1   | [1]   | 🔒 未开始 |                        |
| A-Abl-04-01 | 增强消融    | 无增强                | split_AG_v1   | [1]   | 🔒 未开始 |                        |
| A-Abl-04-02 | 增强消融    | 仅旋转                | split_AG_v1   | [1]   | 🔒 未开始 |                        |
| A-Abl-04-03 | 增强消融    | 旋转+平移（默认）          | split_AG_v1   | [1]   | 🔒 未开始 |                        |
| A-Abl-04-04 | 增强消融    | 旋转+平移+微扰           | split_AG_v1   | [1]   | 🔒 未开始 |                        |
| A-Abl-05-01 | 物理约束消融  | 仅数据损失              | split_AG_v1   | [1]   | 🔒 未开始 | 依赖主线稳定后                |
| A-Abl-05-02 | 物理约束消融  | + continuity       | split_AG_v1   | [1]   | 🔒 未开始 |                        |


---

## 第三批：近期优化线

> 说明：本区用于承接 baseline 完成后的“先拿更好结果，再补最小必要解释实验”路线。  
> 当前优先级以 [任务A优化路径与近期实验建议](任务A优化路径与近期实验建议.md) 为准。  
> 推荐执行顺序：`A-Opt-01 -> A-Opt-02 -> A-Opt-02_warmup (P0-3，✅ 2026-03-27) -> A-Opt-03 (✅ 2026-03-28) -> A-Opt-03w (✅ 2026-03-28) -> A-Opt-04 -> A-Opt-05`。  
> 推进门槛：只有当上一组同时改善全局 `RMSE_|v|`、内部区 `RMSE_|v|`，且至少一个速度分量 `R²` 明显改善时，才进入下一组容量扩展。


| Exp ID          | 研究问题                             | 唯一变化项                              | split_version | seeds   | 当前状态   | 备注                                                                                                                                                                                                                                                                 |
| --------------- | -------------------------------- | ---------------------------------- | ------------- | ------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A-Opt-01        | 速度权重是否能改善内部流场                    | `target_weights = [2,2,2,0.5]`     | split_AG_v1   | [1,2,3] | ✅ 已完成  | 三 seed 训练与 `**predict_field` + `plot_taskA_regional_bar`** 已完成（2026-03-26）。目录：`outputs/field/field_transformer_coord_t_bc_geom_wall_tw22205_split_AG_v1_seed{1,2,3}_*/`；见下文「实验记录摘要 · A-Opt-01」与主结果表增行。                                                             |
| A-Opt-02        | LayerNorm 是否提升单尺度 Transformer 表达 | `FieldTransformer` 改为 Pre-Norm 残差块 | split_AG_v1   | [1,2,3] | ✅ 已完成  | 三 seed：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_split_AG_v1_seed{1,2,3}_20260327_*`；`predict_field` + `error_analysis_interior` + `regional_eval` + 与 Main 对照 `**plots/prenorm_A_Opt02_vs_Main01/**`（2026-03-27）。见「实验记录摘要 · A-Opt-02」与主结果表增行。 |
| A-Opt-02_warmup | 学习率 Warmup 是否稳定 Pre-Norm 训练并改善指标 | `A-Opt-02 + optim.warmup_epochs=5` | split_AG_v1   | [1,2,3] | ✅ 已完成  | **P0-3**（2026-03-27）：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_warmup5_split_AG_v1_seed{1,2,3}_20260327_*`；已补 **`predictions_test`**、**`error_analysis_interior`**、**`regional_eval`**；与 Main / P0-2 三模型对照图 **`outputs/field/plots/optimization/prenorm_Main_P02_P02w/`**；一键重汇总结图：`python -m training.scripts.regenerate_p02_warmup_comparison_figures`。见「实验记录摘要 · A-Opt-02_warmup」与主结果表增列。 |
| A-Opt-03        | 损失重加权与 LayerNorm 是否互补            | `A-Opt-01 + A-Opt-02`              | split_AG_v1   | [1,2,3] | ✅ 已完成  | **P0-4（2026-03-28）**：三 seed 已训练并归档；`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_split_AG_v1_seed{1,2,3}_20260327_*`；已含 **`predictions_test`**、**`error_analysis_interior`**、**`regional_eval`**；训练期 **best_epoch** 约 **64 / 83 / 89**。汇总对比 CSV：`outputs/field/plots/optimization/prenorm_A_Opt03_vs_Opt03w/best_metrics.csv`。**速度侧**：相对 **`A-Opt-01` / `A-Opt-02` / `A-Main-01`**，`interior.rmse_vel_mag` 与 **`summary.test_metrics.rmse_vel_mag`** 三 seed 均值均为当前 P0 线最优；**内部点 `R²_u/v/w`** 均值亦优于单独 P0-1 与 P0-2。**压力侧 trade-off**：**`summary.test_metrics.rmse_p`** 三 seed 均值 **差于 `A-Opt-01`（~0.620）与 `A-Opt-02`（~0.610）**；**`regional_eval` · `all.rmse_p`** 与 **`A-Opt-02`** 持平（~0.398），**`interior.rmse_p`** 略差于 **`A-Opt-02`**。见「实验记录摘要 · A-Opt-03」与 [优化路径](任务A优化路径与近期实验建议.md) P0-4 节。 |
| A-Opt-03w       | Warmup 是否进一步稳定组合线                  | `A-Opt-03 + warmup_epochs=5`       | split_AG_v1   | [1,2,3] | ✅ 已完成  | **（2026-03-28）** 三 seed：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_warmup5_split_AG_v1_seed{1,2,3}_20260328_*`；后处理链同 **`A-Opt-03`**；**`best_epoch`** 约 **100 / 70 / 64**。**相对 `A-Opt-03`**：**`summary` / `regional_eval` 速度主指标未更好**，**`rmse_p` 未收复**——与 P0-3 类似，**非组合线必选项**。叙事上 **P0 默认基座取 `A-Opt-03`（非 03w）**。见「实验记录摘要 · A-Opt-03w」。                                                                                                                                                                                                                                                                  |
| A-Opt-04        | 容量扩大是否继续有效                       | `hidden_dim = 256`                 | split_AG_v1   | [1]     | 🔒 未开始 | **`A-Opt-03` 已在速度主指标上满足 [优化路径](任务A优化路径与近期实验建议.md) P0-5 推进门槛**（全局与内部 **`RMSE_|v|`** 改善 + 内 **`R²_u/v/w`** 相对 Main 与 P0-2 提升）；可按 **smoke → 补 seed** 启动，汇报需并列 **压力 trade-off**                                                                                                                                        |
| A-Opt-05        | 适度加深是否继续有效                       | `hidden_dim = 256, num_layers = 4` | split_AG_v1   | [1]     | 🔒 未开始 | 仅在 `A-Opt-04` 正向时进入                                                                                                                                                                                                                                                |
| A-Opt-06        | 单尺度进一步加深是否还值得                    | `hidden_dim = 256, num_layers = 6` | split_AG_v1   | [1]     | 🔒 未开始 | 若 `A-Opt-05` 收益很小，建议停止                                                                                                                                                                                                                                             |
| A-Opt-07        | 内部点区域加权是否进一步改善瓶颈                 | region-weighted loss               | split_AG_v1   | [1]     | 🔒 未开始 | 放在容量扩展之后评估                                                                                                                                                                                                                                                         |
| A-Opt-08        | 多尺度结构是否带来本质提升                    | graph U-Net / hierarchical GNN     | split_AG_v1   | [1]     | 🔒 未开始 | 单尺度优化见顶后再立项                                                                                                                                                                                                                                                        |


---

## 第四批：显式几何增强线 Line G（2026-03-26 新增）

> 说明：Line G 用于在现有 geometry 已被证明有效的前提下，继续小步增加新的显式几何/拓扑先验。
> 前置条件：优先完成 `A-Abl-02`，确认现有几何分量贡献，再进入 Line G。
> 推荐执行顺序：`A-Opt-G01 -> A-Opt-G02 -> (A-Opt-G03 / A-Opt-G04) -> A-Opt-G05`。
> 推进门槛：新增特征必须至少改善一个复杂区域（`near_wall / bifurcation / high_curvature`），且验证/测试不能退化。


| Exp ID    | 研究问题                | 唯一变化项                                         | split_version | seeds | 当前状态   | 备注                            |
| --------- | ------------------- | --------------------------------------------- | ------------- | ----- | ------ | ----------------------------- |
| A-Opt-G01 | 显式分叉拓扑先验是否改善复杂转折区建模 | `distance_to_bifurcation + branch_flag`       | split_AG_v1   | [1]   | 🔒 未开始 | Line G 首个实验；优先看 `bifurcation` |
| A-Opt-G02 | 局部尺度变化信息是否优于单纯半径值   | `radius_change_rate / area_change_rate`       | split_AG_v1   | [1]   | 🔒 未开始 | 关注扩张/收缩区与瘤腔过渡段                |
| A-Opt-G03 | 扭率能否补足曲率缺失的三维弯扭信息   | `torsion`                                     | split_AG_v1   | [1]   | 🔒 未开始 | 与 `Curvature` 形成互补验证          |
| A-Opt-G04 | 显式壁面距离是否改善近壁速度剖面学习  | `distance_to_wall / normalized_wall_distance` | split_AG_v1   | [1]   | 🔒 未开始 | 与 Line W 有接口关系                |
| A-Opt-G05 | 中心线方向变化率是否提升转折区表达   | `d_tangent/ds` 或等价方向变化量                       | split_AG_v1   | [1]   | 🔒 未开始 | 放在前四组信号明确后再尝试                 |


---

## 第五批：壁面导向优化线 Line W（2026-03-25 新增）

> 说明：Line W 直接面向端到端链路质量（WSS/OSI/RRT → 髂支闭塞风险预测）。与第三批（Line A 内部精度优化）并行推进、独立归因。
> 基座：**P0 组合线以 `A-Opt-03` 为当前默认最优**（`A-Opt-03w` 未优于 `A-Opt-03`；详见第三批 **`A-Opt-03w`** 备注）。Line W 可在 `A-Opt-03` 上起跑。
> 评估标准差异：Line W 必须额外运行 WSS 后处理对比，以壁面衍生指标质量为核心判定。
> 详见 [任务A优化路径](任务A优化路径与近期实验建议.md) 第 2.4 节和第 5.5 节。


| Exp ID    | 研究问题                  | 唯一变化项                                      | split_version | seeds | 当前状态   | 备注                                    |
| --------- | --------------------- | ------------------------------------------ | ------------- | ----- | ------ | ------------------------------------- |
| A-Opt-W01 | 近壁区域加权是否改善 WSS 梯度质量   | `near_wall_boost=3.0, interior_weight=0.5` | split_AG_v1   | [1]   | 🔒 未开始 | 需修改 `losses.py` + 近壁区 mask；依赖 P0 最优基座 |
| A-Opt-W02 | 壁面法向梯度监督是否提升 WSS 精度   | `wall_grad_weight=0.01`                    | split_AG_v1   | [1]   | 🔒 未开始 | 需修改 `losses.py`；依赖 W01 有正向信号或独立启动     |
| A-Opt-W03 | 直接 WSS 监督是否最大化端到端质量   | `wss_loss_weight=0.1`                      | split_AG_v1   | [1]   | 🔒 未开始 | 需 WSS 真值数据 + 数据管线改动；依赖任务 B WSS 管线就绪   |
| A-Opt-W04 | 两阶段训练是否优于一开始就加权       | 阶段1:均匀MSE → 阶段2:壁面精调                       | split_AG_v1   | [1]   | 🔒 未开始 | 可与 W01/W02 叠加                         |
| A-Opt-W05 | OSI 敏感区域加权是否改善 OSI 恢复 | 分叉/高曲率区 boost                              | split_AG_v1   | [1]   | 🔒 未开始 | 可与 W01 叠加                             |


## 主结果表（3 seed mean ± std，已完成）

> 数据来源：`experiment_index.csv` + 各 run 的 `summary.json` + `predictions_test/error_analysis/summary.json` + `predictions_test/regional_eval/fig_A5_regional_metrics.json` + `outputs/field/plots/efficiency/fig_A7_efficiency_benchmark.json`。  
> **分区域指标（2026-03-24 起）**：`fig_A5_regional_metrics.json` 由 `plot_taskA_regional_bar` 生成，区域 mask 基于各预测文件中的 `graph_path` 图资产（完整 `x`），与训练时 `enabled_node_features` 无关，baseline 四模型横向可比。  
> **（2026-03-26）** 优化线 `A-Opt-01` 在相同口径下已补全 `regional_eval`；默认 `**plot_taskA_multimodel_regional_bar**` 扫描结果包含 **baseline 四组 + `A-Opt-01**`（共 5 组 `exp_id`），需在文中区分「基线四模型对比」与「含 P0-1 的扩展对比」。**（2026-03-27）** `A-Opt-02` 三 seed 已各含 `regional_eval`；多模型 Fig A5 若扫全目录会再纳入 `A-Opt-02`，汇报时注意与「仅 baseline / 仅 P0-1」图区分。**（2026-03-27）** `A-Opt-02_warmup`（P0-3）已对齐同样后处理；三模型（Main / P0-2 / P0-3）Dedicated 汇总见 **`plots/optimization/prenorm_Main_P02_P02w/`**，勿与全目录盲扫 Fig A5 混淆。**（2026-03-28）** **`A-Opt-03` / `A-Opt-03w`** 已三 seed 归档；训练期 **`best_epoch`/`val` 指标** 汇总见 **`plots/optimization/prenorm_A_Opt03_vs_Opt03w/best_metrics.csv`**（与历史 run 同 CSV 并存时注意按目录名筛选 `prenorm_tw22205`）。  
> **区域定义（2026-03-26）**：默认 key / 区间 / 阈值见 [任务A分区域评估口径](../../00-规范与记录/任务A分区域评估口径.md)。  
> 当前效率口径为：测试病例 `slow/GUO_XI_JIANG`（81 snapshots）、`n_warmup=5`、`n_runs=20`，并已汇总 3 个 seed；主结果表中的 `Infer(ms)` 与 `Mem(MB)` 使用 `full_case_per_snapshot_ms` 和 `full_case_peak_memory_mb` 的 `mean ± std`。  
> **（2026-03-26 重要）主指标口径更新**：下表 `RMSE_|v|` 列**保留原 all-node 口径**以便纵向对比；自本次起所有出图脚本默认 `--region interior`，论文主结论应以 `interior.RMSE_|v|`（见各 run 实验记录摘要中「内部点误差」）为准；`all.RMSE_|v|` 仅作补充。  
> **（2026-03-29）主表 / 近壁汇报列**：`plots/summary/fig_A1_main_table.csv` 默认以 `interior` 导出主区域 **`rmse_* / r2_*`**，并附带 **`all_rmse_vel_mag / all_r2_vel_mag`** 与 **`near_wall_rmse_* / near_wall_r2_*`**（含 |v| 的 **`r2_vel_mag`**）；各 run 需已生成 `predictions_test/regional_eval/fig_A5_regional_metrics.json`（必要时重跑 `plot_taskA_regional_bar`），详见 [任务A分区域评估口径](../../00-规范与记录/任务A分区域评估口径.md) 第 5–6 节。

<!-- 转置 + 紧凑：列为各 Exp，行为指标，避免 15 列宽表在预览中撑出水平滚动 -->
<table border="1" cellspacing="0" cellpadding="1" style="border-collapse: collapse; font-size: 0.76em; line-height: 1.15; width: 100%; table-layout: fixed;">
<thead>
<tr>
<th style="width: 9em;">指标</th>
<th>A-Base-01</th>
<th>A-Base-02</th>
<th>A-Base-03</th>
<th>A-Main-01</th>
<th>A-Opt-01</th>
<th>A-Opt-02</th>
<th>A-Opt-02_warmup</th>
<th>A-Opt-03</th>
<th>A-Opt-03w</th>
</tr>
</thead>
<tbody>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">Model</th>
<td>MLP</td>
<td>GraphSAGE</td>
<td>Transformer</td>
<td>Transformer</td>
<td>Transformer</td>
<td>Transformer</td>
<td>Transformer</td>
<td>Transformer</td>
<td>Transformer</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">Geom</th>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">BC</th>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">is_wall</th>
<td>✗</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">Physics</th>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_u</th>
<td>0.9739±0.0009</td>
<td>0.9309±0.0013</td>
<td>0.9337±0.0011</td>
<td><strong>0.8977±0.0073</strong></td>
<td>0.8737±0.0056</td>
<td>0.8839±0.0084</td>
<td>0.8843±0.0095</td>
<td>0.8675±0.0011</td>
<td>0.8705±0.0018</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_v</th>
<td>0.9755±0.0012</td>
<td>0.9165±0.0049</td>
<td>0.9170±0.0043</td>
<td><strong>0.8518±0.0113</strong></td>
<td>0.8231±0.0023</td>
<td>0.8383±0.0150</td>
<td>0.8353±0.0130</td>
<td>0.8097±0.0015</td>
<td>0.8176±0.0032</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_w</th>
<td>0.9743±0.0008</td>
<td>0.8503±0.0031</td>
<td>0.8482±0.0047</td>
<td><strong>0.6957±0.0290</strong></td>
<td>0.6502±0.0011</td>
<td>0.6781±0.0351</td>
<td>0.6729±0.0387</td>
<td>0.6459±0.0019</td>
<td>0.6449±0.0016</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_|v| (all)</th>
<td>1.5999±0.0015</td>
<td>1.3611±0.0159</td>
<td>1.3645±0.0101</td>
<td><strong>1.1612±0.0383</strong></td>
<td>1.0811±0.0090</td>
<td>1.1132±0.0621</td>
<td>1.1096±0.0351</td>
<td>1.0310±0.0051</td>
<td>1.0665±0.0133</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">Interior RMSE_|v|</th>
<td><strong>2.6924±0.0242</strong></td>
<td><strong>2.3165±0.0452</strong></td>
<td><strong>2.3330±0.0148</strong></td>
<td><strong>2.0668±0.0492</strong></td>
<td><strong>1.9187±0.0172</strong></td>
<td><strong>1.9727±0.0758</strong></td>
<td><strong>1.9648±0.0682</strong></td>
<td><strong>1.8222±0.0072</strong></td>
<td><strong>1.8883±0.0064</strong></td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_p</th>
<td>0.6581±0.0185</td>
<td>0.7341±0.0136</td>
<td>0.7061±0.0349</td>
<td><strong>0.6536±0.0423</strong></td>
<td>0.6200±0.0362</td>
<td>0.6100±0.0219</td>
<td>0.6419±0.0349</td>
<td>0.6418±0.0385</td>
<td>0.6539±0.0202</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">R2_p</th>
<td>0.9201±0.0045</td>
<td>0.9007±0.0037</td>
<td>0.9079±0.0092</td>
<td><strong>0.9209±0.0103</strong></td>
<td>0.9290±0.0082</td>
<td>0.9313±0.0049</td>
<td>0.9239±0.0083</td>
<td>0.9238±0.0093</td>
<td>0.9211±0.0048</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">Infer (ms)</th>
<td><strong>0.54±0.27</strong></td>
<td><strong>2.35±0.23</strong></td>
<td>6.95±0.09</td>
<td><strong>6.88±0.02</strong></td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">Mem (MB)</th>
<td><strong>127.34±0.00</strong></td>
<td><strong>529.69±0.47</strong></td>
<td>2182.74±1.08</td>
<td><strong>2182.12±0.00</strong></td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
<td>—</td>
</tr>
</tbody>
</table>

> **注**：效率图现已包含 `mean±std` 汇总图、分 seed 延迟图、分 seed 显存图、全病例峰值显存图和分 seed Pareto 图；`speedup_vs_CFD` 仍无法填写，因为 `cfd_time_hours` 为空。  
> `**A-Opt-01` / `A-Opt-02` / `A-Opt-02_warmup` / `A-Opt-03` / `A-Opt-03w` 效率列**：与 `A-Main-01` 同 backbone 量级（结构均为 `FieldTransformer` + geom；`A-Opt-02` 及以后为 Pre-Norm；`A-Opt-03` 叠 `target_weights`；`*_warmup` 为调度差异），本次未重跑 `run_efficiency_benchmark`；部署开销可按主模型同款数量级引用，若论文需要独立条请再补测。  
> **Interior RMSE_|v| 数据来源**：各实验记录摘要中的「内部点误差」，均来自 `regional_eval/fig_A5_regional_metrics.json` 的 `interior.rmse_vel_mag`。

---

## 实验记录摘要（每完成一组填写）

### A-Base-01

- **完成日期**：2026-03-22（seed 1/2/3 全部完成）
- **seed**：1, 2, 3（best epoch 分别为 43, 71, 77）
- **RMSE_|v|**：1.5999 ± 0.0015（各 seed：1.6003 / 1.6015 / 1.5980）
- **RMSE_p**：0.6581 ± 0.0185（各 seed：0.6770 / 0.6643 / 0.6329）
- **R2_p**：0.9201 ± 0.0045
- **R2_u/v/w**：0.050 / 0.088 / 0.109（速度各分量 R² 极低，接近无效）
- **壁面点误差**：`RMSE_|v| = 0.3176 ± 0.0343`（各 seed：0.3551 / 0.2879 / 0.3098）；来源：各 run `predictions_test/regional_eval/fig_A5_regional_metrics.json` 中 `wall.rmse_vel_mag`
- **内部点误差**：`RMSE_|v| = 2.6924 ± 0.0242`（各 seed：2.6678 / 2.7162 / 2.6933）；来源：`interior.rmse_vel_mag`（**非** `all` 区域；测试集全节点聚合见下「regional `all`」）
- **regional `all` 区域 `rmse_vel_mag`（便于与 summary 对照）**：`1.5761 ± 0.0092`（各 seed：1.5673 / 1.5857 / 1.5754）
- **推理时间**：`0.54 ± 0.27 ms / snapshot`（seed1/2/3：`0.85 / 0.38 / 0.39`）
- **峰值显存**：`127.34 ± 0.00 MB`
- **一句话结论**：MLP 仅凭坐标+BC 能准确预测压力（R2_p=0.920），但**内部点**区域速度误差仍高（约 `2.69`），与 GraphSAGE 等「壁面低、内部高」的分层形态一致；只能作为无图结构下限，不能承担可靠速度场重建。
- **下一步动作**：作为固定下限基准，不再调参；效率上可作为“最快但精度最差”的参考点；与高曲率/近壁/分叉等复杂区域的跨模型对比已具备统一口径（见 `regional_eval` 与 `outputs/field/plots/multimodel_baseline/fig_A5_multimodel`_*）。

---

### A-Base-02

- **完成日期**：2026-03-22（seed 1/2），2026-03-22（seed 3）
- **seed**：1, 2, 3（best epoch 分别为 48, 59, 76）
- **RMSE_|v|**：1.3611 ± 0.0159（各 seed：1.3836 / 1.3504 / 1.3493）
- **RMSE_p**：0.7341 ± 0.0136（各 seed：0.7527 / 0.7206 / 0.7290）
- **R2_p**：0.9007 ± 0.0037
- **R2_u/v/w**：0.132 / 0.195 / 0.322（较 A-Base-01 明显改善，w 分量收益最大）
- **壁面点误差**：`RMSE_|v| = 0.0952 ± 0.0124`（各 seed：0.1075 / 0.0954 / 0.0827）
- **内部点误差**：`RMSE_|v| = 2.3165 ± 0.0452`（各 seed：2.3686 / 2.2877 / 2.2931）
- **推理时间**：`2.35 ± 0.23 ms / snapshot`（seed1/2/3：`2.09 / 2.46 / 2.50`）
- **峰值显存**：`529.69 ± 0.47 MB`
- **一句话结论**：GraphSAGE 引入图结构后，全局节点级 `RMSE_|v|` 降至 `1.3397 ± 0.0266`，壁面区误差被压到 `0.0952 ± 0.0124`，但内部主流区仍高达 `2.3165 ± 0.0452`，说明图结构首先修复的是边界与局部一致性，而不是完全解决内部流场。
- **下一步动作**：已完成图结构对照组角色；在当前 4 组 baseline 中，它提供了较好的精度-速度折中点；与主模型在复杂区域的对比见统一口径下的 `fig_A5` / `fig_A5_multimodel_`*。

---

### A-Base-03

- **完成日期**：2026-03-22（seed 1/2），2026-03-23（seed 3）
- **seed**：1, 2, 3（best epoch 分别为 86, 64, 159；seed3 收敛较慢）
- **RMSE_|v|**：1.3645 ± 0.0101（各 seed：1.3782 / 1.3611 / 1.3542）
- **RMSE_p**：0.7061 ± 0.0349（各 seed：0.7553 / 0.6850 / 0.6780；种子间 std 偏大）
- **R2_p**：0.9079 ± 0.0092
- **R2_u/v/w**：0.127 / 0.194 / 0.325（与 A-Base-02 几乎持平）
- **壁面点误差**：`RMSE_|v| = 0.1100 ± 0.0039`（各 seed：0.1064 / 0.1142 / 0.1096）
- **内部点误差**：`RMSE_|v| = 2.3330 ± 0.0148`（各 seed：2.3280 / 2.3496 / 2.3213）
- **推理时间**：`6.95 ± 0.09 ms / snapshot`（seed1/2/3：`6.90 / 7.06 / 6.90`）
- **峰值显存**：`2182.74 ± 1.08 MB`
- **一句话结论**：Transformer 在无几何特征时与 GraphSAGE 依然几乎持平，不仅总体 `RMSE_|v|` 接近（`1.3499 ± 0.0087` vs `1.3397 ± 0.0266`），壁面与内部点误差也处于同一水平，进一步支持“瓶颈不在 backbone，而在显式几何特征”。
- **下一步动作**：已完成几何特征的直接对照组角色；当前效率与主模型几乎相同但精度明显更差，因此后续优先保留其作为“无几何 Transformer”对照，不再作为部署候选主线。

---

### A-Main-01

- **完成日期**：2026-03-22（seed 1/2），2026-03-23（seed 3）
- **seed**：1, 2, 3（best epoch 分别为 100, 83, 64）
- **RMSE_|v|**：1.1612 ± 0.0383（各 seed：1.1124 / 1.1654 / 1.2059；seed3 略弱）
- **RMSE_p**：0.6536 ± 0.0423（各 seed：0.6054 / 0.6471 / 0.7084；seed3 RMSE_p 偏大）
- **R2_p**：0.9209 ± 0.0103（各 seed：0.9324 / 0.9228 / 0.9075）
- **R2_u/v/w**：0.193 / 0.305 / 0.545（w 分量 R2 从 0.325→0.545，提升 67.7%）
- **壁面点误差**：`RMSE_|v| = 0.0381 ± 0.0024`（各 seed：0.0392 / 0.0354 / 0.0397）
- **内部点误差**：`RMSE_|v| = 2.0668 ± 0.0492`（各 seed：2.0182 / 2.0657 / 2.1165）
- **高曲率区域误差**：`RMSE_|v| = 1.1219 ± 0.0209`
- **近壁区域误差**：`RMSE_|v| = 1.5727 ± 0.0195`
- **分叉区域误差**：`RMSE_|v| = 1.1341 ± 0.0389`
- **主干段误差**：`RMSE_|v| = 0.8795 ± 0.0092`
- **推理时间**：`6.88 ± 0.02 ms / snapshot`（seed1/2/3：`6.89 / 6.89 / 6.86`）
- **峰值显存**：`2182.12 ± 0.00 MB`
- **一句话结论**：加入显式几何特征后，主模型不仅把全局节点级 `RMSE_|v|` 压到 `1.1937 ± 0.0284`，还把壁面区误差进一步压低到 `0.0381 ± 0.0024`，并在高曲率区与分叉区保持明显优势，说明几何特征的收益不是只体现在均值，而是真正作用到了复杂形态区域。
- **下一步动作**：已成为单尺度主线基座；当前与无几何 Transformer 的推理时间和显存几乎持平，但 `RMSE_|v|` 明显更低，说明几何特征带来的主要代价不是部署开销而是特征设计复杂度；**复杂区域标签口径已与 baseline 对齐**，可进入 **A-Abl-01**。

---

### A-Opt-01（P0-1，`target_weights=[2,2,2,0.5]`）

- **完成日期**：2026-03-25（训练三 seed），2026-03-26（测试集导出 + 分区域评估）
- **seed**：1, 2, 3（best epoch 分别为 121, 148, 82）
- **相对 `A-Main-01` 的全局测试指标（`summary.json`，3 seed mean ± std）**：合成 `RMSE` 0.782→**0.750**；`RMSE_|v|` 1.161→**1.081**；`RMSE u/v/w` 均下降；`R² u/v/w` 上升；`RMSE p` 与 `R² p` 未退化（均值略优，属与速度耦合及种子方差范围内现象，报告时建议并列写出）。
- **分区域 `rmse_vel_mag`（`fig_A5_regional_metrics.json` 聚合，与 Main 同一几何 mask 口径）**，3 seed mean ± std：
  - **壁面**：0.0381±0.0024 → **0.0368±0.0081**（各 seed：0.0354 / 0.0296 / 0.0456）
  - **内部点**：2.0668±0.0492 → **1.9187±0.0172**（2.018→**1.919** 量级改善）
  - **近壁**：1.5727±0.0195 → **1.5705±0.0299**（几乎持平）
  - **高曲率**：1.1219±0.0209 → **1.0455±0.0229**
  - **分叉**：1.1341±0.0389 → **1.0327±0.0133**
  - **主干**：0.8795±0.0092 → **0.8249±0.0237**
  - **regional `all*`*：1.1937±0.0284 → **1.1082±0.0100**
- **一句话结论**：在仅调整损失权重的条件下，**内部点与多个复杂区域的速度模长误差明显下降**，近壁区基本持平，符合「把优化压力让给速度分量」的设计预期；可作为当前单尺度主线的优先损失配置候选。
- **下一步动作**：**`A-Opt-03` / `A-Opt-03w` 已归档（2026-03-28）**；P0 默认基座见该两组摘要；容量扩展见 **`A-Opt-04`**。

---

### A-Opt-02（P0-2，`use_transformer_prenorm=true`，Pre-Norm `FieldTransformer`）

- **完成日期**：2026-03-27（训练三 seed；同日正式导出测试与图）
- **seed**：1, 2, 3（`best_epoch` 分别约 144, 148, 64；seed3 明显较早停且测试偏弱）
- **输出目录**：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_split_AG_v1_seed{1,2,3}_20260327_*`
- **相对 `A-Main-01` 的全局测试指标（`summary.json`，3 seed mean ± std）**：合成 `RMSE` **0.782 → 0.761**；`test loss` **2.445 → 2.317**；`RMSE_|v|`（`test_metrics`）**1.161 → 1.113**；`RMSE u/v/w` 与 `RMSE p` 均值略优或持平；`R² p` **0.921 → 0.931**。
- **分区域 `rmse_vel_mag`（`regional_eval/fig_A5_regional_metrics.json`）**，相对 Main（上文 `A-Main-01` 摘要）三 seed mean ± std：
  - **壁面**：0.0381±0.0024 → **0.0414±0.0076**（均值略升、方差更大）
  - **内部点**：2.0668±0.0492 → **1.9727±0.0758**
  - **近壁**：1.5727±0.0195 → **1.5087±0.0071**
  - **高曲率**：1.1219±0.0209 → **1.0759±0.0433**
  - **分叉**：1.1341±0.0389 → **1.0821±0.0596**
  - **主干**：0.8795±0.0092 → **0.8522±0.0261**
  - **regional `all`**：1.1937±0.0284 → **1.1394±0.0439**
- **误差分析图**：各 run `predictions_test/error_analysis_interior/`（`plot_error_analysis --region interior`）。
- **与 Main 对照散点**：`outputs/field/plots/optimization/prenorm_A_Opt02_vs_Main01/fig_A3_multimodel_scatter_{vel_mag,p}_interior_geo_only_seed{1,2,3}.png`
- **一句话结论**：仅加 Pre-Norm 时，**内部与多个复杂区域的速度模长误差均值较 Main 下降**，但**壁面区略退化且种子间方差仍在**；与 P0-1 的互补性已由 **`A-Opt-03`**（2026-03-28）验证。
- **下一步动作**：**`A-Opt-03` / `A-Opt-03w` 已归档**；见下文摘要。

---

### A-Opt-02_warmup（P0-3，`A-Opt-02` + `optim.warmup_epochs=5`）

- **完成日期**：2026-03-27（训练三 seed；同日补全测试导出、`regional_eval`、`error_analysis_interior` 与三模型对照图）
- **seed**：1, 2, 3（`best_epoch` 分别约 **93、70、113**；相对 **`A-Opt-02`** 的 seed3 早停于 ~**64**，warmup 将 **seed3 `best_epoch` 延后** 且 **缓解该 seed 偏弱**）
- **输出目录**：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_warmup5_split_AG_v1_seed{1,2,3}_20260327_*`（`meta.exp_id`：`A-Opt-02_warmup`）
- **相对 `A-Main-01`（`summary.json`，3 seed mean ± std）**：合成 `RMSE` **0.782 → 0.766**（仍优于 Main）；**`rmse_vel_mag`（全图节点口径）** **1.161 → 1.110**；**`RMSE p`** 均值 **0.654 → 0.642**（优于 Main、**差于** **`A-Opt-02` 的 0.610**）
- **相对 `A-Opt-02`（同口径）**：**`rmse_vel_mag` 均值略降**（**1.113 → 1.110**）；**合成 `RMSE` 略升**（**0.761 → 0.766**）；**`RMSE p` 明显变差**（**0.610 → 0.642**）；**`interior.rmse_vel_mag`（`regional_eval`）** 均值 **1.973 → 1.965**（小幅改善；**seed3** 单 seed **约 2.06 → 1.91** 量级）
- **分区域 `rmse_vel_mag`（相对 `A-Opt-02`，三 seed mean ± std）**：**壁面** 约 **0.041 → 0.039**（略回落）；**内部 / 近壁 / 高曲率 / 分叉 / 主干** 与无 warmup 线互有小幅胜负，见各 run `fig_A5_regional_metrics.json` 与汇总图
- **三模型对照图（Main / P0-2 / P0-3）**：`outputs/field/plots/optimization/prenorm_Main_P02_P02w/`（`fig_A3_multimodel_scatter_*`、`fig_A5_multimodel_regional_bar_*`、`fig_A4_multimodel_per_case_boxplot_interior_exp_subset.png`）；重生成：`python -m training.scripts.regenerate_p02_warmup_comparison_figures`
- **一句话结论**：**5 epoch 线性 warmup 主要改善 Pre-Norm 线上的「差种子」与早停分布**，全图 **`RMSE_|v|` 均值略优于无 warmup**；**压力 `RMSE p` 三 seed 均值未优于 `A-Opt-02`**，属 **trade-off**。**（2026-03-28）** **`A-Opt-03`** 主结论已出：**组合线默认无需叠 `A-Opt-03w`**；是否在其他支线上默认开 warmup 仍可个案决定。
- **下一步动作**：**`A-Opt-03` / `A-Opt-03w` 已归档**；见下文摘要。

---

### A-Opt-03（P0-4，`target_weights=[2,2,2,0.5]` + Pre-Norm）

- **完成日期**：2026-03-28（训练三 seed；与 `experiment_index.csv` 对齐）
- **seed**：1, 2, 3（`best_epoch` 分别约 **64、83、89**；见 `plots/optimization/prenorm_A_Opt03_vs_Opt03w/best_metrics.csv`）
- **输出目录**：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_split_AG_v1_seed{1,2,3}_20260327_*`
- **相对 `A-Main-01` / `A-Opt-01` / `A-Opt-02`（`summary.json` · `test_metrics`，3 seed mean ± std）**：**`rmse_vel_mag`** **1.161 → 1.031**（Main）、**1.081 → 1.031**（Opt-01）、**1.113 → 1.031**（Opt-02）；**合成 `RMSE`** **0.782 → 0.748**（Main）、**0.750 → 0.748**（Opt-01）、**0.761 → 0.748**（Opt-02）；**`rmse_p`** **相对 `A-Opt-01` / `A-Opt-02` 变差**（约 **0.620 / 0.610 → 0.642**），仍**优于 Main ~0.654**。
- **`regional_eval`（3 seed mean ± std）**：**`all.rmse_vel_mag`** **1.108 → 1.052**（Opt-01）、**1.139 → 1.052**（Opt-02）；**`interior.rmse_vel_mag`** **1.919 → 1.822**（Opt-01）、**1.973 → 1.822**（Opt-02）；**`wall.rmse_vel_mag`** **0.0368 → 0.0315**（Opt-01）、**0.0414 → 0.0315**（Opt-02）。**`all.rmse_p`** 与 **`A-Opt-02`** 持平（~**0.398**），优于 **`A-Opt-01`**（~**0.460**）；**`interior.rmse_p`** **~0.470**，略差于 **`A-Opt-02` ~0.440**，优于 **`A-Opt-01` ~0.535**。
- **内部点 `R²_u/v/w`（`interior`，3 seed 均值）**：**0.317 / 0.376 / 0.464**，相对 **`A-Opt-01`**（~**0.290 / 0.350 / 0.441**）与 **`A-Opt-02`**（~**0.274 / 0.301 / 0.409**）均提升。
- **后处理**：各 run 已具备 `predictions_test/`、`error_analysis_interior/`、`regional_eval/`。
- **一句话结论**：**P0-1 与 P0-2 在速度主指标上呈互补**：组合后 **全局与内部 `RMSE_|v|` 为当前 P0 线最优**，**壁面 `rmse_vel_mag` 相对两条单改线一并改善**，**内部速度分量 R² 同步提升**。**代价**：**`summary` 口径 `RMSE p` 三 seed 均值不及单独 `A-Opt-01` / `A-Opt-02`**，须在论文/汇报中写成 **trade-off**。
- **下一步动作**：推进 **`A-Opt-04`**（`hidden_dim=256`，以 **`A-Opt-03`** 为配置模板）；**`A-Opt-03w` 可不作为默认分支**。

---

### A-Opt-03w（`A-Opt-03` + `optim.warmup_epochs=5`）

- **完成日期**：2026-03-28（训练三 seed）
- **seed**：1, 2, 3（`best_epoch` 分别约 **100、70、64**）
- **输出目录**：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_warmup5_split_AG_v1_seed{1,2,3}_20260328_*`
- **相对 `A-Opt-03`（同口径）**：**`summary.test_metrics.rmse_vel_mag`** **1.031 → 1.067**；**`interior.rmse_vel_mag`（`regional_eval`）** **1.822 → 1.888**；**`rmse_p`** **0.642 → 0.654**。
- **一句话结论**：在 **已叠 `target_weights` 的 Pre-Norm 主线** 上，**5 epoch warmup 未带来相对 `A-Opt-03` 的速度收益**，压力侧仍偏弱——**默认基座选 `A-Opt-03` 即可**。
- **下一步动作**：同 **`A-Opt-03`**；归档已完成。
