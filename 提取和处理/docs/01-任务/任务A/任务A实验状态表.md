# 任务A 实验状态总表

> 本表是任务 A 所有实验的唯一执行状态追踪文件。  
> 每次启动、完成或失败一组实验，**必须更新此表**。  
> 上位文档：[任务A实验清单](任务A实验清单.md) / [任务A冻结卡](任务A冻结卡.md)

> **路线说明（2026-04-01）**：
> - 现有 `A-Base-* / A-Opt-* / Line G / Line W` 统一视为 **`Route-KNN-GNN-V1`** 历史结果
> - 新增修正路线 **`Route-PhysicsAware-V2`** 的执行矩阵见：[任务A V2修正路线实验矩阵](任务A_V2修正路线实验矩阵.md)
> - V2 Gate-0 的具体执行细节见：[任务A V2准备执行清单](任务A_V2准备执行清单.md)
> - V2 首轮 5 组训练实验的统一判定口径见：[任务A V2首轮判定与汇报模板](任务A_V2首轮判定与汇报模板.md)
> - 本状态表自今日起同时追踪 **V1 历史结果** 与 **V2 新实验**

---

## V2：阶段 0 准备与首轮路线对照（2026-04-01 新增）

> 说明：本区只追踪 **`Route-PhysicsAware-V2`**。  
> 命名规则：
> - `V2-Ref-*`：V2 数据口径下的公共参考实验
> - `V2G-*`：`Route-MeshGNN-V2`
> - `V2P-*`：`Route-PointCloud-V2`
> 执行细节见：[任务A V2修正路线实验矩阵](任务A_V2修正路线实验矩阵.md)。

| Exp ID | 类型 | 研究问题 / 任务 | split_version | seeds | 当前状态 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| `V2-Prep-01` | 数据准备 | `.cas/.msh` 是否能稳定导出 mesh 邻接与壁面区域 | `split_AG_v2` | - | 🔒 未开始 | Gate-0，未通过则 `V2G-*` 不得开跑 |
| `V2-Prep-02` | 数据准备 | 采样点能否回溯到原始 mesh 或可信局部邻域 | `split_AG_v2` | - | 🔒 未开始 | 与 V2 采样策略绑定 |
| `V2-Prep-03` | 数据准备 | WSS 真值/壁面节点能否稳定对齐 | `split_AG_v2` | - | 🔒 未开始 | Gate-0，未通过则 WSS 主评估延后 |
| `V2-Prep-04` | 数据准备 | V2 数据生成与评估链路 smoke test | `split_AG_v2` | - | 🔒 未开始 | 2~3 病例最小闭环 |
| `V2-Ref-Base-01` | 训练实验 | V2 数据口径下的点级统一下限 | `split_AG_v2` | [1,2,3] | 🔒 未开始 | 建议 `coords + t + BC` |
| `V2G-Base-01` | 训练实验 | mesh-aware GNN 在无 geometry 下的基线能力 | `split_AG_v2` | [1,2,3] | 🔒 未开始 | `coords + t + BC + is_wall` |
| `V2G-Main-01` | 训练实验 | geometry 在 MeshGNN 上是否成立 | `split_AG_v2` | [1,2,3] | 🔒 未开始 | `coords + t + BC + geometry + is_wall` |
| `V2P-Base-01` | 训练实验 | point-cloud 主干在无 geometry 下的基线能力 | `split_AG_v2` | [1,2,3] | 🔒 未开始 | PointNet++/Point Transformer 二选一 |
| `V2P-Main-01` | 训练实验 | geometry 在 point-cloud 主干上是否成立 | `split_AG_v2` | [1,2,3] | 🔒 未开始 | 与 `V2P-Base-01` 同 backbone |

---

## 战略锚点（2026-03-31）

> **作用域说明（2026-04-01）**：本节锚点仅适用于 **`Route-KNN-GNN-V1`** 的历史延展实验，不自动外推到 V2。

后续**消融实验（第二批及以后的新开跑）**、**Line G（显式几何增强）**与 **Line W（壁面导向）**的主线，统一以 **`A-Opt-05`** 为**配置母版**（`hidden_dim=256`，`num_layers=4`，`target_weights=[2,2,2,0.5]` + Pre-Norm `FieldTransformer`；目录见第三批 **`A-Opt-05`**）。

- **为何从 03 切到 05**：在 **`interior.rmse_vel_mag`** 三 seed 均值上 05 略优于 03；在 **`near_wall` 等紧邻壁面的内部区域**上 05 略优于 03，更贴合端到端链路对梯度/WSS 前置质量的需求（详见各 run `regional_eval/fig_A5_regional_metrics.json` 及「实验记录摘要 · A-Opt-05」）。
- **仍保留 03 的角色**：**`A-Opt-03`（`hidden_dim=128`）**作为 **P0-4 历史锚点**与 **轻量、方差较小、部署成本更低**的对照线；汇报 Pareto/效率时建议 **03 与 05 并列**，避免读者误以为只有一条 P0 线。
- **操作约定**：生成新消融 JSON 时，以 **`A-Opt-05`** 某 seed 的 **`config.snapshot.json`** 为复制基准，仅改「消融项」对应字段；与四组 **baseline**（`A-Base-*` / `A-Main-01`）的叙事对比不变，**控制变量母版**改为 05。

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
| ⏹ 已结案           | 仅完成计划内 seed；判定无需补种，作弱/负向结果归档 |
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
| A-Abl-01-01 | 输入特征消融  | coords + t 仅坐标+时间  | split_AG_v1   | [1]   | 🔒 未开始 | baseline ✅；**新开跑母版 `A-Opt-05`**（见上「战略锚点」） |
| A-Abl-01-02 | 输入特征消融  | + BC               | split_AG_v1   | [1]   | 🔒 未开始 |                        |
| A-Abl-01-03 | 输入特征消融  | + is_wall          | split_AG_v1   | [1]   | 🔒 未开始 |                        |
| A-Abl-01-04 | 输入特征消融  | + geometry（无 wall） | split_AG_v1   | [1]   | 🔒 未开始 |                        |
| A-Abl-02-01 | 几何分量消融  | 去掉 Abscissa        | split_AG_v1   | [1,2,3] | ✅ 已完成 | **（2026-04-09）** 母版 **`A-Opt-05`**；`outputs/field/field_transformer_*_abl02_no_abscissa_split_AG_v1_seed{1,2,3}_*`；**`interior.rmse_vel_mag`** 三 seed 均值 **1.868±0.026**（相对 05 **+0.052**，paired *t* **p≈0.26**）。见「实验记录摘要 · A-Abl-02」。 |
| A-Abl-02-02 | 几何分量消融  | 去掉 NormRadius      | split_AG_v1   | [1,2,3] | ✅ 已完成 | 同上；`*_abl02_no_normradius_*`；均值 **2.043±0.020**（**+0.227**，**p≈0.0015**）— **NormRadius 最关键**。 |
| A-Abl-02-03 | 几何分量消融  | 去掉 Curvature       | split_AG_v1   | [1,2,3] | ✅ 已完成 | 同上；`*_abl02_no_curvature_*`；均值 **1.851±0.030**（**+0.035**，**p≈0.26**）— **影响最弱**。 |
| A-Abl-02-04 | 几何分量消融  | 去掉 Tangent         | split_AG_v1   | [1,2,3] | ✅ 已完成 | 同上；`*_abl02_no_tangent_*`；均值 **1.933±0.038**（**+0.117**，**p≈0.073**）— **Tangent 次之**。 |
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
> 推荐执行顺序：`A-Opt-01 -> A-Opt-02 -> A-Opt-02_warmup (P0-3，✅ 2026-03-27) -> A-Opt-03 (✅ 2026-03-28) -> A-Opt-03w (✅ 2026-03-28) -> A-Opt-04 (✅ 2026-03-29) -> A-Opt-05 (✅ 2026-03-29) -> A-Opt-07 (✅ 2026-04-02，**负结果**)`。  
> 推进门槛：只有当上一组同时改善全局 `RMSE_|v|`、内部区 `RMSE_|v|`，且至少一个速度分量 `R²` 明显改善时，才进入下一组容量扩展。  
> **（2026-03-31）** 容量线执行到此为止；**新开跑优化/消融/Line G** 以 **`A-Opt-05`** 为母版（见篇首「战略锚点」），**`A-Opt-03`** 保留为轻量对照。


| Exp ID          | 研究问题                             | 唯一变化项                              | split_version | seeds   | 当前状态   | 备注                                                                                                                                                                                                                                                                 |
| --------------- | -------------------------------- | ---------------------------------- | ------------- | ------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| A-Opt-01        | 速度权重是否能改善内部流场                    | `target_weights = [2,2,2,0.5]`     | split_AG_v1   | [1,2,3] | ✅ 已完成  | 三 seed 训练与 `**predict_field` + `plot_taskA_regional_bar`** 已完成（2026-03-26）。目录：`outputs/field/field_transformer_coord_t_bc_geom_wall_tw22205_split_AG_v1_seed{1,2,3}_*/`；见下文「实验记录摘要 · A-Opt-01」与主结果表增行。                                                             |
| A-Opt-02        | LayerNorm 是否提升单尺度 Transformer 表达 | `FieldTransformer` 改为 Pre-Norm 残差块 | split_AG_v1   | [1,2,3] | ✅ 已完成  | 三 seed：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_split_AG_v1_seed{1,2,3}_20260327_*`；`predict_field` + `error_analysis_interior` + `regional_eval` + 与 Main 对照 `**plots/prenorm_A_Opt02_vs_Main01/**`（2026-03-27）。见「实验记录摘要 · A-Opt-02」与主结果表增行。 |
| A-Opt-02_warmup | 学习率 Warmup 是否稳定 Pre-Norm 训练并改善指标 | `A-Opt-02 + optim.warmup_epochs=5` | split_AG_v1   | [1,2,3] | ✅ 已完成  | **P0-3**（2026-03-27）：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_warmup5_split_AG_v1_seed{1,2,3}_20260327_*`；已补 **`predictions_test`**、**`error_analysis_interior`**、**`regional_eval`**；与 Main / P0-2 三模型对照图 **`outputs/field/plots/optimization/prenorm_Main_P02_P02w/`**；一键重汇总结图：`python -m training.scripts.regenerate_p02_warmup_comparison_figures`。见「实验记录摘要 · A-Opt-02_warmup」与主结果表增列。 |
| A-Opt-03        | 损失重加权与 LayerNorm 是否互补            | `A-Opt-01 + A-Opt-02`              | split_AG_v1   | [1,2,3] | ✅ 已完成  | **P0-4（2026-03-28）**：三 seed 已训练并归档；`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_split_AG_v1_seed{1,2,3}_20260327_*`；已含 **`predictions_test`**、**`error_analysis_interior`**、**`regional_eval`**；训练期 **best_epoch** 约 **64 / 83 / 89**。汇总对比 CSV：`outputs/field/plots/optimization/prenorm_A_Opt03_vs_Opt03w/best_metrics.csv`。**速度侧**：相对 **`A-Opt-01` / `A-Opt-02` / `A-Main-01`**，`interior.rmse_vel_mag` 与 **`summary.test_metrics.rmse_vel_mag`** 三 seed 均值在 **h128 P0 线**上最优；**内部点 `R²_u/v/w`** 均值亦优于单独 P0-1 与 P0-2。**压力侧 trade-off**：**`summary.test_metrics.rmse_p`** 三 seed 均值 **差于 `A-Opt-01`（~0.620）与 `A-Opt-02`（~0.610）**；**`regional_eval` · `all.rmse_p`** 与 **`A-Opt-02`** 持平（~0.398），**`interior.rmse_p`** 略差于 **`A-Opt-02`**。**（2026-03-31）** **轻量/效率对照**；**新开跑母版见 `A-Opt-05`**（篇首「战略锚点」）。见「实验记录摘要 · A-Opt-03」与 [优化路径](任务A优化路径与近期实验建议.md) P0-4 节。 |
| A-Opt-03w       | Warmup 是否进一步稳定组合线                  | `A-Opt-03 + warmup_epochs=5`       | split_AG_v1   | [1,2,3] | ✅ 已完成  | **（2026-03-28）** 三 seed：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_warmup5_split_AG_v1_seed{1,2,3}_20260328_*`；后处理链同 **`A-Opt-03`**；**`best_epoch`** 约 **100 / 70 / 64**。**相对 `A-Opt-03`**：**`summary` / `regional_eval` 速度主指标未更好**，**`rmse_p` 未收复**——与 P0-3 类似，**非组合线必选项**。**（2026-03-31）** 后续新开跑**不以 03w 为母版**；轻量对照见 **`A-Opt-03`**，主母版见 **`A-Opt-05`**。见「实验记录摘要 · A-Opt-03w」。                                                                                                                                                                                                                                                                  |
| A-Opt-04        | 容量扩大是否继续有效                       | `hidden_dim = 256`                 | split_AG_v1   | [1,2,3] | ✅ 已完成 | **（2026-03-29）** 三 seed：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_h256_split_AG_v1_seed{1,2,3}_20260328_*`；已含 **`predictions_test`**、**`error_analysis_interior`**、**`regional_eval`**；**`best_epoch`** 约 **80 / 70 / 82**。相对 **`A-Opt-03`**：**`interior.rmse_vel_mag`** 三 seed 均值 **变差**（约 **1.822 → 1.849**）；**`summary.test_metrics.rmse_p`** 略优。**角色**：**`A-Opt-05` 的中间态**（仅加宽）；**不作为**后续消融母版——母版取 **`A-Opt-05`（加宽+4L）**。 |
| A-Opt-05        | 适度加深是否继续有效                       | `hidden_dim = 256, num_layers = 4` | split_AG_v1   | [1,2,3] | ✅ 已完成 | **（2026-03-29）** 三 seed：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_h256_l4_split_AG_v1_seed{1,2,3}_20260328_*`；后处理链同 **`A-Opt-04`**。**相对 `A-Opt-04`**：**`interior.rmse_vel_mag` 均值回落**（约 **1.849 → 1.816**），**略优于 `A-Opt-03` 的 1.822**。**近壁等区域**相对 03 **略优**（见各 run **`regional_eval`**）。**（2026-03-31）** 定为 **后续消融 / Line G / Line W 的配置母版**；**trade-off**：显存/时延/跨 seed 方差高于 03。**不启动 `A-Opt-06`** 见优化路径。 |
| A-Opt-05t_wu10  | 10ep warmup 是否稳定 A-Opt-05 训练      | `A-Opt-05 + warmup_epochs=10`      | split_AG_v1   | [1,2]   | ✅ 已完成 | **（2026-03-29）** seed1/2：`*_h256_l4_wu10_split_AG_v1_seed{1,2}_20260329_*`。seed1：RMSE_\|v\|=1.0485，interior=1.8512；**全图与内部均略差于 A-Opt-05 seed1（1.0477/1.8028）**，warmup 在 05 上无明显收益，不作为默认配置。 |
| A-Opt-05t_lr3e4 | lr=3e-4 是否改善 A-Opt-05 精度         | `A-Opt-05_wu10 + lr=3e-4`          | split_AG_v1   | [1]     | ✅ 已完成 | **（2026-03-29）** seed1：`*_h256_l4_wu10_lr3e4_split_AG_v1_seed1_20260329_222839`。RMSE_\|v\|=1.0433，interior=1.8990，RMSE_p=**0.6364**，R2_p=**0.9254**（本组压力最优）。内部速度不如 A-Opt-05 均值；**压力端有潜力，建议补 seed2/3 再确认**。 |
| A-Opt-05t_wd2e4 | wd=2e-4 是否改善正则化效果               | `A-Opt-05_wu10 + wd=2e-4`          | split_AG_v1   | [1]     | ✅ 已完成 | **（2026-03-29）** seed1：`*_h256_l4_wu10_wd2e4_split_AG_v1_seed1_20260329_222849`。RMSE_\|v\|=1.1871，interior=2.0648，RMSE_p=0.7711——**明显变差，不可取**。 |
| A-Opt-05t_sch15 | schedpat=15 是否优于默认调度             | `A-Opt-05_wu10 + scheduler_patience=15` | split_AG_v1 | [1] | ✅ 已完成 | **（2026-03-30）** seed1：`*_h256_l4_wu10_schpat15_split_AG_v1_seed1_20260330_134411`。RMSE_\|v\|=1.0549，interior=1.8464，RMSE_p=0.6640——**未形成主指标稳定优势**；母版仍为 A-Opt-05 骨架，调参分支单独归因。 |
| A-Opt-06        | 单尺度进一步加深是否还值得                    | `hidden_dim = 256, num_layers = 6` | split_AG_v1   | [1]     | 🔒 未开始 | 若 `A-Opt-05` 收益很小，建议停止                                                                                                                                                                                                                                             |
| A-Opt-07        | 内部点区域加权是否进一步改善瓶颈                 | `optim.interior_loss_boost = 3.0`（余同 **A-Opt-05**） | split_AG_v1   | [1,2,3] | ✅ 已完成 | **（2026-04-02）** 三 seed：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_h256_l4_iboost3_split_AG_v1_seed{1,2,3}_20260331_175619`；已 **`predictions_test`** + **`regional_eval`**；与 **`A-Main-01` / `A-Opt-05`** 对照图：`plots/optimization/A_Opt07_vs_Opt05_Main01/`（`python -m training.scripts.regenerate_opt07_vs_opt05_main_figures`）。**相对母版 `A-Opt-05`**：全图 **`rmse_vel_mag`** 与 **`interior.rmse_vel_mag`** 均未更好，**`near_wall` / `wall`** 变差；**`rmse_p`** 略差——**负结果**，母版仍为 **`A-Opt-05`**。见「实验记录摘要 · A-Opt-07」。 |
| A-Opt-08        | 多尺度结构是否带来本质提升                    | graph U-Net / hierarchical GNN     | split_AG_v1   | [1]     | 🔒 未开始 | 单尺度优化见顶后再立项                                                                                                                                                                                                                                                        |


---

## 第四批：显式几何增强线 Line G（2026-03-26 新增）

> 说明：Line G 用于在现有 geometry 已被证明有效的前提下，继续小步增加新的显式几何/拓扑先验。
> 前置条件：**`A-Abl-02` 已于 2026-04-09 三 seed 归档**（见「实验记录摘要 · A-Abl-02」）；可进入 Line G 小步验证。**新开跑一律以 `A-Opt-05` 为母版**（篇首「战略锚点」）。
> 推荐执行顺序：`A-Opt-G01 -> A-Opt-G02 -> (A-Opt-G03 / A-Opt-G04) -> A-Opt-G05`。
> 推进门槛：新增特征必须至少改善一个复杂区域（`near_wall / bifurcation / high_curvature`），且验证/测试不能退化。


| Exp ID    | 研究问题                | 唯一变化项                                         | split_version | seeds | 当前状态   | 备注                            |
| --------- | ------------------- | --------------------------------------------- | ------------- | ----- | ------ | ----------------------------- |
| A-Opt-G01 | 显式分叉拓扑先验是否改善复杂转折区建模 | `dist_to_bifurcation + branch_id`（与 JSON 一致） | split_AG_v1   | [1,2,3] | ✅ 已完成 | **（2026-04-12～04-13）** 三 seed：`outputs/field/field_transformer_opt05_g01_bifurcation_split_AG_v1_seed*_20260412_*` / `*_20260413_113440`；`predictions_test` + 标准后处理齐。配置：`training/configs/field/generated/line_g/A-Opt-G01_seed*.json`。详见「实验记录摘要 · Line G」。 |
| A-Opt-G02 | 局部尺度变化信息是否优于单纯半径值   | `dR_ds`                                         | split_AG_v1   | [1]     | ⏹ 已结案 | **（2026-04-12）** seed1：`outputs/field/field_transformer_opt05_g02_dRds_split_AG_v1_seed1_20260412_231304`。**（2026-04-17）** 单 seed 相对 **`A-Opt-05`** 无收益，**不补 seed2/3**。配置：`training/configs/field/generated/line_g/A-Opt-G02_seed1.json`。详见「实验记录摘要 · Line G」。 |
| A-Opt-G03 | 扭率能否补足曲率缺失的三维弯扭信息   | `torsion`                                       | split_AG_v1   | [1]     | ⏹ 已结案 | **（2026-04-12）** seed1：`outputs/field/field_transformer_opt05_g03_torsion_split_AG_v1_seed1_20260412_231304`。**（2026-04-17）** 单 seed 相对 **`A-Opt-05`** 无收益，**不补 seed2/3**。配置：`training/configs/field/generated/line_g/A-Opt-G03_seed1.json`。详见「实验记录摘要 · Line G」。 |
| A-Opt-G04 | 显式壁面距离是否改善近壁速度剖面学习  | `dist_to_wall` 等                               | split_AG_v1   | [1,2,3] | ✅ 已完成 | **（2026-04-13）** 三 seed：`outputs/field/field_transformer_opt05_g04_wall_distance_split_AG_v1_seed*_20260413_*`。配置：`training/configs/field/generated/line_g/A-Opt-G04_seed*.json`。详见「实验记录摘要 · Line G」。 |
| A-Opt-G05 | 中心线方向变化率是否提升转折区表达   | `d_tangent_ds`                                  | split_AG_v1   | [1,2,3] | ✅ 已完成 | **（2026-04-13）** 三 seed：`outputs/field/field_transformer_opt05_g05_tangent_change_rate_split_AG_v1_seed*_20260413_*`。配置：`training/configs/field/generated/line_g/A-Opt-G05_seed*.json`。详见「实验记录摘要 · Line G」。 |


---

## 第五批：壁面导向优化线 Line W（2026-03-25 新增）

> 说明：Line W 直接面向端到端链路质量（WSS/OSI/RRT → 髂支闭塞风险预测）。与第三批（Line A 内部精度优化）并行推进、独立归因。
> 基座：**（2026-03-31）Line W 与 Line A 后续实验统一以 `A-Opt-05` 为默认起跑配置**（见篇首「战略锚点」：近壁等区域略优、更贴近 WSS 前置质量）。**`A-Opt-03`** 仍可作为 **低开销对照分支**（显存/时延/方差）与高显存线并列汇报；**`A-Opt-03w` 不作为母版**。
> 评估标准差异：Line W 必须额外运行 WSS 后处理对比，以壁面衍生指标质量为核心判定。
> 详见 [任务A优化路径](任务A优化路径与近期实验建议.md) 第 2.4 节和第 5.5 节。


| Exp ID    | 研究问题                  | 唯一变化项                                      | split_version | seeds | 当前状态   | 备注                                    |
| --------- | --------------------- | ------------------------------------------ | ------------- | ----- | ------ | ------------------------------------- |
| A-Opt-W01 | 近壁区域加权是否改善 WSS 梯度质量   | `near_wall_boost=3.0, interior_weight=0.5` | split_AG_v1   | [1]   | 🔒 未开始 | 需修改 `losses.py` + 近壁区 mask；依赖 P0 最优基座 |
| A-Opt-W02 | 壁面法向梯度监督是否提升 WSS 精度   | `wall_grad_weight=0.01`                    | split_AG_v1   | [1]   | 🔒 未开始 | 需修改 `losses.py`；依赖 W01 有正向信号或独立启动     |
| A-Opt-W03 | 直接 WSS 监督是否最大化端到端质量   | **纯 WSS**：`target_weights=0` + `wss_loss_weight=1`（草案 `A-Opt-W03-wss-only`） | split_AG_v1   | [1]   | 🔒 未开始 | **先行批已归档**：场 + WSS 联合见 **`A-*-wss-multi`**（`wss_loss_weight=0.1`，第五批附表），**不等同**于本行「仅 WSS」 |
| A-Opt-W04 | 两阶段训练是否优于一开始就加权       | 阶段1:均匀MSE → 阶段2:壁面精调                       | split_AG_v1   | [1]   | 🔒 未开始 | 可与 W01/W02 叠加                         |
| A-Opt-W05 | OSI 敏感区域加权是否改善 OSI 恢复 | 分叉/高曲率区 boost                              | split_AG_v1   | [1]   | 🔒 未开始 | 可与 W01 叠加                             |

### 第五批附：WSS 多任务辅助监督（`wss_multitask`，2026-04-14～04-17）

> **说明**：在 **`A-Base-*` / `A-Main-01` / `A-Opt-05`** 各自输入与场损失权重不变的前提下，增加 **`model.wss_dim=4`** 与 **`optim.wss_loss_weight=0.1`**。配置：**`training/configs/field/generated/baseline_wss_multitask/`**。登记：**`outputs/field/experiment_index.csv`**；测试集壁面 WSS 指标：**`outputs/field/wss_multitask_test_wall_wss_metrics.tsv`**；场 **R² \|v\|**、**R²_p** 见各 run **`summary.json`**。**R² 汇报主表**见本节下方「**实验记录摘要 · WSS 多任务**」。若以 **`line_g`** 为配置入口：本批跑批与 **`training/configs/field/generated/line_g/`** 无关；**`A-Opt-G02` / `G03`** 为 **⏹ 已结案**（仅 seed1，不补种），见第四批表。

| Exp ID | 研究问题 | 唯一变化项 | split_version | seeds | 当前状态 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| A-Base-01-wss-multi | MLP 基线 + WSS 头 | 同 `A-Base-01` + WSS 多任务 | split_AG_v1 | [1,2,3] | ✅ 训练已完成 | `field_mlp_coord_t_bc_wss_multitask_split_AG_v1_seed{1,2,3}_20260415_*` / `20260416_*` |
| A-Base-02-wss-multi | GraphSAGE + wall + WSS 头 | 同 `A-Base-02` + WSS 多任务 | split_AG_v1 | [1,2,3] | ✅ 训练已完成 | `field_graphsage_coord_t_bc_wall_wss_multitask_split_AG_v1_seed{1,2,3}_20260416_*` |
| A-Base-03-wss-multi | Transformer + wall + WSS 头 | 同 `A-Base-03` + WSS 多任务 | split_AG_v1 | [1,2,3] | ✅ 训练已完成 | `field_transformer_coord_t_bc_wall_wss_multitask_split_AG_v1_seed{1,2,3}_20260416_*` |
| A-Main-01-wss-multi | 全几何 Transformer + WSS 头 | 同 `A-Main-01` + WSS 多任务 | split_AG_v1 | [1,2,3] | ✅ 训练已完成 | `field_transformer_coord_t_bc_geom_wall_wss_multitask_split_AG_v1_seed{1,2,3}_20260416_*` / `20260417_*` |
| A-Opt-05-wss-multi | **05 母版** + WSS 头 | 同 `A-Opt-05` + WSS 多任务 | split_AG_v1 | [1,2,3] | ✅ 训练已完成 | seed1：`..._seed1_20260415_232824`；seed2/3：`..._seed{2,3}_20260414_204752` |

详见「**实验记录摘要 · WSS 多任务**」。**`predictions_test` / Fig A3–A5 全链**已于 **2026-04-17** 闭环（阵列 **Job 2704**、补点 **Job 2715**，见 **`docs/02-推进与变更/代码修改与实验推进记录.md`** 文首条目）。


## 主结果表（3 seed mean ± std，已完成）

> 数据来源：`experiment_index.csv` + 各 run 的 `summary.json` + `predictions_test/error_analysis/summary.json` + `predictions_test/regional_eval/fig_A5_regional_metrics.json` + `outputs/field/plots/efficiency/fig_A7_efficiency_benchmark.json`。  
> **分区域指标（2026-03-24 起）**：`fig_A5_regional_metrics.json` 由 `plot_taskA_regional_bar` 生成，区域 mask 基于各预测文件中的 `graph_path` 图资产（完整 `x`），与训练时 `enabled_node_features` 无关，baseline 四模型横向可比。  
> **（2026-03-26）** 优化线 `A-Opt-01` 在相同口径下已补全 `regional_eval`；默认 `**plot_taskA_multimodel_regional_bar**` 扫描结果包含 **baseline 四组 + `A-Opt-01**`（共 5 组 `exp_id`），需在文中区分「基线四模型对比」与「含 P0-1 的扩展对比」。**（2026-03-27）** `A-Opt-02` 三 seed 已各含 `regional_eval`；多模型 Fig A5 若扫全目录会再纳入 `A-Opt-02`，汇报时注意与「仅 baseline / 仅 P0-1」图区分。**（2026-03-27）** `A-Opt-02_warmup`（P0-3）已对齐同样后处理；三模型（Main / P0-2 / P0-3）Dedicated 汇总见 **`plots/optimization/prenorm_Main_P02_P02w/`**，勿与全目录盲扫 Fig A5 混淆。**（2026-03-28）** **`A-Opt-03` / `A-Opt-03w`** 已三 seed 归档；训练期 **`best_epoch`/`val` 指标** 汇总见 **`plots/optimization/prenorm_A_Opt03_vs_Opt03w/best_metrics.csv`**（与历史 run 同 CSV 并存时注意按目录名筛选 `prenorm_tw22205`）。  
> **区域定义（2026-03-26）**：默认 key / 区间 / 阈值见 [任务A分区域评估口径](../../00-规范与记录/任务A分区域评估口径.md)。  
> 当前效率口径为：测试病例 `slow/GUO_XI_JIANG`（81 snapshots）、`n_warmup=5`、`n_runs=20`，并已汇总 3 个 seed；主结果表中的 `Infer(ms)` 与 `Mem(MB)` 使用 `full_case_per_snapshot_ms` 和 `full_case_peak_memory_mb` 的 `mean ± std`。  
> **（2026-03-26 重要）主指标口径更新**：下表 `RMSE_|v|` 列**保留原 all-node 口径**以便纵向对比；自本次起所有出图脚本默认 `--region interior`，论文主结论应以 `interior.RMSE_|v|`（见各 run 实验记录摘要中「内部点误差」）为准；`all.RMSE_|v|` 仅作补充。  
> **（2026-03-29）主表 / 近壁汇报列**：`plots/summary/fig_A1_main_table.csv` 默认以 `interior` 导出主区域 **`rmse_* / r2_*`**，并附带 **`all_rmse_vel_mag / all_r2_vel_mag`** 与 **`near_wall_rmse_* / near_wall_r2_*`**（含 |v| 的 **`r2_vel_mag`**）；各 run 需已生成 `predictions_test/regional_eval/fig_A5_regional_metrics.json`（必要时重跑 `plot_taskA_regional_bar`），详见 [任务A分区域评估口径](../../00-规范与记录/任务A分区域评估口径.md) 第 5–6 节。  
> **（2026-03-29）** 已归档 **`A-Opt-04`**（`hidden_dim=256`）；下表已增 **`A-Opt-04`** 列（3 seed mean ± std）。  
> **（2026-03-31）** 已归档 **`A-Opt-05`**（`hidden_dim=256, num_layers=4`）并补入 ③ 容量线子表；同时入账 **`A-Opt-05_tune`** 四组小步超参实验（`warmup10` / `lr3e-4` / `wd2e-4` / `schedpat15`，均 seed=1）——详见第三批跟踪表与「实验记录摘要 · A-Opt-05_tune」。**（2026-04-02）** 已归档 **`A-Opt-07`**（`interior_loss_boost=3`，三 seed）并增列 ③ 子表；见「实验记录摘要 · A-Opt-07」。**（2026-04-14）** **Line G**：**`A-Opt-G01` / `G04` / `G05` 三 seed** 已具备 `summary.json` 与 `predictions_test`；主表增列若纳入 Line G，请以状态表「实验记录摘要 · Line G」数值为准。**（2026-04-17）** **`A-*-wss-multi`**（场 + WSS 多任务）三 seed 已入账 `experiment_index.csv`，壁面 WSS 汇总见 **`wss_multitask_test_wall_wss_metrics.tsv`**（第五批附表与「实验记录摘要 · WSS 多任务」）。

<!-- 拆为三张子表：① Baseline 组  ② P0 优化前半（Opt-01～03）  ③ P0 容量线（Opt-03w/04） -->

**① Baseline 组**

<table border="1" cellspacing="0" cellpadding="1" style="border-collapse: collapse; font-size: 0.78em; line-height: 1.15; width: 100%; table-layout: fixed;">
<thead>
<tr>
<th style="width: 9em;">指标</th>
<th>A-Base-01</th>
<th>A-Base-02</th>
<th>A-Base-03</th>
<th>A-Main-01</th>
</tr>
</thead>
<tbody>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">Model</th>
<td>MLP</td>
<td>GraphSAGE</td>
<td>Transformer</td>
<td>Transformer</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">Geom</th>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>✓</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">BC</th>
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
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">Physics</th>
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
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_v</th>
<td>0.9755±0.0012</td>
<td>0.9165±0.0049</td>
<td>0.9170±0.0043</td>
<td><strong>0.8518±0.0113</strong></td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_w</th>
<td>0.9743±0.0008</td>
<td>0.8503±0.0031</td>
<td>0.8482±0.0047</td>
<td><strong>0.6957±0.0290</strong></td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_|v| (all)</th>
<td>1.5999±0.0015</td>
<td>1.3611±0.0159</td>
<td>1.3645±0.0101</td>
<td><strong>1.1612±0.0383</strong></td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">Interior RMSE_|v|</th>
<td><strong>2.6924±0.0242</strong></td>
<td><strong>2.3165±0.0452</strong></td>
<td><strong>2.3330±0.0148</strong></td>
<td><strong>2.0668±0.0492</strong></td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_p</th>
<td>0.6581±0.0185</td>
<td>0.7341±0.0136</td>
<td>0.7061±0.0349</td>
<td><strong>0.6536±0.0423</strong></td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">R2_p</th>
<td>0.9201±0.0045</td>
<td>0.9007±0.0037</td>
<td>0.9079±0.0092</td>
<td><strong>0.9209±0.0103</strong></td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">Infer (ms)</th>
<td><strong>0.54±0.27</strong></td>
<td><strong>2.35±0.23</strong></td>
<td>6.95±0.09</td>
<td><strong>6.88±0.02</strong></td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">Mem (MB)</th>
<td><strong>127.34±0.00</strong></td>
<td><strong>529.69±0.47</strong></td>
<td>2182.74±1.08</td>
<td><strong>2182.12±0.00</strong></td>
</tr>
</tbody>
</table>

**② P0 优化前半（A-Main-01★ 为对照基准；全部 Transformer + Geom + BC + is_wall）**

<table border="1" cellspacing="0" cellpadding="1" style="border-collapse: collapse; font-size: 0.78em; line-height: 1.15; width: 100%; table-layout: fixed;">
<thead>
<tr>
<th style="width: 9em;">指标</th>
<th>A-Main-01 ★</th>
<th>A-Opt-01</th>
<th>A-Opt-02</th>
<th>A-Opt-02_warmup</th>
<th>A-Opt-03</th>
</tr>
</thead>
<tbody>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">PreNorm</th>
<td>✗</td>
<td>✗</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">TW[2,2,2,.5]</th>
<td>✗</td>
<td>✓</td>
<td>✗</td>
<td>✗</td>
<td>✓</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">Warmup</th>
<td>✗</td>
<td>✗</td>
<td>✗</td>
<td>5ep</td>
<td>✗</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_u</th>
<td>0.8977±0.0073</td>
<td>0.8737±0.0056</td>
<td>0.8839±0.0084</td>
<td>0.8843±0.0095</td>
<td><strong>0.8675±0.0011</strong></td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_v</th>
<td>0.8518±0.0113</td>
<td>0.8231±0.0023</td>
<td>0.8383±0.0150</td>
<td>0.8353±0.0130</td>
<td><strong>0.8097±0.0015</strong></td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_w</th>
<td>0.6957±0.0290</td>
<td>0.6502±0.0011</td>
<td>0.6781±0.0351</td>
<td>0.6729±0.0387</td>
<td><strong>0.6459±0.0019</strong></td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_|v| (all)</th>
<td>1.1612±0.0383</td>
<td>1.0811±0.0090</td>
<td>1.1132±0.0621</td>
<td>1.1096±0.0351</td>
<td><strong>1.0310±0.0051</strong></td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">Interior RMSE_|v|</th>
<td>2.0668±0.0492</td>
<td><strong>1.9187±0.0172</strong></td>
<td>1.9727±0.0758</td>
<td>1.9648±0.0682</td>
<td>1.8222±0.0072</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_p</th>
<td>0.6536±0.0423</td>
<td>0.6200±0.0362</td>
<td><strong>0.6100±0.0219</strong></td>
<td>0.6419±0.0349</td>
<td>0.6418±0.0385</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">R2_p</th>
<td>0.9209±0.0103</td>
<td>0.9290±0.0082</td>
<td><strong>0.9313±0.0049</strong></td>
<td>0.9239±0.0083</td>
<td>0.9238±0.0093</td>
</tr>
</tbody>
</table>

**③ P0 容量线（A-Opt-03★ 为对照基准；全部 PreNorm + TW[2,2,2,.5]；A-Opt-07 叠内部监督加权）**

<table border="1" cellspacing="0" cellpadding="1" style="border-collapse: collapse; font-size: 0.78em; line-height: 1.15; width: 100%; table-layout: fixed;">
<thead>
<tr>
<th style="width: 9em;">指标</th>
<th>A-Opt-03 ★</th>
<th>A-Opt-03w</th>
<th>A-Opt-04</th>
<th>A-Opt-05</th>
<th>A-Opt-07</th>
</tr>
</thead>
<tbody>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">hidden_dim</th>
<td>128</td>
<td>128</td>
<td>256</td>
<td>256</td>
<td>256</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">num_layers</th>
<td>3</td>
<td>3</td>
<td>3</td>
<td>4</td>
<td>4</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">Warmup</th>
<td>✗</td>
<td>5ep</td>
<td>✗</td>
<td>✗</td>
<td>✗</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">interior_loss_boost</th>
<td>1</td>
<td>1</td>
<td>1</td>
<td>1</td>
<td><strong>3</strong></td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_u</th>
<td><strong>0.8675±0.0011</strong></td>
<td>0.8705±0.0018</td>
<td>0.8723±0.0058</td>
<td>0.8697±0.0061</td>
<td>0.8709±0.0056</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_v</th>
<td><strong>0.8097±0.0015</strong></td>
<td>0.8176±0.0032</td>
<td>0.8182±0.0072</td>
<td>0.8119±0.0039</td>
<td>0.8149±0.0051</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_w</th>
<td>0.6459±0.0019</td>
<td>0.6449±0.0016</td>
<td><strong>0.6416±0.0010</strong></td>
<td>0.6449±0.0045</td>
<td>0.6470±0.0029</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_|v| (all)</th>
<td><strong>1.0310±0.0051</strong></td>
<td>1.0665±0.0133</td>
<td>1.0519±0.0099</td>
<td>1.0399±0.0082</td>
<td>1.0433±0.0120</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">Interior RMSE_|v|</th>
<td>1.8222±0.0072</td>
<td>1.8883±0.0064</td>
<td>1.8493±0.0121</td>
<td><strong>1.8162±0.0275</strong></td>
<td>1.8207±0.0211</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">RMSE_p</th>
<td>0.6418±0.0385</td>
<td>0.6539±0.0202</td>
<td><strong>0.6386±0.0114</strong></td>
<td>0.6449±0.0022</td>
<td>0.6489±0.0143</td>
</tr>
<tr>
<th scope="row" style="text-align: left; font-weight: normal;">R2_p</th>
<td>0.9238±0.0093</td>
<td>0.9211±0.0048</td>
<td><strong>0.9248±0.0030</strong></td>
<td>0.9234±0.0005</td>
<td>0.9224±0.0035</td>
</tr>
</tbody>
</table>

> **注**：效率图现已包含 `mean±std` 汇总图、分 seed 延迟图、分 seed 显存图、全病例峰值显存图和分 seed Pareto 图；`speedup_vs_CFD` 仍无法填写，因为 `cfd_time_hours` 为空。  
> `**A-Opt-01` / `A-Opt-02` / `A-Opt-02_warmup` / `A-Opt-03` / `A-Opt-03w` / `A-Opt-04` / `A-Opt-05` 效率列**：与 `A-Main-01` 相比，**`A-Opt-04`/`05`（`hidden_dim=256`，05 另 `num_layers=4`）显存与时延更高**；其余与同 backbone 量级叙述类似（**`A-Opt-02` 及以后为 Pre-Norm；`A-Opt-03` 叠 `target_weights`；`*_warmup` 为调度差异**）。**本次仍未重跑 `run_efficiency_benchmark`**；部署开销若需进主文，建议对 **`A-Opt-03` vs `A-Opt-05` 各补 1～3 seed 的独立条**。  
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
- **下一步动作**：**`A-Opt-03` / `A-Opt-03w`、`A-Opt-04` / `A-Opt-05` 已依次归档（至 2026-03-29）**；**（2026-03-31）** 后续消融与几何增强线以 **`A-Opt-05`** 为母版；**`A-Opt-03`** 作轻量对照，见篇首「战略锚点」。

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
- **下一步动作**：**`A-Opt-04` / `A-Opt-05` 已跑完（2026-03-29）**；**（2026-03-31）** 以 **`A-Opt-05`** 为后续实验母版（近壁略优、内部均值略优；方差/成本高于 03）；**`A-Opt-03`** 作 P0-4 轻量对照。容量线细节见 **`A-Opt-04` / `A-Opt-05`** 摘要。**`A-Opt-03w` 可不作为默认分支**。

---

### A-Opt-03w（`A-Opt-03` + `optim.warmup_epochs=5`）

- **完成日期**：2026-03-28（训练三 seed）
- **seed**：1, 2, 3（`best_epoch` 分别约 **100、70、64**）
- **输出目录**：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_warmup5_split_AG_v1_seed{1,2,3}_20260328_*`
- **相对 `A-Opt-03`（同口径）**：**`summary.test_metrics.rmse_vel_mag`** **1.031 → 1.067**；**`interior.rmse_vel_mag`（`regional_eval`）** **1.822 → 1.888**；**`rmse_p`** **0.642 → 0.654**。
- **一句话结论**：在 **已叠 `target_weights` 的 Pre-Norm 主线** 上，**5 epoch warmup 未带来相对 `A-Opt-03` 的速度收益**，压力侧仍偏弱——**03 仅作轻量对照；母版见 `A-Opt-05`**（2026-03-31）。
- **下一步动作**：同 **`A-Opt-03`**；归档已完成。

---

### A-Opt-04（P0-5 容量①，`A-Opt-03` + `hidden_dim = 256`）

- **完成日期**：2026-03-28～03-29（训练三 seed；与仓库根目录维护的 **`outputs/field/experiment_index.csv`** 对齐）
- **seed**：1, 2, 3（`best_epoch` 分别约 **80、70、82**）
- **输出目录**：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_h256_split_AG_v1_seed{1,2,3}_20260328_*`
- **相对 `A-Opt-03`（`summary.json` · `test_metrics`，3 seed mean ± std）**：**`rmse_vel_mag`（全图节点）** **1.031 → 1.052**；**`rmse_p`** **0.642 → 0.639**（略优）；**`R²_p`** **~0.925**（与 03 基本持平略好）。
- **`regional_eval`**：**`interior.rmse_vel_mag`** **1.822 → 1.849**（**变差**；各 seed 约 **1.858 / 1.856 / 1.834**）；**`all.rmse_vel_mag`** **~1.052 → ~1.068**。**内部点 `R²_u/v/w`（3 seed 均值）**：约 **0.325 / 0.356 / 0.470**，其中 **`R²_v` 较 `A-Opt-03`（0.376）回落**。**`wall.rmse_vel_mag`** 三 seed 约 **0.027 / 0.034 / 0.049**，均值 **差于 `A-Opt-03` ~0.032**。
- **后处理**：各 run 已具备 `predictions_test/`、`error_analysis_interior/`、`regional_eval/`。
- **一句话结论**：**仅放大 hidden width 在未同步正则/训练预算调整时，未继续改善论文主口径内部速度误差**；**压力 `summary.rmse_p` 有轻微收复**。本组为 **`A-Opt-05` 的中间态**，**不作为母版**。
- **下一步动作**：**`A-Opt-05`（h256+4L）** 已定母版；**Line G 子组 G01/G04/G05 已三 seed 归档**（见下「实验记录摘要 · Line G」）；其余见 [优化路径](任务A优化路径与近期实验建议.md)。

---

### A-Opt-05（P0-5 容量②，`A-Opt-04` + `num_layers = 4`）

- **完成日期**：2026-03-28～03-29（训练三 seed）
- **seed**：1, 2, 3（`best_epoch` 分别为 **79、97、82**）
- **输出目录**：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_h256_l4_split_AG_v1_seed{1,2,3}_20260328_*`（seed3 时间戳 `20260329_*`）
- **RMSE_u**：0.8697 ± 0.0061（各 seed：0.8776 / 0.8626 / 0.8689）
- **RMSE_v**：0.8119 ± 0.0039（各 seed：0.8131 / 0.8160 / 0.8067）
- **RMSE_w**：0.6449 ± 0.0045（各 seed：0.6485 / 0.6477 / 0.6385）
- **RMSE_|v|**：1.0399 ± 0.0082（各 seed：1.0477 / 1.0433 / 1.0286）
- **RMSE_p**：0.6449 ± 0.0022（各 seed：0.6477 / 0.6447 / 0.6423）
- **R2_p**：0.9234 ± 0.0005（各 seed：0.9227 / 0.9234 / 0.9240）
- **R2_u/v/w**：0.243 / 0.368 / 0.610（3 seed 均值）
- **分区域 `rmse_vel_mag`（`fig_A5_regional_metrics.json`，3 seed mean ± std）**：
  - **壁面**：0.0420 ± 0.0041（各 seed：0.0439 / 0.0363 / 0.0458）
  - **内部点**：**1.8162 ± 0.0275**（各 seed：1.8028 / 1.8545 / 1.7914）
  - **近壁**：1.4535 ± 0.0309（各 seed：1.4163 / 1.4919 / 1.4522）
  - **高曲率**：0.9970 ± 0.0095（各 seed：0.9836 / 1.0045 / 1.0029）
  - **分叉**：0.9879 ± 0.0139（各 seed：0.9772 / 1.0076 / 0.9790）
  - **主干**：0.7589 ± 0.0210（各 seed：0.7295 / 0.7769 / 0.7703）
  - **regional `all`**：1.0492 ± 0.0157（各 seed：1.0415 / 1.0711 / 1.0350）
- **相对 `A-Opt-04`**：`interior.rmse_vel_mag` **1.849 → 1.816**（回补了加宽带来的内部速度退化）；`summary.rmse_vel_mag` **1.052 → 1.040**。**相对 `A-Opt-03`**：`interior.rmse_vel_mag` 均值略优（1.822 → 1.816），但差距小于跨 seed 方差，不宜单独宣称新 SOTA；`near_wall` 均值明显优于 03（~1.573 → 1.454），更贴近端到端梯度质量需求；`summary.rmse_p` 三 seed 均值 ~0.645，弱于 `A-Opt-01`/`A-Opt-02` 的压力最优区间。
- **后处理**：各 run 已具备 `predictions_test/`、`error_analysis_interior/`、`regional_eval/`。
- **一句话结论**：**加深一层主要「修复」了加宽带来的内部速度回退**，`near_wall` 区域相对 `A-Opt-03` 明显改善（1.573 → 1.454），更利于端到端 WSS 梯度质量叙事；**Trade-off**：显存/时延/跨 seed 方差均高于 03。
- **下一步动作**：**（2026-03-31）** 本组为 **消融 / Line G / Line W 的统一母版**；按 P0-5 **停止条件**不继续 `A-Opt-06`；**`A-Opt-05_tune`** 见下节；**`A-Abl-02` 已完成（2026-04-09）**；**`A-Opt-G01 / G04 / G05` 已三 seed 归档（2026-04-12～04-14）**；**`G02` / `G03` 已结案（仅 seed1）** → 优先 **`A-Abl-01`** 或 **Line W**（**WSS 全量对比暂缓**）。

---

### A-Opt-07（P1-2 内部监督加权，`A-Opt-05` + `interior_loss_boost = 3`）

- **完成日期**：2026-03-31～04-01（训练三 seed）；**（2026-04-02）** 测试集预测与区域评估闭环
- **seed**：1, 2, 3（`best_epoch` 分别为 **106、55、72**）
- **配置**：`training/configs/field/generated/optimization/A-Opt-07_seed{1,2,3}.json`；`run.experiment_name` 后缀 **`_iboost3`**
- **输出目录**：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_h256_l4_iboost3_split_AG_v1_seed{1,2,3}_20260331_175619`
- **RMSE_u**：0.8709 ± 0.0056（各 seed：0.8662 / 0.8794 / 0.8670）
- **RMSE_v**：0.8149 ± 0.0051（各 seed：0.8143 / 0.8214 / 0.8091）
- **RMSE_w**：0.6470 ± 0.0029（各 seed：0.6466 / 0.6507 / 0.6437）
- **RMSE_|v|**：1.0433 ± 0.0120（各 seed：1.0517 / 1.0500 / 1.0283）
- **RMSE_p**：0.6489 ± 0.0143（各 seed：0.6350 / 0.6684 / 0.6432）
- **R2_p**：0.9224 ± 0.0035（各 seed：0.9257 / 0.9177 / 0.9237）
- **R2_u/v/w**：0.241 / 0.364 / 0.607（3 seed 均值，`summary.json`）
- **分区域 `rmse_vel_mag`（`fig_A5_regional_metrics.json`，3 seed mean ± std）**：
  - **壁面**：0.0669 ± 0.0079（各 seed：0.0584 / 0.0774 / 0.0651）— **差于 `A-Opt-05` 的 ~0.042**
  - **内部点**：1.8207 ± 0.0211（各 seed：1.8450 / 1.8067 / 1.8104）— **略差于 `A-Opt-05` 的 1.8162**
  - **近壁**：1.4888 ± 0.0424（各 seed：1.4584 / 1.5417 / 1.4662）— **差于 `A-Opt-05` 的 ~1.454**
  - **高曲率**：0.9982 ± 0.0092（各 seed：1.0026 / 1.0071 / 0.9850）
  - **regional `all`**：1.0527 ± 0.0098（各 seed：1.0663 / 1.0450 / 1.0466）
- **相对 `A-Opt-05`**：**未满足**优化路径「全局 + 内部 `RMSE_|v|` 同步改善」门槛；提高非壁面节点损失权重在本实现下**更偏向拉内部，但损害壁面/近壁恢复**，全图速度略变差。
- **后处理**：各 run 已具备 `predictions_test/`、`regional_eval/`；seed2/3 已生成 `fig_A4_per_case_metrics_interior.csv` 与 `fig_A3_scatter_interior.png`。**建议对 seed1 再执行** `plot_taskA_per_case_boxplot` 与 `plot_taskA_scatter`（与同 seed 的 05/Main 对照箱线/散点时需各 exp 均已具备 `fig_A4_*`）。
- **多模型对照图（Main / 05 / 07）**：`outputs/field/plots/optimization/A_Opt07_vs_Opt05_Main01/` — `python -m training.scripts.regenerate_opt07_vs_opt05_main_figures`
- **一句话结论**：**内部 `interior_loss_boost=3` 未带来相对母版 `A-Opt-05` 的主指标收益**，属**清晰负结果**；后续主线仍取 **`A-Opt-05`**；**`A-Abl-02` 已闭环**；**Line G 之 G01/G04/G05 已三 seed 归档（2026-04-14）**。
- **下一步动作**：不必追加深或再扫 `interior_boost`；将本组作为「仅加权内部监督不够」写入正文/附录即可。

---

### Line G（显式几何增强，母版 `A-Opt-05`）

> **（2026-04-14）** 摘要：**`A-Opt-G01` / `G04` / `G05`** 已 **三 seed** 训练、`predictions_test` 与 Fig A3–A5 多模型对比就绪。**（2026-04-17）** **`G02` / `G03`** 仅 **seed=1**，单 seed 上相对 **`A-Opt-05`** 无收益，**已决定不补 seed2/3**，记 **⏹ 已结案**。

- **配置目录**：`training/configs/field/generated/line_g/`（`A-Opt-G0*_seed*.json`）。
- **与母版 `A-Opt-05`（`summary.json` · `test_metrics`，三 seed 均值）对照（速览）**：
  - **`rmse_vel_mag`**：05 约 **1.040**；**G04 / G05** 约 **1.038**（略优）；**G01** 约 **1.045**（略差）。
  - **`rmse`（合成）**：G01 约 **0.543** 略优于 05 约 **0.562**；G04 / G05 与 05 接近。
  - **分通道 `R²`**：G01 的 **`R²_u` / `R²_p`** 略优于 05；**`R²_vel_mag`** 三组 Line G 约在 **0.58～0.59**（05 的 `summary` 未统一写入 `r2_vel_mag`，不宜直接并列）。
- **多模型对比图（G01/G04/G05 × 三 seed 均值 vs baseline / Main / 05）**：`outputs/field/plots/line_g/G01_G04_G05_vs_baselines_mean3seed/`（详见 `docs/02-推进与变更/代码修改与实验推进记录.md` 2026-04-14 条目）。
- **一句话结论**：在 **05 母版**上叠加 **壁面距离（G04）** 或 **切向变化率（G05）** 对 **`summary.rmse_vel_mag` 三 seed 均值** 有 **小幅正向**；**分叉拓扑（G01）** 在 **合成 `rmse` 与压力/分速度 R²** 上有信号，但 **速度模**略差于 05；均 **未形成大幅碾压**，适合作为正文「几何先验小步扩展」而非新母版。**`G02`（`dR_ds`）/ `G03`（`torsion`）** 在 **seed=1** 上相对05 **未见收益**，**不再补种**。
- **下一步动作**：转入 **`A-Abl-01`** / **Line W**；汇报时与 **`A-Base-*` / `A-Main-01`** 对照保留叙事完整性。

---

### WSS 多任务（场监督 + `wss_loss_weight=0.1`，2026-04-14～04-17）

- **完成日期**：训练三 seed 已登记 **`outputs/field/experiment_index.csv`**（2026-04-14～04-17）；测试集**壁面节点** WSS RMSE/R² 见 **`outputs/field/wss_multitask_test_wall_wss_metrics.tsv`**；**（2026-04-17）** **`predictions_test` + Fig A3/A4/误差/Fig A5** 已按推进记录闭环（阵列 2704、补点 2715）。
- **配置**：**`training/configs/field/generated/baseline_wss_multitask/`**（`A-Base-01-wss-multi`～`A-Opt-05-wss-multi`）
- **与 Line W `A-Opt-W03` 草案的区别**：本批为 **速度场/压力 + WSS 联合**（`target_weights` 与各自无 WSS 母版一致）；**`A-Opt-W03-wss-only`** 为 **关闭场监督、仅 WSS**（`training/configs/field/generated/wss_multitask/`，**未跑**）

**汇报用 R² 主表（多任务各 `exp_id`，三 seed 均值 ± 总体标准差）**

| Exp ID | **R² 壁面 WSS**（`wss_r2_wss`，TSV） | **R² \|v\|**（`summary.test_metrics.r2_vel_mag`） | **R²_p**（`summary.test_metrics.r2_p`） |
| --- | --- | --- | --- |
| A-Base-01-wss-multi | −0.052 ± 0.008 | 0.016 ± 0.002 | 0.923 ± 0.004 |
| A-Base-02-wss-multi | 0.383 ± 0.006 | 0.289 ± 0.009 | 0.902 ± 0.003 |
| A-Base-03-wss-multi | 0.384 ± 0.006 | 0.288 ± 0.001 | 0.912 ± 0.009 |
| A-Main-01-wss-multi | 0.460 ± 0.004 | 0.468 ± 0.007 | 0.928 ± 0.007 |
| A-Opt-05-wss-multi | 0.463 ± 0.004 | 0.584 ± 0.013 | 0.922 ± 0.003 |

> **口径**：**R² 壁面 WSS** 为测试集 **壁面节点**、WSS **向量（4 维合成）** 的 R²，与 `training/scripts/_eval_wss_metrics_once.py` / **`WSSMeter`** 一致。**R² \|v\|**、**R²_p** 来自各 run **`summary.json`** 的 **`test_metrics`**（训练结束全图 eval；多任务 run 均含 `r2_vel_mag` 字段）。

- **`summary.test_metrics` 其它速览（三 seed 均值）**：**`A-Opt-05-wss-multi`** — `rmse_vel_mag` **约 1.043**，合成 **`rmse`** **约 0.752**；**`A-Main-01-wss-multi`** — `rmse_vel_mag` **约 1.179**，合成 **`rmse`** **约 0.784**（与上表 **R²_p** 同源）。
- **壁面 WSS RMSE（TSV · `wss_rmse`，三 seed 均值）**：Base-01-wss-multi **约 1.262**；Base-02/03-wss-multi **约 1.13**；Main-01-wss-multi **约 1.084**；Opt-05-wss-multi **约 1.077**（相对 Main-wss-multi **略优**）。
- **一句话结论**：多任务头下，**壁面 WSS R²** 在 **MLP** 上为**负**；在 **GraphSAGE / Transformer-wall** 上约 **0.38**；在 **Main / Opt-05 母版** 上约 **0.46～0.46**，**Opt-05-wss-multi** 与 **Main-wss-multi** 基本持平、壁面 RMSE 略优。**速度模 R²**（`r2_vel_mag`）随骨干由 MLP →05 **单调抬升**（约 **0.02 → 0.58**），**压力 R²** 在 **Main-wss-multi** 上最高（**约 0.928**），**Opt-05-wss-multi** 略低但仍与 **0.92** 档对齐。与无 WSS 同配置的 **逐对 trade-off** 仍以各 **`summary.json`** / **区域 Fig A5** 为准。
- **下一步动作**：按需运行 **`plot_taskA_multimodel_regional_bar`** 做多模型柱图；或接 **Line W** 加权 / **纯 WSS（`A-Opt-W03-wss-only`）** 分支。

---

### A-Abl-02（显式几何分量消融，母版 `A-Opt-05`）

- **完成日期**：2026-04-02～04-07（各子组训练）；**（2026-04-09）** 三 seed 汇总与文档归档
- **对照母版**：**`A-Opt-05`**（`interior.rmse_vel_mag` 三 seed 均值 **1.816±0.034**）
- **指标口径**：**`regional_eval` · `interior` · `rmse_vel_mag`**；统计汇总见 **`outputs/field/plots/ablation/geometry_opt05_mean3seed/fig_A6_ablation_summary_stats_interior.json`**（paired *t* 相对母版，*n*=3 seed）
- **输出目录模式**：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_h256_l4_abl02_no_{abscissa|normradius|curvature|tangent}_split_AG_v1_seed{1,2,3}_*`（已登记 **`outputs/field/experiment_index.csv`**）
- **Fig A6（五组合计）**：`outputs/field/plots/ablation/geometry_opt05_mean3seed/fig_A6_ablation_summary_interior.{png,csv}`（复现命令见 **`docs/02-推进与变更/代码修改与实验推进记录.md`** 2026-04-07 条目）

| 子组 | 去掉的分量 | interior `rmse_vel_mag`（mean±std） | 相对 05 Δmean | paired *t* *p* | 解读 |
| --- | --- | --- | --- | --- | --- |
| `A-Abl-02-01` | Abscissa | 1.868±0.026 | +0.052 | ~0.26 | 有退化趋势，**三 seed 不足以宣称显著** |
| `A-Abl-02-02` | NormRadius | 2.043±0.020 | +0.227 | ~**0.0015** | **影响最大**，与母版差异**统计显著** |
| `A-Abl-02-03` | Curvature | 1.851±0.030 | +0.035 | ~0.26 | **与母版最接近**，单项贡献**最弱** |
| `A-Abl-02-04` | Tangent（3 维） | 1.933±0.038 | +0.117 | ~0.073 | **次之**，接近常见显著性阈值 |

- **一句话结论**：在 **`A-Opt-05` 母版**上，**归一化半径 NormRadius 是显式几何通道中的主贡献项**；**切向 Tangent** 次之；**弧长 Abscissa** 与 **曲率 Curvature** 的边际更大但 **3 seed 下相对母版未达常规显著**——叙事上可写「半径与局部方向框架最关键，曲率单项可视为弱贡献」。
- **下一步动作**：推进 **`A-Abl-01`**（输入层级消融）；**Line G 侧** **`G01/G04/G05` 已闭环**，**`G02`/`G03` 已结案（不补种）**；可接 **Line W**；复杂区域对比可继续用各 run 的 **`regional_eval`** 补充正文。

---

### `A-Opt-05_tune`（P0-5 后小步超参，`A-Opt-05` 骨架上的试跑）

> **配置目录**：`training/configs/field/generated/optimization/A-Opt-05_tune/`。批量 manifest 示例：`training/cluster/manifest_list_A-Opt-05_tune.tsv`。

- **（2026-03-31）本仓库已入账（`outputs/field/experiment_index.csv`）**：**`A-Opt-05t_warmup10`**（seed **1、2**）、**`A-Opt-05t_warmup10_lr3e-4`**（seed **1**）、**`A-Opt-05t_warmup10_wd2e-4`**（seed **1**）、**`A-Opt-05t_warmup10_schedpat15`**（seed **1**，run：`..._wu10_schpat15_..._20260330_134411`）。上述 run 已具备 **`predictions_test/`**、**`error_analysis_interior/`**、**`regional_eval/`**（与 **`A-Opt-05`** 后处理链一致）。
- **清单与目录差**：`manifest_list_A-Opt-05_tune.tsv` 中含 **`A-Opt-05t_warmup5`（三 seed）** 与 **`lr3e-4` seed2/3**；**当前 `outputs/field/` 无同名实验目录**——若已在其他机器跑完，需拷回并登记 `experiment_index.csv` 后再算「闭环」。
- **数值倾向（seed1 主表，摘要）**：**`lr3e-4`** 在 **`interior.rmse_vel_mag`** 上相对基线 **`A-Opt-05`** **略优**，**内部 `R²` 分量略好**；**`wd2e-4`** **明显变差**；**`schedpat15`** **未稳定优于** **`warmup10` 默认调度**（详见各 run **`summary.json`** 与 **`regional_eval/fig_A5_regional_metrics.json`**）。
- **多模型横向图（文件夹名含 `Opt03`，seed=1 子集）**：`outputs/field/plots/optimization/A_Opt05_tune_vs_Opt03_seed1/`（Fig A3 / A5 / A4；**解读时**以 **母版 `A-Opt-05`** 为主视角，`A-Opt-03` 为对照）。
- **WSS**：**未做** `compare_hemo_wss_runs` **全量导出**；后续可用 **`training/cluster/wss_runs_A_Opt03_vs_Opt05tune_seed1.tsv`** 在集群提交。
