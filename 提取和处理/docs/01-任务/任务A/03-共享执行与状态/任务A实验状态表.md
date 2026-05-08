# 任务A 实验状态总表

> 本表是任务 A 所有实验的唯一执行状态追踪文件。
> 每次启动、完成或失败一组实验，**必须更新此表**。
> 上位文档：[任务A实验清单](../01-V1路线/任务A_V1实验清单.md) / [任务A冻结卡](../01-V1路线/任务A_V1冻结卡.md)

> **路线说明（2026-04-01）**：
>
> - 现有 `A-Base-* / A-Opt-* / Line G / Line W` 统一视为 `**Route-KNN-GNN-V1`** 历史结果
> - 新增修正路线 `**Route-PhysicsAware-V2**` 的执行矩阵见：[任务A V2修正路线实验矩阵](../02-V2路线/任务A_V2修正路线实验矩阵.md)
> - V2 Gate-0 的具体执行细节见：[任务A V2准备执行清单](../02-V2路线/任务A_V2准备执行清单.md)
> - V2 首轮 5 组训练实验的统一判定口径见：[任务A V2首轮判定与汇报模板](../02-V2路线/任务A_V2首轮判定与汇报模板.md)
> - 本状态表自今日起同时追踪 **V1 历史结果** 与 **V2 新实验**

---

## V3：Route-DualDomain-PointNeXt-V3（2026-05-04 新增）

> 说明：本区追踪 `**Route-DualDomain-PointNeXt-V3**`。
> 路线计划文档：[V3 PointNeXt 双域 WSS 优先路线计划](../03-V3路线/任务A_V3_PointNeXt双域WSS优先路线计划.md)
> 实验跟踪日志：[V3 实验执行跟踪日志](../03-V3路线/V3_实验执行跟踪日志.md)
> 执行顺序：`Diag-00 → Probe-P/V/WSS → Probe-PWSS/VP/VWSS → Anchor/Base/Main → WSS-*`，严格按层推进。

| Exp ID | 类型 | 研究问题 / 任务 | split | seeds | 当前状态 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| `V3P-Diag-00` | 诊断 | mask/loss 尺度/壁面真值/WSS 分布/归一化基准 | `split_AG_v1` | 1 | ✅ 已完成 | 作业 3415；产出 `outputs/field/diagnostics/v3p_diag00_seed1/`；校准决策：`lambda_vel_int 0.3→0.15`（§4.2.1）、`lambda_vel_noslip 0.1→1.0`（§4.1.1 raw_truth）、augment.rotation 关闭 |
| `V3P-Probe-P-01` | 单目标 probe | 压力单目标上限（只 p） | `split_AG_v1` | 1 | ✅ 已完成 | 作业 **3472**（新数据重跑）；**r2_p=0.962** ✅；best_epoch=100；17 test cases（PENG 已移除）；旧作业 3418 已作废 |
| `V3P-Probe-V-01` | 单目标 probe | 速度单目标上限（只 vel） | `split_AG_v1` | 1 | ✅ 已完成 | 作业 **3473**（新数据重跑）；**r2_vel_mag=0.294**（较旧版下降，新数据归一化变化所致）；best_epoch=41；旧作业 3419 已作废 |
| `V3P-Probe-WSS-01` | 单目标 probe | WSS 单目标上限（只 WSS） | `split_AG_v1` | 1 | ✅ 已完成 | 作业 **3474**（新数据重跑）；**wss_r2_wss=0.397** ✅；best_epoch=2（极早收敛）；旧作业 3420 已作废 |
| `V3P-Probe-PWSS-01` | 双目标 probe | p + WSS 干扰诊断 | `split_AG_v1` | 1 | ✅ 已完成 | 作业 **3476**；**r2_p=0.929, wss_r2_wss=0.366**；P+WSS 可共存（各下降~3%）；wss_y 意外改善（0.011→0.067） |
| `V3P-Probe-VP-01` | 双目标 probe | 速度 + 压力 field 诊断 | `split_AG_v1` | 1 | ⏸️ 降级（可选） | VWSS-01 负结果已确认速度监督有害；VP-01 诊断价值降低，暂不排队 |
| `V3P-Probe-VWSS-01` | 双目标 probe | 速度上下文是否帮助 WSS | `split_AG_v1` | 1 | ✅ 已完成 | 作业 **3475**；**wss_r2_wss=0.343**（低于单目标0.397），**r2_vel_mag=0.074**（低于单目标0.294）；**负结果**：速度监督同时压低两目标，主线不含速度监督 |
| `V3P-Anchor-01` | 锚点 | 同采样 V1 Transformer 锚点 | `split_AG_v1` | 1→[1,2,3] | 📋 seed=1 完成，待补 seed | 作业 **3537**；`wss_r2_wss=0.367`，`r2_p=0.898`；**3623**：predict+图件；目录 `…/anchor01_…_20260506_134310/` |
| `V3P-Base-01` | 主线对照 | 无几何 PointNeXt（含弱速度） | `split_AG_v1` | 1→[1,2,3] | 📋 seed=1 完成，待补 seed | 作业 **3538**；`wss_r2_wss=0.331`，`r2_p=0.936`；**3623**；`…/base01_nogeom_…_20260506_134310/` |
| `V3P-Base-01-PW` | 正式主线 | 无几何 PointNeXt（纯 P+WSS） | `split_AG_v1` | 1→[1,2,3] | 📋 seed=1 完成，待补 seed | 作业 **3543**；`wss_r2_wss=0.343`，`r2_p=0.916`；**3623**；`…/base01_nogeom_pw_…_20260506_230830/` |
| `V3P-Main-01` | 主线对照 | 几何 PointNeXt（含弱速度） | `split_AG_v1` | 1→[1,2,3] | 📋 seed=1 完成，待补 seed | 作业 **3539**；`wss_r2_wss=0.336`，`r2_vel_mag=0.679`；**3623**；`…/main01_geom_…_20260506_134310/` |
| `V3P-Main-01-PW` | **V3 核心主线** | 几何 PointNeXt（纯 P+WSS） | `split_AG_v1` | 1→[1,2,3] | 📋 seed=1 **trainer 修复后**已重训+出图；待补 seed2/3 | **3544**（旧）：`best_epoch=11`，`wss_r2_wss=0.367`，`r2_p=0.920`，`…20260506_230831/`（旧 `val_score` bug）。**3634+3643**（新）：run `…20260508_001936/`；`best_epoch=104`；`test_metrics`：`wss_r2_wss=0.344`，`r2_p=0.936`；`test_metrics_best_wss`：`wss_r2_wss=**0.365**`，`r2_p=0.935`；**3643** 已闭环 predict + Task A 图件。跟踪见 [V3_实验执行跟踪日志](../03-V3路线/V3_实验执行跟踪日志.md) |
| `V3P-WSS-01-a/b/c` | WSS 穷扫（含速度） | lambda_wss = 0.05/0.10/0.20 | `split_AG_v1` | 1→[1,2,3] | 📋 seed=1 完成，待补 seed | 作业 **3540/3541/3542**；WSS **0.368 / 0.338 / 0.358**；**3623** |
| `V3P-WSS-01-a/b/c-PW` | WSS 穷扫（纯 P+WSS） | lambda_wss = 0.05/0.10/0.20 | `split_AG_v1` | 1→[1,2,3] | 📋 seed=1 完成，待补 seed | 作业 **3545/3546/3547**；WSS **0.395 / 0.390 / 0.376**（**a-PW 超锚点**）；**3623** |
| `V3P-WSS-02` | 条件实验 | 速度上下文增强 | `split_AG_v1` | 1→[1,2,3] | ❌ 取消 | VWSS-01 负结果：速度监督不帮助 WSS，条件不触发 |

---

## V2：阶段 0 准备与首轮路线对照（2026-04-01 新增）

> 说明：本区只追踪 `**Route-PhysicsAware-V2**`。
> 命名规则：
>
> - `V2-Ref-*`：V2 数据口径下的公共参考实验
> - `V2G-*`：`Route-MeshGNN-V2`
> - `V2P-*`：`Route-PointCloud-V2`
> 执行细节见：[任务A V2修正路线实验矩阵](../02-V2路线/任务A_V2修正路线实验矩阵.md)。


| Exp ID           | 类型   | 研究问题 / 任务                                                                                                                                                                                                                           | split_version                  | seeds   | 当前状态                                                   | 备注                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ---------------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `V2-Prep-01`     | 数据准备 | `.cas/.msh` 是否能稳定导出 mesh 邻接与壁面区域                                                                                                                                                                                                    | `split_AG_v2`                  | -       | 🔒 未开始                                                 | Gate-0，未通过则 `V2G-*` 不得开跑                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `V2-Prep-02`     | 数据准备 | 采样点能否回溯到原始 mesh 或可信局部邻域                                                                                                                                                                                                             | `split_AG_v2`                  | -       | 🔒 未开始                                                 | 与 V2 采样策略绑定                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `V2-Prep-03`     | 数据准备 | WSS 真值/壁面节点能否稳定对齐                                                                                                                                                                                                                   | `split_AG_v2`                  | -       | 🔒 未开始                                                 | Gate-0，未通过则 WSS 主评估延后                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `V2-Prep-04`     | 数据准备 | V2 数据生成与评估链路 smoke test                                                                                                                                                                                                             | `split_AG_v2`                  | -       | 🔒 未开始                                                 | 2~3 病例最小闭环                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `V2-Ref-Base-01` | 训练实验 | V2 数据口径下的点级统一下限                                                                                                                                                                                                                     | `split_AG_v2`                  | [1,2,3] | 🔒 未开始                                                 | 建议 `coords + t + BC`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `V2G-Base-01`    | 训练实验 | mesh-aware GNN 在无 geometry 下的基线能力                                                                                                                                                                                                   | `split_AG_v2`                  | [1,2,3] | 🔒 未开始                                                 | `coords + t + BC + is_wall`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `V2G-Main-01`    | 训练实验 | geometry 在 MeshGNN 上是否成立                                                                                                                                                                                                            | `split_AG_v2`                  | [1,2,3] | 🔒 未开始                                                 | `coords + t + BC + geometry + is_wall`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `V2P-Base-01`    | 训练实验 | point-cloud 主干在无 geometry 下的基线能力                                                                                                                                                                                                    | `split_AG_v1`（bootstrap）       | [1]     | 🌱 待补 seed                                             | seed=1 已完成（bootstrap）；r2_vel_mag=0.354，best_epoch=87；**暂不补 seed**，等正式 split_AG_v2 后重跑                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `V2P-Main-01`    | 训练实验 | geometry 在 point-cloud 主干上是否成立                                                                                                                                                                                                      | `split_AG_v1`（bootstrap）       | [1]     | 🌱 待补 seed                                             | seed=1 已完成（bootstrap）；r2_vel_mag=0.609，best_epoch=68；**geometry 增益显著，已满足首轮通过条件**；暂不补 seed，等 split_AG_v2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `V2P-WSSP-01`    | 训练实验 | PointNeXt + **壁面血流动力学专线**：13000 壁面点 + 2000 近壁内部点；**速度 loss=0**，直接监督 **p + WSS**（`wss_loss_weight=1`）                                                                                                                                | `split_AG_v1`（bootstrap，同 V2P） | [1]     | ✅ 已完成（seed=1）                                          | **（2026-04-24）** 配置 `training/configs/field/generated/v2_pointcloud/V2P-WSSP-01_seed1.json`；run：`outputs/field/field_v2_pointnext_wssp01_geom_wall13000_near2000_split_AG_v1_seed1_20260424_164609/`；已登记 `**experiment_index.csv**`；`**predictions_test` + `error_analysis_interior`（含 `wss/`）+ `regional_eval`（含 `fig_A5_regional_wss_metrics.json`）** 已闭环。数值摘要见下「实验记录摘要 · V2P-WSSP-01」。                                                                                                                                                                                                                                              |
| `V2P-WSSP-02`    | 训练实验 | PointNeXt + **全场监督 + WSS 辅助头**：恢复 u/v/w/p 完整监督（`target_weights=[2,2,2,0.5]`）+WSS 头（`wss_loss_weight=0.5`）+ **混合验证指标早停**（`early_stop_wss_weight=1.0`）；wall-rich 采样（13000 壁面 + 2000 近壁，同 WSSP-01 图资产）~~原文档误标为「标准采样」，2026-04-26 核实修正~~ | `split_AG_v1`（bootstrap，同 V2P） | [1]     | ✅ 已完成（seed=1）— **负结果**                                 | **（2026-04-25）** 配置 `training/configs/field/generated/v2_pointcloud/V2P-WSSP-02_seed1.json`；run：`outputs/field/field_v2_pointnext_wssp02_geom_full_supervision_wss_split_AG_v1_seed1_20260425_093041/`；**负结果**：`r2_p=0.004`（崩溃）、`wss_r2_wss=-0.018`。根因：`wss_loss≈19.5 × 0.5=9.77` 是 `data_loss≈1.95` 的 5 倍，梯度被 WSS 劫持；混合早停被不可降的 WSS loss 主导，epoch 58 过早终止。结论：`wss_loss_weight=0.5` 过高。                                                                                                                                                                                                                                                 |
| `V2P-WSSP-03`    | 训练实验 | PointNeXt + **全场监督 + 轻量 WSS 辅助头**：同 Main-01 配方 + WSS 头（`wss_loss_weight=0.01`），场主导早停（`early_stop_wss_weight=0`）；wall-rich 采样（同 WSSP-01 图资产）~~原文档误标为「标准采样」，2026-04-26 核实修正~~                                                         | `split_AG_v1`（bootstrap，同 V2P） | [1]     | ✅ 已完成（seed=1）                                          | **（2026-04-26）** 配置 `training/configs/field/generated/v2_pointcloud/V2P-WSSP-03_seed1.json`；run：`outputs/field/field_v2_pointnext_wssp03_geom_full_supervision_wss_low_split_AG_v1_seed1_20260425_230427/`；`best_epoch=34`；`**predictions_test` + Fig A3–A5 / `error_analysis_interior`（含 `wss/`）** 已闭环。测试集节选：`r2_vel_mag≈0.592`，`r2_p≈0.025`，`wss_r2_wss≈−0.017`；`wall` 区 WSS 见 `regional_eval/fig_A5_regional_wss_metrics.json`。摘要见下「实验记录摘要 · V2P-WSSP-03」。                                                                                                                                                                          |
| `V2P-WSSP-04`    | 训练实验 | PointNeXt + **压力/WSS 主线，速度弱辅助**：`target_weights=[0.1,0.1,0.1,1.0]`，`wss_loss_weight=0.5`，`early_stop_wss_weight=0`；主指标为压力与壁面 WSS                                                                                                    | `split_AG_v1`（bootstrap，同 V2P） | [1]     | ✅ 已完成（seed=1）— **相对 03，主指标未改善**                        | **（2026-04-26）** 配置 `training/configs/field/generated/v2_pointcloud/V2P-WSSP-04_seed1.json`；历史 run：`…/20260426_004522/`；**（2026-04-27）** 集群重启后重训落盘 `**…/20260427_103849/**`（`best_epoch=100`）；`predict_field`+分区域+与 **Line W 三实验** 同批多模型已闭环——**分区域 `r2_p` 与 Transformer+WSS 多任务不在同档**，详见下「实验记录摘要 · V2P-WSSP-04」与「**Line W：A-Opt-W03 权重复扫**」。早期测试集节选：`r2_vel_mag≈0.579`，`r2_p≈0.014`，`wss_r2_wss≈−0.017`；**03 vs 04** 对照 `**outputs/field/plots/v2p_wssp03_vs_04_seed1/**`；**四实验总览** `**outputs/field/plots/wssp04_w03_line_seed1_20260427/**`。                                                                                         |
| `V2P-WSSP-05`    | 训练实验 | 复刻 **WSSP-01** 配方：**仅 p+WSS**（`target_weights=[0,0,0,1]`），`wss_loss_type=mse`，`wss_loss_weight=1`；wall13000+near2000；三 seed 建立固定采样下基线                                                                                               | `split_AG_v1`（bootstrap，同 V2P） | [1,2,3] | ✅ 已完成（训练+三 seed `predict`+主图件）— **略负 WSS R²、r2_p 方差大** | **（2026-04-28）** 配置 `training/configs/field/generated/v2_pointcloud/V2P-WSSP-05_seed{1,2,3}.json`；runs `outputs/field/field_v2_pointnext_wssp05_baseline_p_wss_wall13000_near2000_split_AG_v1_seed{1,2,3}_20260428_000004/`；`**summary.test_metrics**`（测试集）节选：`r2_p` **0.215 / 0.038 / −0.022**，`wss_r2_wss` **−0.025 / −0.022 / −0.016**；**未达** Go/No-Go 中 V1 锚点 **~0.46**。已登记 `**experiment_index.csv**`；`**predictions_test` + `regional_eval` + Fig A3/A4 + `error_analysis_interior`（interior）** 已闭环；若需壁面 WSS 误差子图可补 `**plot_error_analysis --manifest … --wss**`。训练 Slurm **3187–3189**，后处理阵列 **3201**。详见「实验记录摘要 · V2P-WSSP-05」。 |
| `V2P-WSSP-06`    | 训练实验 | **同 05**，唯一变更：`optim.wss_loss_type=huber`（Smooth L1，`beta=1`）；单变量对照 MSE vs 鲁棒 WSS loss                                                                                                                                              | `split_AG_v1`（bootstrap，同 V2P） | [1,2,3] | ✅ 训练完成；🌱 **predict 缺 seed=3**                         | **（2026-04-28）** 配置 `…/V2P-WSSP-06_seed{1,2,3}.json`；runs `…/field_v2_pointnext_wssp06_wss_huber_only_else_like_wssp05_split_AG_v1_seed1_20260428_000339/` 等；**有 `predictions_test` 的 seed**：`r2_p` **0.003 / 0.015**，`wss_r2_wss` **−0.016 / −0.015**；**seed=3**：仅 `**summary.json` 测试集指标**（`r2_p≈−0.031`，`wss_r2_wss≈−0.013`），**待补 `predict_field**`。**Huber 相对 05 未稳定抬高 `wss_r2_wss` 或 `r2_p**`，跨 seed 方差大。Slurm **3190–3192**、后处理 **3201/3205**。详见「实验记录摘要 · V2P-WSSP-06」。                                                                                                                                                          |


---

## 战略锚点（2026-03-31）

> **作用域说明（2026-04-01）**：本节锚点仅适用于 `**Route-KNN-GNN-V1**` 的历史延展实验，不自动外推到 V2。

后续**消融实验（第二批及以后的新开跑）**、**Line G（显式几何增强）**与 **Line W（壁面导向）的主线，统一以 `A-Opt-05` 为配置母版**（`hidden_dim=256`，`num_layers=4`，`target_weights=[2,2,2,0.5]` + Pre-Norm `FieldTransformer`；目录见第三批 `**A-Opt-05**`）。

- **为何从 03 切到 05**：在 `**interior.rmse_vel_mag**` 三 seed 均值上 05 略优于 03；在 `**near_wall` 等紧邻壁面的内部区域**上 05 略优于 03，更贴合端到端链路对梯度/WSS 前置质量的需求（详见各 run `regional_eval/fig_A5_regional_metrics.json` 及「实验记录摘要 · A-Opt-05」）。
- **仍保留 03 的角色**：`**A-Opt-03`（`hidden_dim=128`）**作为 **P0-4 历史锚点**与 **轻量、方差较小、部署成本更低**的对照线；汇报 Pareto/效率时建议 **03 与 05 并列**，避免读者误以为只有一条 P0 线。
- **操作约定**：生成新消融 JSON 时，以 `**A-Opt-05`** 某 seed 的 `**config.snapshot.json**` 为复制基准，仅改「消融项」对应字段；与四组 **baseline**（`A-Base-*` / `A-Main-01`）的叙事对比不变，**控制变量母版**改为 05。

---

## 状态说明


| 状态标记            | 含义                           |
| --------------- | ---------------------------- |
| 🔒 未开始          | 尚未生成配置或提交                    |
| 🔬 待 smoke test | 配置已就绪，等待最小闭环验证               |
| 🚀 进行中          | 至少一个 seed 已启动                |
| 🌱 待补 seed      | seed=1 已通过，等待补 seed=2,3      |
| 📋 待汇总          | 训练完成，等待写入记录表                 |
| ✅ 已完成           | 结果、图表、实验记录均已归档               |
| ⏹ 已结案           | 仅完成计划内 seed；判定无需补种，作弱/负向结果归档 |
| ❌ 失败待重跑         | 出现错误或配置问题，需修复后重跑             |


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


| Exp ID      | 研究问题    | 唯一变化项              | split_version | seeds   | 当前状态   | 备注                                                                                                                                                                                                                                     |
| ----------- | ------- | ------------------ | ------------- | ------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A-Abl-01-01 | 输入特征消融  | coords + t 仅坐标+时间  | split_AG_v1   | [1]     | 🔒 未开始 | baseline ✅；**新开跑母版 `A-Opt-05`**（见上「战略锚点」）                                                                                                                                                                                              |
| A-Abl-01-02 | 输入特征消融  | + BC               | split_AG_v1   | [1]     | 🔒 未开始 |                                                                                                                                                                                                                                        |
| A-Abl-01-03 | 输入特征消融  | + is_wall          | split_AG_v1   | [1]     | 🔒 未开始 |                                                                                                                                                                                                                                        |
| A-Abl-01-04 | 输入特征消融  | + geometry（无 wall） | split_AG_v1   | [1]     | 🔒 未开始 |                                                                                                                                                                                                                                        |
| A-Abl-02-01 | 几何分量消融  | 去掉 Abscissa        | split_AG_v1   | [1,2,3] | ✅ 已完成  | **（2026-04-09）** 母版 `**A-Opt-05**`；`outputs/field/field_transformer_*_abl02_no_abscissa_split_AG_v1_seed{1,2,3}_*`；`**interior.rmse_vel_mag**` 三 seed 均值 **1.868±0.026**（相对 05 **+0.052**，paired *t* **p≈0.26**）。见「实验记录摘要 · A-Abl-02」。 |
| A-Abl-02-02 | 几何分量消融  | 去掉 NormRadius      | split_AG_v1   | [1,2,3] | ✅ 已完成  | 同上；`*_abl02_no_normradius_*`；均值 **2.043±0.020**（**+0.227**，**p≈0.0015**）— **NormRadius 最关键**。                                                                                                                                          |
| A-Abl-02-03 | 几何分量消融  | 去掉 Curvature       | split_AG_v1   | [1,2,3] | ✅ 已完成  | 同上；`*_abl02_no_curvature_*`；均值 **1.851±0.030**（**+0.035**，**p≈0.26**）— **影响最弱**。                                                                                                                                                       |
| A-Abl-02-04 | 几何分量消融  | 去掉 Tangent         | split_AG_v1   | [1,2,3] | ✅ 已完成  | 同上；`*_abl02_no_tangent_*`；均值 **1.933±0.038**（**+0.117**，**p≈0.073**）— **Tangent 次之**。                                                                                                                                                  |
| A-Abl-03-01 | 坐标归一化消融 | 原始坐标               | split_AG_v1   | [1]     | 🔒 未开始 |                                                                                                                                                                                                                                        |
| A-Abl-03-02 | 坐标归一化消融 | 仅中心化               | split_AG_v1   | [1]     | 🔒 未开始 |                                                                                                                                                                                                                                        |
| A-Abl-03-03 | 坐标归一化消融 | 中心化+PCA对齐          | split_AG_v1   | [1]     | 🔒 未开始 |                                                                                                                                                                                                                                        |
| A-Abl-03-04 | 坐标归一化消融 | 中心化+PCA+缩放（当前版本）   | split_AG_v1   | [1]     | 🔒 未开始 |                                                                                                                                                                                                                                        |
| A-Abl-04-01 | 增强消融    | 无增强                | split_AG_v1   | [1]     | 🔒 未开始 |                                                                                                                                                                                                                                        |
| A-Abl-04-02 | 增强消融    | 仅旋转                | split_AG_v1   | [1]     | 🔒 未开始 |                                                                                                                                                                                                                                        |
| A-Abl-04-03 | 增强消融    | 旋转+平移（默认）          | split_AG_v1   | [1]     | 🔒 未开始 |                                                                                                                                                                                                                                        |
| A-Abl-04-04 | 增强消融    | 旋转+平移+微扰           | split_AG_v1   | [1]     | 🔒 未开始 |                                                                                                                                                                                                                                        |
| A-Abl-05-01 | 物理约束消融  | 仅数据损失              | split_AG_v1   | [1]     | 🔒 未开始 | 依赖主线稳定后                                                                                                                                                                                                                                |
| A-Abl-05-02 | 物理约束消融  | + continuity       | split_AG_v1   | [1]     | 🔒 未开始 |                                                                                                                                                                                                                                        |


---

## 第三批：近期优化线

> 说明：本区用于承接 baseline 完成后的“先拿更好结果，再补最小必要解释实验”路线。
> 当前优先级以 [任务A优化路径与近期实验建议](../01-V1路线/任务A_V1优化路径与近期实验建议.md) 为准。
> 推荐执行顺序：`A-Opt-01 -> A-Opt-02 -> A-Opt-02_warmup (P0-3，✅ 2026-03-27) -> A-Opt-03 (✅ 2026-03-28) -> A-Opt-03w (✅ 2026-03-28) -> A-Opt-04 (✅ 2026-03-29) -> A-Opt-05 (✅ 2026-03-29) -> A-Opt-07 (✅ 2026-04-02，**负结果**)`。
> 推进门槛：只有当上一组同时改善全局 `RMSE_|v|`、内部区 `RMSE_|v|`，且至少一个速度分量 `R²` 明显改善时，才进入下一组容量扩展。
> **（2026-03-31）** 容量线执行到此为止；**新开跑优化/消融/Line G** 以 `**A-Opt-05**` 为母版（见篇首「战略锚点」），`**A-Opt-03**` 保留为轻量对照。


| Exp ID          | 研究问题                             | 唯一变化项                                              | split_version | seeds   | 当前状态   | 备注                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| --------------- | -------------------------------- | -------------------------------------------------- | ------------- | ------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A-Opt-01        | 速度权重是否能改善内部流场                    | `target_weights = [2,2,2,0.5]`                     | split_AG_v1   | [1,2,3] | ✅ 已完成  | 三 seed 训练与 `**predict_field` + `plot_taskA_regional_bar`** 已完成（2026-03-26）。目录：`outputs/field/field_transformer_coord_t_bc_geom_wall_tw22205_split_AG_v1_seed{1,2,3}_*/`；见下文「实验记录摘要 · A-Opt-01」与主结果表增行。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| A-Opt-02        | LayerNorm 是否提升单尺度 Transformer 表达 | `FieldTransformer` 改为 Pre-Norm 残差块                 | split_AG_v1   | [1,2,3] | ✅ 已完成  | 三 seed：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_split_AG_v1_seed{1,2,3}_20260327_*`；`predict_field` + `error_analysis_interior` + `regional_eval` + 与 Main 对照 `**plots/prenorm_A_Opt02_vs_Main01/**`（2026-03-27）。见「实验记录摘要 · A-Opt-02」与主结果表增行。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| A-Opt-02_warmup | 学习率 Warmup 是否稳定 Pre-Norm 训练并改善指标 | `A-Opt-02 + optim.warmup_epochs=5`                 | split_AG_v1   | [1,2,3] | ✅ 已完成  | **P0-3**（2026-03-27）：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_warmup5_split_AG_v1_seed{1,2,3}_20260327_*`；已补 `**predictions_test**`、`**error_analysis_interior**`、`**regional_eval**`；与 Main / P0-2 三模型对照图 `**outputs/field/plots/optimization/prenorm_Main_P02_P02w/**`；一键重汇总结图：`python -m training.scripts.regenerate_p02_warmup_comparison_figures`。见「实验记录摘要 · A-Opt-02_warmup」与主结果表增列。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| A-Opt-03        | 损失重加权与 LayerNorm 是否互补            | `A-Opt-01 + A-Opt-02`                              | split_AG_v1   | [1,2,3] | ✅ 已完成  | **P0-4（2026-03-28）**：三 seed 已训练并归档；`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_split_AG_v1_seed{1,2,3}_20260327_*`；已含 `**predictions_test**`、`**error_analysis_interior**`、`**regional_eval**`；训练期 **best_epoch** 约 **64 / 83 / 89**。汇总对比 CSV：`outputs/field/plots/optimization/prenorm_A_Opt03_vs_Opt03w/best_metrics.csv`。**速度侧**：相对 `**A-Opt-01` / `A-Opt-02` / `A-Main-01**`，`interior.rmse_vel_mag` 与 `**summary.test_metrics.rmse_vel_mag**` 三 seed 均值在 **h128 P0 线**上最优；**内部点 `R²_u/v/w**` 均值亦优于单独 P0-1 与 P0-2。**压力侧 trade-off**：`**summary.test_metrics.rmse_p**` 三 seed 均值 **差于 `A-Opt-01`（~~0.620）与 `A-Opt-02`（~~0.610）**；`**regional_eval` · `all.rmse_p**` 与 `**A-Opt-02**` 持平（~0.398），`**interior.rmse_p**` 略差于 `**A-Opt-02**`。**（2026-03-31）** **轻量/效率对照**；**新开跑母版见 `A-Opt-05**`（篇首「战略锚点」）。见「实验记录摘要 · A-Opt-03」与 [优化路径](../01-V1路线/任务A_V1优化路径与近期实验建议.md) P0-4 节。 |
| A-Opt-03w       | Warmup 是否进一步稳定组合线                | `A-Opt-03 + warmup_epochs=5`                       | split_AG_v1   | [1,2,3] | ✅ 已完成  | **（2026-03-28）** 三 seed：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_warmup5_split_AG_v1_seed{1,2,3}_20260328_*`；后处理链同 `**A-Opt-03**`；`**best_epoch**` 约 **100 / 70 / 64**。**相对 `A-Opt-03**`：`**summary` / `regional_eval` 速度主指标未更好**，`**rmse_p` 未收复**——与 P0-3 类似，**非组合线必选项**。**（2026-03-31）** 后续新开跑**不以 03w 为母版**；轻量对照见 `**A-Opt-03**`，主母版见 `**A-Opt-05**`。见「实验记录摘要 · A-Opt-03w」。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| A-Opt-04        | 容量扩大是否继续有效                       | `hidden_dim = 256`                                 | split_AG_v1   | [1,2,3] | ✅ 已完成  | **（2026-03-29）** 三 seed：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_h256_split_AG_v1_seed{1,2,3}_20260328_*`；已含 `**predictions_test**`、`**error_analysis_interior**`、`**regional_eval**`；`**best_epoch**` 约 **80 / 70 / 82**。相对 `**A-Opt-03**`：`**interior.rmse_vel_mag**` 三 seed 均值 **变差**（约 **1.822 → 1.849**）；`**summary.test_metrics.rmse_p**` 略优。**角色**：`**A-Opt-05` 的中间态**（仅加宽）；**不作为**后续消融母版——母版取 `**A-Opt-05`（加宽+4L）**。                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| A-Opt-05        | 适度加深是否继续有效                       | `hidden_dim = 256, num_layers = 4`                 | split_AG_v1   | [1,2,3] | ✅ 已完成  | **（2026-03-29）** 三 seed：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_h256_l4_split_AG_v1_seed{1,2,3}_20260328_*`；后处理链同 `**A-Opt-04**`。**相对 `A-Opt-04**`：`**interior.rmse_vel_mag` 均值回落**（约 **1.849 → 1.816**），**略优于 `A-Opt-03` 的 1.822**。**近壁等区域**相对 03 **略优**（见各 run `**regional_eval**`）。**（2026-03-31）** 定为 **后续消融 / Line G / Line W 的配置母版**；**trade-off**：显存/时延/跨 seed 方差高于 03。**不启动 `A-Opt-06**` 见优化路径。                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| A-Opt-05t_wu10  | 10ep warmup 是否稳定 A-Opt-05 训练     | `A-Opt-05 + warmup_epochs=10`                      | split_AG_v1   | [1,2]   | ✅ 已完成  | **（2026-03-29）** seed1/2：`*_h256_l4_wu10_split_AG_v1_seed{1,2}_20260329_*`。seed1：RMSE_|v|=1.0485，interior=1.8512；**全图与内部均略差于 A-Opt-05 seed1（1.0477/1.8028）**，warmup 在 05 上无明显收益，不作为默认配置。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| A-Opt-05t_lr3e4 | lr=3e-4 是否改善 A-Opt-05 精度         | `A-Opt-05_wu10 + lr=3e-4`                          | split_AG_v1   | [1]     | ✅ 已完成  | **（2026-03-29）** seed1：`*_h256_l4_wu10_lr3e4_split_AG_v1_seed1_20260329_222839`。RMSE_|v|=1.0433，interior=1.8990，RMSE_p=**0.6364**，R2_p=**0.9254**（本组压力最优）。内部速度不如 A-Opt-05 均值；**压力端有潜力，建议补 seed2/3 再确认**。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| A-Opt-05t_wd2e4 | wd=2e-4 是否改善正则化效果                | `A-Opt-05_wu10 + wd=2e-4`                          | split_AG_v1   | [1]     | ✅ 已完成  | **（2026-03-29）** seed1：`*_h256_l4_wu10_wd2e4_split_AG_v1_seed1_20260329_222849`。RMSE_|v|=1.1871，interior=2.0648，RMSE_p=0.7711——**明显变差，不可取**。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| A-Opt-05t_sch15 | schedpat=15 是否优于默认调度             | `A-Opt-05_wu10 + scheduler_patience=15`            | split_AG_v1   | [1]     | ✅ 已完成  | **（2026-03-30）** seed1：`*_h256_l4_wu10_schpat15_split_AG_v1_seed1_20260330_134411`。RMSE_|v|=1.0549，interior=1.8464，RMSE_p=0.6640——**未形成主指标稳定优势**；母版仍为 A-Opt-05 骨架，调参分支单独归因。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| A-Opt-06        | 单尺度进一步加深是否还值得                    | `hidden_dim = 256, num_layers = 6`                 | split_AG_v1   | [1]     | 🔒 未开始 | 若 `A-Opt-05` 收益很小，建议停止                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| A-Opt-07        | 内部点区域加权是否进一步改善瓶颈                 | `optim.interior_loss_boost = 3.0`（余同 **A-Opt-05**） | split_AG_v1   | [1,2,3] | ✅ 已完成  | **（2026-04-02）** 三 seed：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_h256_l4_iboost3_split_AG_v1_seed{1,2,3}_20260331_175619`；已 `**predictions_test**` + `**regional_eval**`；与 `**A-Main-01` / `A-Opt-05**` 对照图：`plots/optimization/A_Opt07_vs_Opt05_Main01/`（`python -m training.scripts.regenerate_opt07_vs_opt05_main_figures`）。**相对母版 `A-Opt-05**`：全图 `**rmse_vel_mag**` 与 `**interior.rmse_vel_mag**` 均未更好，`**near_wall` / `wall**` 变差；`**rmse_p**` 略差——**负结果**，母版仍为 `**A-Opt-05**`。见「实验记录摘要 · A-Opt-07」。                                                                                                                                                                                                                                                                                                                                                                  |
| A-Opt-08        | 多尺度结构是否带来本质提升                    | graph U-Net / hierarchical GNN                     | split_AG_v1   | [1]     | 🔒 未开始 | 单尺度优化见顶后再立项                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |


---

## 第四批：显式几何增强线 Line G（2026-03-26 新增）

> 说明：Line G 用于在现有 geometry 已被证明有效的前提下，继续小步增加新的显式几何/拓扑先验。
> 前置条件：`**A-Abl-02` 已于 2026-04-09 三 seed 归档**（见「实验记录摘要 · A-Abl-02」）；可进入 Line G 小步验证。**新开跑一律以 `A-Opt-05` 为母版**（篇首「战略锚点」）。
> 推荐执行顺序：`A-Opt-G01 -> A-Opt-G02 -> (A-Opt-G03 / A-Opt-G04) -> A-Opt-G05`。
> 推进门槛：新增特征必须至少改善一个复杂区域（`near_wall / bifurcation / high_curvature`），且验证/测试不能退化。


| Exp ID    | 研究问题                | 唯一变化项                                        | split_version | seeds   | 当前状态  | 备注                                                                                                                                                                                                                                                               |
| --------- | ------------------- | -------------------------------------------- | ------------- | ------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A-Opt-G01 | 显式分叉拓扑先验是否改善复杂转折区建模 | `dist_to_bifurcation + branch_id`（与 JSON 一致） | split_AG_v1   | [1,2,3] | ✅ 已完成 | **（2026-04-12～04-13）** 三 seed：`outputs/field/field_transformer_opt05_g01_bifurcation_split_AG_v1_seed*_20260412_*` / `*_20260413_113440`；`predictions_test` + 标准后处理齐。配置：`training/configs/field/generated/line_g/A-Opt-G01_seed*.json`。详见「实验记录摘要 · Line G」。      |
| A-Opt-G02 | 局部尺度变化信息是否优于单纯半径值   | `dR_ds`                                      | split_AG_v1   | [1]     | ⏹ 已结案 | **（2026-04-12）** seed1：`outputs/field/field_transformer_opt05_g02_dRds_split_AG_v1_seed1_20260412_231304`。**（2026-04-17）** 单 seed 相对 `**A-Opt-05**` 无收益，**不补 seed2/3**。配置：`training/configs/field/generated/line_g/A-Opt-G02_seed1.json`。详见「实验记录摘要 · Line G」。    |
| A-Opt-G03 | 扭率能否补足曲率缺失的三维弯扭信息   | `torsion`                                    | split_AG_v1   | [1]     | ⏹ 已结案 | **（2026-04-12）** seed1：`outputs/field/field_transformer_opt05_g03_torsion_split_AG_v1_seed1_20260412_231304`。**（2026-04-17）** 单 seed 相对 `**A-Opt-05**` 无收益，**不补 seed2/3**。配置：`training/configs/field/generated/line_g/A-Opt-G03_seed1.json`。详见「实验记录摘要 · Line G」。 |
| A-Opt-G04 | 显式壁面距离是否改善近壁速度剖面学习  | `dist_to_wall` 等                             | split_AG_v1   | [1,2,3] | ✅ 已完成 | **（2026-04-13）** 三 seed：`outputs/field/field_transformer_opt05_g04_wall_distance_split_AG_v1_seed*_20260413_*`。配置：`training/configs/field/generated/line_g/A-Opt-G04_seed*.json`。详见「实验记录摘要 · Line G」。                                                            |
| A-Opt-G05 | 中心线方向变化率是否提升转折区表达   | `d_tangent_ds`                               | split_AG_v1   | [1,2,3] | ✅ 已完成 | **（2026-04-13）** 三 seed：`outputs/field/field_transformer_opt05_g05_tangent_change_rate_split_AG_v1_seed*_20260413_*`。配置：`training/configs/field/generated/line_g/A-Opt-G05_seed*.json`。详见「实验记录摘要 · Line G」。                                                      |


---

## 第五批：壁面导向优化线 Line W（2026-03-25 新增）

> 说明：Line W 直接面向端到端链路质量（WSS/OSI/RRT → 髂支闭塞风险预测）。与第三批（Line A 内部精度优化）并行推进、独立归因。
> 基座：**（2026-03-31）Line W 与 Line A 后续实验统一以 `A-Opt-05` 为默认起跑配置**（见篇首「战略锚点」：近壁等区域略优、更贴近 WSS 前置质量）。`**A-Opt-03**` 仍可作为 **低开销对照分支**（显存/时延/方差）与高显存线并列汇报；`**A-Opt-03w` 不作为母版**。
> 评估标准差异：Line W 必须额外运行 WSS 后处理对比，以壁面衍生指标质量为核心判定。
> 详见 [任务A优化路径](../01-V1路线/任务A_V1优化路径与近期实验建议.md) 第 2.4 节和第 5.5 节。


| Exp ID    | 研究问题                  | 唯一变化项                                                                       | split_version | seeds | 当前状态   | 备注                                                                                                                                                                                                        |
| --------- | --------------------- | --------------------------------------------------------------------------- | ------------- | ----- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A-Opt-W01 | 近壁区域加权是否改善 WSS 梯度质量   | `near_wall_boost=3.0, interior_weight=0.5`                                  | split_AG_v1   | [1]   | 🔒 未开始 | 需修改 `losses.py` + 近壁区 mask；依赖 P0 最优基座                                                                                                                                                                     |
| A-Opt-W02 | 壁面法向梯度监督是否提升 WSS 精度   | `wall_grad_weight=0.01`                                                     | split_AG_v1   | [1]   | 🔒 未开始 | 需修改 `losses.py`；依赖 W01 有正向信号或独立启动                                                                                                                                                                         |
| A-Opt-W03 | 直接 WSS 监督是否最大化端到端质量   | **纯 WSS**：`target_weights=0` + `wss_loss_weight=1`（草案 `A-Opt-W03-wss-only`） | split_AG_v1   | [1]   | 🔒 未开始 | **先行批已归档**：场 + WSS 联合见 `**A-*-wss-multi**`（`wss_loss_weight=0.1`，第五批附表），**不等同**于本行「仅 WSS」。**（2026-04-24）** **V2P-WSSP-01** 为 PointNeXt 上 **p + WSS**（`target_weights=[0,0,0,1]`），**亦非**「纯 WSS」草案，见篇首 V2 表 |
| A-Opt-W04 | 两阶段训练是否优于一开始就加权       | 阶段1:均匀MSE → 阶段2:壁面精调                                                        | split_AG_v1   | [1]   | 🔒 未开始 | 可与 W01/W02 叠加                                                                                                                                                                                             |
| A-Opt-W05 | OSI 敏感区域加权是否改善 OSI 恢复 | 分叉/高曲率区 boost                                                               | split_AG_v1   | [1]   | 🔒 未开始 | 可与 W01 叠加                                                                                                                                                                                                 |


### 第五批附：WSS 多任务辅助监督（`wss_multitask`，2026-04-14～04-17）

> **说明**：在 `**A-Base-*` / `A-Main-01` / `A-Opt-05**` 各自输入与场损失权重不变的前提下，增加 `**model.wss_dim=4**` 与 `**optim.wss_loss_weight=0.1**`。配置：`**training/configs/field/generated/baseline_wss_multitask/**`。登记：`**outputs/field/experiment_index.csv**`；测试集壁面 WSS 指标：`**outputs/field/wss_multitask_test_wall_wss_metrics.tsv**`；场 **R² v**、**R²_p** 见各 run `**summary.json**`。**R² 汇报主表**见本节下方「**实验记录摘要 · WSS 多任务**」。若以 `**line_g**` 为配置入口：本批跑批与 `**training/configs/field/generated/line_g/**` 无关；`**A-Opt-G02` / `G03**` 为 **⏹ 已结案**（仅 seed1，不补种），见第四批表。


| Exp ID              | 研究问题                       | 唯一变化项                   | split_version | seeds   | 当前状态    | 备注                                                                                                       |
| ------------------- | -------------------------- | ----------------------- | ------------- | ------- | ------- | -------------------------------------------------------------------------------------------------------- |
| A-Base-01-wss-multi | MLP 基线 + WSS 头             | 同 `A-Base-01` + WSS 多任务 | split_AG_v1   | [1,2,3] | ✅ 训练已完成 | `field_mlp_coord_t_bc_wss_multitask_split_AG_v1_seed{1,2,3}_20260415_*` / `20260416_*`                   |
| A-Base-02-wss-multi | GraphSAGE + wall + WSS 头   | 同 `A-Base-02` + WSS 多任务 | split_AG_v1   | [1,2,3] | ✅ 训练已完成 | `field_graphsage_coord_t_bc_wall_wss_multitask_split_AG_v1_seed{1,2,3}_20260416_*`                       |
| A-Base-03-wss-multi | Transformer + wall + WSS 头 | 同 `A-Base-03` + WSS 多任务 | split_AG_v1   | [1,2,3] | ✅ 训练已完成 | `field_transformer_coord_t_bc_wall_wss_multitask_split_AG_v1_seed{1,2,3}_20260416_*`                     |
| A-Main-01-wss-multi | 全几何 Transformer + WSS 头    | 同 `A-Main-01` + WSS 多任务 | split_AG_v1   | [1,2,3] | ✅ 训练已完成 | `field_transformer_coord_t_bc_geom_wall_wss_multitask_split_AG_v1_seed{1,2,3}_20260416_*` / `20260417_*` |
| A-Opt-05-wss-multi  | **05 母版** + WSS 头          | 同 `A-Opt-05` + WSS 多任务  | split_AG_v1   | [1,2,3] | ✅ 训练已完成 | seed1：`..._seed1_20260415_232824`；seed2/3：`..._seed{2,3}_20260414_204752`                                |


详见「**实验记录摘要 · WSS 多任务**」。`**predictions_test` / Fig A3–A5 全链**已于 **2026-04-17** 闭环（阵列 **Job 2704**、补点 **Job 2715**，见 `**docs/02-推进与变更/代码修改与实验推进记录.md**` 文首条目）。

## 主结果表（3 seed mean ± std，已完成）

> 数据来源：`experiment_index.csv` + 各 run 的 `summary.json` + `predictions_test/error_analysis/summary.json` + `predictions_test/regional_eval/fig_A5_regional_metrics.json` + `outputs/field/plots/efficiency/fig_A7_efficiency_benchmark.json`。
> **分区域指标（2026-03-24 起）**：`fig_A5_regional_metrics.json` 由 `plot_taskA_regional_bar` 生成，区域 mask 基于各预测文件中的 `graph_path` 图资产（完整 `x`），与训练时 `enabled_node_features` 无关，baseline 四模型横向可比。
> **（2026-03-26）** 优化线 `A-Opt-01` 在相同口径下已补全 `regional_eval`；默认 `**plot_taskA_multimodel_regional_bar**` 扫描结果包含 **baseline 四组 + `A-Opt-01**`（共 5 组 `exp_id`），需在文中区分「基线四模型对比」与「含 P0-1 的扩展对比」。**（2026-03-27）** `A-Opt-02` 三 seed 已各含 `regional_eval`；多模型 Fig A5 若扫全目录会再纳入 `A-Opt-02`，汇报时注意与「仅 baseline / 仅 P0-1」图区分。**（2026-03-27）** `A-Opt-02_warmup`（P0-3）已对齐同样后处理；三模型（Main / P0-2 / P0-3）Dedicated 汇总见 `**plots/optimization/prenorm_Main_P02_P02w/**`，勿与全目录盲扫 Fig A5 混淆。**（2026-03-28）** `**A-Opt-03` / `A-Opt-03w**` 已三 seed 归档；训练期 `**best_epoch`/`val` 指标** 汇总见 `**plots/optimization/prenorm_A_Opt03_vs_Opt03w/best_metrics.csv**`（与历史 run 同 CSV 并存时注意按目录名筛选 `prenorm_tw22205`）。
> **区域定义（2026-03-26）**：默认 key / 区间 / 阈值见 [任务A分区域评估口径](../../../00-规范与记录/任务A分区域评估口径.md)。
> 当前效率口径为：测试病例 `slow/GUO_XI_JIANG`（81 snapshots）、`n_warmup=5`、`n_runs=20`，并已汇总 3 个 seed；主结果表中的 `Infer(ms)` 与 `Mem(MB)` 使用 `full_case_per_snapshot_ms` 和 `full_case_peak_memory_mb` 的 `mean ± std`。
> **（2026-03-26 重要）主指标口径更新**：下表 `RMSE_|v|` 列**保留原 all-node 口径**以便纵向对比；自本次起所有出图脚本默认 `--region interior`，论文主结论应以 `interior.RMSE_|v|`（见各 run 实验记录摘要中「内部点误差」）为准；`all.RMSE_|v|` 仅作补充。
> **（2026-03-29）主表 / 近壁汇报列**：`plots/summary/fig_A1_main_table.csv` 默认以 `interior` 导出主区域 `**rmse_* / r2_***`，并附带 `**all_rmse_vel_mag / all_r2_vel_mag**` 与 `**near_wall_rmse_* / near_wall_r2_***`（含 |v| 的 `**r2_vel_mag**`）；各 run 需已生成 `predictions_test/regional_eval/fig_A5_regional_metrics.json`（必要时重跑 `plot_taskA_regional_bar`），详见 [任务A分区域评估口径](../../../00-规范与记录/任务A分区域评估口径.md) 第 5–6 节。
> **（2026-03-29）** 已归档 `**A-Opt-04**`（`hidden_dim=256`）；下表已增 `**A-Opt-04**` 列（3 seed mean ± std）。
> **（2026-03-31）** 已归档 `**A-Opt-05**`（`hidden_dim=256, num_layers=4`）并补入 ③ 容量线子表；同时入账 `**A-Opt-05_tune**` 四组小步超参实验（`warmup10` / `lr3e-4` / `wd2e-4` / `schedpat15`，均 seed=1）——详见第三批跟踪表与「实验记录摘要 · A-Opt-05_tune」。**（2026-04-02）** 已归档 `**A-Opt-07**`（`interior_loss_boost=3`，三 seed）并增列 ③ 子表；见「实验记录摘要 · A-Opt-07」。**（2026-04-14）** **Line G**：`**A-Opt-G01` / `G04` / `G05` 三 seed** 已具备 `summary.json` 与 `predictions_test`；主表增列若纳入 Line G，请以状态表「实验记录摘要 · Line G」数值为准。**（2026-04-17）** `**A-*-wss-multi**`（场 + WSS 多任务）三 seed 已入账 `experiment_index.csv`，壁面 WSS 汇总见 `**wss_multitask_test_wall_wss_metrics.tsv**`（第五批附表与「实验记录摘要 · WSS 多任务」）。**（2026-04-24）** **V2 点云壁面专线 `V2P-WSSP-01**`（seed=1，bootstrap）：见篇首 V2 表与「实验记录摘要 · V2P-WSSP-01」。**（2026-04-25）** `**V2P-WSSP-02**`（全场+WSS 头，seed=1）已归档，**流场可用 `interior`，WSS 仍以 `wall` 为主**——见「实验记录摘要 · V2P-WSSP-02」。**（2026-04-26）** `**V2P-WSSP-03` / `V2P-WSSP-04**`（seed=1）已训练并完成 `**predict_field` + 标准 Fig A3–A5 / 误差 / WSS 后处理**（wall-rich 采样，同 WSSP-01 图资产；~~原误标「标准采样」~~）；**03 vs 04**：04 在 `**r2_p` / `wall` WSS** 上**未优于** 03，见「实验记录摘要 · V2P-WSSP-03 / V2P-WSSP-04」与 `**outputs/field/plots/v2p_wssp03_vs_04_seed1/**`。**后处理约定**：若 `**target_weights` 中 u/v/w 为 0**，流场散点/误差的 u/v/w **不得作主结论**；WSS 须用 `**plot_error_analysis --wss --wss-region wall**`、`**plot_taskA_regional_bar --wss**`，**WSS 指标以 `wall` 为主**（`interior` 常不可靠）。`**plot_taskA_regional_bar**` 对此类实验请显式 `**--metric-key rmse_p**` 等，避免默认 `**rmse_vel_mag**` 误读。`**plot_training_history**`：仅画单 run 时用 `**--run-dir**`（已避免默认扫全 `outputs/field`）；若需自定义 glob 再传 `**--pattern**`。**（2026-04-27）** **Line W 权重复扫** 三配置 `**A-Opt-W03-p025-w02` / `A-Opt-W03-w02` / `A-Opt-W03-w05**` 与 `**V2P-WSSP-04**` 已同批 `**predict_field` + 分区域 + 多模型**（Slurm **3142–3145** 重训 run、见下「**Line W：A-Opt-W03 权重复扫**」）；**多模型总览** `**outputs/field/plots/wssp04_w03_line_seed1_20260427/**`。**V2P-WSSP-04** 分区域 `**r2_p**` 与 Transformer+WSS 线**差距极大**，**不得无注释混表**。



**① Baseline 组**


| 指标                | A-Base-01         | A-Base-02         | A-Base-03         | A-Main-01         |
| ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| Model             | MLP               | GraphSAGE         | Transformer       | Transformer       |
| Geom              | ✗                 | ✗                 | ✗                 | ✓                 |
| BC                | ✓                 | ✓                 | ✓                 | ✓                 |
| is_wall           | ✗                 | ✓                 | ✓                 | ✓                 |
| Physics           | ✗                 | ✗                 | ✗                 | ✗                 |
| RMSE_u            | 0.9739±0.0009     | 0.9309±0.0013     | 0.9337±0.0011     | **0.8977±0.0073** |
| RMSE_v            | 0.9755±0.0012     | 0.9165±0.0049     | 0.9170±0.0043     | **0.8518±0.0113** |
| RMSE_w            | 0.9743±0.0008     | 0.8503±0.0031     | 0.8482±0.0047     | **0.6957±0.0290** |
| RMSE_|v| (all)    | 1.5999±0.0015     | 1.3611±0.0159     | 1.3645±0.0101     | **1.1612±0.0383** |
| Interior RMSE_|v| | **2.6924±0.0242** | **2.3165±0.0452** | **2.3330±0.0148** | **2.0668±0.0492** |
| RMSE_p            | 0.6581±0.0185     | 0.7341±0.0136     | 0.7061±0.0349     | **0.6536±0.0423** |
| R2_p              | 0.9201±0.0045     | 0.9007±0.0037     | 0.9079±0.0092     | **0.9209±0.0103** |
| Infer (ms)        | **0.54±0.27**     | **2.35±0.23**     | 6.95±0.09         | **6.88±0.02**     |
| Mem (MB)          | **127.34±0.00**   | **529.69±0.47**   | 2182.74±1.08      | **2182.12±0.00**  |


**② P0 优化前半（A-Main-01★ 为对照基准；全部 Transformer + Geom + BC + is_wall）**


| 指标                | A-Main-01 ★   | A-Opt-01          | A-Opt-02          | A-Opt-02_warmup | A-Opt-03          |
| ----------------- | ------------- | ----------------- | ----------------- | --------------- | ----------------- |
| PreNorm           | ✗             | ✗                 | ✓                 | ✓               | ✓                 |
| TW[2,2,2,.5]      | ✗             | ✓                 | ✗                 | ✗               | ✓                 |
| Warmup            | ✗             | ✗                 | ✗                 | 5ep             | ✗                 |
| RMSE_u            | 0.8977±0.0073 | 0.8737±0.0056     | 0.8839±0.0084     | 0.8843±0.0095   | **0.8675±0.0011** |
| RMSE_v            | 0.8518±0.0113 | 0.8231±0.0023     | 0.8383±0.0150     | 0.8353±0.0130   | **0.8097±0.0015** |
| RMSE_w            | 0.6957±0.0290 | 0.6502±0.0011     | 0.6781±0.0351     | 0.6729±0.0387   | **0.6459±0.0019** |
| RMSE_|v| (all)    | 1.1612±0.0383 | 1.0811±0.0090     | 1.1132±0.0621     | 1.1096±0.0351   | **1.0310±0.0051** |
| Interior RMSE_|v| | 2.0668±0.0492 | **1.9187±0.0172** | 1.9727±0.0758     | 1.9648±0.0682   | 1.8222±0.0072     |
| RMSE_p            | 0.6536±0.0423 | 0.6200±0.0362     | **0.6100±0.0219** | 0.6419±0.0349   | 0.6418±0.0385     |
| R2_p              | 0.9209±0.0103 | 0.9290±0.0082     | **0.9313±0.0049** | 0.9239±0.0083   | 0.9238±0.0093     |


**③ P0 容量线（A-Opt-03★ 为对照基准；全部 PreNorm + TW[2,2,2,.5]；A-Opt-07 叠内部监督加权）**


| 指标                  | A-Opt-03 ★        | A-Opt-03w     | A-Opt-04          | A-Opt-05          | A-Opt-07      |
| ------------------- | ----------------- | ------------- | ----------------- | ----------------- | ------------- |
| hidden_dim          | 128               | 128           | 256               | 256               | 256           |
| num_layers          | 3                 | 3             | 3                 | 4                 | 4             |
| Warmup              | ✗                 | 5ep           | ✗                 | ✗                 | ✗             |
| interior_loss_boost | 1                 | 1             | 1                 | 1                 | **3**         |
| RMSE_u              | **0.8675±0.0011** | 0.8705±0.0018 | 0.8723±0.0058     | 0.8697±0.0061     | 0.8709±0.0056 |
| RMSE_v              | **0.8097±0.0015** | 0.8176±0.0032 | 0.8182±0.0072     | 0.8119±0.0039     | 0.8149±0.0051 |
| RMSE_w              | 0.6459±0.0019     | 0.6449±0.0016 | **0.6416±0.0010** | 0.6449±0.0045     | 0.6470±0.0029 |
| RMSE_|v| (all)      | **1.0310±0.0051** | 1.0665±0.0133 | 1.0519±0.0099     | 1.0399±0.0082     | 1.0433±0.0120 |
| Interior RMSE_|v|   | 1.8222±0.0072     | 1.8883±0.0064 | 1.8493±0.0121     | **1.8162±0.0275** | 1.8207±0.0211 |
| RMSE_p              | 0.6418±0.0385     | 0.6539±0.0202 | **0.6386±0.0114** | 0.6449±0.0022     | 0.6489±0.0143 |
| R2_p                | 0.9238±0.0093     | 0.9211±0.0048 | **0.9248±0.0030** | 0.9234±0.0005     | 0.9224±0.0035 |


> **注**：效率图现已包含 `mean±std` 汇总图、分 seed 延迟图、分 seed 显存图、全病例峰值显存图和分 seed Pareto 图；`speedup_vs_CFD` 仍无法填写，因为 `cfd_time_hours` 为空。
> `**A-Opt-01` / `A-Opt-02` / `A-Opt-02_warmup` / `A-Opt-03` / `A-Opt-03w` / `A-Opt-04` / `A-Opt-05` 效率列**：与 `A-Main-01` 相比，`**A-Opt-04`/`05`（`hidden_dim=256`，05 另 `num_layers=4`）显存与时延更高**；其余与同 backbone 量级叙述类似（`**A-Opt-02` 及以后为 Pre-Norm；`A-Opt-03` 叠 `target_weights`；`*_warmup` 为调度差异**）。**本次仍未重跑 `run_efficiency_benchmark**`；部署开销若需进主文，建议对 `**A-Opt-03` vs `A-Opt-05` 各补 1～3 seed 的独立条**。
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
- **下一步动作**：已完成图结构对照组角色；在当前 4 组 baseline 中，它提供了较好的精度-速度折中点；与主模型在复杂区域的对比见统一口径下的 `fig_A5` / `fig_A5_multimodel`_*。

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
- **下一步动作**：`**A-Opt-03` / `A-Opt-03w`、`A-Opt-04` / `A-Opt-05` 已依次归档（至 2026-03-29）**；**（2026-03-31）** 后续消融与几何增强线以 `**A-Opt-05`** 为母版；`**A-Opt-03**` 作轻量对照，见篇首「战略锚点」。

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
- **一句话结论**：仅加 Pre-Norm 时，**内部与多个复杂区域的速度模长误差均值较 Main 下降**，但**壁面区略退化且种子间方差仍在**；与 P0-1 的互补性已由 `**A-Opt-03`**（2026-03-28）验证。
- **下一步动作**：`**A-Opt-03` / `A-Opt-03w` 已归档**；见下文摘要。

---

### A-Opt-02_warmup（P0-3，`A-Opt-02` + `optim.warmup_epochs=5`）

- **完成日期**：2026-03-27（训练三 seed；同日补全测试导出、`regional_eval`、`error_analysis_interior` 与三模型对照图）
- **seed**：1, 2, 3（`best_epoch` 分别约 **93、70、113**；相对 `**A-Opt-02`** 的 seed3 早停于 ~**64**，warmup 将 **seed3 `best_epoch` 延后** 且 **缓解该 seed 偏弱**）
- **输出目录**：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_warmup5_split_AG_v1_seed{1,2,3}_20260327_*`（`meta.exp_id`：`A-Opt-02_warmup`）
- **相对 `A-Main-01`（`summary.json`，3 seed mean ± std）**：合成 `RMSE` **0.782 → 0.766**（仍优于 Main）；`**rmse_vel_mag`（全图节点口径）** **1.161 → 1.110**；`**RMSE p`** 均值 **0.654 → 0.642**（优于 Main、**差于** `**A-Opt-02` 的 0.610**）
- **相对 `A-Opt-02`（同口径）**：`**rmse_vel_mag` 均值略降**（**1.113 → 1.110**）；**合成 `RMSE` 略升**（**0.761 → 0.766**）；`**RMSE p` 明显变差**（**0.610 → 0.642**）；`**interior.rmse_vel_mag`（`regional_eval`）** 均值 **1.973 → 1.965**（小幅改善；**seed3** 单 seed **约 2.06 → 1.91** 量级）
- **分区域 `rmse_vel_mag`（相对 `A-Opt-02`，三 seed mean ± std）**：**壁面** 约 **0.041 → 0.039**（略回落）；**内部 / 近壁 / 高曲率 / 分叉 / 主干** 与无 warmup 线互有小幅胜负，见各 run `fig_A5_regional_metrics.json` 与汇总图
- **三模型对照图（Main / P0-2 / P0-3）**：`outputs/field/plots/optimization/prenorm_Main_P02_P02w/`（`fig_A3_multimodel_scatter_*`、`fig_A5_multimodel_regional_bar_*`、`fig_A4_multimodel_per_case_boxplot_interior_exp_subset.png`）；重生成：`python -m training.scripts.regenerate_p02_warmup_comparison_figures`
- **一句话结论**：**5 epoch 线性 warmup 主要改善 Pre-Norm 线上的「差种子」与早停分布**，全图 `**RMSE_|v|` 均值略优于无 warmup**；**压力 `RMSE p` 三 seed 均值未优于 `A-Opt-02`**，属 **trade-off**。**（2026-03-28）** `**A-Opt-03`** 主结论已出：**组合线默认无需叠 `A-Opt-03w`**；是否在其他支线上默认开 warmup 仍可个案决定。
- **下一步动作**：`**A-Opt-03` / `A-Opt-03w` 已归档**；见下文摘要。

---

### A-Opt-03（P0-4，`target_weights=[2,2,2,0.5]` + Pre-Norm）

- **完成日期**：2026-03-28（训练三 seed；与 `experiment_index.csv` 对齐）
- **seed**：1, 2, 3（`best_epoch` 分别约 **64、83、89**；见 `plots/optimization/prenorm_A_Opt03_vs_Opt03w/best_metrics.csv`）
- **输出目录**：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_split_AG_v1_seed{1,2,3}_20260327_*`
- **相对 `A-Main-01` / `A-Opt-01` / `A-Opt-02`（`summary.json` · `test_metrics`，3 seed mean ± std）**：`**rmse_vel_mag`** **1.161 → 1.031**（Main）、**1.081 → 1.031**（Opt-01）、**1.113 → 1.031**（Opt-02）；**合成 `RMSE`** **0.782 → 0.748**（Main）、**0.750 → 0.748**（Opt-01）、**0.761 → 0.748**（Opt-02）；`**rmse_p`** **相对 `A-Opt-01` / `A-Opt-02` 变差**（约 **0.620 / 0.610 → 0.642**），仍**优于 Main ~0.654**。
- `**regional_eval`（3 seed mean ± std）**：`**all.rmse_vel_mag`** **1.108 → 1.052**（Opt-01）、**1.139 → 1.052**（Opt-02）；`**interior.rmse_vel_mag`** **1.919 → 1.822**（Opt-01）、**1.973 → 1.822**（Opt-02）；`**wall.rmse_vel_mag`** **0.0368 → 0.0315**（Opt-01）、**0.0414 → 0.0315**（Opt-02）。`**all.rmse_p`** 与 `**A-Opt-02**` 持平（~~**0.398**），优于 `**A-Opt-01`**（~~**0.460**）；`**interior.rmse_p`** **~0.470**，略差于 `**A-Opt-02` ~0.440**，优于 `**A-Opt-01` ~0.535**。
- **内部点 `R²_u/v/w`（`interior`，3 seed 均值）**：**0.317 / 0.376 / 0.464**，相对 `**A-Opt-01`**（~~**0.290 / 0.350 / 0.441**）与 `**A-Opt-02`**（~~**0.274 / 0.301 / 0.409**）均提升。
- **后处理**：各 run 已具备 `predictions_test/`、`error_analysis_interior/`、`regional_eval/`。
- **一句话结论**：**P0-1 与 P0-2 在速度主指标上呈互补**：组合后 **全局与内部 `RMSE_|v|` 为当前 P0 线最优**，**壁面 `rmse_vel_mag` 相对两条单改线一并改善**，**内部速度分量 R² 同步提升**。**代价**：`**summary` 口径 `RMSE p` 三 seed 均值不及单独 `A-Opt-01` / `A-Opt-02`**，须在论文/汇报中写成 **trade-off**。
- **下一步动作**：`**A-Opt-04` / `A-Opt-05` 已跑完（2026-03-29）**；**（2026-03-31）** 以 `**A-Opt-05`** 为后续实验母版（近壁略优、内部均值略优；方差/成本高于 03）；`**A-Opt-03**` 作 P0-4 轻量对照。容量线细节见 `**A-Opt-04` / `A-Opt-05**` 摘要。`**A-Opt-03w` 可不作为默认分支**。

---

### A-Opt-03w（`A-Opt-03` + `optim.warmup_epochs=5`）

- **完成日期**：2026-03-28（训练三 seed）
- **seed**：1, 2, 3（`best_epoch` 分别约 **100、70、64**）
- **输出目录**：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_warmup5_split_AG_v1_seed{1,2,3}_20260328_*`
- **相对 `A-Opt-03`（同口径）**：`**summary.test_metrics.rmse_vel_mag`** **1.031 → 1.067**；`**interior.rmse_vel_mag`（`regional_eval`）** **1.822 → 1.888**；`**rmse_p`** **0.642 → 0.654**。
- **一句话结论**：在 **已叠 `target_weights` 的 Pre-Norm 主线** 上，**5 epoch warmup 未带来相对 `A-Opt-03` 的速度收益**，压力侧仍偏弱——**03 仅作轻量对照；母版见 `A-Opt-05`**（2026-03-31）。
- **下一步动作**：同 `**A-Opt-03`**；归档已完成。

---

### A-Opt-04（P0-5 容量①，`A-Opt-03` + `hidden_dim = 256`）

- **完成日期**：2026-03-28～03-29（训练三 seed；与仓库根目录维护的 `**outputs/field/experiment_index.csv`** 对齐）
- **seed**：1, 2, 3（`best_epoch` 分别约 **80、70、82**）
- **输出目录**：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_h256_split_AG_v1_seed{1,2,3}_20260328_*`
- **相对 `A-Opt-03`（`summary.json` · `test_metrics`，3 seed mean ± std）**：`**rmse_vel_mag`（全图节点）** **1.031 → 1.052**；`**rmse_p`** **0.642 → 0.639**（略优）；`**R²_p`** **~0.925**（与 03 基本持平略好）。
- `**regional_eval`**：`**interior.rmse_vel_mag**` **1.822 → 1.849**（**变差**；各 seed 约 **1.858 / 1.856 / 1.834**）；`**all.rmse_vel_mag`** **~1.052 → ~1.068**。**内部点 `R²_u/v/w`（3 seed 均值）**：约 **0.325 / 0.356 / 0.470**，其中 `**R²_v` 较 `A-Opt-03`（0.376）回落**。`**wall.rmse_vel_mag`** 三 seed 约 **0.027 / 0.034 / 0.049**，均值 **差于 `A-Opt-03` ~0.032**。
- **后处理**：各 run 已具备 `predictions_test/`、`error_analysis_interior/`、`regional_eval/`。
- **一句话结论**：**仅放大 hidden width 在未同步正则/训练预算调整时，未继续改善论文主口径内部速度误差**；**压力 `summary.rmse_p` 有轻微收复**。本组为 `**A-Opt-05` 的中间态**，**不作为母版**。
- **下一步动作**：`**A-Opt-05`（h256+4L）** 已定母版；**Line G 子组 G01/G04/G05 已三 seed 归档**（见下「实验记录摘要 · Line G」）；其余见 [优化路径](../01-V1路线/任务A_V1优化路径与近期实验建议.md)。

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
- **下一步动作**：**（2026-03-31）** 本组为 **消融 / Line G / Line W 的统一母版**；按 P0-5 **停止条件**不继续 `A-Opt-06`；`**A-Opt-05_tune`** 见下节；`**A-Abl-02` 已完成（2026-04-09）**；`**A-Opt-G01 / G04 / G05` 已三 seed 归档（2026-04-12～04-14）**；`**G02` / `G03` 已结案（仅 seed1）** → 优先 `**A-Abl-01`** 或 **Line W**（**WSS 全量对比暂缓**）。

---

### A-Opt-07（P1-2 内部监督加权，`A-Opt-05` + `interior_loss_boost = 3`）

- **完成日期**：2026-03-31～04-01（训练三 seed）；**（2026-04-02）** 测试集预测与区域评估闭环
- **seed**：1, 2, 3（`best_epoch` 分别为 **106、55、72**）
- **配置**：`training/configs/field/generated/optimization/A-Opt-07_seed{1,2,3}.json`；`run.experiment_name` 后缀 `**_iboost3`**
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
- **一句话结论**：**内部 `interior_loss_boost=3` 未带来相对母版 `A-Opt-05` 的主指标收益**，属**清晰负结果**；后续主线仍取 `**A-Opt-05`**；`**A-Abl-02` 已闭环**；**Line G 之 G01/G04/G05 已三 seed 归档（2026-04-14）**。
- **下一步动作**：不必追加深或再扫 `interior_boost`；将本组作为「仅加权内部监督不够」写入正文/附录即可。

---

### Line G（显式几何增强，母版 `A-Opt-05`）

> **（2026-04-14）** 摘要：`**A-Opt-G01` / `G04` / `G05`** 已 **三 seed** 训练、`predictions_test` 与 Fig A3–A5 多模型对比就绪。**（2026-04-17）** `**G02` / `G03`** 仅 **seed=1**，单 seed 上相对 `**A-Opt-05`** 无收益，**已决定不补 seed2/3**，记 **⏹ 已结案**。

- **配置目录**：`training/configs/field/generated/line_g/`（`A-Opt-G0*_seed*.json`）。
- **与母版 `A-Opt-05`（`summary.json` · `test_metrics`，三 seed 均值）对照（速览）**：
  - `**rmse_vel_mag`**：05 约 **1.040**；**G04 / G05** 约 **1.038**（略优）；**G01** 约 **1.045**（略差）。
  - `**rmse`（合成）**：G01 约 **0.543** 略优于 05 约 **0.562**；G04 / G05 与 05 接近。
  - **分通道 `R²`**：G01 的 `**R²_u` / `R²_p**` 略优于 05；`**R²_vel_mag**` 三组 Line G 约在 **0.58～0.59**（05 的 `summary` 未统一写入 `r2_vel_mag`，不宜直接并列）。
- **多模型对比图（G01/G04/G05 × 三 seed 均值 vs baseline / Main / 05）**：`outputs/field/plots/line_g/G01_G04_G05_vs_baselines_mean3seed/`（详见 `docs/02-推进与变更/代码修改与实验推进记录.md` 2026-04-14 条目）。
- **一句话结论**：在 **05 母版**上叠加 **壁面距离（G04）** 或 **切向变化率（G05）** 对 `**summary.rmse_vel_mag` 三 seed 均值** 有 **小幅正向**；**分叉拓扑（G01）** 在 **合成 `rmse` 与压力/分速度 R²** 上有信号，但 **速度模**略差于 05；均 **未形成大幅碾压**，适合作为正文「几何先验小步扩展」而非新母版。`**G02`（`dR_ds`）/ `G03`（`torsion`）** 在 **seed=1** 上相对05 **未见收益**，**不再补种**。
- **下一步动作**：转入 `**A-Abl-01`** / **Line W**；汇报时与 `**A-Base-*` / `A-Main-01`** 对照保留叙事完整性。

---

### WSS 多任务（场监督 + `wss_loss_weight=0.1`，2026-04-14～04-17）

- **完成日期**：训练三 seed 已登记 `**outputs/field/experiment_index.csv`**（2026-04-14～04-17）；测试集**壁面节点** WSS RMSE/R² 见 `**outputs/field/wss_multitask_test_wall_wss_metrics.tsv`**；**（2026-04-17）** `**predictions_test` + Fig A3/A4/误差/Fig A5** 已按推进记录闭环（阵列 2704、补点 2715）。
- **配置**：`**training/configs/field/generated/baseline_wss_multitask/`**（`A-Base-01-wss-multi`～`A-Opt-05-wss-multi`）
- **与 Line W `A-Opt-W03` 草案的区别**：本批为 **速度场/压力 + WSS 联合**（`target_weights` 与各自无 WSS 母版一致）；`**A-Opt-W03-wss-only`** 为 **关闭场监督、仅 WSS**（`training/configs/field/generated/wss_multitask/`，**未跑**）

**汇报用 R² 主表（多任务各 `exp_id`，三 seed 均值 ± 总体标准差）**


| Exp ID              | **R² 壁面 WSS**（`wss_r2_wss`，TSV） | **R² |v|**（`summary.test_metrics.r2_vel_mag`） | **R²_p**（`summary.test_metrics.r2_p`） |
| ------------------- | ------------------------------- | --------------------------------------------- | ------------------------------------- |
| A-Base-01-wss-multi | −0.052 ± 0.008                  | 0.016 ± 0.002                                 | 0.923 ± 0.004                         |
| A-Base-02-wss-multi | 0.383 ± 0.006                   | 0.289 ± 0.009                                 | 0.902 ± 0.003                         |
| A-Base-03-wss-multi | 0.384 ± 0.006                   | 0.288 ± 0.001                                 | 0.912 ± 0.009                         |
| A-Main-01-wss-multi | 0.460 ± 0.004                   | 0.468 ± 0.007                                 | 0.928 ± 0.007                         |
| A-Opt-05-wss-multi  | 0.463 ± 0.004                   | 0.584 ± 0.013                                 | 0.922 ± 0.003                         |


> **口径**：**R² 壁面 WSS** 为测试集 **壁面节点**、WSS **向量（4 维合成）** 的 R²，与 `training/scripts/_eval_wss_metrics_once.py` / `**WSSMeter`** 一致。**R² v**、**R²_p** 来自各 run `**summary.json`** 的 `**test_metrics**`（训练结束全图 eval；多任务 run 均含 `r2_vel_mag` 字段）。

- `**summary.test_metrics` 其它速览（三 seed 均值）**：`**A-Opt-05-wss-multi`** — `rmse_vel_mag` **约 1.043**，合成 `**rmse`** **约 0.752**；`**A-Main-01-wss-multi`** — `rmse_vel_mag` **约 1.179**，合成 `**rmse`** **约 0.784**（与上表 **R²_p** 同源）。
- **壁面 WSS RMSE（TSV · `wss_rmse`，三 seed 均值）**：Base-01-wss-multi **约 1.262**；Base-02/03-wss-multi **约 1.13**；Main-01-wss-multi **约 1.084**；Opt-05-wss-multi **约 1.077**（相对 Main-wss-multi **略优**）。
- **一句话结论**：多任务头下，**壁面 WSS R²** 在 **MLP** 上为**负**；在 **GraphSAGE / Transformer-wall** 上约 **0.38**；在 **Main / Opt-05 母版** 上约 **0.46～0.46**，**Opt-05-wss-multi** 与 **Main-wss-multi** 基本持平、壁面 RMSE 略优。**速度模 R²**（`r2_vel_mag`）随骨干由 MLP →05 **单调抬升**（约 **0.02 → 0.58**），**压力 R²** 在 **Main-wss-multi** 上最高（**约 0.928**），**Opt-05-wss-multi** 略低但仍与 **0.92** 档对齐。与无 WSS 同配置的 **逐对 trade-off** 仍以各 `**summary.json`** / **区域 Fig A5** 为准。
- **下一步动作**：按需运行 `**plot_taskA_multimodel_regional_bar`** 做多模型柱图；或接 **Line W** 加权 / **纯 WSS（`A-Opt-W03-wss-only`）** 分支。

---

### A-Abl-02（显式几何分量消融，母版 `A-Opt-05`）

- **完成日期**：2026-04-02～04-07（各子组训练）；**（2026-04-09）** 三 seed 汇总与文档归档
- **对照母版**：`**A-Opt-05`**（`interior.rmse_vel_mag` 三 seed 均值 **1.816±0.034**）
- **指标口径**：`**regional_eval` · `interior` · `rmse_vel_mag`**；统计汇总见 `**outputs/field/plots/ablation/geometry_opt05_mean3seed/fig_A6_ablation_summary_stats_interior.json**`（paired *t* 相对母版，*n*=3 seed）
- **输出目录模式**：`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_h256_l4_abl02_no_{abscissa|normradius|curvature|tangent}_split_AG_v1_seed{1,2,3}_`*（已登记 `**outputs/field/experiment_index.csv**`）
- **Fig A6（五组合计）**：`outputs/field/plots/ablation/geometry_opt05_mean3seed/fig_A6_ablation_summary_interior.{png,csv}`（复现命令见 `**docs/02-推进与变更/代码修改与实验推进记录.md`** 2026-04-07 条目）


| 子组            | 去掉的分量        | interior `rmse_vel_mag`（mean±std） | 相对 05 Δmean | paired *t* *p* | 解读                       |
| ------------- | ------------ | --------------------------------- | ----------- | -------------- | ------------------------ |
| `A-Abl-02-01` | Abscissa     | 1.868±0.026                       | +0.052      | ~0.26          | 有退化趋势，**三 seed 不足以宣称显著** |
| `A-Abl-02-02` | NormRadius   | 2.043±0.020                       | +0.227      | ~**0.0015**    | **影响最大**，与母版差异**统计显著**   |
| `A-Abl-02-03` | Curvature    | 1.851±0.030                       | +0.035      | ~0.26          | **与母版最接近**，单项贡献**最弱**    |
| `A-Abl-02-04` | Tangent（3 维） | 1.933±0.038                       | +0.117      | ~0.073         | **次之**，接近常见显著性阈值         |


- **一句话结论**：在 `**A-Opt-05` 母版**上，**归一化半径 NormRadius 是显式几何通道中的主贡献项**；**切向 Tangent** 次之；**弧长 Abscissa** 与 **曲率 Curvature** 的边际更大但 **3 seed 下相对母版未达常规显著**——叙事上可写「半径与局部方向框架最关键，曲率单项可视为弱贡献」。
- **下一步动作**：推进 `**A-Abl-01*`*（输入层级消融）；**Line G 侧** `**G01/G04/G05` 已闭环**，`**G02`/`G03` 已结案（不补种）**；可接 **Line W**；复杂区域对比可继续用各 run 的 `**regional_eval`** 补充正文。

---

### `A-Opt-05_tune`（P0-5 后小步超参，`A-Opt-05` 骨架上的试跑）

> **配置目录**：`training/configs/field/generated/optimization/A-Opt-05_tune/`。批量 manifest 示例：`training/cluster/manifest_list_A-Opt-05_tune.tsv`。

- **（2026-03-31）本仓库已入账（`outputs/field/experiment_index.csv`）**：`**A-Opt-05t_warmup10`**（seed **1、2**）、`**A-Opt-05t_warmup10_lr3e-4`**（seed **1**）、`**A-Opt-05t_warmup10_wd2e-4`**（seed **1**）、`**A-Opt-05t_warmup10_schedpat15`**（seed **1**，run：`..._wu10_schpat15_..._20260330_134411`）。上述 run 已具备 `**predictions_test/`**、`**error_analysis_interior/**`、`**regional_eval/**`（与 `**A-Opt-05**` 后处理链一致）。
- **清单与目录差**：`manifest_list_A-Opt-05_tune.tsv` 中含 `**A-Opt-05t_warmup5`（三 seed）** 与 `**lr3e-4` seed2/3**；**当前 `outputs/field/` 无同名实验目录**——若已在其他机器跑完，需拷回并登记 `experiment_index.csv` 后再算「闭环」。
- **数值倾向（seed1 主表，摘要）**：`**lr3e-4`** 在 `**interior.rmse_vel_mag**` 上相对基线 `**A-Opt-05**` **略优**，**内部 `R²` 分量略好**；`**wd2e-4`** **明显变差**；`**schedpat15`** **未稳定优于** `**warmup10` 默认调度**（详见各 run `**summary.json`** 与 `**regional_eval/fig_A5_regional_metrics.json**`）。
- **多模型横向图（文件夹名含 `Opt03`，seed=1 子集）**：`outputs/field/plots/optimization/A_Opt05_tune_vs_Opt03_seed1/`（Fig A3 / A5 / A4；**解读时**以 **母版 `A-Opt-05`** 为主视角，`A-Opt-03` 为对照）。
- **WSS**：**未做** `compare_hemo_wss_runs` **全量导出**；后续可用 `**training/cluster/wss_runs_A_Opt03_vs_Opt05tune_seed1.tsv`** 在集群提交。

---

### V2P Bootstrap（PointNeXt，split_AG_v1，2026-04-20～04-21）

> **口径说明**：本节为 **bootstrap 结果**，使用 `split_AG_v1` 而非正式 `split_AG_v2`；结论仅用于回答"PointNeXt 主干能否稳定训练"与"geometry 在 V2P 上是否仍有正信号"，**不作为正式 V2 结论入账**。

- **完成日期**：`V2P-Base-01` seed=1（2026-04-20）；`V2P-Main-01` seed=1（2026-04-21）
- **配置**：`training/configs/field/generated/v2_pointcloud/V2P-Base-01_seed1.json` / `V2P-Main-01_seed1.json`
- **manifest**：`training/configs/field/generated/v2_pointcloud/manifest_bootstrap_split_AG_v1.json`
- **输出目录**：
  - `outputs/field/field_v2_pointnext_base01_coord_t_bc_wall_bootstrap_split_AG_v1_seed1_20260420_202710/`
  - `outputs/field/field_v2_pointnext_main01_coord_t_bc_geom_wall_bootstrap_split_AG_v1_seed1_20260421_034607/`

**全局测试指标对比（seed=1，`summary.json · test_metrics`）**


| 指标             | **V2P-Base-01**（无 geom） | **V2P-Main-01**（有 geom） | A-Opt-05 V1 锚点（3 seed 均值） |
| -------------- | ----------------------- | ----------------------- | ------------------------- |
| `best_epoch`   | 87                      | **68**                  | 79                        |
| `rmse`（全局合成）   | 0.836                   | **0.746**               | 0.754                     |
| `rmse_vel_mag` | 1.300                   | **1.011**               | 1.048                     |
| `r2_vel_mag`   | 0.354                   | **0.609**               | ~0.576（regional/all）      |
| `rmse_u`       | 0.927                   | **0.864**               | 0.878                     |
| `rmse_v`       | 0.897                   | **0.810**               | 0.813                     |
| `rmse_w`       | 0.829                   | **0.648**               | 0.649                     |
| `rmse_p`       | 0.666                   | **0.634**               | 0.648                     |
| `r2_u`         | 0.139                   | **0.253**               | 0.229                     |
| `r2_v`         | 0.228                   | **0.371**               | 0.366                     |
| `r2_w`         | 0.355                   | **0.606**               | 0.605                     |
| `r2_p`         | 0.918                   | **0.926**               | 0.923                     |


> A-Opt-05 的 `r2_vel_mag`（~0.576）取自 `regional_eval/all`；V2P 实验暂无分区域评估文件。

- **Geometry 增益（Main-01 vs Base-01）**：`rmse_vel_mag` 下降 **22.2%**（1.300→1.011）；`r2_vel_mag` 提升 **+0.255**（0.354→0.609，相对+72%）；速度分量 R²（u/v/w）全线大幅改善，信号清晰。
- **V2P-Main-01 vs V1 锚点 A-Opt-05**：`rmse_vel_mag` -3.5%，`r2_vel_mag` +5.7%；满足文档规定的首轮通过条件（`r2_vel_mag` 口径）。
- **短板**：仅 seed=1；无分区域评估（near_wall/interior 缺失）；bootstrap 口径（split_AG_v1）。
- **一句话结论**：**PointNeXt 主干可稳定训练；显式 geometry 增益显著（rmse_vel_mag -22%，r2_vel_mag +72%）；V2P-Main-01 seed=1 已在全局 r2_vel_mag 口径上超过 V1 锚点 A-Opt-05。**
- **下一步动作**：① 运行分区域评估补全 near_wall/interior 指标；② 推进 Gate-0（split_AG_v2）；③ Gate-0 通过后重跑正式 V2P-Base-01/V2P-Main-01；④ 结果稳定后开放 V2P-Opt-01 与 V2P-Abl-02-*。

---

### V2P-WSSP-01（PointNeXt，p + WSS 直接监督，bootstrap `split_AG_v1`，2026-04-24）

> **定位**：**不计入** V2「首轮 5 组」最小集合；属于 **V2 点云路线**上、与 **Line W / 壁面血流动力学**对齐的 **bootstrap** 实验（采样：`wall_max_points=13000`、近壁内部 2000，总 15000 点；详见配置 `meta.notes`）。**正式结论**仍待 `split_AG_v2` 与多 seed。

- **完成日期**：2026-04-24（seed=1）；**best_epoch**：68（`summary.json`）
- **配置**：`training/configs/field/generated/v2_pointcloud/V2P-WSSP-01_seed1.json`；manifest 登记：`training/configs/field/generated/v2_pointcloud/manifest_wssp_wall13000_near2000.json`
- **输出目录**：`outputs/field/field_v2_pointnext_wssp01_geom_wall13000_near2000_split_AG_v1_seed1_20260424_164609/`
- **监督口径**：`optim.target_weights = [0,0,0,1]`（**仅 p**，u/v/w 无数据项）；`optim.wss_loss_weight = 1.0`，`model.wss_dim = 4`

`**summary.json` · `test_metrics`（全图 eval，seed=1）**


| 键                             | 值      | 解读                  |
| ----------------------------- | ------ | ------------------- |
| `rmse_p`                      | ~0.691 | 压力可拟合               |
| `r2_p`                        | ~0.911 | 与 p 监督一致            |
| `r2_u`, `r2_v`, `r2_w`        | ~0     | **未监督速度，不得据此否定主干**  |
| `rmse_vel_mag` / `r2_vel_mag` | 差 / 负  | 同上                  |
| `wss_loss`                    | ~4.10  | 验证 batch 内 WSS 项占主导 |
| `data_loss`                   | ~0.12  | 主要来自 p              |


**WSS 分区域（`predictions_test/regional_eval/fig_A5_regional_wss_metrics.json`，测试集全节点聚合）**

- `**wall`**（主汇报）：`rmse`（四维合成均方根）~**1.01**；`**r2_wss`（标量通道）** ~**0.44**；`rmse_wss_x/y/z`、分项 `r2_*` 见 JSON。
- `**interior`**：`r2_wss` 等**大幅为负**——与 **WSS 真值/监督主要在壁面**一致，**勿与 wall 混读**。

**已生成产物（相对 run 目录）**

- `predictions_test/manifest.json`、`*.pt`
- `predictions_test/error_analysis_interior/`（流场；**u/v/w 面板仅作对照**）
- `predictions_test/error_analysis_interior/wss/`（`**plot_error_analysis --wss --wss-region wall`**）
- `predictions_test/regional_eval/`（`**plot_taskA_regional_bar --metric-key rmse_p --metric-key rmse --wss**`）
- `fig_training_curves.png`（`plot_training_history --run-dir <run>`）
- `fig_A3_scatter_interior.png`（标题建议英文，避免中文字体警告）
- **一句话结论**：在 **bootstrap 口径**下，**p 监督有效**；**WSS 直接监督下壁面区标量 WSS 有一定可解释拟合（`wall.r2_wss` ~0.44）**，分量与 interior WSS 仍弱；速度场未训练，**不能**与 `A-Opt-05` 等全速度监督实验直接比 `rmse_vel_mag`。
- **下一步动作**：Gate-0 / `split_AG_v2` 后 **复跑 seed 1～3**；与 `**A-Opt-05-wss-multi`** 等对照时 **分列「场监督 / WSS 权重 / 采样」**，避免混表。

### V2P-WSSP-02（PointNeXt，全场 u/v/w/p + WSS 辅助头 + 混合早停，bootstrap `split_AG_v1`，2026-04-25）

> **定位**：与 `**V2P-WSSP-01`** 对照——同 PointNeXt + WSS 头，但恢复 **全速度+压力** 场监督（`target_weights=[2,2,2,0.5]`）、`**wss_loss_weight=0.5`**、`**early_stop_wss_weight=1.0**`，wall-rich 采样（同 WSSP-01 图资产；~~原配置 `meta.notes` 误标为「标准采样」，2026-04-26 核实修正~~）。**不计入** V2 首轮 5 组；结论仍为 bootstrap。

- **完成日期**：2026-04-25（seed=1）；`**best_epoch`**：58；`**best_val_loss`（混合）**：~0.742（`summary.json`）
- **配置**：`training/configs/field/generated/v2_pointcloud/V2P-WSSP-02_seed1.json`
- **输出目录**：`outputs/field/field_v2_pointnext_wssp02_geom_full_supervision_wss_split_AG_v1_seed1_20260425_093041/`
- **监督口径**：`target_weights=[2,2,2,0.5]`；`wss_loss_weight=0.5`；`early_stop_wss_weight=1.0`；`model.wss_dim=4`

`**summary.json` · `test_metrics`（全图 eval，seed=1，节选）**


| 键                       | 值             | 解读                                                                                |
| ----------------------- | ------------- | --------------------------------------------------------------------------------- |
| `r2_vel_mag`            | **~0.635**    | 相对 **WSSP-01**（速度未训、`r2_vel_mag` 为负）**场量主指标大幅改善**；可与 `**V2P-Main-01` / 全速度监督**线对照 |
| `r2_p`                  | ~0.004        | 与 `**target_weights` 压低 p 权重** 一致，压力可解释方差近零                                       |
| `r2_u/v/w`              | 正但中等          | 场监督下速度分量有信号                                                                       |
| `wss_r2_wss`            | **~−0.018**   | 测试集壁面 WSS 向量 R² 仍为**略负**；**未**实现「加全场监督即转正 WSS」                                    |
| `wss_rmse` / `wss_loss` | ~2.31 / ~19.5 | WSS 误差项仍大；与 **WSSP-01** 的 WSS 口径不可仅比绝对值（损失权重与主任务不同）                               |


`**regional_eval/fig_A5_regional_wss_metrics.json`（测试集聚合）**

- `**wall`（主汇报 WSS）**：`r2_wss` **~−0.018**，`rmse` ~2.31（与 `summary` 中 `wss_*` 同量级）
- `**interior`**：WSS 的 R² 常因方差近零**数值爆炸**，**勿作主结论**；**以 `wall` / `all` 为准**（同 WSSP-01 约定）

**已生成产物（相对 run 目录）**

- `predictions_test/manifest.json`、**1458** 个 `*.pt`
- `fig_A3_scatter_interior.png`、`fig_A4_per_case_boxplot_interior.png`、`fig_A4_per_case_metrics_interior.csv`
- `predictions_test/error_analysis_interior/`（流场；**interior** 散点/箱线口径与全速度监督实验一致，**可作主 readout**）
- `predictions_test/error_analysis_interior/wss/`（`**plot_error_analysis --wss`**）
- `predictions_test/regional_eval/`（`**plot_taskA_regional_bar --wss**`，含场与 WSS 柱图及 JSON）
- 训练曲线：`plot_training_history --run-dir <run_dir>`
- **与 V2P-WSSP-01 的粗对比（同 seed、bootstrap）**：**WSSP-02** 在 `**r2_vel_mag`** 上 **明显更好**；**WSSP-01** 在 `**r2_p`** 与 **壁面 WSS 可解释性（`wall.r2_wss` ~0.44，regional）** 上 **更好**——两者 **监督目标不同**（采样口径实际一致，均为 wall-rich 13000+2000），归因须围绕监督配方分列。**WSSP-02** 未在 **WSS R²** 上超过 **WSSP-01 壁面指标**。
- **一句话结论**：**全速度场监督**显著改善 **速度模** 测试表现，但 **WSS 头测试 R² 仍未转正**；不支持「仅加当前配方的辅助头+混合早停即可解决 WSS」的强结论。
- **下一步动作**：Gate-0 后 `split_AG_v2` 上 **补 seed 2/3**；与 `**A-Opt-05-wss-multi`** 对照时明列 **backbone 与 `wss_loss_weight` / 采样**；若重跑 WSS 归一化需同步 **preprocess 版本** 标签。

### V2P-WSSP-03（PointNeXt，全场 u/v/w/p + 轻量 WSS 辅助头，bootstrap `split_AG_v1`，2026-04-26）

> **定位**：修复 **WSSP-02** 的梯度失衡——保持 `**target_weights=[2,2,2,0.5]`**（同 Main-01），将 `**wss_loss_weight` 降至 0.01**，`**early_stop_wss_weight=0`**（场 loss 驱动早停）。wall-rich 采样（同 WSSP-01 图资产；~~原文档误标为「标准采样」，2026-04-26 核实修正~~）；**不计入** V2 首轮 5 组。

- **完成日期**：2026-04-25～04-26（训练落盘至 04-26）；`**best_epoch`**：34；`**best_val_loss**`：~0.741（`summary.json`）
- **配置**：`training/configs/field/generated/v2_pointcloud/V2P-WSSP-03_seed1.json`
- **输出目录**：`outputs/field/field_v2_pointnext_wssp03_geom_full_supervision_wss_low_split_AG_v1_seed1_20260425_230427/`
- **监督口径**：`target_weights=[2,2,2,0.5]`；`wss_loss_weight=0.01`；`early_stop_wss_weight=0`；`model.wss_dim=4`

`**summary.json` · `test_metrics`（全图 eval，seed=1，节选）**


| 键              | 值           | 解读                                                                            |
| -------------- | ----------- | ----------------------------------------------------------------------------- |
| `r2_vel_mag`   | **~0.591**  | 相对 **WSSP-02**（`r2_p` 崩溃）压力恢复可读；相对 `**V2P-Main-01`**（~0.609）略低，属同主干+WSS 头合理范围 |
| `r2_p`         | **~0.025**  | 仍低，但相对 **WSSP-02（~0.004）** 明显修复                                               |
| `wss_r2_wss`   | **~−0.017** | 与 **WSSP-02** 同量级，**轻量 WSS 权重未带来 WSS R² 转正**                                  |
| `wss_rmse_wss` | ~2.31       | 与 **WSSP-04** 几乎相同                                                            |


`**regional_eval/fig_A5_regional_wss_metrics.json` · `wall`（主汇报 WSS）**

- `rmse_wss` ~**2.308**，`r2_wss` ~**−0.017**（与 `summary` 一致）

**已生成产物（相对 run 目录）**

- `predictions_test/manifest.json`、**1458** 个 `*.pt`（2026-04-26 导出）
- `fig_A3_scatter_interior.png`、`fig_A3_scatter_wall.png`、`fig_A4_per_case_boxplot_interior.png`、`fig_A4_per_case_metrics_interior.csv`
- `predictions_test/error_analysis_interior/`（含 `wss/`）
- `predictions_test/regional_eval/`（`plot_taskA_regional_bar --wss`）
- **一句话结论**：**WSSP-03 成功修复 WSSP-02 的「压力崩溃」**；**WSS 测试 R² 仍为略负**，未达配置备注中「向 A-*-wss-multi ~0.46 靠拢」的预期。
- **下一步动作**：与 **WSSP-04** 对照已见「实验记录摘要 · V2P-WSSP-04」；若要坚持压力+WSS 主线，需 **多 seed、`split_AG_v2` 或架构/数据口径** 层面的改动，而非仅损失重加权。

### V2P-WSSP-04（PointNeXt，压力 + WSS 主线、速度弱辅助，bootstrap `split_AG_v1`，2026-04-26）

> **定位**：将 **p 与 WSS** 作为主优化对象，`u/v/w` 仅 **0.1** 权重辅助近壁上下文；用于检验「压低速度、抬高 p/WSS 损失」能否改善 `**r2_p` / `wall` WSS**。

- **完成日期**：2026-04-26（seed=1）；`**best_epoch`**：41；`**best_val_loss**`：~0.041（`summary.json`；标度与 03 不可直接比，因 total loss 配方不同）
- **配置**：`training/configs/field/generated/v2_pointcloud/V2P-WSSP-04_seed1.json`
- **Manifest（批量登记用）**：`training/configs/field/generated/v2_pointcloud/manifest_wssp_pressure_wss_primary.json`
- **输出目录**：`outputs/field/field_v2_pointnext_wssp04_geom_pressure_wss_weakvel_split_AG_v1_seed1_20260426_004522/`（**2026-04-27 集群重训+全后处理规范 run**：`…/20260427_103849/`，与 Line W 三实验、多模型同批，见上「**（2026-04-27）**…」与 [代码修改与实验推进记录](../../../02-推进与变更/代码修改与实验推进记录.md) 文首 **2026-04-27** 条目。）
- **监督口径**：`target_weights=[0.1,0.1,0.1,1.0]`；`wss_loss_weight=0.5`；`early_stop_wss_weight=0.0`；`model.wss_dim=4`

`**summary.json` · `test_metrics`（全图 eval，seed=1，节选）**


| 键              | 值           | 与 WSSP-03 对比      |
| -------------- | ----------- | ----------------- |
| `r2_vel_mag`   | ~**0.579**  | **低于** 03（~0.591） |
| `r2_p`         | ~**0.014**  | **低于** 03（~0.025） |
| `wss_r2_wss`   | ~**−0.017** | 与 03 **同量级**      |
| `wss_rmse_wss` | ~2.31       | 与 03 **几乎相同**     |


`**regional_eval/fig_A5_regional_wss_metrics.json` · `wall`**

- `rmse_wss` ~**2.307**，`r2_wss` ~**−0.017**（相对 03 **无实质增益**）

**双 run 可视化（仓库根执行后落盘）**

- `outputs/field/plots/v2p_wssp03_vs_04_seed1/`：`plot_taskA_multimodel_scatter`（`p` / `vel_mag` × `interior` / `wall`）、`plot_taskA_multimodel_regional_bar`（`rmse_vel_mag`、`rmse_p`、`rmse_wss`，`--seed 1`）

**已生成产物**：与 **WSSP-03** 相同后处理链（`predictions_test` + `error_analysis_interior` + `regional_eval`）。

- **一句话结论**：在 **seed=1、wall-rich 采样（同 WSSP-01）** 下，**损失重加权（04）未实现「压力 + 壁面 WSS 优于 03」**；主读 **p** 与 **wall WSS** 均未改善，**速度模略差**。不支持「仅靠 04 配方即可达成导师主线」的结论。
- **下一步动作**：若仍走压力+WSS 主线（采样口径已确认与 WSSP-01 一致），优先 `**split_AG_v2` + 多 seed** 再判；架构/归一化级问题见 [代码修改与实验推进记录](../../../02-推进与变更/代码修改与实验推进记录.md) **2026-04-26** 条目。

**（2026-04-27）集群重启后同配置重训与分区域主读数（run `…/20260427_103849/`）**

- `**summary.json`**（全图 eval，节选）：`r2_p≈0.021`，`wss_r2_wss≈−0.012`，`r2_vel_mag≈0.566`（与上表 2026-04-26 节选同量级；`best_epoch=100`）。
- `**regional_eval`（论文/内参主读数）**：`interior` `**r2_p`≈0.025**、`wall` 场 `**r2_p`≈0.019**；`fig_A5_regional_wss_metrics.json` 的 `**wall`**：`r2_wss`≈**−0.012**，`rmse_wss`≈**2.30**。与 **仅看 `summary`** 的叙事须一致标注 **区域**。
- **与 Line W 三实验（见下节）对比**：Transformer+场全监督+WSS 权重复扫在 `**interior` `r2_p`≈0.63～0.64** 量级，**V2P-WSSP-04 不可与其无注释混表**。

### V2P-WSSP-05（PointNeXt，复刻 WSSP-01：**仅 p+WSS**，MSE WSS，`split_AG_v1` 三 seed，2026-04-28）

> **定位**：在 **wall13000+near2000** 图资产与计划矩阵命名下复核 **WSSP-01**，补全 seed=1/2/3，建立 **仅监督压力 + WSS（速度 loss=0）** 的正式三 seed 基线。

- **集群**：训练 Slurm **3187～3189**（`training/cluster/run_train_field.slurm`）；`**predict_field` + 任务 A 标准图件**阵列 **3201**（`manifest_list_v2p_wssp05_06_predict.tsv` 第 1～3 行）。
- **配置**：`training/configs/field/generated/v2_pointcloud/V2P-WSSP-05_seed{1,2,3}.json`
- **输出目录**：`outputs/field/field_v2_pointnext_wssp05_baseline_p_wss_wall13000_near2000_split_AG_v1_seed{1,2,3}_20260428_000004/`

`**summary.json` · `test_metrics`（测试集聚合 eval，节选）**


| seed | `best_epoch` | `r2_p`    | `wss_r2_wss` | `wss_rmse_wss` |
| ---- | ------------ | --------- | ------------ | -------------- |
| 1    | 34           | **0.215** | −0.025       | ~2.32          |
| 2    | 45           | 0.038     | −0.022       | ~2.31          |
| 3    | 24           | −0.022    | −0.016       | ~2.31          |


- **读数口径**：速度分量 `r2_u/v/w`、`r2_vel_mag` **未监督**，仅作文档占位，不作为主结论。
- `**predictions_test` + `regional_eval`**：三 seed 已生成；若要 `**error_analysis_interior/wss/**` 壁面 WSS 图，对上述 manifest 执行 `**plot_error_analysis --wss**`（阵列 **3205** 为可选补跑）。
- **一句话结论**：三 seed 下 `**wss_r2_wss` 仍为略负**，`**r2_p` 方差大**（含一次 **~0.22**）；**未达到** Go/No-Go 中 **V1 `A-Opt-05-wss-multi` ~0.46**。**相对 WSSP-01 单 seed 的「复核」已齐三 seed**；与 **06** 对照见下节（**Huber 未单列解决**）——**下一步**：`**V2P-WSSP-07`** 或 **08**（见 [计划讨论.md](../计划讨论.md)）。

### V2P-WSSP-06（PointNeXt，同 05，`wss_loss_type=huber`，三 seed，2026-04-28）

> **定位**：在 **05** 上 **单变量** 将壁面 WSS 监督由 **MSE** 换成 **Smooth L1（文档称 Huber，`beta=1`）**，检验异常值/梯度尺度是否是主瓶颈。

- **集群**：训练 **3190～3192**；后处理与 **05** 同清单（`manifest` 第 4～6 行），**3201 / 可选 3205**。
- **配置**：`training/configs/field/generated/v2_pointcloud/V2P-WSSP-06_seed{1,2,3}.json`
- **输出目录**：`outputs/field/field_v2_pointnext_wssp06_wss_huber_only_else_like_wssp05_split_AG_v1_seed1_20260428_000339/`、`…_seed2_20260428_025645/`、`…_seed3_20260428_031748/`

`**summary.json` · `test_metrics`（节选；含仅训练 eval、尚无 `predictions_test` 的 seed）**


| seed | `best_epoch` | `r2_p` | `wss_r2_wss` | 备注                                                             |
| ---- | ------------ | ------ | ------------ | -------------------------------------------------------------- |
| 1    | 22           | 0.003  | −0.016       | 已 `**predict_field`**                                          |
| 2    | 16           | 0.015  | −0.015       | 已 `**predict_field**`                                          |
| 3    | 24           | −0.031 | −0.013       | 缺 predict 导出（目录无 `predictions_test/`）；数值来自 `summary` 全图测试 eval |


**与 05 的粗对照（同为 bootstrap，`split_AG_v1`）**

- `**wss_r2_wss`**：**未**呈现 Huber **稳定优于** MSE（三 seed **互有胜负**，幅度 ~0～0.012）。
- `**r2_p`**：**05 seed1** 明显高于 **任一 06 seed**；跨线 **方差大于**「MSE vs Huber」单侧效应。
- `**summary` 中的 `data_loss` / `wss_loss` 数值标度**：06 上 `**wss_loss` ~1 量级**，05 上 **~19.5 量级**，来自 **loss 定义差异**，不可直接横向比绝对值大小。
- **一句话结论**：**Huber（06）在当前设置下未完成「稳定抬高 wall WSS R²」**；不排除需 **补全 06 seed3 预测**、`split_AG_v2` 或多 seed **方差缩减**后再判。**下一步**：优先 `**V2P-WSSP-07`**（弱速度辅助）；若仍不显式改善，按 [计划讨论.md](../计划讨论.md) 再议 **08（两阶段 / 结构解耦）**。

### Line W：A-Opt-W03 权重复扫（`A-Opt-W03-p025-w02` / `w02` / `w05`，`split_AG_v1`，2026-04-27）

> **定位**：在 `**A-Opt-W03-multi` 母版**（`A-Opt-05` + 场 + `wss_loss_weight` 扫参）上，检验 **压低压力项权重 / 提高 WSS loss 权重** 能否在保压力的前提下改善 **壁面 WSS**。母版与配置见 `training/configs/field/generated/wss_multitask/A-Opt-W03-*_seed1.json`；推进记录见 [代码修改与实验推进记录](../../../02-推进与变更/代码修改与实验推进记录.md) **2026-04-27**（四实验 batch）条目。

- **完成日期**：2026-04-27（seed=1）；**集群作业**：Slurm **3143** / **3144** / **3145**（与 `V2P-WSSP-04` **3142** 同批提交）。
- **输出目录**：
  - `A-Opt-W03-p025-w02` → `outputs/field/field_transformer_opt05_wss_multitask_p025_w02_split_AG_v1_seed1_20260427_103849/`（`target_weights` 末维 **0.25**，`wss_loss_weight` **0.2**）
  - `A-Opt-W03-w02` → `outputs/field/field_transformer_opt05_wss_multitask_w02_split_AG_v1_seed1_20260427_103849/`（`wss_loss_weight` **0.2**）
  - `A-Opt-W03-w05` → `outputs/field/field_transformer_opt05_wss_multitask_w05_split_AG_v1_seed1_20260427_103849/`（`wss_loss_weight` **0.5**）

`**regional_eval` 分区域节选（`fig_A5_regional_metrics.json` + `fig_A5_regional_wss_metrics.json`，测试集聚合）**


| exp_id             | interior `r2_p` | wall 场 `r2_p` | wall WSS `r2_wss` | wall WSS `rmse_wss`（约） |
| ------------------ | --------------- | ------------- | ----------------- | ---------------------- |
| A-Opt-W03-p025-w02 | ~0.628          | ~0.682        | ~−0.006           | ~2.29                  |
| A-Opt-W03-w02      | ~0.635          | ~0.690        | **~−0.001**       | **~2.29**              |
| A-Opt-W03-w05      | ~0.630          | ~0.685        | ~−0.014           | ~2.30                  |


**已生成产物**

- 各 run：`predictions_test/manifest.json`、`regional_eval/`（含 WSS JSON 与柱图）、`error_analysis_interior/`（含 `wss/`）、`fig_A3_scatter_interior.png`、`fig_A4_per_case_boxplot_interior.png`。
- **四实验 + `V2P-WSSP-04` 多模型**：`outputs/field/plots/wssp04_w03_line_seed1_20260427/`（`plot_taskA_multimodel_scatter` / `plot_taskA_multimodel_regional_bar` / `plot_taskA_multimodel_per_case_boxplot`，`--exp-filter` 四 `exp_id`，`--seed 1`）。
- **一句话结论**：三者在 **压力与速度** 上分区域**差距很小**；`**w02` 在 `wall` WSS `r2_wss` 上略优**；`**p025-w02` 未明显释放 WSS**；`**w05` 未优于 `w02`，且 wall WSS 转更差**，与配置中「不再向更大权重扩展」一致。以分区域为准，**均未达到** 配置备注中 **「`r2_p`≥0.9 左右」** 的保底叙述（约 **0.63～0.64** / **0.68～0.69**）。
- **下一步动作**：Line W 若继续，建议以 `**w02` 为短名单锚点**；**不再单扫更大 `wss_loss_weight`**。**（2026-04-28）** `**V2P-WSSP-05` / `V2P-WSSP-06` 三 seed 已训练并完成主后处理**（见上两节摘要）；优先接 `**V2P-WSSP-07`** 或 **纯 WSS 草案**（见 [计划讨论.md](../计划讨论.md)）。