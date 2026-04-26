# 任务A本周实验启动清单

> **（2026-04-25 导师沟通后修正）**：后续 WSSP 线若速度场相关系数难以上升，则主目标调整为 **压力场 + 壁面 WSS 快速预测**。主指标固定为 **`r2_p / rmse_p`** 与 **`wall.r2_wss / wall.rmse_wss`**；速度仅作近壁上下文的弱辅助监督和诊断项。新增配置：**`training/configs/field/generated/v2_pointcloud/V2P-WSSP-04_seed1.json`**，manifest：**`training/configs/field/generated/v2_pointcloud/manifest_wssp_pressure_wss_primary.json`**。

> **（2026-04-25）进度注记**：**`V2P-WSSP-02`**（PointNeXt，**全场 u/v/w/p + WSS 辅助头 + 混合早停**，**标准采样**）seed=1 **已完成**（bootstrap `split_AG_v1`）：**`predictions_test`、Fig A3/A4、`error_analysis_interior`（含 `wss/`）、`regional_eval`（场+WSS）** 已闭环；**流场可读 `interior`，WSS 主读 `wall`**。数值与 **WSSP-01 对照结论**见 [任务A实验状态表](任务A实验状态表.md)「V2P-WSSP-02」、[任务A配置与启动说明](任务A配置与启动说明.md) §10 示例 B。

> **（2026-04-24）进度注记**：**`V2P-WSSP-01`**（PointNeXt，**壁面 13000 + 近壁 2000** 采样，**仅监督 p + WSS**）seed=1 **已完成**（bootstrap `split_AG_v1`）：`predictions_test`、`error_analysis_interior/wss`、`regional_eval`（含 **`fig_A5_regional_wss_metrics.json`**）已闭环；**勿用 u/v/w 散点作主结论**。详见 [任务A实验状态表](任务A实验状态表.md)「V2P-WSSP-01」、[任务A配置与启动说明](任务A配置与启动说明.md) §10。

> **（2026-04-22）进度注记**：**V2P Bootstrap** — `V2P-Base-01`（PointNeXt，无 geometry）与 `V2P-Main-01`（PointNeXt，有 geometry）seed=1 **已在 `split_AG_v1` 上完成训练（bootstrap 口径）**。关键结果：geometry 增益显著（`rmse_vel_mag` -22%，`r2_vel_mag` +72%）；`V2P-Main-01` seed=1 全局 `r2_vel_mag=0.609`，已在该口径上超过 V1 锚点 `A-Opt-05`（~0.576）。当前状态：仅 seed=1、无分区域评估、使用 bootstrap split，不作为正式 V2 结论；等待 Gate-0 通过后在 `split_AG_v2` 上复现正式版。详见 [任务A实验状态表](任务A实验状态表.md)「V2P Bootstrap」。

> **（2026-04-17）进度注记**：**WSS 多任务**（`A-*-wss-multi`，配置 **`baseline_wss_multitask/`**）训练 **三 seed** 已入账 **`outputs/field/experiment_index.csv`**；壁面 WSS RMSE/R² 见 **`outputs/field/wss_multitask_test_wall_wss_metrics.tsv`**；**`predictions_test` + Fig A3–A5 全链**已闭环（推进记录文首 **Job 2704 / 2715**）。**多任务 R² 主表**（壁面 WSS / **R² \|v\|** / **R²_p**）见 [任务A实验状态表](任务A实验状态表.md)「实验记录摘要 · WSS 多任务」。**与 `line_g`（Line G）无冲突**：**`G02`/`G03`** 为 **⏹ 已结案**（仅 seed1，不补种）。
> **（2026-04-14）进度注记**：**Line G** 之 **`A-Opt-G01` / `G04` / `G05`** 已 **三 seed** 归档（`predictions_test` + Fig A3–A5 多模型对比：`outputs/field/plots/line_g/G01_G04_G05_vs_baselines_mean3seed/`）；**`G02` / `G03`** 仅 **seed1**，**（2026-04-17）** 已决定 **不补 seed2/3**，记 **⏹ 已结案**。详见 [任务A实验状态表](任务A实验状态表.md)「实验记录摘要 · Line G」。
> **（2026-04-09）进度注记**：**`A-Abl-02`（几何分量消融）** 已在 **`A-Opt-05` 母版**上 **三 seed 闭环**，汇总图 **`outputs/field/plots/ablation/geometry_opt05_mean3seed/`** ——见 [任务A实验状态表](任务A实验状态表.md)「实验记录摘要 · A-Abl-02」。下列「本周 baseline」清单仍为 **2026-03 首周模板**；当前 V1 主线母版为 **`A-Opt-05`**，**`A-Opt-07`（内部监督加权）** 已闭环且结论为负。

> 配套文档：[任务A实验清单](任务A实验清单.md) / [任务A周度检查表](任务A周度检查表.md) / [实验记录填写规范](../../00-规范与记录/实验记录填写规范.md)

这份清单是给你本周直接开跑用的，不是长期总纲。目标只有一个：在本周内把 `A-Base-01` 到 `A-Main-01` 跑通，并启动 `A-Abl-01`。

---

## 1. 本周唯一主线

配置生成和批量启动方式见：[任务A配置与启动说明](任务A配置与启动说明.md)

本周只回答 3 个问题：

1. 图结构是否比点级模型更有价值。
2. geometry 是否在当前数据和划分下有稳定收益。
3. `BC / is_wall` 是否应该保留在主线输入中。

本周先不要做：

- physics loss
- hierarchy
- 大量 backbone 横向比较
- 无节制调参

---

## 2. 开跑前检查

下面项目不打勾，不建议正式起跑多组实验。

- [ ] 已固定 `data_version`
- [ ] 已固定 `split_version`
- [ ] 已列出训练/验证/测试患者清单
- [ ] 已核对 `data.x` 的 10 维索引
- [ ] 已核对 `data.global_cond` 的 6 维索引
- [ ] 已核对 `data.y` 的 4 维索引
- [ ] 已确认归一化统计量只来自训练集
- [ ] 已确认评估脚本能独立读取 checkpoint
- [ ] 已确定统一输出目录 `outputs/field/...`
- [ ] 已跑通一个 1 epoch 的 smoke test

---

## 3. 本周必须跑的实验

### 3.1 第一优先级：4 个 baseline

| 优先级 | Exp ID | 目的 | 本周最低要求 |
| --- | --- | --- | --- |
| P0 | `A-Base-01` | 建立无图下限 | 完成 smoke test + `seed=1` |
| P0 | `A-Base-02` | 验证图结构是否必要 | 完成 smoke test + `seed=1` |
| P0 | `A-Base-03` | 作为无 geometry 的 Transformer 对照 | 完成 smoke test + `seed=1` |
| P0 | `A-Main-01` | 形成单尺度主线基座 | 完成 smoke test + `seed=1` |

### 3.2 第二优先级：补 seed

当 4 个 baseline 的 `seed=1` 结果可读后，再补：

- [ ] `A-Base-01` 的剩余 seed
- [ ] `A-Base-02` 的剩余 seed
- [ ] `A-Base-03` 的剩余 seed
- [ ] `A-Main-01` 的剩余 seed

### 3.3 第三优先级：启动 `A-Abl-01`

本周至少启动下面 2 到 3 组，不要求全部跑完，但要把配置和队列建好：

- [ ] `A-Abl-01-01` `coords + t`
- [ ] `A-Abl-01-02` `coords + t + BC`
- [ ] `A-Abl-01-03` `coords + t + BC + is_wall`
- [ ] `A-Abl-01-04` `coords + t + BC + geometry`

---

## 4. 每天执行清单

### Day 1

- [ ] 固定数据与 split
- [ ] 核对输入输出维度
- [ ] 跑通 `A-Base-01` smoke test
- [ ] 记录第一份 smoke test 日志

### Day 2

- [ ] 配好 `A-Base-02`
- [ ] 配好 `A-Base-03`
- [ ] 配好 `A-Main-01`
- [ ] 实现统一特征掩膜逻辑

### Day 3

- [ ] 跑完 4 个 baseline 的 `seed=1`
- [ ] 判断是否存在明显 bug 或异常不收敛
- [ ] 清理掉明显不合理配置

### Day 4

- [ ] 统一评估 4 个 baseline
- [ ] 生成第一版主结果表
- [ ] 生成壁面/内部点分层表
- [ ] 导出至少一个病例图

### Day 5

- [ ] 决定先补哪些 baseline 的多 seed
- [ ] 启动 `A-Abl-01`
- [ ] 写出本周第一版一句话结论

### Day 6

- [ ] 补齐最关键 baseline 的剩余 seed
- [ ] 整理 `A-Abl-01` 首轮结果
- [ ] 判断 geometry 是否值得继续深挖

### Day 7

- [ ] 汇总本周完成实验
- [ ] 固定当前单尺度最优主线
- [ ] 只保留下周 1 到 2 个最高优先级动作

---

## 5. 每组实验最小记录

每跑完一组，至少补下面这些信息：

- `Exp ID`：
- `seed`：
- `best epoch`：
- `RMSE_|v|`：
- `RMSE_p`：
- `壁面点误差`：
- `内部点误差`：
- `高曲率区域误差`：
- `推理时间`：
- `峰值显存`：
- `一句话结论`：

---

## 6. 周末验收标准

这周结束前，至少应满足：

- [ ] 4 个 baseline 都有可读结果
- [ ] 至少 2 个 baseline 已补多 seed
- [ ] 已有第一版主结果表
- [ ] 已有至少一张分层指标表
- [ ] `A-Abl-01` 已启动

如果这 5 项还没齐，不建议下周直接进入 hierarchy。

---

## 7. 优化线（P0）进度指针（非本周唯一主线）

Baseline 与 `A-Abl-01` 完成后，**内部精度优化线**见 [任务A实验状态表](任务A实验状态表.md) 第三批与 [任务A优化路径与近期实验建议](任务A优化路径与近期实验建议.md)。**（2026-03-27）** **`A-Opt-01`、`A-Opt-02`、`A-Opt-02_warmup`（P0-3）均已三 seed 归档**（含 `predictions_test`、`regional_eval`、`error_analysis_interior` 及 `plots/optimization/prenorm_Main_P02_P02w`）。**（2026-03-28）** **`A-Opt-03`、`A-Opt-03w`（P0-4）已三 seed 归档**（同上后处理链；`plots/optimization/prenorm_A_Opt03_vs_Opt03w/best_metrics.csv`）。**（2026-03-29）** **`A-Opt-04`（`hidden_dim=256`）、`A-Opt-05`（`num_layers=4`）已三 seed 归档**（`outputs/field/field_transformer_coord_t_bc_geom_wall_prenorm_tw22205_h256_*` / `*_h256_l4_*`；后处理链同前）。**（2026-03-31）** **`A-Opt-05_tune`**：见 **`outputs/field/experiment_index.csv`** 与 [任务A实验状态表](任务A实验状态表.md)「`A-Opt-05_tune`」；**多模型图** `outputs/field/plots/optimization/A_Opt05_tune_vs_Opt03_seed1/`（标题虽含 03，**读图时以 05 为母版视角**）；**WSS 全量对比暂缓**。**（2026-03-31）后续消融 / Line G / Line W 的配置母版：`A-Opt-05`**（`h256×4L` + P0-4 组合）；**`A-Opt-03` 保留轻量、低开销对照**。详见状态表篇首「战略锚点」。**（2026-04-14）Line G**：**`G01`/`G04`/`G05` 三 seed 已归档**；**`G02`/`G03` ⏹ 已结案**（不补种）。**下一优先**：**不堆 `A-Opt-06`**；推进 **`A-Abl-01`** 或 **Line W**；按需 **`A-Opt-05` vs `A-Opt-03` 效率条**（WSS 放集群、须你方明确再动）。
