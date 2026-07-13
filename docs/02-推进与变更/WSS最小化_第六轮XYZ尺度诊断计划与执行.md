# WSS 最小化·第六轮总路线与执行入口

> 状态：`WSS MAINLINE ACTIVE / HORIZONTAL MINIMAL DIAGNOSTICS`
>
> 更新：2026-07-13
>
> 总原则：**WSS 精度是主线，压力/速度横向对比只保留能改变机制判断的最小矩阵**。不再以补齐 36 runs 作为 WSS 训练的前置条件。
>
> 边界：开发阶段仅读 `val`，不读 legacy `test16`；当前 61 例结论不外推数百/数千例学习上限。

## 1. 本文档的职责

本文只作第六轮的总导航、优先级和状态入口，不再堆叠每条路线的实验细节。

| 文档 | 职责 | 当前优先级 |
|---|---|---|
| [WSS 精度突破计划与执行](WSS最小化_第六轮_WSS精度突破计划与执行.md) | C/E 尺度、L1/L2 loss、M1/M2 架构与确认性验证 | **P0，当前主线** |
| [横向多目标对比计划与执行](WSS最小化_第六轮_横向多目标对比计划与执行.md) | 壁面/内部压力、近壁速度的最小机制诊断 | **P1，按 Gate 触发** |
| [边界条件与速度条件路线](WSS最小化_第六轮_边界条件与速度路线.md) | BC 可辨识性、oracle BC、入口裁剪分层、速度→WSS oracle/V2 | **P2，条件路线保留** |
| [跨路线评估与横向对比口径](../00-规范与记录/WSS跨路线评估与横向对比口径.md) | V3P/wss_min 可比性、WSS 指标层级、Bridge 协议 | 共享硬口径 |
| [训练实验跟踪](WSS最小化_训练实验跟踪.md) | 已完成 run 的指标、Job 与裁决证据 | 事实真源 |
| [代码修改与实验推进记录](WSS最小化_代码修改与实验推进记录.md) | 代码/配置/文档变更的倒序日志 | 工程真源 |

## 2. 第六轮总调度

### Phase W｜WSS 突破（当前主线）

1. 完成 `coord_scale`/裁剪/邻域覆盖只读审计。
2. 独立跑 E 嵌套因子和 C 全局共享物理尺度坐标。
3. 在冻结 D 输入上做 L1/L2 小 Gate，避免与 C/E 同时改变。
4. 只组合一次“最佳输入 × 最佳 loss”，然后再考虑 M1/M2。
5. 候选须经 duplicate-grouped repeated validation，才能进入最终 OOF/test 决策。

### Phase H｜横向最小诊断（与 W 互不阻塞）

1. 壁面 gauge pressure `xyz` vs `xyz+geom` 已完成，作为任务难度诊断收口。
2. 落地同一份 `data_new` adapter，主要用于近壁速度和速度→WSS V0/V1。
3. 先跑 `|v|+geom` 与联合 `u,v,w+geom` 单 seed sanity。独立三分量和内部压力按科学需要触发。
4. 只有单 seed 改变 WSS 机制判断或确需进入论文主表，才扩展为三 seed。

### Phase C｜条件路线（保留）

- BC 路线保留 I0/I1/I2，但 CFD RCR/压力强制标记 `oracle_non_deployable=true`。
- 速度横向回归与速度→WSS 是两个问题：前者用于补齐基线，后者必须先过 V0/V1 oracle Gate。
- 入口裁剪只做预处理敏感性分层，不允许根据 val 结果反复调裁剪阈值。

## 3. 已完成的 XYZ 尺度诊断（P0）

冻结协议：`split_AG_wss_min_v2_dev1.json`（train 53 / val 8）、peak step 1162 / index 21、FPS-2000、PointNeXt-S、B1 fixed target-weight、160 epoch、seed `1234/7/2025`、val-only。

| 组 | 输入 | `R²_field_raw` | `R²_casemean` | 结论 |
|---|---|---:|---:|---|
| A | `xyz` | `0.164±0.070` | `-0.085±0.055` | 无尺度对照 |
| B | `xyz+coord_scale` | `0.208±0.038` | `0.008±0.099` | 尺度信号可辨识 |
| D | `xyz+geom` | `0.311±0.014` | `0.197±0.020` | 显式几何仍主导 |

- `B−A` field `+0.044`、casemean `+0.093`，均过 `+0.02` 门槛且三 seed 方向一致。
- `D−B` 只作非嵌套探索对照，不称为严格几何增量。
- 严格归因还需 E：`E−B`、`E−D`和交互项 `E−D−B+A`。
- 绝对精度仍不高；`0.70` 仅作长期理想参考，当前开发裁决以相对 control 改善和护栏为准。

Jobs `7029–7037` 首批曾因 config 目录迁移发生 5/9 路径失效；Jobs `7039–7043` 补齐，A/B/D 9/9 均已可评估。完整作业表、指标和失败复盘以[训练实验跟踪](WSS最小化_训练实验跟踪.md)为准。

## 4. 全路线公共硬约束

1. split/stats 只由 train 拟合；原病例与其衍生样本必须同 fold。
2. 压力、速度和 WSS 使用各自的归一化、指标和选模规则，不复用 WSS 热点惩罚做通用分数，也不共用 `R²=0.70` 单门槛。
3. 每个比较必须冻结唯一变量、单位、坐标帧、采样域、点数和统计来源。
4. 三 seed 只描述训练随机性；候选稳健性另用按病例配对的差值、bootstrap CI 和 grouped repeated split 评估。
5. 空余 GPU 可提交已预注册且互不依赖的 run；不因资源空闲跳过 adapter/指标/坐标帧 QA，也不为补齐横向表而自动扩展。
6. 未达预注册开发门槛时，不触发 OOF/test16，也不把 oracle 特征包装成可部署输入。

## 5. 当前状态

- 壁面 gauge pressure Track A 的 6 个作业 `7551–7556` 已完成，结果见横向对比文档。
- Track B adapter 未验收；完成后只启动最小速度 sanity 与 V0/V1 所需产物，不直接启动 30 个剩余 run。
- WSS C/E、L1/L2 是当前主线；M1/M2 等待 W1–W3 证据。
- BC 和速度→WSS 保留为条件路线，不从计划中删除。
> 状态：`ACTIVE / 当前停在 P0-Metric`
> 更新：2026-07-13（最终审查收敛版）
> 唯一主线：`P0-Metric → P0-Density → 2×2 F3 → 分支优化 → repeated validation`

## 1. 本文档职责

本文只回答“现在看哪份文档、当前做到哪里、下一步是什么”，不重复实验表和历史论证。

| 文档 | 唯一职责 | 当前状态 |
|---|---|---|
> 状态：`ACTIVE / 当前停在 P0-Metric`
> 更新：2026-07-13（最终审查收敛版）
> 唯一主线：`P0-Metric → P0-Density → 2×2 F3 → 分支优化 → repeated validation`

## 1. 本文档职责

本文只回答“现在看哪份文档、当前做到哪里、下一步是什么”，不重复实验表和历史论证。

| 文档 | 唯一职责 | 当前状态 |
|---|---|---|
| [WSS 精度突破计划](WSS最小化_第六轮_WSS精度突破计划与执行.md) | 最新指标合同、Gate、主实验顺序和待办 | **P0 主执行文档** |
| [横向多目标计划](WSS最小化_第六轮_横向多目标对比计划与执行.md) | 已完成压力诊断、V3P 可比性和暂停项 | P1 条件诊断 |
| [BC/速度条件路线](WSS最小化_第六轮_边界条件与速度路线.md) | oracle-BC、临床可获得性、速度→WSS V0/V1/V2 Gate | P2 条件路线 |
| [训练实验跟踪](WSS最小化_训练实验跟踪.md) | Job、逐 seed 指标、历史实验判读 | 事实真源 |
| [WSS 专用推进记录](WSS最小化_代码修改与实验推进记录.md) | 代码/配置/文档变更历史 | 工程真源 |

## 2. 当前有效状态

- 固定任务：`peak step 1162 / index 21`，可部署输入下的 WSS 单目标预测。
- W0–W3 已完成；其详细过程不再放入当前计划。
- E 和 D+raw-Huber 均冻结为临时候选；E×raw-Huber 已 No-Go。
- 旧版 train-only F3、单一 `train R²≥0.85` Gate 和“RCR 已确认是根因”的结论全部停止使用。
- 当前没有经过新版指标合同和病例划分确认的最终可部署候选。
- test16 保持未读；`R²=0.70` 仅是长期理想目标。

## 3. 当前执行顺序

1. **P0-Metric**：统一指标、训练 history、早停、checkpoint、Gate 和单元测试；只读重评旧 top-k/last。
2. **P0-Density**：单变量验证 `nsample`/训练评估密度错配，生成新的冻结 control。
3. **2×2 F3**：架构×RCR，加入 shuffled-RCR、RCR-only/随机病例特征与病例外验证。
4. **分支优化**：只按 F3 主效应选择 density/global-context、可获得 BC、物理先验或数据/标签路线。
5. **确认性验证**：唯一候选进入 duplicate-grouped repeated validation；过门后才申请 OOF/test16。

详细公式、矩阵和 Gate 只看[WSS 精度突破计划](WSS最小化_第六轮_WSS精度突破计划与执行.md)，本文不再复制。

## 4. 公共硬约束

1. split/stats 只由 train 拟合；原病例与衍生样本必须同 fold。
2. `R²_casemean` 只表示逐病例 spatial R² 平均，不得解释为病例 mean-WSS 的跨病例 R²。
3. level、pattern、hotspot 和病例等权 Pa 误差必须分列。
4. 文档指标、Gate 代码、训练早停和 checkpoint 选择必须使用同一冻结合同。
5. 每例热点先逐病例计算再等权聚合；全病例拼接 top10 只作历史衔接。
6. 三 seed 只描述训练随机性；病例稳健性必须用 grouped repeated split。
7. oracle 特征必须标记 `oracle_non_deployable=true`，不得进入部署候选。
8. 未通过预注册 Gate，不触发 OOF/test16。

## 5. 暂停项

- 横向 36-run 压力/速度矩阵不补齐。
- Track B adapter 不因资源空闲自动启动。
- V3P 多时相 P+WSS 联训不恢复。
- V2 速度 surrogate 只有 V0/V1 过门后才立项。

## 6. 当前完成条件

- [ ] P0-Metric 完成并通过测试。
- [ ] P0-Density 产生冻结 control。
- [ ] 2×2 F3 完成病例外判读。
- [ ] 选出且只选出一个后续分支。
- [ ] 唯一候选通过 repeated validation。
