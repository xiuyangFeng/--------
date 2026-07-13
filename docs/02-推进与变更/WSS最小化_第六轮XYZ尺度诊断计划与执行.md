# WSS 最小化·第六轮总路线与执行入口

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
