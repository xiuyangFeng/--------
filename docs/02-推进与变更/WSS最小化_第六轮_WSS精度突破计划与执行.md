# WSS 最小化·第六轮 WSS 精度突破计划与执行

> 状态：`ACTIVE / P0-Metric 待执行；旧版 F3 已停止`
> 更新：2026-07-13（最终对抗性审查后收敛版）
> 当前任务：固定 `peak step 1162 / WSS-min index 21`，只做可部署输入下的 WSS 单目标预测。
> 当前顺序：`P0-Metric → P0-Density → 2×2 F3 → 分支优化 → repeated validation`

## 1. 当前有效结论

1. **WSS 是唯一 P0 训练主线。** 压力和速度均为独立条件诊断，不共享 backbone、loss 或 checkpoint，也不能替代 WSS 验收。
2. **E 不是已确认最佳模型。** E=`xyz+coord_scale+geom` 在 dev1-val8 的旧口径下 `field_cb` 相对 D 提升 `+0.021`，仅列为临时候选；新版指标重评和病例划分确认前不作最终排名。
3. **继续扫 loss 的优先级低。** D+raw-Huber 能改善 field/top10/high-WSS MAE，但 seed 脆弱、病例稳健性无稳定改善；E×raw-Huber 组合阴性，正式停止该组合线。
4. **密度/邻域错配必须先修。** 完整点评估下 `nsample=16` 截断广泛，旧 baseline 混入 train/eval 密度差异，不适合直接解释信息天花板。
5. **RCR 是待验证的信息源，不是已确认根因。** RCR 有跨病例变异且不可直接部署，但现有证据尚未证明它能在病例外解释 WSS level/pattern 残差。
6. **`1/r³` 是候选先验，不是 AAA 局部真值公式。** 必须先做 held-out 零训练探针，再决定是否实现 residual head。
7. **`R²=0.70` 仅是长期理想目标。** 当前开发裁决依赖相对冻结 control 的病例等权改善、病例外稳健性和热点护栏。

## 2. 已完成证据摘要

冻结开发协议：`split_AG_wss_min_v2_dev1.json`（train 53 / val 8）、固定 peak、FPS-2000、PointNeXt-S、三 seed、val-only。

| 组/路线 | 已有结果 | 当前解释 |
|---|---|---|
| A `xyz` | raw field/旧 per-case mean R² `0.164±0.070 / -0.085±0.055` | 无尺度/无显式几何对照 |
| B `xyz+coord_scale` | `0.208±0.038 / 0.008±0.099` | 尺度信号存在，但不等于 WSS level 可预测 |
| D `xyz+geom` | `0.311±0.014 / 0.197±0.020` | 当前历史 control；显式几何贡献最大 |
| E `xyz+scale+geom` | field `0.331±0.014`，field_cb `0.330` | dev1 临时候选；尺度与几何无协同证据 |
| D+raw-Huber | field 约 `+0.036`，high-WSS MAE 约 `-7%` | 幅值有定向收益，但 seed 脆弱、负例未改善 |
| E×raw-Huber | 未同时优于两个单项 control | No-Go，停止组合 |

完整 Job、逐 seed 数值和历史实验过程只维护在[训练实验跟踪](WSS最小化_训练实验跟踪.md)，不再在本文重复。

## 3. 冻结指标合同

### 3.1 四列主结果

| 维度 | 必报指标 | 回答的问题 |
|---|---|---|
| **A 病例整体水平** | 跨病例 case mean/p95/p99 WSS 的 R²、MAE、Spearman | 病例级 scale 是否可预测 |
| **B 病例内空间型态** | 每例分别中心化后的 spatial R²、逐病例中位数/P10/负例数 | 相对场型和最差病例是否改善 |
| **C 高 WSS 幅值与定位** | 每例 top10 IoU/Dice、high-WSS MAE、top10/p95/p99 幅值比 | 热点位置和幅值是否恢复 |
| **总体工程误差** | 病例等权 MAE/RMSE（Pa）、`R²_field_casebalanced` | 总体病例等权表现 |

### 3.2 历史指标边界

- `R²_field_raw`：仅作历史衔接，容易被点数和高方差病例主导。
- 现有 `R²_casemean`：实际是逐病例 spatial R² 的算术平均，不是病例 mean-WSS 的跨病例 R²；后续代码/报告改名为 `R²_percase_mean` 或提供同等明确注释。
- 全病例拼接 top10：仅作历史衔接；正式热点指标必须逐病例计算再等权聚合。

### 3.3 checkpoint 选择

采用预注册的分层/Pareto 规则，不再由旧 `r4_composite_v1` 永久决定：

1. 先按病例等权 Pa 误差和 B 稳健性保留 Pareto top-k。
2. 负 R² 病例数不得增加。
3. C 指标作为硬护栏和 tie-break。
4. test 不参与选模；不得在看完 val/test 后修改规则。

## 4. 当前执行计划

### P0-Metric｜指标与选模合同闭环

需要完成：

1. 在 `metrics.py/evaluate.py` 实现 A/B/C 与总体工程误差四列指标。
2. `train.py` history、早停和 top-k 保存同步使用新合同。
3. `gate1_compare.py` 与文档使用相同字段和公式。
4. 增加单元测试，覆盖病例等权、level/pattern 分离、每例热点聚合和 checkpoint 规则。
5. 对已有 top-k/last checkpoint 做 val-only 只读重评；若新指标最优 epoch 未保存，旧结果继续标记探索性。

**Gate**：代码、测试、训练 history、checkpoint 规则和报告字段完全一致后，才启动新训练。

### P0-Density｜密度/邻域协议修复

只做单变量比较：

- `nsample=16/32/64`；或
- 训练/评估密度匹配方案。

固定输入、loss、seed 和训练预算；同时报告各层物理邻域半径、截断率及 A/B/C 指标。

**Gate**：确认密度修复相对旧 control 的病例外收益后，生成新的冻结 control；否则保留旧结构并记录密度假设 No-Go。

### P1-F3｜2×2 oracle 诊断

| 架构 \ 输入 | geometry | geometry + true RCR/flow split | geometry + shuffled RCR |
|---|---|---|---|
| 密度修复后的 PointNeXt control | F3-A | F3-B | F3-Bneg |
| 无下采样/高容量对照 | F3-C | F3-D | F3-Dneg |

附加要求：

- 增加 `RCR-only` 或等维随机病例特征对照。
- 固定 RCR 与出口/分支对应关系，并完成 permutation QA。
- 同时报 train-fit 和 duplicate-grouped 病例外 validation。
- `oracle 逐病例 WSS scale` 只能作为标签派生校准上限单列，不进入主 Gate。

**Gate**：真实 RCR 的病例外配对增量必须方向稳定且明显超过 shuffled-RCR，才能认定存在可泛化信息效应；高容量/密度修复架构的病例外增量稳定，才能认定架构效应。两个效应允许同时成立，不再使用单一 `train R²≥0.85` 二分。

### P1/P2｜按证据分支

| F3 结果 | 后续动作 |
|---|---|
| 架构效应大、RCR 效应弱 | density/global-context 单变量优化 |
| RCR 病例外效应大、架构效应弱 | 调查临床可获得 BC/分流信息；oracle 不进入部署模型 |
| 两者都大 | 分别确认后再组合，不直接堆叠 |
| 两者都弱 | 优先检查标签、网格、目标定义和输出分辨率 |

物理先验仅保留两个条件动作：

1. 比较 `1/r³`、oracle branch-`Q/r³`、train-fitted `Q̂/r³` 的 held-out 指标；C 过门后才实现 residual head。
2. 只有 A 指标证明病例级 scale 可预测，才启动 `scale × pattern` 目标分解。

## 5. 确认性验证与止损

1. 单 seed 只用于廉价筛选；三 seed 只描述训练随机性。
2. 候选升级为新基线前，必须在 dev2/dev3 或 duplicate-grouped repeated split 上方向一致。
3. 报告病例配对差值、区间、失败病例与 A/B/C 全部护栏。
4. 未稳定优于冻结 control 时停止，不读 OOF/test16。
5. 只有通过开发与 repeated-validation Gate，才申请最终 OOF/test16 和后续临床/生产验证。

## 6. 冻结与暂停项

- 冻结：E、D+raw-Huber、L2、旧 M1–M4，等待 P0/F3 后再决定是否重启。
- No-Go：E×raw-Huber 组合。
- 暂停：V3P 多时相 P+WSS 联训、Track B 36-run 横向矩阵、速度 surrogate V2。
- 禁止：把 CFD RCR/压力/真值速度包装成可部署输入；使用 test 重新选模；仅凭 train-fit 宣称部署上限。

## 7. 当前待办

- [ ] 完成 P0-Metric 代码/测试/只读重评。
- [ ] 完成 P0-Density 单变量实验并冻结 control。
- [ ] 按新 control 执行 2×2 F3 与负对照。
- [ ] 按主效应选择一个后续分支。
- [ ] 对唯一候选执行 duplicate-grouped repeated validation。

路线导航见[第六轮总入口](WSS最小化_第六轮XYZ尺度诊断计划与执行.md)；压力和跨目标诊断见[横向多目标计划](WSS最小化_第六轮_横向多目标对比计划与执行.md)；BC/速度条件 Gate 见[边界条件与速度路线](WSS最小化_第六轮_边界条件与速度路线.md)。
