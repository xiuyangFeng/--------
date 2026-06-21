# 任务D论文可视化与指标建议

> 上位文档：[实验设计总纲](../../实验设计总纲.md) / [任务D端到端验证清单](./任务D端到端验证清单.md)

## 1. 文档定位

本文件用于固定任务 D 在论文中的端到端闭环验证图表与指标。任务 D 不是新增模块，而是把任务 A、B、C 的证据链串成最终结论。

---

## 2. 论文中建议保留的核心可视化

### 2.1 必做图

1. 三链路 ROC 图
2. 三链路 PR 图
3. 校准曲线对比图
4. DCA 对比图
5. 速度收益柱状图

三条固定链路：

1. `Clinical + Geometry`
2. `Clinical + Geometry + CFD-hemodynamics`
3. `Clinical + Geometry + AI-hemodynamics`

### 2.2 强烈建议补充的图

1. 链路级性能差值图
   - 看 AI 相对 CFD 退化了多少
2. 指标替代敏感性分析图
   - 只替换 `TAWSS`
   - 只替换 `OSI`
   - 只替换 `RRT`
   - 全部替换
3. 临床流程时间线示意图
   - 展示 CFD 小时级与 AI 秒级差异

---

## 3. 论文中建议报告的指标

### 3.1 主性能指标

- `AUROC`
- `AUPRC`
- `Brier`
- Calibration
- `DCA`
- Runtime

### 3.2 效率收益指标

- 单病例 CFD 时间
- 单病例 AI 推理时间
- 总加速比

### 3.3 替代性分析指标

- 逐个替换关键指标后的性能下降幅度
- AI 相对 CFD 的 gap
- 结果稳定性

---

## 4. 写论文时的推荐组织方式

### 4.1 闭环主结论

- 一张三链路主表
- 一张 ROC / PR 组合图
- 一张校准 + DCA 组合图

### 4.2 替代机制分析

- 一张“逐项替代”敏感性图
- 一张 AI vs CFD 性能差值图

### 4.3 临床可用性分析

- 一张速度收益图
- 一张流程示意图

---

## 5. 完整出图规划

### 5.1 Figure D1：端到端三链路主结果表

回答的问题：

- 数字孪生闭环是否成立。

建议表头：

- `Chain`
- `Feature Source`
- `AUROC`
- `AUPRC`
- `Brier`
- `Calibration`
- `DCA`
- `Runtime`
- `Notes`

三条链固定为：

1. `Clinical + Geometry`
2. `Clinical + Geometry + CFD-hemodynamics`
3. `Clinical + Geometry + AI-hemodynamics`

### 5.2 Figure D2：ROC 与 PR 对照图

回答的问题：

- AI 链路是否接近 CFD 链路，并优于无血流链路。

推荐版式：

- 左：三链路 ROC
- 右：三链路 PR

### 5.3 Figure D3：校准与 DCA 对照图

回答的问题：

- AI 链路是否不仅“能分开”，而且“能用于临床决策”。

推荐版式：

- 左：Calibration curve
- 右：DCA

### 5.4 Figure D4：替代敏感性分析图

回答的问题：

- AI 替代 CFD 时，哪个指标最关键，哪个最容易造成性能退化。

推荐版式：

- 横轴为替代方案：
  - `replace TAWSS`
  - `replace OSI`
  - `replace RRT`
  - `replace all`
- 纵轴为性能变化：
  - `delta AUROC`
  - 或 `delta AUPRC`

### 5.5 Figure D5：速度收益与流程图

回答的问题：

- AI 替代 CFD 的实际工程价值有多大。

推荐版式：

- 左：runtime bar chart
  - `single-case CFD`
  - `single-case AI`
  - `speedup`
- 右：临床流程示意图
  - 几何输入
  - AI 场重建
  - hemodynamic 指标
  - 风险输出

### 5.6 Figure D6：整篇论文总结图

回答的问题：

- 论文整条证据链如何闭环。

建议做成 graphical abstract 风格：

1. `Geometry + BC`
2. `AI field reconstruction`
3. `Hemodynamic indicators`
4. `Risk prediction`

每个模块下方标一个关键数字：

- A：`RMSE / runtime`
- B：`Pearson / bias`
- C：`AUROC / Brier`
- D：`speedup`

这张图很适合做摘要图或论文最后一张总览图。

---

## 6. 当前状态判断

- 任务 D 的文档要求已经比较明确，重点不在新增更多图，而在于确保三条链路严格同口径比较。
- 论文里最容易写弱的地方不是模型性能，而是“为什么 AI 可以替代 CFD”这件事的证据链是否完整。
- 因此任务 D 的图一定要同时覆盖：
  - 风险预测性能
  - 校准与临床价值
  - 速度收益

---

## 7. 与前序任务的关系

- 任务 D 对任务 A、B、C 都有硬依赖。
- 任务 A 要先证明场重建稳定。
- 任务 B 要先证明关键血流指标恢复可信。
- 任务 C 要先证明血流特征对风险预测确有增益。
- 只有这三层都成立，任务 D 才能作为论文最终闭环结论。
