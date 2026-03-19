# 任务C论文可视化与指标建议

> 上位文档：[实验设计总纲](../../实验设计总纲.md) / [任务C风险建模规范](./任务C风险建模规范.md) / [项目思路](../../paper_idea/项目思路.md)

## 1. 文档定位

本文件用于固定任务 C 在未来论文中需要呈现的风险预测图表与指标，核心目标是证明：

- CFD 血流动力学特征是否真的提升风险预测；
- AI 血流动力学特征是否能接近 CFD 版本的效果；
- 结论是否具有统计稳定性与临床可解释性。

---

## 2. 论文中建议保留的核心可视化

### 2.1 必做图

1. ROC 曲线
   - 至少比较 `Clinical + Geometry`
   - `Clinical + Geometry + CFD-hemodynamics`
   - `Clinical + Geometry + AI-hemodynamics`
2. PR 曲线
   - 样本不平衡时尤其重要
3. 校准曲线
4. DCA
5. SHAP summary plot
6. 典型高风险病例 SHAP force plot
7. 典型低风险病例 SHAP force plot

### 2.2 强烈建议补充的图

1. 不同特征组 AUROC / AUPRC 箱线图
   - 展示不同折、不同 seed 的波动
2. CFD 特征组 vs AI 特征组校准对比图
3. 特征稳定性条形图
   - 哪些特征在不同折里持续进入前列
4. 高低风险患者特征分布对比图
   - 提升医学可解释性

---

## 3. 论文中建议报告的指标

### 3.1 区分度指标

- `AUROC`
- `AUPRC`
- `Accuracy`
- `Sensitivity`
- `Specificity`
- `F1`

### 3.2 校准度指标

- `Brier Score`
- Calibration curve
- 可选：calibration slope
- 可选：calibration intercept

### 3.3 临床价值指标

- `DCA`

### 3.4 稳定性与统计指标

- 外层交叉验证 `mean ± std`
- `AUROC / AUPRC / Brier` 的 95% bootstrap CI
- 必要时补 paired test

---

## 4. 论文里建议固定的特征组对照

任务 C 的主线不建议写成“很多模型混战”，而应写成“特征来源对照”。

建议至少固定以下组：

1. `Clinical-only`
2. `Geometry-only`
3. `Clinical + Geometry`
4. `Clinical + Geometry + CFD-hemodynamics`
5. `Clinical + Geometry + AI-hemodynamics`

扩展组：

- `Clinical + Geometry + AI-hemodynamics + h_global`

建议：

- `h_global` 更适合作为扩展实验，不建议和主结果混在一起；
- 主结果优先证明“AI 血流特征是否能近似替代 CFD 血流特征”。

---

## 5. 写论文时的推荐组织方式

### 5.1 主结果段

- 一张主结果表
- 一张 ROC / PR 组合图
- 一张校准曲线图

### 5.2 临床解释段

- 一张 DCA 图
- 一张 SHAP summary plot
- 两张典型病例 SHAP force plot

### 5.3 稳健性段

- 一张各折 AUROC / AUPRC 箱线图
- 一个 95% CI 表

---

## 6. 完整出图规划

### 6.1 Figure C1：任务 C 主结果表

回答的问题：

- 哪类特征组真正带来风险预测增益。

建议固定特征组：

1. `Clinical-only`
2. `Geometry-only`
3. `Clinical + Geometry`
4. `Clinical + Geometry + CFD-hemodynamics`
5. `Clinical + Geometry + AI-hemodynamics`

建议表头：

- `Feature Set`
- `Model`
- `AUROC`
- `AUPRC`
- `Accuracy`
- `Sensitivity`
- `Specificity`
- `F1`
- `Brier`
- `Notes`

### 6.2 Figure C2：ROC 与 PR 主图

回答的问题：

- AI-hemodynamics 是否接近 CFD-hemodynamics，且优于无血流特征版本。

推荐版式：

- 左：ROC
- 右：PR

固定显示三条主链：

- `Clinical + Geometry`
- `Clinical + Geometry + CFD-hemodynamics`
- `Clinical + Geometry + AI-hemodynamics`

### 6.3 Figure C3：校准与临床决策图

回答的问题：

- 即使区分度接近，AI 链路是否在校准和临床决策价值上仍然可用。

推荐版式：

- 左：Calibration curve
- 右：DCA

建议：

- 主文只放三条主链
- 其他扩展组放补充材料

### 6.4 Figure C4：全局解释图

回答的问题：

- 模型为什么做出这样的风险判断。

推荐版式：

- 左：SHAP summary plot
- 右：top-10 feature importance 条形图

建议：

- 把 CFD 和 AI 的重要特征并排比较，观察是否保留了相近的医学解释方向

### 6.5 Figure C5：个体解释图

回答的问题：

- 高风险和低风险病例各自的关键驱动因素是什么。

推荐版式：

- 上：典型高风险病例 SHAP force / waterfall
- 下：典型低风险病例 SHAP force / waterfall

建议：

- 选择最具代表性的病例，而不是随机病例
- 同时标注真实标签与预测概率

### 6.6 Figure C6：稳健性图

回答的问题：

- 结果是否稳定，而不是依赖单次 split 或单次 seed。

推荐版式：

- 左：各折 `AUROC` 箱线图
- 中：各折 `AUPRC` 箱线图
- 右：`Brier` 箱线图

可选补充：

- 95% bootstrap CI 表
- 特征入选稳定性图

### 6.7 Supplementary C：建议放附录的图

- 更多子组分析
- 不同模型家族对比
- `h_global` 扩展实验
- 更多病例解释图

---

## 7. 当前状态判断

- `docs` 已经把任务 C 的指标口径写得比较完整；
- 当前最重要的不是再发散更多模型，而是把“特征来源对照 + 统计稳定性 + 解释性”做扎实；
- 如果只报 `AUROC` 而不报校准和 DCA，论文说服力会明显不足；
- 如果只证明 AI 接近 CFD 的 `AUROC`，但校准或解释性明显劣化，也不能轻易宣称数字孪生链成立。

---

## 8. 与前后任务的关系

- 任务 C 可以先跑 `Clinical-only / Geometry-only / Clinical + Geometry`。
- 但真正与论文主线相关的两组：
  - `Clinical + Geometry + CFD-hemodynamics`
  - `Clinical + Geometry + AI-hemodynamics`
  都依赖任务 B 输出的标准化特征表。
- 任务 D 是否成立，取决于任务 C 是否已经证明：
  - CFD 血流特征确实有增益；
  - AI 血流特征与 CFD 方向一致且差距有限。
