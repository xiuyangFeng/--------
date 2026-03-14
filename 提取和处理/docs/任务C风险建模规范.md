# 任务C风险建模规范

> 上位文档：[实验设计总纲](实验设计总纲.md) | 相关文档：[任务A](任务A实验清单.md) / [任务B](任务B指标计算规范.md) / [任务D](任务D端到端验证清单.md)

## 1. 文档定位

本规范用于统一“临床风险预测”阶段的建模逻辑、输入特征、数据划分、评价指标和解释分析方式，避免后续因为样本划分、特征来源和模型口径变化导致结果不可比。

当前优先适用任务：

- 髂支闭塞风险预测

后续可扩展：

- AAA 破裂风险预测
- AAA 生长风险预测

---

## 2. 任务目标

任务 C 的核心不是单纯提高一个分类分数，而是回答以下 3 个问题：

1. 临床和几何特征能否建立稳定风险预测基线。
2. 真实 CFD 衍生血流动力学特征能否显著提升风险预测。
3. AI 衍生血流动力学特征能否接近 CFD 衍生特征的临床价值。

因此，任务 C 必须始终围绕“特征来源对比”来组织，而不是一味堆模型。

---

## 3. 任务定义

## 3.1 输入

风险模型输入是患者级或患者-阶段级特征表，不再是节点级图数据。

输入特征应来自以下 4 类来源。

### A 类：临床特征

示例：

- 年龄
- 性别
- 吸烟史
- 高血压
- 糖尿病
- 高脂血症
- 心血管病史
- 冠状动脉病史
- 抗血栓方案
- 是否使用球囊扩张
- 支架品牌
- 支架尺寸

### B 类：几何/解剖特征

示例：

- `Torsion-CIA-pre`
- `Radius-EIA-pre`
- `Torsion-CIA-DRE-DRI`
- `Proximal_neck_angulation`
- `Aneurysm_max_diameter`
- 左右几何差异指标

### C 类：CFD 衍生血流动力学特征

示例：

- `TAWSS-pre`
- `OSI-pre-DRI`
- `RRT-IA-post`
- `RRT-DRE`
- `OSI-post-DRI`

### D 类：AI 衍生血流动力学特征

定义与 C 类一致，但来源于任务 B 的 AI 指标表。

### E 类：深度隐式特征

示例：

- `h_global`
- 其他图池化全局嵌入

这一类属于扩展特征，不作为第一阶段必须项。

## 3.2 输出

优先输出风险概率，而不是硬标签。

### 推荐输出

- 风险概率 `p(y=1)`

### 保留输出

- 二分类标签

其中硬标签只作为派生结果，由概率配合阈值产生。

---

## 4. 数据组织与主键规范

任务 C 的所有表必须至少包含以下主键：

- `patient_id`
- `phase`
- `prediction_target`
- `split_version`
- `source`
  - `clinical`
  - `geometry`
  - `CFD`
  - `AI`

建议以“每位患者或每位患者-阶段一行”的表格组织。

---

## 5. 标签定义规范

## 5.1 髂支闭塞风险预测

### 标签

- `1`：未来发生闭塞
- `0`：未发生闭塞

### 注意事项

- 必须明确标签对应的时间窗口
- 必须明确输入特征是术前、术后早期还是两者结合
- 必须避免把未来信息泄漏到输入特征里

## 5.2 AAA 破裂风险预测

### 标签

- `1`：发生破裂
- `0`：未破裂

### 注意事项

- 需明确是否按直径分层
- 需明确是否只使用术前信息

## 5.3 AAA 生长风险预测

### 标签

- 二分类：快速生长 / 缓慢生长
- 或连续值：年增长率

当前若样本较少，优先做二分类。

---

## 6. 特征组设计

任务 C 的核心不是比较很多模型，而是比较不同特征组。

## 6.1 必做特征组

### C-Feat-01：Clinical-only

目的：

建立纯临床表格基线。

### C-Feat-02：Geometry-only

目的：

验证纯几何信息的风险解释能力。

### C-Feat-03：Clinical + Geometry

目的：

建立传统多模态基线。

### C-Feat-04：Clinical + Geometry + CFD-hemodynamics

目的：

验证真实高保真血流特征是否带来增益。

### C-Feat-05：Clinical + Geometry + AI-hemodynamics

目的：

验证数字孪生链条是否成立。

## 6.2 可选特征组

### C-Feat-06：Clinical + Geometry + AI-hemodynamics + h_global

目的：

验证深度隐式流场表征是否额外有益。

### C-Feat-07：Knowledge-only

仅保留文献已知关键风险因子。

目的：

验证“复现文献已知因素”与“加入 AI 特征”的差别。

---

## 7. 风险模型选择规范

## 7.1 首选模型

优先使用表格学习模型。

### 推荐首选

- `Logistic Regression`
- `XGBoost`
- `LightGBM`

## 7.2 备选模型

- `Random Forest`
- `MLP`

## 7.3 建议比较策略

第一阶段不建议模型太多。建议固定：

1. `Logistic Regression`
2. `XGBoost`

理由：

- 一个线性且可解释
- 一个非线性且适合表格

这样足够覆盖主要情况。

---

## 8. 数据划分与交叉验证规范

## 8.1 总原则

患者级划分必须与上游任务保持一致或至少能回溯到统一的 `split_version`。

## 8.2 推荐方案

### 样本较少时

- 重复分层 `5-fold CV`

### 样本极少且类别不平衡时

- 重复分层 `5-fold CV`
- 同时报告多次重复均值和标准差

### 如需独立测试集

- 训练/验证使用交叉验证
- 独立测试集仅用于最终报告

## 8.3 嵌套交叉验证（Nested CV）

当样本量有限（< 100）且需要同时进行超参数调优和公平评估时，
必须使用嵌套 CV 以避免评估偏差：

- **外层**：5-fold 患者级分层 CV，用于生成无偏的泛化性能估计
- **内层**：3-fold CV 或留一法，在外层训练集上执行超参数搜索
- 外层每折评估指标汇总后，报告 mean ± std 及 95% bootstrap CI

不允许在单层 CV 上同时调参和评估，否则结果有乐观偏差。

## 8.4 Bootstrap 置信区间

所有主结果（AUROC、AUPRC、Brier Score）必须附带 95% CI：

- 推荐方法：bootstrap resampling (n=2000, stratified)
- 从外层 CV 各折预测概率汇总后进行 bootstrap
- 报告格式：`metric (95% CI: lower–upper)`

## 8.5 特征选择策略

为了避免高维小样本过拟合，推荐以下流程：

1. **单变量筛选**：卡方检验（类别）或 Mann-Whitney U（连续）预筛选 p < 0.1 的特征
2. **多变量选择**：在嵌套 CV 内层使用递归特征消除（RFE）或 L1 正则化
3. **最终入选**：取多折交集或出现频率 ≥ 80% 的特征
4. **稳定性报告**：论文中报告各折选中的特征列表，以证明选择稳定性

禁止在全量数据上做特征选择后再拆分训练/测试集。

## 8.6 类别不平衡处理

仅允许以下方式：

- 类别权重
- 训练集内部重采样

禁止：

- 在验证集或测试集上做重采样

---

## 9. 预处理规范

## 9.1 数值特征

建议：

- `StandardScaler`

注意：

- 缩放统计量必须只由训练集计算

## 9.2 类别特征

建议：

- One-hot encoding

## 9.3 缺失值

必须固定一种策略，不得实验间随意变化。

可选：

- 中位数填补
- 众数填补
- 单独缺失标志位

推荐：

- 数值特征：训练集内中位数
- 类别特征：众数或单独 `missing`

---

## 10. 必做实验清单

## 10.1 C-Core-01：Clinical-only

### 目的

建立最低临床基线。

### 模型

- Logistic Regression
- XGBoost

## 10.2 C-Core-02：Geometry-only

### 目的

验证纯几何风险信息。

## 10.3 C-Core-03：Clinical + Geometry

### 目的

建立不使用血流动力学特征的强基线。

## 10.4 C-Core-04：Clinical + Geometry + CFD-hemodynamics

### 目的

验证真实血流动力学信息的临床增益。

这是端到端链路中的“参考上限”。

## 10.5 C-Core-05：Clinical + Geometry + AI-hemodynamics

### 目的

验证 AI 数字孪生链条。

这是任务 C 最关键的实验。

## 10.6 C-Core-06：Knowledge-only vs Knowledge + AI

### 目的

验证 AI 特征是否在文献关键风险因子之外仍有补充价值。

---

## 11. 评价指标

任务 C 必须同时报告区分度、校准度和临床决策价值。

## 11.1 区分度

- `AUROC`
- `AUPRC`
- `Accuracy`
- `Sensitivity`
- `Specificity`
- `F1`

## 11.2 校准度

- `Brier Score`
- Calibration curve
- 可选：calibration slope/intercept

## 11.3 临床决策价值

- Decision Curve Analysis `DCA`

## 11.4 推荐主指标

如果类别不平衡明显，建议主指标优先：

1. `AUROC`
2. `AUPRC`
3. `Brier Score`

---

## 12. 结果表规范

## 12.1 主结果表

| Feature Set | Model | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | F1 | Brier | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## 12.2 消融结果表

| Exp ID | Feature Group | Added Feature Type | AUROC | AUPRC | Brier | Interpretation |
| --- | --- | --- | --- | --- | --- | --- |

## 12.3 端到端对照表

| Chain | Features | Source | AUROC | AUPRC | Brier | Gap vs CFD |
| --- | --- | --- | --- | --- | --- | --- |

---

## 13. 解释分析规范

风险模型不能只给一个分数，必须解释为什么。

## 13.1 全局解释

- SHAP summary plot
- 特征重要性排序

## 13.2 个体解释

- 典型高风险病例 SHAP force plot
- 典型低风险病例 SHAP force plot

## 13.3 分组解释

如果样本允许，可按以下分组：

- 闭塞 vs 通畅
- 不同术前解剖类型
- 不同 AAA 直径组

---

## 14. 风险模型与任务B的接口要求

任务 C 直接依赖任务 B 的特征输出，因此必须统一接口。

## 14.1 必须保证

- CFD 与 AI 版本特征定义一致
- 列名一致
- 单位一致
- 聚合方式一致

## 14.2 禁止情况

- CFD 用区域平均，AI 用区域最大值
- CFD 用全周期指标，AI 用局部时间步指标
- 不同实验更改 DRI 定义

---

## 15. 通过标准与止损点

## 15.1 可以宣称“数字孪生链成立”的最低标准

至少满足以下 3 条：

1. `Clinical + Geometry + CFD-hemodynamics` 优于 `Clinical + Geometry`
2. `Clinical + Geometry + AI-hemodynamics` 也优于 `Clinical + Geometry`
3. AI 版本与 CFD 版本差距有限且方向一致
4. 多个 seed 或多折结果稳定

## 15.2 需要止损复盘的情况

1. CFD-hemodynamics 本身不提升风险模型
2. AI-hemodynamics 完全破坏风险排序
3. 风险模型对 split 极度敏感
4. 加深度隐式特征后性能不升反降且解释性消失

出现以上情况，应优先检查：

- 风险标签定义
- 任务 B 指标质量
- 数据泄漏
- 样本不平衡处理

---

## 16. 建议执行顺序

1. 先固定闭塞风险特征表结构。
2. 先跑 `Clinical-only` 和 `Clinical+Geometry`。
3. 再跑 `Clinical+Geometry+CFD-hemodynamics`。
4. 确认真实血流特征有价值后，再接入 AI-hemodynamics。
5. 最后再考虑 `h_global` 和更复杂融合模型。

核心原则：

先证明真实血流特征有效，再证明 AI 可以替代，不要顺序反过来。
