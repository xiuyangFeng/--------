# 任务B论文可视化与指标建议

> 上位文档：[实验设计总纲](../../实验设计总纲.md) / [任务B指标计算规范](./任务B指标计算规范.md) / [项目思路](../../paper_idea/项目思路.md)

## 1. 文档定位

本文件用于固定任务 B 在论文中建议展示的图表和指标，重点回答：

1. AI 重建场能否恢复 CFD 衍生血流动力学指标。
2. 这些指标是否足够稳定，能支撑任务 C 的风险预测。

---

## 2. 当前代码中已有的相关基础

当前仓库已经有任务 B 的基础骨架，不是从零开始：

- `training/analysis/hemo.py`
  - 已有 `WSS / TAWSS / OSI / RRT` 计算骨架
  - 已支持 point-level 与 per-case / per-region 汇总思路
- `training/analysis/visualization.py`
  - 已有 `bland_altman()`
  - 已有 `plot_regional_bar()`
  - 已有 scatter、boxplot 等通用函数

当前判断：

- 指标计算链已经有方向；
- 论文图表中最核心的 Bland-Altman、scatter、区域对比图已经有代码基础；
- 区域热图、排序一致性图仍更像“论文阶段需要补齐的专用图”。

---

## 3. 论文中建议保留的核心可视化

### 3.1 必做图

1. CFD vs AI 指标散点图
   - 分别展示 `WSS / TAWSS / OSI / RRT`
2. Bland-Altman 图
   - 用于展示 AI 与 CFD 的一致性和系统偏差
3. 区域级箱线图
   - 比较不同区域上的指标误差分布
4. 左右差异指标对比图
   - 尤其是髂动脉左右差异、差值、比值、`DRI`
5. 典型病例高风险区域热图
   - 展示 AI 是否保留风险区域位置与形态

### 3.2 强烈建议补充的图

1. 病例级排序一致性图
   - 看 AI 与 CFD 对病例风险顺序的保真度
2. 区域级误差条形图
   - 对比 `AAA 主体 / 左右髂总 / 左右髂外 / 左右髂内`
3. 高风险病例与低风险病例对照图
   - 说明 AI 指标能否支持风险分层
4. waterfall 或 rank plot
   - 展示病例排序变化

---

## 4. 论文中建议报告的指标

### 4.1 指标对象

任务 B 至少围绕以下对象：

- `WSS`
- `TAWSS`
- `OSI`
- `RRT`

### 4.2 点级或壁面点级一致性指标

- `MAE`
- `RMSE`
- `R²`

适用对象：

- `WSS`
- 壁面点级 `RRT`
- 局部 `TAWSS`

### 4.3 区域级一致性指标

- `MAE`
- `RMSE`
- `R²`
- Pearson
- Spearman

区域输出建议至少包含：

- 平均值
- 最大值
- 95th percentile
- 高风险阈值面积占比

### 4.4 病例级一致性指标

- Pearson
- Spearman
- Bland-Altman
- 排序一致性
- 高低风险分层一致性

### 4.5 下游风险相关附加指标

如果论文主线要服务任务 C，建议额外重点保留：

- `TAWSS-pre`
- `OSI-pre-DRI`
- `RRT-IA-post`
- `OSI-post-DRI`
- 左右差异比
- 左右差值

---

## 5. 写论文时的推荐组织方式

### 5.1 一致性主结果

- 一张散点图组
- 一张 Bland-Altman 图组
- 一张区域级结果表

### 5.2 风险可用性结果

- 一张左右差异指标图
- 一张病例排序一致性图
- 一张典型高风险区域热图

### 5.3 讨论重点

建议重点讨论两件事：

1. AI 和 CFD 是否“数值接近”
2. AI 和 CFD 是否“排序一致”

对任务 C 来说，第二件事往往比第一件事更重要。

---

## 6. 完整出图规划

### 6.1 Figure B1：任务 B 主结果表

回答的问题：

- AI 衍生指标能否整体接近 CFD。

建议表头：

- `Model`
- `WSS_RMSE`
- `WSS_R2`
- `TAWSS_RMSE`
- `TAWSS_R2`
- `OSI_RMSE`
- `OSI_R2`
- `RRT_RMSE`
- `RRT_R2`
- `Case-level Corr`

建议模型：

- `CFD upper bound`
- `MLP -> hemo`
- `Graph model -> hemo`
- `Transformer(no geometry) -> hemo`
- `Transformer(geometry) -> hemo`

### 6.2 Figure B2：四指标一致性散点图组

回答的问题：

- `WSS / TAWSS / OSI / RRT` 是否都被恢复，而不是只恢复其中一个。

推荐版式：

- `2 x 2`
- 每个子图一个指标
- 每个子图统一标：
  - Pearson
  - Spearman
  - RMSE
  - 对角参考线

当前代码支撑：

- scatter 类函数可复用
- 具体“多指标统一排版”脚本仍建议后续补

### 6.3 Figure B3：Bland-Altman 图组

回答的问题：

- AI 是否存在系统偏差，误差是否在可接受范围内。

推荐版式：

- `2 x 2`
- 与 Figure B2 指标顺序保持一致
- 统一 y 轴写法为 `AI - CFD`

当前代码支撑：

- `training.analysis.visualization.bland_altman()`

### 6.4 Figure B4：区域级结果图

回答的问题：

- AI 在不同解剖区域上的指标恢复是否一致。

推荐版式：

- 左：区域级箱线图
- 右：区域级条形图

区域建议固定为：

- `AAA 主体`
- `左髂总`
- `右髂总`
- `左髂外`
- `右髂外`
- 如后续区域更细，再扩展到内髂

当前代码支撑：

- 区域聚合思路已具备
- 真实多区域 mask 仍需后续补齐

### 6.5 Figure B5：病例级排序一致性图

回答的问题：

- AI 能否保留病例相对高低风险顺序。

推荐版式：

- 左：CFD rank vs AI rank 散点
- 右：前 10 个高风险病例的 rank comparison 或 slope chart

这张图很重要，因为它直接服务任务 C。

### 6.6 Figure B6：左右差异指标图

回答的问题：

- AI 是否保留双侧不对称性相关信息。

推荐内容：

- `OSI_pre_DRI`
- `OSI_post_DRI`
- 左右差值
- 左右比值

推荐版式：

- grouped bar 或 paired scatter

### 6.7 Figure B7：典型病例高风险区域图

回答的问题：

- AI 是否保留了高风险区域的位置和面积。

推荐版式：

- `2 x 2`
- `CFD risk map`
- `AI risk map`
- `Absolute error map`
- `Region overlay`

建议只挑最有代表性的 1 到 2 个病例。

### 6.8 Supplementary B：建议放附录的图

- 不同上游模型驱动的 hemo 结果图
- 更多病例排序图
- 各区域详细统计表
- 多 seed 指标稳定性图

---

## 7. 当前状态判断

- 文档层面已经很清楚地规定了任务 B 必须同时输出点级、区域级、病例级结果。
- 代码层面已经有指标计算和 Bland-Altman 的基础，但“论文级全套出图”还没有完全补齐。
- 如果后续只做相关系数而不做排序一致性和区域热图，论文论证会偏弱。

---

## 8. 与后续任务的关系

- 任务 B 直接决定任务 C 中 `CFD-hemodynamics` 和 `AI-hemodynamics` 特征表的质量。
- 如果任务 B 不能证明关键指标在病例级上稳定一致，任务 C 和任务 D 都不应直接推进主结论。
