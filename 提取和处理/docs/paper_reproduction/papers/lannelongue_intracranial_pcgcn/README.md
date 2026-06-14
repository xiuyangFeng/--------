# Lannelongue et al. physics-constrained intracranial aneurysm GNN

## 基本信息

- 论文：Physics constrained graph neural network for real time prediction of intracranial aneurysm hemodynamics
- 发表：npj Digital Medicine 2026
- 数据：105 例 patient-derived intracranial aneurysm benchmark dataset
- 本项目优先级：P1

## 方法要点

该工作用 physics-constrained GNN 预测颅内动脉瘤 3D transient hemodynamics，覆盖 WSS、OSI 等血流动力学指标，并公开 benchmark dataset。

## 与本项目的关系

适合做强引用和潜在医学 mesh/GNN baseline：

- 同属患者特异性动脉瘤血流 surrogate；
- 输出覆盖 WSS/OSI；
- 数据集公开，有助于论文 related work 和外部对照。

## 适配清单

- [ ] 确认代码是否随文章或数据集公开
- [ ] 若只有数据，先作为外部 benchmark 结构参考
- [ ] 若代码可用，检查是否依赖颅内动脉瘤 mesh 特定字段
- [ ] 将输出目标对齐本项目 WSS / TAWSS / OSI

## 风险

- 颅内动脉瘤和 AAA 的几何尺度、分支结构、边界条件不同。
- 若无代码，不能列为已复现实验，只能列为文献对照。
