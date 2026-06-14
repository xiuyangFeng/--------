# PINGS / PINGS-X / physics-informed carotid flow

## 基本信息

- 论文：Physics-informed graph neural networks for flow field estimation in carotid arteries
- 相关代码：
  - `https://github.com/sukjulian/physics-informed-gnn`
  - `https://github.com/SpatialAILab/PINGS-X`
- 发表：Medical Image Analysis 2026
- 本项目优先级：P1

## 方法要点

该类工作面向颈动脉血流场估计，使用 PointNet++ / GNN / SIREN / physics-informed 约束，从医学血管数据学习 4D flow 或 velocity field。不同仓库命名和论文版本可能不完全一致，后续复现前必须先做代码审计，避免把 `PINGS`、`PINGS-X` 和 `physics-informed-gnn` 混成同一个实验。

## 与本项目的关系

适合借鉴：

- 医学血管小样本设置；
- physics-informed loss 的实现方式；
- group-steerable / equivariant 思路；
- 4D flow 或非 CFD 标签下的泛化讨论。

不宜直接照搬：

- 它不是 AAA WSS 直接预测；
- 数据来源和噪声结构与 CFD 监督不同；
- 物理约束不能直接替代本项目 GNN autograd PINN 口径。

## 适配清单

- [ ] 分别审计两个仓库输入格式、license、数据下载和训练入口
- [ ] 对齐本项目 `u/v/w/p` 或 WSS 输出
- [ ] 单独标注 physics loss 是否在连续坐标映射上成立
- [ ] 与 V1 PINN / V3 G2 文档口径分开记录

## 当前判断

该方向适合引用和方法借鉴，但短期不应放在 AAA/WSS 第一复现位：它更偏 4D carotid flow / velocity field 重建，不是本项目的 AAA wall WSS 直接 baseline。
