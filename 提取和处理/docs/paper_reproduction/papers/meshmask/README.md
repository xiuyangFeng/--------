# MeshMask

## 基本信息

- 论文：MeshMask: Physics-Based Simulations with Masked Graph Neural Networks
- 年份：ICLR 2025
- 链接：OpenReview / arXiv
- 本项目优先级：P2

## 方法要点

MeshMask 对 CFD mesh 节点做 masked pretraining，随机掩掉部分节点，训练 GNN 学习鲁棒物理表征。论文报告在多个 CFD 数据集上提升长期预测，其中包含 3D 颅内动脉瘤大网格数据。

## 与本项目的关系

它更像一种预训练策略，而不是第一轮独立 baseline。当前本项目已经试过一轮 SSL/预训练思路但未稳定突破；MeshMask 的价值在于提供更强的 masked mesh pretraining 参考。

## 适配清单

- [ ] 确认官方代码是否公开
- [ ] 若无代码，先按论文方法实现 masked node reconstruction 原型
- [ ] 必须在 P0/P1 某个可跑 backbone 上叠加，不单独混表
- [ ] 对照同架构无 pretraining 版本

## 风险

- 如果没有真实 mesh，大部分优势无法发挥。
- 不应把 masked pretraining 的收益与 backbone 变更混在同一张表里。

## 梳理记录

- **状态**：待首轮计划实验矩阵完成
- **文档**：矩阵跑齐后填写 [`梳理记录.md`](梳理记录.md)（模板 [`_template/梳理记录.md`](../_template/梳理记录.md)）
- **规范**：[04-梳理记录规范](../../04-梳理记录规范.md)
