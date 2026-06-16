# cerebrovascular PointNet/PINN point-cloud paper

## 基本信息

- 论文：Physics-informed neural networks for three-dimensional cerebrovascular hemodynamic prediction: A point cloud preprocessing strategy based on limited data
- 发表：Engineering Applications of Artificial Intelligence 2025
- 本项目优先级：P2

## 方法要点

该论文在 51 例脑血管 CFD 数据上，将点云重采样到 10,000 点，用 PointNet-style 网络预测 `Vx/Vy/Vz/P`，并通过 PINN residual 和 no-slip 条件增强训练。FR/PD 是从预测速度/压力导出的临床 surrogate。

## 与本项目的关系

可借鉴：

- 小样本医学血管点云；
- 点云重采样和插值；
- PINN residual 的动态权重；
- FR/PD 这类病例级 surrogate 叙事。

不能直接照搬：

- 主输出不是 WSS；
- 速度/压力 NMAE 不能证明 WSS 点级 R² 可突破；
- 10,000 点均匀化可能损害近壁梯度和 high-WSS hotspot。

## 适配清单

- [x] 已拿到 CROWN/Beihang 源码，位置：`external_baselines/CROWN_Beihang/`
- [x] 已完成源码初步审计与私有数据适配计划：[`CROWN_Beihang私有数据适配计划.md`](CROWN_Beihang私有数据适配计划.md)
- [ ] 本项目若借鉴 PINN，必须区分连续 PointNet 映射与 GNN autograd residual
- [ ] 任何 WSS 结论必须走完整后处理链
- [ ] 第一轮先跑 `u,v,w,p` 的 paper-original 非 PINN / PINN 对照，再做显式几何特征消融

## 梳理记录

- **状态**：待首轮计划实验矩阵完成
- **文档**：矩阵跑齐后填写 [`梳理记录.md`](梳理记录.md)（模板 [`_template/梳理记录.md`](../_template/梳理记录.md)）
- **规范**：[04-梳理记录规范](../../04-梳理记录规范.md)
