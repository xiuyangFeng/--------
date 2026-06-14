# PointNetCFD

## 基本信息

- 论文：A point-cloud deep learning framework for prediction of fluid flow fields on irregular geometries
- 代码：`https://github.com/Ali-Stanford/PointNetCFD`
- 本项目优先级：P0

## 方法要点

PointNetCFD 使用 shared MLP + global max pooling + point-wise decoder 在不规则几何点云上预测 CFD 流场。它不依赖 mesh connectivity，是建立点云外部 baseline 的最低成本入口。

## 与本项目的关系

适合回答：在不使用图结构时，公开 PointNet-style 方法能否在本项目私有数据上达到可接受的 `u/v/w/p` 或近壁质量。

## 最小复现路线

1. 将本项目采样点导出为 PointNetCFD 可读格式。
2. 输入先只用 `x,y,z,t,BC,is_wall`。
3. 输出先对齐 `u/v/w/p`。
4. 再加入显式几何特征做 ablation。
5. 最后才尝试 WSS head。

## 风险

- PointNet 的全局 pooling 对局部边界层梯度可能不足。
- 只要速度/压力改善，不代表 WSS 后处理一定改善。
