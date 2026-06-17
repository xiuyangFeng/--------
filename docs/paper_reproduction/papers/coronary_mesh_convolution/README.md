# Coronary mesh convolution / SE(3)-equivariant artery wall WSS

## 基本信息

- 相关论文：Mesh Neural Networks for SE(3)-Equivariant Hemodynamics Estimation on the Artery Wall
- 代码：`https://github.com/sukjulian/coronary-mesh-convolution`
- 任务：artery wall surface mesh 上估计 WSS vector
- 本项目优先级：P0

## 方法要点

该方向在动脉壁面 mesh 上做 SE(3)-equivariant hemodynamics estimation，代码包含 artery wall WSS 相关实现、数据处理和 baseline。虽然原始任务不是 AAA，但数据结构与本项目“壁面点/面片上的 WSS 向量预测”高度接近。

## 与本项目的关系

这是比通用 MeshGraphNet 更贴近本项目 WSS 终点的 surface mesh baseline：

- 输出是壁面 WSS vector；
- 关注旋转/平移等变性；
- 可直接检验 V3 中 `wss_x/y` 坐标系负担问题；
- 有成熟代码和 baseline，可优先审计。

## 适配清单

- [ ] 拉取仓库并记录环境
- [ ] 确认输入 mesh 顶点、面片、法向、标签格式
- [ ] 将本项目 STL/壁面点转换成目标格式
- [ ] 输出 WSS vector 后与 `wss_x/y/z` 对齐
- [ ] 同时保留 DiffusionNet 或非等变 baseline 作为消融

## 风险

- 冠脉和 AAA 的拓扑、尺度与边界条件不同。
- 若本项目只有采样壁面点而没有可靠面片 connectivity，需要先完成 surface graph 构建。

## 梳理记录

- **状态**：待首轮计划实验矩阵完成
- **文档**：矩阵跑齐后填写 [`梳理记录.md`](梳理记录.md)（模板 [`_template/梳理记录.md`](../_template/梳理记录.md)）
- **规范**：[04-梳理记录规范](../../04-梳理记录规范.md)
