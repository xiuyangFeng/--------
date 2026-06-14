# E(3)-equivariant AAA WSS

## 基本信息

- 论文：Wall Shear Stress Estimation in Abdominal Aortic Aneurysms: Towards Generalisable Neural Surrogate Models
- 年份：2025
- 任务：AAA transient WSS surrogate
- 本项目优先级：P0

## 方法要点

该工作使用 E(3)-equivariant geometric deep learning、robust geometrical descriptors 和 projective geometric algebra 来估计 AAA transient WSS，并强调跨几何、边界条件、拓扑和网格分辨率的泛化。

## 与本项目的关系

这是当前列表里与本项目最直接竞争的 baseline：

- 同为 AAA；
- 同样关注 WSS；
- 方法核心是等变性和几何描述符；
- 正好对应本项目 V3 中 `wss_x/y` 长期难学、坐标系负担重的问题。

## 适配清单

- [ ] 定位作者公开代码和 license
- [ ] 确认输入是 surface mesh、point cloud 还是 artery tree graph
- [ ] 建立本项目壁面 surface graph / STL 顶点映射
- [ ] 输出 transient WSS，反归一化后报告 Pa
- [ ] 指标必须包括 `wss_x/y/z`、向量角误差、TAWSS/OSI

## 风险

- 若依赖完整 surface mesh 拓扑，本项目需要先补齐 mesh 映射层。
- 如果只能复现论文结构而没有官方代码，需单独标注为“architecture reproduction”。
