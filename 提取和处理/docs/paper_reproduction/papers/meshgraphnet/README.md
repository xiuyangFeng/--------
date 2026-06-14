# MeshGraphNet

## 基本信息

- 论文：Learning Mesh-Based Simulation with Graph Networks
- 年份：ICLR 2021
- 代码/框架：NVIDIA PhysicsNeMo 已集成 MeshGraphNet；社区也有 PyG 复现
- 本项目优先级：P1

## 方法要点

MeshGraphNet 是经典 Encode-Process-Decode 架构：节点和边特征编码到 latent space，多层 message passing 处理，再解码为节点物理量。它是 mesh-based physical simulation 的标准基线。

## 与本项目的关系

适合回答：真实 mesh 拓扑上的 EPD GNN 是否优于当前历史 kNN / PointNeXt 路线。

关键前置是本项目必须能提供物理可信的 mesh connectivity。若直接使用当前 kNN 图作为 MeshGraphNet 输入，会重复 V1 的物理邻接问题，不能称为严格 mesh baseline。

## 适配清单

- [ ] 确认 `.cas/.msh` 或 VTK mesh connectivity 是否可导出
- [ ] 定义 node feature：坐标、时间、BC、显式几何特征、wall mask
- [ ] 定义 edge feature：相对位移、距离、mesh 边类型
- [ ] 输出：先 `u/v/w/p`，再扩展 WSS
- [ ] 指标：near_wall、WSS、TAWSS、病例级 p95

## 风险

- 真实 mesh 拓扑和采样点之间的映射是主要工程难点。
- PhysicsNeMo 依赖栈可能较重，需独立环境审计。
