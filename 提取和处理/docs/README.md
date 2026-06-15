# 项目文档索引

> 更新时间：2026-06-14
> 主入口：[实验设计总纲](实验设计总纲.md)

本目录存放实验设计、任务规范、路线文档、推进日志、汇报材料和外部论文 baseline 复现记录。当前项目已经不再按“单一 GNN 优化线”推进，而是分成：

- **任务 A 当前主线**：V3 PointNeXt / 双域 WSS / 路径 G-I 诊断与 Go/No-Go
- **V1 补充验证**：PINN / physics loss 阶梯消融，用于回答早期速度压力场与物理损失问题
- **外部 baseline 复现**：公开医学血管点云、mesh、等变网络、2D 展开方法在私有 AAA/WSS 数据上的对照
- **后处理与可视化**：预测点云到 CFD 面片/体网格的公平映射和论文图件规范

## 1. 现在先看什么

### 日常推进任务 A

1. [任务A总入口](01-任务/任务A/README.md)
2. [V3 路线 README](01-任务/任务A/03-V3路线/README.md)
3. [V3 实验执行跟踪日志](01-任务/任务A/03-V3路线/01-执行与待办/V3_实验执行跟踪日志.md)
4. [V3 后续优化待办](01-任务/任务A/03-V3路线/01-执行与待办/V3_后续优化待办.md)
5. [代码修改与实验推进记录](02-推进与变更/代码修改与实验推进记录.md)

### 查当前总设计

1. [实验设计总纲](实验设计总纲.md)
2. [项目缺陷分析与修正路径](02-推进与变更/项目缺陷分析与修正路径.md)
3. [任务A实验状态表](01-任务/任务A/03-共享执行与状态/任务A实验状态表.md)

### 准备外部论文 baseline

1. [外部论文 baseline 复现库](paper_reproduction/README.md)
2. [文献筛选总表](paper_reproduction/00-文献筛选总表.md)
3. [复现优先级与适配策略](paper_reproduction/01-复现优先级与适配策略.md)
4. [私有数据适配统一口径](paper_reproduction/02-私有数据适配统一口径.md)
5. [后处理可视化与插值方法](paper_reproduction/03-后处理可视化与插值方法.md)
6. [点云预测结果回构与 CFD 后处理可视化流程](paper_reproduction/visualization_pipeline/README.md)

### 准备论文图和汇报

1. [任务E血流动力学三维可视化执行清单](01-任务/任务E/任务E血流动力学三维可视化执行清单.md)
2. [任务E论文可视化规范](01-任务/任务E/任务E论文可视化规范.md)
3. [任务A基线成果与创新点汇报思路](03-汇报材料/任务A基线成果与创新点汇报思路.md)

## 2. 当前路线状态

| 路线 | 当前定位 | 最近状态 |
| --- | --- | --- |
| `Route-KNN-GNN-V1` | 历史基线与消融证据 | A-Base / A-Main / A-Opt / Line G / Line W 保留为历史对照；新增 V1 PINN 阶梯用于回答物理损失问题 |
| `Route-PhysicsAware-V2` | V2 修正路线历史框架 | V2P-WSSP 已形成一批 p+WSS / WSS loss 对照结果，当前不再作为日常主攻 |
| `Route-DualDomain-PointNeXt-V3` | 当前任务 A 主攻 | V3P post5463 band 约 `wss_r2_wss=0.425±0.012`；G1/G2/G3/G4-a/G4-b 多轮 No-Go 后进入 Q0/I 诊断与后续路线裁决 |
| 外部论文 baseline | 下一阶段论文必需对照 | 已建立 `docs/paper_reproduction/`；第一批优先 MultiViewUNet、coronary mesh convolution、LaB-GATr、E(3)-equivariant AAA WSS、PointNetCFD |
| 后处理可视化 | 论文与答辩支撑链 | 已明确同点指标优先、插值只作展示、WSS 后处理必须先做 CFD velocity oracle；新增 `paper_reproduction/visualization_pipeline/` 管理点云回面片与 CFD-Post/Fluent 交付流程 |

## 3. 目录结构

| 目录 | 用途 |
| --- | --- |
| **根目录** | [实验设计总纲](实验设计总纲.md)：当前论文实验设计、路线状态与任务拆分 |
| **00-规范与记录/** | 实验记录填写规范、区域评估口径、实验记录表 |
| **01-任务/** | 任务 A/B/C/D/E 的执行文档总目录 |
| **02-推进与变更/** | 代码修改与实验推进记录、缺陷分析、训练脚手架记录和归档交接材料 |
| **03-汇报材料/** | PPT、汇报思路、图件生成脚本和汇报图 |
| **paper_idea/** | 项目思路、老师给定论文阅读材料、基准模型推荐与论文素材 |
| **paper_reproduction/** | 外部论文 baseline 复现库，记录公开模型、私有数据适配口径与后处理映射方法 |
| **../external_baselines/** | 外部论文 baseline 复现代码，当前包含 PointNetCFD 独立训练入口 |

## 4. 快速跳转

| 类别 | 链接 |
| --- | --- |
| 总纲 | [实验设计总纲](实验设计总纲.md) |
| 任务 A 总入口 | [01-任务/任务A/README](01-任务/任务A/README.md) |
| 任务 A V1 | [V1 README](01-任务/任务A/01-V1路线/README.md) · [V1 实验清单](01-任务/任务A/01-V1路线/任务A_V1实验清单.md) |
| 任务 A V2 | [V2 README](01-任务/任务A/02-V2路线/README.md) · [V2 修正路线实验矩阵](01-任务/任务A/02-V2路线/任务A_V2修正路线实验矩阵.md) |
| 任务 A V3 | [V3 README](01-任务/任务A/03-V3路线/README.md) · [V3 实验日志](01-任务/任务A/03-V3路线/01-执行与待办/V3_实验执行跟踪日志.md) · [V3 待办](01-任务/任务A/03-V3路线/01-执行与待办/V3_后续优化待办.md) |
| 任务 A 状态 | [任务A实验状态表](01-任务/任务A/03-共享执行与状态/任务A实验状态表.md) · [任务A配置与启动说明](01-任务/任务A/03-共享执行与状态/任务A配置与启动说明.md) |
| 任务 B | [任务B指标计算规范](01-任务/任务B/任务B指标计算规范.md) |
| 任务 C | [任务C风险建模规范](01-任务/任务C/任务C风险建模规范.md) |
| 任务 D | [任务D端到端验证清单](01-任务/任务D/任务D端到端验证清单.md) |
| 任务 E | [任务E执行清单](01-任务/任务E/任务E血流动力学三维可视化执行清单.md) · [任务E论文可视化规范](01-任务/任务E/任务E论文可视化规范.md) |
| 推进记录 | [代码修改与实验推进记录](02-推进与变更/代码修改与实验推进记录.md) |
| 项目思路 | [项目思路](paper_idea/项目思路.md) · [基准模型推荐](paper_idea/基准模型推荐与引用参考.md) |
| 外部 baseline | [paper_reproduction/README](paper_reproduction/README.md) · [文献筛选总表](paper_reproduction/00-文献筛选总表.md) · [后处理插值方法](paper_reproduction/03-后处理可视化与插值方法.md) · [点云回构可视化流程](paper_reproduction/visualization_pipeline/README.md) |
| 外部 baseline 代码 | [PointNetCFD 复现代码](../external_baselines/pointnetcfd/README.md) |

## 5. 维护规则

- 任务 A 新实验事实优先写入 V3 实验日志或对应路线状态表，再同步推进记录。
- 代码、脚本、配置或实验文档变更后，必须在 [代码修改与实验推进记录](02-推进与变更/代码修改与实验推进记录.md) 文首新增记录。
- 仅修改 PPT/PPTX 等汇报文件时，不需要更新推进记录；若同时修改实验文档或脚本，则仍需更新。
- 外部 baseline 复现相关内容统一写入 `paper_reproduction/`，不要混进 V3 内部优化路线。
