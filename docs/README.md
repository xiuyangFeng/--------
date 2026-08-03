# 项目文档索引

> 更新时间：2026-08-02
> 主入口：[实验设计总纲](实验设计总纲.md)

本目录存放实验设计、任务规范、路线文档、推进日志、汇报材料和外部论文 baseline 复现记录。当前项目已经不再按“单一 GNN 优化线”推进，而是分成：

- **任务 A 当前主线**：V3 PointNeXt / 双域 WSS / 路径 G-I 诊断与 Go/No-Go
- **V1 历史补充验证**：旧 PINN / physics loss 阶梯已归档，只作早期路线证据
- **外部 baseline 复现**：公开医学血管点云、mesh、等变网络、2D 展开方法在私有 AAA/WSS 数据上的对照
- **后处理与可视化**：预测点云到 CFD 面片/体网格的公平映射和论文图件规范
- **WSS-only 最小化线**：`pipeline_wss_min/` + `training_wss_min/`；当前单 seed 开发锚点为 LSA2 SAME-H2 + `log(local_radius)`；数据口径 v4（AG76 / AAA 白名单 / ILO before41）
- **velocity→WSS 物理估算**：`wss_mri_calculator/`（CFD 点云适配）；正式池绑定 PINN `train138/test35`；adaptive-CV v1 已冻结，multiscale v2 见该目录 `experiments/`
- **峰值体域 `u,v,w,p` PINN**：8 臂 Slurm array `11128_[0-7%4]` 已全部完成并审计；每臂 400 epoch / 27,600 step。物理残差与压力普遍改善，速度收益依赖架构与几何输入

## 1. 现在先看什么

### 日常推进任务 A

1. [任务A总入口](01-任务/任务A/README.md)
2. [V3 路线 README](01-任务/任务A/03-V3路线/README.md)
3. [V3 实验执行跟踪日志](01-任务/任务A/03-V3路线/01-执行与待办/V3_实验执行跟踪日志.md)
4. [V3 后续优化待办](01-任务/任务A/03-V3路线/01-执行与待办/V3_后续优化待办.md)
5. [代码修改与实验推进记录](02-推进与变更/代码修改与实验推进记录.md)

### 当前复现与 V3P 状态速读

| 主题 | 先看 | 只记录什么 |
| --- | --- | --- |
| V3P / 后平台期精度优化 | [V3 路线 README](01-任务/任务A/03-V3路线/README.md) + [V3P 后平台期结构与训练优化路线](01-任务/任务A/03-V3路线/01-执行与待办/V3P_后平台期结构与训练优化路线_2026-07-01.md) + [V3P 精度平台期复盘](01-任务/任务A/03-V3路线/01-执行与待办/V3P_精度平台期复盘与下一轮想法_2026-06-30.md) | 对照既有 No-Go 证据后，只保留 K1-lite 5886 作为当前内部 GPU 判断；K3/K5 已 No-Go |
| 外部 CROWN/Beihang 复现 | [CROWN 代码 README](../external_baselines/crown_beihang/README.md) + [hemodynamics_pointcloud_pinn](paper_reproduction/papers/hemodynamics_pointcloud_pinn/README.md) | `u,v,w,p` 速度/压力 paper-original 复现，不写成 WSS baseline |
| 外部 baseline 批次总结 | [paper_reproduction/README](paper_reproduction/README.md) + [梳理记录规范](paper_reproduction/04-梳理记录规范.md) | 一轮矩阵跑齐后的批次结论，单个 Job 只放 `external_baselines/<name>/experiments/` |
| WSS-only 最小化与体域 PINN | [峰值体域 PINN 当前入口](02-推进与变更/WSS_PINN/README.md) + [历史体域物理约束讨论稿](02-推进与变更/WSS最小化_体域物理约束与PINN训练路线_2026-07-29.md) + [高值区域预测优化方案](02-推进与变更/WSS高值区域预测优化方案.md) + [PointNet baseline 实验矩阵与进度](02-推进与变更/WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md) + [训练实验跟踪](02-推进与变更/WSS最小化_训练实验跟踪.md) + [ILO 术前最终审核](02-推进与变更/ILO术前队列最终审核通过_2026-07-17.md) + [跨路线指标口径](00-规范与记录/WSS跨路线评估与横向对比口径.md) | WSS-only 锚点不变；体域 PINN 8/8 completed / audited，结果图与机器可读汇总已落盘；15,000 epoch 早停协议仍为候选、未提交 |
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
6. [点云预测值与真值回插到面片方法](paper_reproduction/05-点云预测值与真值回插到面片方法.md)
7. [CFD-Post 云图导出与交付工具](../tools/cfdpost_cloud_export/README.md)

### 准备论文图和汇报

1. [任务E血流动力学三维可视化执行清单](01-任务/任务E/任务E血流动力学三维可视化执行清单.md)
2. [任务E论文可视化规范](01-任务/任务E/任务E论文可视化规范.md)
3. [任务A基线成果与创新点汇报思路](03-汇报材料/任务A基线成果与创新点汇报思路.md)

## 2. 当前路线状态

| 路线 | 当前定位 | 状态摘要 |
| --- | --- | --- |
| `Route-KNN-GNN-V1` | 历史基线与消融证据 | A-Base / A-Main / A-Opt / Line G / Line W 保留为历史对照；新增 V1 PINN 阶梯用于回答物理损失问题 |
| `Route-PhysicsAware-V2` | V2 修正路线历史框架 | V2P-WSSP 已形成一批 p+WSS / WSS loss 对照结果，当前不再作为日常主攻 |
| `Route-DualDomain-PointNeXt-V3` | 当前任务 A 主攻 | V3P post5463 / I6-diag 平台约 `wss_r2_wss=0.425±0.012`，I6-diag **0.429**；E-J、G4-c、J4/J5、K3/K5 均已封口或 No-Go；M-N/G5 叙事可收口，M-E 当前只等 K1-lite BranchFeat Job **5886** 判读 |
| `Route-Volume-UVWP-PINN` | 峰值体域四通道补充线 | PointNet / PointNet++ 八实验 8/8 completed；PINN 物理一致性和压力普遍改善，但速度泛化收益不统一，PointNet + `xyz+geom` 是唯一五个数据 R² 全部改善的配对 |
| 外部论文 baseline | 下一阶段论文必需对照 | PointNetCFD 首轮矩阵已完成；CROWN/Beihang raw_ascii v1 已进入 lazy 非 PINN 重训 Job 5751，PINN 5740 已判 No-Go |
| 后处理可视化 | 论文与答辩支撑链 | 已明确同点指标优先、插值只作展示、WSS 后处理必须先做 CFD velocity oracle；新增 `paper_reproduction/visualization_pipeline/` 管理点云回面片与 CFD-Post/Fluent 交付流程 |

## 3. 目录结构

| 目录 | 用途 |
| --- | --- |
| **根目录** | [实验设计总纲](实验设计总纲.md)：当前论文实验设计、路线状态与任务拆分 |
| **00-规范与记录/** | 实验记录填写规范、区域评估口径、分析对比图与 postview 可视化目录说明、[集群 node04 使用要点](00-规范与记录/集群node04使用要点.md) |
| **01-任务/** | 任务 A/B/C/D/E 的执行文档总目录 |
| **02-推进与变更/** | 代码修改与实验推进记录、缺陷分析、训练脚手架记录和归档交接材料 |
| **03-汇报材料/** | PPT、汇报思路、图件生成脚本和汇报图 |
| **paper_idea/** | 项目思路、老师给定论文阅读材料、基准模型推荐与论文素材 |
| **paper_reproduction/** | 外部论文 baseline 复现库，记录公开模型、私有数据适配口径与后处理映射方法 |
| **../external_baselines/** | 外部论文 baseline 复现代码，当前包含 PointNetCFD 独立训练入口 |
| **../pipeline_wss_min/** | WSS-only 最小化预处理代码，独立输出到 `data_wss_min/` |
| **../training_wss_min/** | WSS-min 训练/评估，只读 `data_wss_min` |
| **../wss_mri_calculator/** | MRI WSS 计算器 + CFD 点云 velocity→WSS 适配与冻结实验 |
| **../wss_pinn/** | 峰值体域 `u,v,w,p` 活动实现；旧 WSS-target 文档归档，派生数据与结果单独落盘 |

## 4. 快速跳转

| 类别 | 链接 |
| --- | --- |
| 总纲 | [实验设计总纲](实验设计总纲.md) |
| 项目管理规范 | [项目知识库整理规范](00-规范与记录/项目知识库整理规范.md) · [本轮整理审计](02-推进与变更/项目知识库整理审计_2026-06-15.md) · [实验记录填写规范](00-规范与记录/实验记录填写规范.md) · [WSS 跨路线指标口径](00-规范与记录/WSS跨路线评估与横向对比口径.md) · [集群 node04 使用要点](00-规范与记录/集群node04使用要点.md) |
| 可视化落盘 | [实验分析对比图目录说明](00-规范与记录/实验分析对比图目录说明.md)（run 精度） · [点云回插面片可视化目录说明](00-规范与记录/点云回插面片可视化目录说明.md)（postview 面片云图） |
| 任务 A 总入口 | [01-任务/任务A/README](01-任务/任务A/README.md) |
| 任务 A V1 | [V1 README](01-任务/任务A/01-V1路线/README.md) · [V1 实验清单](01-任务/任务A/01-V1路线/任务A_V1实验清单.md) |
| 任务 A V2 | [V2 README](01-任务/任务A/02-V2路线/README.md) · [V2 修正路线实验矩阵](01-任务/任务A/02-V2路线/任务A_V2修正路线实验矩阵.md) |
| 任务 A V3 | [V3 README](01-任务/任务A/03-V3路线/README.md) · [V3 实验日志](01-任务/任务A/03-V3路线/01-执行与待办/V3_实验执行跟踪日志.md) · [V3 待办](01-任务/任务A/03-V3路线/01-执行与待办/V3_后续优化待办.md) |
| V3P 后平台期优化 | [V3P 后平台期结构与训练优化路线](01-任务/任务A/03-V3路线/01-执行与待办/V3P_后平台期结构与训练优化路线_2026-07-01.md) · [V3P 精度平台期复盘](01-任务/任务A/03-V3路线/01-执行与待办/V3P_精度平台期复盘与下一轮想法_2026-06-30.md) |
| 任务 A 状态 | [任务A实验状态表](01-任务/任务A/03-共享执行与状态/任务A实验状态表.md) · [任务A配置与启动说明](01-任务/任务A/03-共享执行与状态/任务A配置与启动说明.md) |
| 任务 B | [任务B指标计算规范](01-任务/任务B/任务B指标计算规范.md) |
| 任务 C | [任务C风险建模规范](01-任务/任务C/任务C风险建模规范.md) |
| 任务 D | [任务D端到端验证清单](01-任务/任务D/任务D端到端验证清单.md) |
| 任务 E | [任务E执行清单](01-任务/任务E/任务E血流动力学三维可视化执行清单.md) · [任务E论文可视化规范](01-任务/任务E/任务E论文可视化规范.md) |
| 推进记录 | [V3P/主线代码修改与实验推进记录](02-推进与变更/代码修改与实验推进记录.md) · [WSS 最小化代码修改与实验推进记录](02-推进与变更/WSS最小化_代码修改与实验推进记录.md) |
| WSS-only 最小化 / velocity→WSS / 体域 PINN | [根 README 主攻入口](../README.md) · [CFD 适配说明](../wss_mri_calculator/README_CFD_ADAPTATION.md) · [adaptive-CV v1](../wss_mri_calculator/experiments/pointcloud_adaptive_v1/README.md) · [multiscale v2](../wss_mri_calculator/experiments/pointcloud_multiscale_v2/README.md) · [峰值体域 PINN 当前入口](02-推进与变更/WSS_PINN/README.md) · [旧 WSS-target PINN 归档](02-推进与变更/WSS_PINN/_archive/wss_target_v1_20260730/README.md) · [历史体域物理约束讨论稿](02-推进与变更/WSS最小化_体域物理约束与PINN训练路线_2026-07-29.md) · [高值区域预测优化方案](02-推进与变更/WSS高值区域预测优化方案.md) · [PointNet baseline 矩阵](02-推进与变更/WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md) · [训练实验跟踪](02-推进与变更/WSS最小化_训练实验跟踪.md) · [推进记录](02-推进与变更/WSS最小化_代码修改与实验推进记录.md) · [历史路线归档](02-推进与变更/_archive/WSS最小化/README.md) |
| 项目思路 | [项目思路](paper_idea/项目思路.md) · [基准模型推荐](paper_idea/基准模型推荐与引用参考.md) |
| 外部 baseline | [paper_reproduction/README](paper_reproduction/README.md) · [文献筛选总表](paper_reproduction/00-文献筛选总表.md) · [后处理插值方法](paper_reproduction/03-后处理可视化与插值方法.md) · [点云回插面片方法](paper_reproduction/05-点云预测值与真值回插到面片方法.md) · [CFD-Post 云图导出工具](../tools/cfdpost_cloud_export/README.md) |
| 外部 baseline 代码 | [PointNetCFD 复现代码](../external_baselines/pointnetcfd/README.md) |

## 5. 维护规则

- 任务 A 新实验事实优先写入 V3 实验日志或对应路线状态表，再同步推进记录。
- V3P / 训练主线 / 通用代码变更后，在 [代码修改与实验推进记录](02-推进与变更/代码修改与实验推进记录.md) 文首新增记录；`pipeline_wss_min/` 相关变更写入 [WSS 最小化代码修改与实验推进记录](02-推进与变更/WSS最小化_代码修改与实验推进记录.md)。
- 数据侧或模型训练侧的集群运行代码必须提供详细进度日志，日志粒度至少细到每一个病例，确保大规模任务可判断是否仍在有效实验/推进，避免因总数据量过大造成无效等待。
- GPU 训练/评估任务只要资源有空闲即可提交使用；若暂无空闲 GPU，则按实验计划顺序提交排队，不再为是否提交或是否排队单独确认。
- 仅修改 PPT/PPTX 等汇报文件时，不需要更新推进记录；若同时修改实验文档或脚本，则仍需更新。
- 外部 baseline 复现相关内容统一写入 `paper_reproduction/`，不要混进 V3 内部优化路线。
