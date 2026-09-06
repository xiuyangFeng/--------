# 项目文档索引

> 更新时间：2026-09-04（formal backbone 选定 P2V/D2；anatomy-only 正式 bundle 与模型改造仍 No-Go）
> 主入口：[实验设计总纲](实验设计总纲.md)

本目录存放实验设计、任务规范、路线文档、推进日志、汇报材料和外部论文 baseline 复现记录。2026-08-08 起，当前模型设计范围为显式 RCR 条件的体域 `u,v,w,p` V4 重构；其他路线按历史、baseline 或冻结下游验证器维护：

- **当前模型设计主线**：旧 `volume_uvwp_bc_rcr_v4` 48-run 为 pre-Centerline-V2 容量配平历史 screen。2026-09-04 用户选择 formal backbone=PointNet P2V / 纯 PointNet++ D2 `c125-k128`；当前代码/配置尚未改造。八例低压力族及 ZHOU/ZUO 原 Q 长尾已完成 raw 修复，但 2026-08-31 staging 已过期；须先完成 anatomy-only train138/test34 正式 bundle、模型/采样合同与新 Gate，继续保持 `training_ready=false`
- **任务 A 直接 WSS 历史线**：V3 PointNeXt / 双域 WSS / 路径 G-I 诊断结果保留，本轮不继续调参
- **V1 历史补充验证**：旧 PINN / physics loss 阶梯已归档，只作早期路线证据
- **外部 baseline 复现**：公开医学血管点云、mesh、等变网络、2D 展开方法在私有 AAA/WSS 数据上的对照
- **后处理与可视化**：预测点云到 CFD 面片/体网格的公平映射和论文图件规范
- **Centerline V2 数据修复**：2026-08-28 已完成当时源产物；2026-08-31 WSS_PINN staging 曾通过工程 Gate，但因后续 raw/真实 mm/anatomy-only 合同更新已过期，仅作中间证据；其他下游和正式训练 route 尚未切换
- **WSS-only 最小化线**：`pipeline_wss_min/` + `training_wss_min/`；当前单 seed 开发锚点为 LSA2 SAME-H2 + `log(local_radius)`；数据口径 v4（AG76 / AAA 白名单 / ILO before41）
- **velocity→WSS 冻结验证器**：`wss_mri_calculator/` 的 Profile-Secant V3；test35 全壁面病例 overall / pooled high-WSS R²=`0.9604/0.9440`，但逐病例 high-WSS R² 均值=`0.7735`、平均峰值低估=`15.81%`。当前不继续调算法，也不参与 `u,v,w,p` 训练或选模

## 1. 现在先看什么

### 当前 `u,v,w,p` 推进

1. [正式重建前剩余整改问题与验收计划](02-推进与变更/WSS_PINN/WSS_PINN_V4正式重建前剩余整改问题与验收计划_2026-09-03.md)
2. [BC/RCR V4 近壁 1.5 mm 与 PN/PNPP 采样核查](03-汇报材料/V4汇报/V4_近壁采样核查_2026-09-04/README.md)
3. [173 例训练数据数值与刚性配准审阅及修复计划](02-推进与变更/WSS_PINN/WSS_PINN_V4_173例训练数据数值与刚性配准审阅及修复计划_2026-08-30.md)
4. [V4 大重构设计方案](02-推进与变更/WSS_PINN/WSS_PINN_V4大重构设计方案_2026-08-08.md)
5. [体域 PINN 路线真源](02-推进与变更/WSS_PINN/README.md)
6. [`wss_pinn` 代码入口](../wss_pinn/README.md)
7. [核心代码诊断（field-v4 历史科学合同，🧊已归档）](02-推进与变更/WSS_PINN/_archive/核心代码诊断与下一轮设计建议_2026-08-05.md)
8. [代码修改与实验推进记录](02-推进与变更/代码修改与实验推进记录.md)

历史任务 A / V3 直接 WSS 路线仍从[任务A总入口](01-任务/任务A/README.md)进入，
本轮不继续其结构或 loss 调优。

### Centerline V2 数据修复

1. [全队列修复与切换记录](02-推进与变更/Centerline_V2全队列修复与切换记录_2026-08-28.md)
2. [全量输出 README](../outputs/centerline_v2_full_173_20260828/README.md)
3. [已执行归档方案](02-推进与变更/_archive/VMTK中心线修复与全队列优化方案_已执行_2026-08-27.md)
4. [`pipeline` 中心线入口与切换边界](../pipeline/README.md#centerline-v2-全队列审计2026-08-28)

WSS_PINN 的 08-31 独立 staging 不覆盖旧 root，但已被后续合同更新判为过期；
`data_wss_min` 和其他旧 bundle 仍未重建。该 staging 不得与旧矩阵混用，也不得作为正式训练输入。

### 当前复现与 V3P 状态速读

| 主题 | 先看 | 只记录什么 |
| --- | --- | --- |
| V3P / 直接 WSS 历史线 | [V3 路线 README](01-任务/任务A/03-V3路线/README.md) + [V3P 后平台期结构与训练优化路线](01-任务/任务A/03-V3路线/01-执行与待办/V3P_后平台期结构与训练优化路线_2026-07-01.md) + [V3P 精度平台期复盘](01-任务/任务A/03-V3路线/01-执行与待办/V3P_精度平台期复盘与下一轮想法_2026-06-30.md) | 保留既有 No-Go 与平台期证据；本轮不继续 WSS 网络优化 |
| 外部 CROWN/Beihang 复现 | [CROWN 代码 README](../external_baselines/crown_beihang/README.md) + [hemodynamics_pointcloud_pinn](paper_reproduction/papers/hemodynamics_pointcloud_pinn/README.md) | `u,v,w,p` 速度/压力 paper-original 复现，不写成 WSS baseline |
| 外部 baseline 批次总结 | [paper_reproduction/README](paper_reproduction/README.md) + [梳理记录规范](paper_reproduction/04-梳理记录规范.md) | 一轮矩阵跑齐后的批次结论，单个 Job 只放 `external_baselines/<name>/experiments/` |
| velocity→WSS V1–V4 | [实验总跟踪](../wss_mri_calculator/experiments/README.md) + [Profile-Secant V3 推荐结果](../wss_mri_calculator/experiments/pointcloud_surface_mls_v4/PROFILE_SECANT_HIGH_TAIL_V3_RESULTS.md) + [全壁面结果](../outputs/wss_mri_calculator/pointcloud_surface_mls_v4/profile_secant_v3_fullwall/README.md) + [CFD 适配说明](../wss_mri_calculator/README_CFD_ADAPTATION.md) | Profile-Secant V3 冻结为 `u,v,w,p` 最终 checkpoint 的下游 sanity check；不继续调算法或参与选模 |
| 当前 `u,v,w,p` 优化与历史 WSS 线 | [173 例数据审阅与修复计划](02-推进与变更/WSS_PINN/WSS_PINN_V4_173例训练数据数值与刚性配准审阅及修复计划_2026-08-30.md) + [V4 大重构设计](02-推进与变更/WSS_PINN/WSS_PINN_V4大重构设计方案_2026-08-08.md) + [体域 PINN 当前入口](02-推进与变更/WSS_PINN/README.md) + [当前代码入口](../wss_pinn/README.md) + [field-v4 Stage 0–1 已归档合同](02-推进与变更/WSS_PINN/_archive/WSS_PINN_下一智能体目标提示词_冻结诊断后执行Stage0至Stage1_已完成_2026-08-06.md) + [V3 六臂结果](../outputs/wss_pinn/volume_uvwp_peak_qs_smooth_v3/summary/README.md) + [历史六臂预注册](02-推进与变更/WSS_PINN/_archive/WSS_PINN_下一智能体目标提示词_准稳态平滑场六臂实验_已完成_2026-08-05.md) + [老师论文与 V2 对照](paper_reproduction/papers/hemodynamics_pointcloud_pinn/README.md) | seed1234 的 0–15 已在 xlsx 主表；旧 173 例只作 pre-Centerline-V2 历史 screen，formal 为 train138/test34 共 172 例且训练 No-Go |

### 查当前总设计

1. [实验设计总纲](实验设计总纲.md)
2. [任务A实验状态表](01-任务/任务A/03-共享执行与状态/任务A实验状态表.md)

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

## 2. 当前路线状态

| 路线 | 当前定位 | 状态摘要 |
| --- | --- | --- |
| `Route-KNN-GNN-V1` | 历史基线与消融证据 | A-Base / A-Main / A-Opt / Line G / Line W 保留为历史对照；新增 V1 PINN 阶梯用于回答物理损失问题 |
| `Route-PhysicsAware-V2` | V2 修正路线历史框架 | V2P-WSSP 已形成一批 p+WSS / WSS loss 对照结果，当前不再作为日常主攻 |
| `Route-DualDomain-PointNeXt-V3` | 直接 WSS 历史内部线 | V3P post5463 / I6-diag 平台约 `wss_r2_wss=0.425±0.012`，I6-diag **0.429**；既有分支保留，当前不新增 WSS 调参 |
| `Route-CFD-Velocity-to-WSS-V4` | 冻结的 WSS 后处理验证器 | Profile-Secant V3 只用于 `u,v,w,p` 主 checkpoint 的 downstream sanity check；不继续调算法，不参与训练或选模 |
| `Route-Volume-UVWP-PINN` | **当前数据/骨干 cutover 主线；正式训练 No-Go** | 旧 BC/RCR V4 为容量配平历史 screen；08-31 staging 已过期。formal split=train138/test34、backbone=P2V/D2 `c125-k128`，待正式 bundle、模型/采样合同与新 preflight 闭环 |
| 外部论文 baseline | 论文必需对照 | PointNetCFD 首轮矩阵已完成；CROWN/Beihang 非 PINN/PINN paper-original 复现均已结案 No-Go。原文 NMAE、病例级 FR/PD R² 不与当前点级 speed R² 混表 |
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
| **../tools/centerline_v2_*.py** | Centerline V2 权威表面冻结、稳健 VMTK、显式图、Gate 和批量渲染 |
| **../training_wss_min/** | WSS-min 训练/评估，只读 `data_wss_min` |
| **../outputs/field/** | [历史任务 A / V3 field 产物入口](../outputs/field/README.md)，含 archive 与可再生性记录 |
| **../wss_mri_calculator/** | MRI WSS 计算器 + CFD 点云 velocity→WSS 适配与冻结实验 |
| **../wss_pinn/** | 峰值体域 `u,v,w,p` 活动实现；旧 WSS-target 文档归档，派生数据与结果单独落盘 |

## 4. 快速跳转

| 类别 | 链接 |
| --- | --- |
| 总纲 | [实验设计总纲](实验设计总纲.md) |
| 项目管理规范 | [项目知识库整理规范](00-规范与记录/项目知识库整理规范.md) · [实验记录填写规范](00-规范与记录/实验记录填写规范.md) · [WSS 跨路线指标口径](00-规范与记录/WSS跨路线评估与横向对比口径.md) · [集群 node04 使用要点](00-规范与记录/集群node04使用要点.md) |
| 可视化落盘 | [实验分析对比图目录说明](00-规范与记录/实验分析对比图目录说明.md)（run 精度） · [点云回插面片可视化目录说明](00-规范与记录/点云回插面片可视化目录说明.md)（postview 面片云图） |
| Field 产物治理 | [预测产物保留与可再生性规范](00-规范与记录/field预测产物保留与可再生性规范.md) · [2026-08-30 已执行清理方案](02-推进与变更/_archive/outputs_field冗余审查与可再生性清理方案_已执行_2026-08-30.md) |
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
| Centerline V2 | [全队列修复与切换记录](02-推进与变更/Centerline_V2全队列修复与切换记录_2026-08-28.md) · [全量输出](../outputs/centerline_v2_full_173_20260828/README.md) · [已执行归档方案](02-推进与变更/_archive/VMTK中心线修复与全队列优化方案_已执行_2026-08-27.md) |
| WSS-only 最小化 / velocity→WSS / 体域 PINN | [根 README 主攻入口](../README.md) · [173 例数据审阅与修复计划](02-推进与变更/WSS_PINN/WSS_PINN_V4_173例训练数据数值与刚性配准审阅及修复计划_2026-08-30.md) · [V4 大重构设计](02-推进与变更/WSS_PINN/WSS_PINN_V4大重构设计方案_2026-08-08.md) · [体域 PINN 当前入口](02-推进与变更/WSS_PINN/README.md) · [体域 PINN 当前代码](../wss_pinn/README.md) · [WSS_PINN 归档索引](02-推进与变更/WSS_PINN/_archive/README.md) · [velocity→WSS V1–V4 总跟踪](../wss_mri_calculator/experiments/README.md) · [CFD 适配说明](../wss_mri_calculator/README_CFD_ADAPTATION.md) · [PointNet baseline 矩阵](02-推进与变更/WSS最小化_PointNet_baseline实验矩阵与进度跟踪.md) · [推进记录](02-推进与变更/WSS最小化_代码修改与实验推进记录.md) |
| 项目思路 | [项目思路](paper_idea/项目思路.md) · [基准模型推荐](paper_idea/基准模型推荐与引用参考.md) |
| 外部 baseline | [paper_reproduction/README](paper_reproduction/README.md) · [文献筛选总表](paper_reproduction/00-文献筛选总表.md) · [后处理插值方法](paper_reproduction/03-后处理可视化与插值方法.md) · [点云回插面片方法](paper_reproduction/05-点云预测值与真值回插到面片方法.md) · [CFD-Post 云图导出工具](../tools/cfdpost_cloud_export/README.md) |
| 外部 baseline 代码 | [PointNetCFD 复现代码](../external_baselines/pointnetcfd/README.md) |

## 5. 维护规则

- 任务 A 新实验事实优先写入 V3 实验日志或对应路线状态表，再同步推进记录。
- V3P / 训练主线 / 通用代码变更后，在 [代码修改与实验推进记录](02-推进与变更/代码修改与实验推进记录.md) 文首新增记录；`pipeline_wss_min/` 相关变更写入 [WSS 最小化代码修改与实验推进记录](02-推进与变更/WSS最小化_代码修改与实验推进记录.md)。
- **推进记录滚动切卷**：两份推进记录只保留当前活跃区间（约 1–2 个月），超过约 1500 行或跨季度时把最旧月份整月切入 `02-推进与变更/_archive/<记录名>_<区间>卷.md`，并同步文首历史卷索引。切卷只搬不改写，卷首注明链接解析规则。
- **归档约定**：结案/被取代的文档与资产就近移入所在目录的 `_archive/`（不删除、不出仓），移动时给文档加 🧊冻结 或 ⏸️暂停 标注，并更新对应 `_archive/README.md` 索引与活跃文档中的入链；归档卷内部旧链接不逐条改写。当前归档索引：[02-推进与变更/_archive](02-推进与变更/_archive/README.md) · [03-汇报材料/_archive](03-汇报材料/_archive/README.md) · [WSS_PINN/_archive](02-推进与变更/WSS_PINN/_archive/README.md)。
- `wss_mri_calculator` 新路线或冻结结论先更新 [V1–V4 总跟踪](../wss_mri_calculator/experiments/README.md)，再同步根 README、本文索引与两级推进记录。
- 数据侧或模型训练侧的集群运行代码必须提供详细进度日志，日志粒度至少细到每一个病例，确保大规模任务可判断是否仍在有效实验/推进，避免因总数据量过大造成无效等待。
- GPU 训练/评估任务只要资源有空闲即可提交使用；若暂无空闲 GPU，则按实验计划顺序提交排队，不再为是否提交或是否排队单独确认。
- 仅修改 PPT/PPTX 等汇报文件时，不需要更新推进记录；若同时修改实验文档或脚本，则仍需更新。
- 外部 baseline 复现相关内容统一写入 `paper_reproduction/`，不要混进 V3 内部优化路线。
