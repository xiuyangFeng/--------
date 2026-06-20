# 外部论文 baseline 复现库

> 创建日期：2026-06-14
> 定位：独立于任务 A V1/V2/V3 训练代码的外部论文复现记录库，用于把公开模型或论文架构适配到本项目私有 AAA / WSS 数据集前，先完成可复现性、数据口径和工程风险梳理。

## 1. 为什么单独建这个目录

当前任务 A 的 V3 路线已经围绕 PointNeXt、WSS 直接监督、速度辅助、SSL 预训练和 2D 壁面展开做过多轮验证，点级 WSS 精度仍主要停在 `wss_r2_wss ~0.40-0.43` 带宽。下一阶段需要从“内部小改”切换到“外部模型家族 baseline 竞争”：

- 用公开论文模型在本项目私有数据上跑出可报告 baseline；
- 判断 mesh、point-cloud、等变网络、2D 壁面展开、物理约束和 masked pretraining 哪类路线最值得继续；
- 为论文实验表提供审稿人可接受的外部对照；
- 避免把外部复现实验混入 V3 当前路线，造成口径混表。

## 2. 目录结构

```text
docs/paper_reproduction/
├── README.md
├── 00-文献筛选总表.md
├── 01-复现优先级与适配策略.md
├── 02-私有数据适配统一口径.md
├── 03-后处理可视化与插值方法.md
├── 04-梳理记录规范.md          ← 每轮实验矩阵完成后的梳理总结（非每 run）
└── papers/
    ├── _template/              ← 梳理记录模板
    ├── meshgraphnet/
    ├── meshmask/
    ├── pings/
    ├── pointnetcfd/
    ├── e3_equivariant_aaa_wss/
    ├── coronary_mesh_convolution/
    ├── lab_gatr/
    ├── aneug_flow/
    ├── multiviewunet_aaa_tawss/
    ├── lannelongue_intracranial_pcgcn/
    └── hemodynamics_pointcloud_pinn/
```

## 3. 阅读顺序

1. [00-文献筛选总表](00-文献筛选总表.md)：先看哪些论文值得复现，哪些只适合引用。
2. [01-复现优先级与适配策略](01-复现优先级与适配策略.md)：看 P0/P1/P2 的执行顺序。
3. [02-私有数据适配统一口径](02-私有数据适配统一口径.md)：所有外部代码适配本项目数据前必须遵守的输入、输出、split、指标口径。
4. [03-后处理可视化与插值方法](03-后处理可视化与插值方法.md)：预测点云如何回映射到面片/体网格，以及如何避免插值平滑造成误判。
5. [04-梳理记录规范](04-梳理记录规范.md)：**一轮实验矩阵全部跑完后**如何写梳理总结（非每个 Job 一条）。
6. `papers/<paper_id>/README.md` + `papers/<paper_id>/梳理记录.md`：方法说明与批次梳理结论。

## 4. 当前推荐的第一批 baseline

| 优先级 | 论文/模型 | 复现目标 | 当前判断 |
| --- | --- | --- | --- |
| P0 | E(3)-equivariant AAA WSS | 直接对齐 AAA transient WSS | 最接近本项目终点，优先核查代码与数据接口 |
| P0 | MultiViewUNet AAA TAWSS | 2D 展开 + TAWSS 非图 baseline | 可作为 G4 换轨和任务 B 的强对照 |
| P0 | Coronary mesh convolution / SE(3) hemodynamics | artery wall mesh WSS vector | 代码成熟，虽非 AAA 但任务结构很接近 |
| P0 | LaB-GATr | biomedical surface/volume mesh geometric algebra transformer | 可作为等变 AAA WSS 的底座候选 |
| P0 | PointNetCFD / PointNet-style | 不建图点云 CFD baseline | **首轮矩阵已完成** · [梳理记录](papers/pointnetcfd/梳理记录.md) |
| P1 | PINGS | PointNet++ / GNN + physics-informed flow field | 医学血管、4D flow MRI，参考实现价值高 |
| P1 | MeshGraphNet | 经典 mesh EPD baseline | 需真实 mesh 拓扑或严格转换层 |
| P1 | AneuG-Flow / IA WSS benchmark | 颅内动脉瘤合成 CFD / WSS benchmark | 数据和任务有价值，但 IA 与 AAA 需分开叙事 |
| P1 | Lannelongue et al. PC-GNN | 颅内动脉瘤 transient hemodynamics | 数据集公开，代码待进一步确认 |
| P2 | MeshMask | masked GNN pretraining | SOTA 思路强，但先作为策略复现或二阶段增强 |
| P2 | cerebrovascular PointNet PINN / CROWN-Beihang | 小样本点云 + PINN v/p | CROWN raw_ascii v1 适配进行中；仅作为 `u,v,w,p` 速度/压力复现线，不能直接写成 WSS baseline |

## 5. 记录规则

- 每篇论文一个文件夹，至少包含 **`README.md`**（方法 + **本轮计划实验矩阵**）与 **`梳理记录.md`**（矩阵跑齐后填写；未完成前可只保留 `_template` 占位）。
- **梳理记录**：每完成 **一轮计划实验矩阵** 更新一次（例如 PointNetCFD 四组齐 → 写 1 篇批次梳理）；**不要**每个 Job / 每个 run 改 `梳理记录.md`。规范见 [04-梳理记录规范](04-梳理记录规范.md)。
- 单次 run 指标仍走 `external_baselines/<baseline>/experiments/` 与 `outputs/.../analysis_report.md`（见 [外部baseline实验记录规范](../00-规范与记录/外部baseline实验记录规范.md)）。
- 不在本目录内粘贴大段论文原文，只保留复现相关事实、链接和适配判断。
- 若后续真正修改外部代码或新增适配脚本，应另建 `external_baselines/` 或等价代码目录，本目录记录设计与 **梳理结论**。
- 每完成一轮矩阵并写好 `梳理记录.md` 后，同步更新 `docs/02-推进与变更/代码修改与实验推进记录.md` 文首（含梳理链接）。
