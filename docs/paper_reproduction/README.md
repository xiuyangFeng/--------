# 外部论文 baseline 复现库

> 创建日期：2026-06-14
> 入口同步：2026-09-15
> 定位：独立于内部训练路线的外部论文复现记录库，记录公开方法在私有 AAA / WSS 数据上的适配设计、实际结果与比较边界。当前实验策略见[实验设计总纲](../实验设计总纲.md)，论文所需证据见[论文规划](../paper_idea/论文规划_几何驱动WSS与部署鲁棒性_2026-09-15.md)。

**当前状态**：PointNetCFD-style 四组是旧 AG 数据、`split_AG_v1`、seed1 的方法适配，已完成；CROWN 非 PINN / PINN 两臂已完成评估并结案。当前 V5 的直接 WSS 强外部 baseline **仍待按同一数据、输入、选模和评估协议建立**，文献候选不等于已完成对照。

## 1. 为什么单独建这个目录

本目录建立时（2026-06），历史 V3 路线围绕 PointNeXt、WSS 直接监督、速度辅助、SSL 预训练和 2D 壁面展开做过多轮验证，点级 WSS 精度主要停在 `wss_r2_wss ~0.40-0.43` 带宽。该数字仅说明历史背景，不是当前 V5 的精度，也不能与 V5 的物理量指标直接比较。本目录继续承担以下职责：

- 用公开论文模型在本项目私有数据上跑出可报告 baseline；
- 判断 mesh、point-cloud、等变网络、2D 壁面展开、物理约束和 masked pretraining 哪类路线最值得继续；
- 为论文实验表提供审稿人可接受的外部对照；
- 将外部方法适配与内部消融分别记录；只有比较合同一致后，才进入同一论文结果表。

## 2. 目录结构

```text
docs/paper_reproduction/
├── README.md
├── 00-文献筛选总表.md
├── 01-复现优先级与适配策略.md
├── 02-私有数据适配统一口径.md
├── 03-后处理可视化与插值方法.md
├── 04-梳理记录规范.md          ← 每轮实验矩阵完成后的梳理总结（非每 run）
├── 05-点云预测值与真值回插到面片方法.md
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

1. [00-文献筛选总表](00-文献筛选总表.md)：2026-06 的文献与代码核查；正式适配前需更新来源和仓库可用性。
2. [01-复现优先级与适配策略](01-复现优先级与适配策略.md)：历史 P0/P1/P2 候选顺序；当前取舍以总纲和论文规划为准。
3. [02-私有数据适配统一口径](02-私有数据适配统一口径.md)：通用输入、输出与指标边界；其中 V3 split 是历史合同，V5 对照需另行固定当前病例清单、峰值目标、真实单位和选模协议。
4. [03-后处理可视化与插值方法](03-后处理可视化与插值方法.md)：预测点云如何回映射到面片/体网格，以及如何避免插值平滑造成误判。
5. [05-点云预测值与真值回插到面片方法](05-点云预测值与真值回插到面片方法.md)：专门看 `wss_pred/wss_cfd/p_pred/p_cfd` 如何从点云插值到 STL/VTK 面片、如何记录参数和覆盖率。
6. [04-梳理记录规范](04-梳理记录规范.md)：**一轮实验矩阵全部跑完后**如何写梳理总结（非每个 Job 一条）。
7. `papers/<paper_id>/README.md` + `papers/<paper_id>/梳理记录.md`：方法说明与批次梳理结论。

## 4. 历史候选清单与实际完成状态

下表保留 2026-06 文献筛选的优先级。代码成熟度等判断属于当时核查，不代表已完成本项目适配；当前峰值 WSS 任务不能直接以 TAWSS 或全周期流场结果替代比较。

| 优先级 | 论文/模型 | 复现目标 | 当前判断 |
| --- | --- | --- | --- |
| P0 | E(3)-equivariant AAA WSS | 直接对齐 AAA transient WSS | WSS 强对照候选；代码与峰值任务接口待核查 |
| P0 | MultiViewUNet AAA TAWSS | 2D 展开 + TAWSS 非图 baseline | 历史 G4 / 任务 B 候选；需明确目标适配，不能直接作为峰值 WSS 同任务对照 |
| P0 | Coronary mesh convolution / SE(3) hemodynamics | artery wall mesh WSS vector | 代码成熟，虽非 AAA 但任务结构很接近 |
| P0 | LaB-GATr | biomedical surface/volume mesh geometric algebra transformer | 可作为等变 AAA WSS 的底座候选 |
| P0 | PointNetCFD / PointNet-style | 不建图点云 CFD baseline | **旧 AG 四组、seed1 已完成**；自写 adapter + 简化 MLP 的 style 适配，非原文 benchmark 数值复现；非 V5 同协议 WSS 强对照 · [梳理记录](papers/pointnetcfd/梳理记录.md) |
| P1 | PINGS | PointNet++ / GNN + physics-informed flow field | 医学血管、4D flow MRI，参考实现价值高 |
| P1 | MeshGraphNet | 经典 mesh EPD baseline | 需真实 mesh 拓扑或严格转换层 |
| P1 | AneuG-Flow / IA WSS benchmark | 颅内动脉瘤合成 CFD / WSS benchmark | 数据和任务有价值，但 IA 与 AAA 需分开叙事 |
| P1 | Lannelongue et al. PC-GNN | 颅内动脉瘤 transient hemodynamics | 数据集公开，代码待进一步确认 |
| P2 | MeshMask | masked GNN pretraining | SOTA 思路强，但先作为策略复现或二阶段增强 |
| P2 | cerebrovascular PointNet PINN / CROWN-Beihang | 小样本点云 + PINN v/p | **raw_ascii v1 非 PINN / PINN 两臂已完成并结案（No-Go）**；仅输出 `u,v,w,p`，不能直接写成 WSS baseline · [合并汇报](../../external_baselines/crown_beihang/experiments/CROWN_非PINN与PINN复现汇报_合并.md) |

## 5. 记录规则

- 每篇论文一个文件夹，至少包含 **`README.md`**（方法 + **本轮计划实验矩阵**）与 **`梳理记录.md`**（矩阵跑齐后填写；未完成前可只保留 `_template` 占位）。
- **梳理记录**：每完成 **一轮计划实验矩阵** 更新一次（例如 PointNetCFD 四组齐 → 写 1 篇批次梳理）；**不要**每个 Job / 每个 run 改 `梳理记录.md`。规范见 [04-梳理记录规范](04-梳理记录规范.md)。
- 单次 run 指标仍走 `external_baselines/<baseline>/experiments/` 与 `outputs/.../analysis_report.md`（见 [外部baseline实验记录规范](../00-规范与记录/外部baseline实验记录规范.md)）。
- 不在本目录内粘贴大段论文原文，只保留复现相关事实、链接和适配判断。
- 若后续真正修改外部代码或新增适配脚本，应另建 `external_baselines/` 或等价代码目录，本目录记录设计与 **梳理结论**。
- 每完成一轮矩阵并写好 `梳理记录.md` 后，同步更新 `docs/02-推进与变更/代码修改与实验推进记录.md` 文首（含梳理链接）。
