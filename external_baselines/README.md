# External Baselines

> 定位：存放外部论文 baseline 的最小复现代码。这里的代码用于在本项目私有数据上复现公开论文方法，不作为任务 A V1/V2/V3 内部训练代码的一部分。

## 当前内容

| baseline | 目录 | 状态 |
| --- | --- | --- |
| PointNetCFD | `pointnetcfd/` | 四组矩阵已完成；[梳理记录](../docs/paper_reproduction/papers/pointnetcfd/梳理记录.md) · [experiments/](pointnetcfd/experiments/README.md) |
| CROWN/Beihang | `crown_beihang/` | raw_ascii v1 已完成；5739 OOM 截断 No-Go，5740/5751 evaluate **No-Go**（非 PINN best_ep=55 · `p_r2=-5.28`）；[README](crown_beihang/README.md) |

## 实验记录与回填（2026-06-16 起）

| 粒度 | 做什么 |
| --- | --- |
| **每个 Job** | `experiments/<name>/实验分析记录.md` + `outputs/.../analysis_report.md`（训练自动） |
| **一轮矩阵跑齐** | `docs/paper_reproduction/papers/<id>/梳理记录.md`（批次总结，**非每 run**） |
| **总账** | `docs/02-推进与变更/代码修改与实验推进记录.md` 文首 1 条 |

- 实验族 / run 规范：[`外部baseline实验记录规范`](../docs/00-规范与记录/外部baseline实验记录规范.md)
- 梳理规范：[`04-梳理记录规范`](../docs/paper_reproduction/04-梳理记录规范.md)
- PointNetCFD 首轮梳理示例：[`papers/pointnetcfd/梳理记录.md`](../docs/paper_reproduction/papers/pointnetcfd/梳理记录.md)

## 使用原则

- 外部 baseline 必须使用与任务 A 一致的患者级 split。
- 第一轮先跑 paper-original 或最接近 paper-original 的输入输出，再做显式几何特征增强。
- 输出表必须和 V3 主线分开记录，避免把外部模型适配写成 V3 内部优化。
- 若外部论文只预测速度/压力，不应直接写成 WSS baseline，除非完整走通 WSS 后处理或直接 WSS head。
