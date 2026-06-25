# External Baselines

> 定位：外部论文 baseline 的**最小复现代码**与**实验记录**。用于在本项目私有数据上复现公开方法；**不是**任务 A V1/V2/V3 内部训练代码。
> **原则**：指标与 V3P/V3D **分表**；可视化对照（如 postview）**不得**写入 V3 母版表。

---

## 1. 当前 baseline 一览

| baseline | 代码目录 | 结论（2026-06-25） | 主文档 |
| --- | --- | --- | --- |
| **PointNetCFD** | [pointnetcfd/](pointnetcfd/) | 四组矩阵 **已完成** | [paper 梳理记录](../docs/paper_reproduction/papers/pointnetcfd/梳理记录.md) |
| **CROWN/Beihang** | [crown_beihang/](crown_beihang/) | 非 PINN **5751/5774** · PINN **5757/5806** · 均 **No-Go**（PINN p NMAE 13.2% · 较非 PINN 16.0% 改善） | [合并汇报](crown_beihang/experiments/CROWN_非PINN与PINN复现汇报_合并.md) |

---

## 2. 与 V3P 的关系（讨论 M-E 时必读）

| 维度 | V3P 母版（I6-diag） | CROWN VP（5751） |
| --- | --- | --- |
| 输入 | 几何 + BC + 双域点云 | 仅 `(x,y,z)` |
| 压力 R²_p | **~0.93** | **深负**（No-Go） |
| WSS | direct head **0.429** | **无** |
| 工程角色 | M-E 主战场 | **不可**作压力/WSS 替代；速度 NMAE 可作 E-B 讨论参考 |

并排可视化（非训练指标）：[V3P_vs_CROWN_对照表_1146.md](../outputs/field/postview/V3P_vs_CROWN_对照表_1146.md) · [PINN 对照表](../outputs/field/postview/V3P_vs_CROWN_PINN_对照表_1146.md)

---

## 3. 文档该读哪份（避免重复打开）

### 总规范

- [外部 baseline 实验记录规范](../docs/00-规范与记录/外部baseline实验记录规范.md)
- [paper_reproduction 梳理规范](../docs/paper_reproduction/04-梳理记录规范.md)

### PointNetCFD

| 文档 | 用途 |
| --- | --- |
| [pointnetcfd/README.md](pointnetcfd/README.md) | 代码入口 |
| [experiments/README.md](pointnetcfd/experiments/README.md) | 实验族索引 |
| [papers/pointnetcfd/梳理记录.md](../docs/paper_reproduction/papers/pointnetcfd/梳理记录.md) | 批次总结 |

### CROWN/Beihang

| 文档 | 用途 |
| --- | --- |
| [crown_beihang/README.md](crown_beihang/README.md) | **主入口**（口径、训练、当前状态） |
| [crown_beihang/docs/README.md](crown_beihang/docs/README.md) | 专项说明索引 |
| [experiments/README.md](crown_beihang/experiments/README.md) | 实验族 ↔ Job |
| [CROWN_非PINN与PINN复现汇报_合并.md](crown_beihang/experiments/CROWN_非PINN与PINN复现汇报_合并.md) | **导师汇报唯一入口**（5751/5774 + 5757/5806 · 1146 可视化） |
| [5739_5740_根因诊断.md](crown_beihang/docs/5739_5740_根因诊断.md) | OOM / TIMEOUT 工程记录 |

**不必通读**：各 Job 的 `analysis_report.md`（训练自动生成）除非查单次 run 细节。

---

## 4. 实验记录回填

| 粒度 | 写到哪里 |
| --- | --- |
| 每个 Job | `experiments/<name>/实验分析记录.md` |
| 一轮矩阵 | `docs/paper_reproduction/papers/<id>/梳理记录.md` |
| 总账 | `docs/02-推进与变更/代码修改与实验推进记录.md` 文首 |

---

## 5. 使用原则

- 必须使用与任务 A 一致的**患者级 split**。
- 第一轮先跑 paper-original，再做显式几何增强。
- 外部论文只预测 u,v,w,p 时，**不得**直接标为 WSS baseline，除非走通 WSS 后处理或独立 WSS head。

---

## 变更记录

| 日期 | 内容 |
| --- | --- |
| 2026-06-25 | 收紧导航：V3P 对照表、M-E 边界、CROWN 文档分层 |
| 2026-06-16 | 实验记录与回填规范 |
