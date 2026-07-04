# V3P 平台期收口 · G5 叙事与主动选数

> 口径：V3P · `split_AG_v1` · post5463 band **0.425±0.012** · I6-diag **0.429**  
> 用途：平台期 **M-N 叙事**、**M-E 数据采购**、Phase1 主动选数审计的**专项入口**；日常执行仍以 [待办](../V3_后续优化待办.md) + [实验跟踪](../V3_实验执行跟踪日志.md) 为准。

---

## 先看哪一份

| 目标 | 文档 | 说明 |
| --- | --- | --- |
| **论文 / 汇报 / 对外叙事（M-N）** | [G5_主叙事与平台期结论.md](G5_主叙事与平台期结论.md) | G0–G4 回顾 · J4/J5 判读 · §7 收口段落 |
| **fast/CHEN 失败机制** | [失败病例档案_CHEN_FAN.md](失败病例档案_CHEN_FAN.md) | 平台期典型 fast 域 WSS 弱例 |
| **当前 Phase1 审计（最新）** | [主动选数清单.md](主动选数清单.md) | v5 · QA **13** · neighbor **5/5** · `phase1_pass` ✅ |
| **257 池采购批次** | [采购_batch1.md](采购_batch1.md) · [采购_batch2.md](采购_batch2.md) | 各 6 例 · failure proxy 排名 · CPU 深审 |
| **继续想 M-E 下一跳** | [平台期复盘](../V3P_精度平台期复盘与下一轮想法_2026-06-30.md) | Ideas A–F · 不在本目录 |

---

## 文档关系

```text
G5_主叙事（总结论）
    ├── 失败病例档案 CHEN（fast 域缩影）
    ├── 主动选数清单 v5（Phase1 门禁 · 当前）
    │       └── _history/ v2→v3→v4（审计迭代，默认不打开）
    └── 采购 batch1 → batch2（257 池小批候选）
```

**主动选数版本线**（审计递增，非重复结论）：

| 版本 | 文件 | 关键变化 |
| --- | --- | --- |
| v2 | [_history/主动选数_v2_20260630.md](_history/主动选数_v2_20260630.md) | J4 No-Go 后初版 · 未含 WSS QA |
| v3 | [_history/主动选数_v3_20260701.md](_history/主动选数_v3_20260701.md) | +QA · AG 内仅 1 例可入训 · 引出 Batch-1 |
| v4 | [_history/主动选数_v4_20260701.md](_history/主动选数_v4_20260701.md) | Batch-1 并入 · `phase1_pass` ✅ → 立项 J5 |
| **v5（当前）** | [主动选数清单.md](主动选数清单.md) | +Batch-2 · QA **13** · J5 弱 No-Go 后 J6 封口 |

---

## 关联产物（代码侧）

| 类型 | 路径 |
| --- | --- |
| 采购排名 JSON | `outputs/field/f0_decision/v3p_procurement_rank_*.json` |
| 采购批次 txt | `outputs/field/f0_decision/v3p_procurement_batch*.txt` |
| Phase1 选数 JSON | `outputs/field/f0_decision/v3p_active_selection_v*.json` |
| split 草案 | `split_AG_active_v*.json` |

脚本输出目录：**本目录**（`run_v3p_phase1_active_data.py` · `run_v3p_pipeline_procurement_rank.py`）。

---

## 当前状态（2026-07-01）

- **G5 叙事**：✅ 收口（§7 含 J4/J5/L3）
- **J5 ActiveData**（+6 AAA）：弱 No-Go **0.433** · **J6 GPU 封口**
- **Phase1 v5**：`phase1_pass` ✅ · split 草案 `split_AG_active_v3.json`
- **下一跳（M-E）**：Batch-3 CPU 采购或 Ideas A–F 改造 · 见 [V3 README](../../README.md) §0
