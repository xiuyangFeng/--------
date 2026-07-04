# V3P 采购优先级 BATCH2（20260701）

> 口径：V3P · 257 池 · failure_cluster 物理 proxy · **非** blind pipeline · **不开 J5 GPU**

## 1. 池状态

- 257 池总量：**257**
- split_AG_v1 外候选：**165**（graphs-ready **165** · QA 通过 **165**）
- 需 pipeline（graphs 缺失）：**0**

## 2. BATCH2（near-domain fast 失败簇 proxy · 排除已采购批次）

| # | case | domain | near_failure_dist | wss_p95 | graphs | QA |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 1 | `AAA/unruputer/LI_YOU_YU` | AAA | 6.881 | 0.943 | ✅ | ✅ |
| 2 | `AAA/unruputer/LIU_JIE` | AAA | 6.887 | 1.744 | ✅ | ✅ |
| 3 | `AAA/ruputer/LIN_LIANG_XIAO` | AAA | 6.993 | 1.636 | ✅ | ✅ |
| 4 | `ILO/XIE_ZHI_FU-0/before` | ILO | 6.799 | 1.450 | ✅ | ✅ |
| 5 | `ILO/LI_FA_XIANG-1/after` | ILO | 6.800 | 1.291 | ✅ | ✅ |
| 6 | `ILO/WEI_QING_FENG-1/after` | ILO | 6.824 | 1.168 | ✅ | ✅ |

## 3. 判读与下一跳

- 257 池 **graphs 已齐** → 本批为 **候选深审 / 入训评估**，非 blind 257 pipeline。
- 优先 **AAA near-domain**（距 failure 簇 wss_p95/p50/curvature 最近）；ILO 域 tier 靠后。
- 本批 QA 通过后：重跑 Phase1 v5 邻居覆盖；**J5/J6 GPU 已封口**，仅评 phase1_pass 与 split 草案。
- **禁止**：full 257 pipeline · J6 GPU · full V3D 长训。
