# V3P 采购优先级 Batch-1（20260701）

> 口径：V3P · 257 池 · failure_cluster 物理 proxy · **非** blind pipeline · **不开 J5 GPU**

## 1. 池状态

- 257 池总量：**257**
- split_AG_v1 外候选：**171**（graphs-ready **171** · QA 通过 **171**）
- 需 pipeline（graphs 缺失）：**0**

## 2. Batch-1（near-domain fast 失败簇 proxy · 排除 J4 已用 ZHAO）

| # | case | domain | near_failure_dist | wss_p95 | graphs | QA |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 1 | `AAA/ruputer/SHI_YUN_XI` | AAA | 6.723 | 1.328 | ✅ | ✅ |
| 2 | `AAA/ruputer/LI_BING_JIANG` | AAA | 6.756 | 1.128 | ✅ | ✅ |
| 3 | `AAA/ruputer/ZUO_DAO_SHENG` | AAA | 6.824 | 1.009 | ✅ | ✅ |
| 4 | `AAA/ruputer/XIE_JIN_QUAN` | AAA | 6.846 | 1.264 | ✅ | ✅ |
| 5 | `AAA/ruputer/ZHOU_KE_XUN` | AAA | 6.908 | 1.973 | ✅ | ✅ |
| 6 | `AAA/ruputer/YANG_BAO_KUI` | AAA | 6.923 | 0.960 | ✅ | ✅ |

## 3. 判读与下一跳

- 257 池 **graphs 已齐** → 本批为 **候选深审 / 入训评估**，非 blind 257 pipeline。
- Batch-1 优先 **AAA near-domain**（距 failure 簇 wss_p95/p50/curvature 最近）；ILO 域 tier 靠后。
- 本批 QA 通过后：重跑 Phase1 邻居覆盖 → 仍须 **QA≥3 且 neighbor≥2** 才评 J5 seed1 40ep。
- **禁止**：full 257 pipeline · J5 GPU（phase1_pass 前）· full V3D 长训。
