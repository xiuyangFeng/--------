# new-unit independent eval: frozen X5D_v51 (train136 stats), 5 seeds, Pa-mean ensemble — training_wss_min/experiments/wss_v52_new95_20260922/eval_wave3b

| unit | n | true p99 Pa | seeds R² (min..max) | ensemble R² | top10 ratio | cohort cv3 folded-out R² mean (p10..p90) | verdict |
|---|---:|---:|---|---:|---:|---|---|
| AG/fast/PENG_JI_MING | 14713 | 20.3 | 0.734..0.801 | **0.795** | 0.95 | 0.744 (0.630..0.839) | in band |
| ILO/AN_GUANG_JIE-0/after | 70636 | 25.4 | 0.677..0.750 | **0.749** | 0.79 | 0.684 (0.580..0.765) | in band |

2 units: case-balanced R²_cb (ensemble) = 0.7677; per-unit ensemble R² mean 0.772 / median 0.772; verdicts {'in band': 2, 'below p10 → inspect': 0, 'far below → suspect data': 0}
per cohort mean ensemble R²: AG 0.795 (n=1), ILO 0.749 (n=1)
cv3 folded-out 136-case reference 0.7100 / test34 0.7749
