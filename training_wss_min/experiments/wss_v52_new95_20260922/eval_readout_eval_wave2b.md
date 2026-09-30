# new-unit independent eval: frozen X5D_v51 (train136 stats), 5 seeds, Pa-mean ensemble — training_wss_min/experiments/wss_v52_new95_20260922/eval_wave2b

| unit | n | true p99 Pa | seeds R² (min..max) | ensemble R² | top10 ratio | cohort cv3 folded-out R² mean (p10..p90) | verdict |
|---|---:|---:|---|---:|---:|---|---|
| ILO/SUN_XU_XIA-1/after | 89708 | 64.5 | 0.700..0.773 | **0.748** | 0.67 | 0.684 (0.580..0.765) | in band |

1 units: case-balanced R²_cb (ensemble) = 0.7480; per-unit ensemble R² mean 0.748 / median 0.748; verdicts {'in band': 1, 'below p10 → inspect': 0, 'far below → suspect data': 0}
per cohort mean ensemble R²: ILO 0.748 (n=1)
cv3 folded-out 136-case reference 0.7100 / test34 0.7749
