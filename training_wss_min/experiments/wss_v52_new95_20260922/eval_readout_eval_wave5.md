# new-unit independent eval: frozen X5D_v51 (train136 stats), 5 seeds, Pa-mean ensemble — training_wss_min/experiments/wss_v52_new95_20260922/eval_wave5

| unit | n | true p99 Pa | seeds R² (min..max) | ensemble R² | top10 ratio | cohort cv3 folded-out R² mean (p10..p90) | verdict |
|---|---:|---:|---|---:|---:|---|---|
| ILO/WANG_LI_MIN-0/before | 54987 | 9.7 | 0.637..0.700 | **0.700** | 0.81 | 0.684 (0.580..0.765) | in band |

1 units: case-balanced R²_cb (ensemble) = 0.6997; per-unit ensemble R² mean 0.700 / median 0.700; verdicts {'in band': 1, 'below p10 → inspect': 0, 'far below → suspect data': 0}
per cohort mean ensemble R²: ILO 0.700 (n=1)
cv3 folded-out 136-case reference 0.7100 / test34 0.7749
