# new-unit independent eval: frozen X5D_v51 (train136 stats), 5 seeds, Pa-mean ensemble — training_wss_min/experiments/wss_v52_new95_20260922/eval_wave4

| unit | n | true p99 Pa | seeds R² (min..max) | ensemble R² | top10 ratio | cohort cv3 folded-out R² mean (p10..p90) | verdict |
|---|---:|---:|---|---:|---:|---|---|
| ILO/LIU_CHUN_YANG-1/before | 85051 | 28.7 | 0.721..0.745 | **0.765** | 0.84 | 0.684 (0.580..0.765) | in band |
| ILO/WANG_LI_MIN-0/after | 61731 | 7.5 | 0.608..0.677 | **0.681** | 0.80 | 0.684 (0.580..0.765) | in band |
| ILO/WEI_QING_FENG-1/after | 100158 | 20.2 | 0.641..0.681 | **0.691** | 0.79 | 0.684 (0.580..0.765) | in band |

3 units: case-balanced R²_cb (ensemble) = 0.7507; per-unit ensemble R² mean 0.712 / median 0.691; verdicts {'in band': 3, 'below p10 → inspect': 0, 'far below → suspect data': 0}
per cohort mean ensemble R²: ILO 0.712 (n=3)
cv3 folded-out 136-case reference 0.7100 / test34 0.7749
