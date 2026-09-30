# new-unit independent eval: frozen X5D_v51 (train136 stats), 5 seeds, Pa-mean ensemble — training_wss_min/experiments/wss_v52_new95_20260922/eval_wave3

| unit | n | true p99 Pa | seeds R² (min..max) | ensemble R² | top10 ratio | cohort cv3 folded-out R² mean (p10..p90) | verdict |
|---|---:|---:|---|---:|---:|---|---|
| AG/slow/NIE_QUAN_ZHONG | 8199 | 35.4 | 0.582..0.626 | **0.629** | 0.71 | 0.744 (0.630..0.839) | below p10 → inspect |
| ILO/GUO_AI_JUN-0/after | 110471 | 34.4 | 0.683..0.775 | **0.752** | 0.71 | 0.684 (0.580..0.765) | in band |
| ILO/GUO_AI_JUN-0/before | 98703 | 26.2 | 0.678..0.741 | **0.741** | 0.72 | 0.684 (0.580..0.765) | in band |
| ILO/GUO_QING_SHAN-0/after | 72704 | 28.4 | 0.730..0.759 | **0.770** | 0.76 | 0.684 (0.580..0.765) | in band |
| ILO/LI_FA_XIANG-1/after | 96907 | 20.8 | 0.708..0.755 | **0.761** | 0.76 | 0.684 (0.580..0.765) | in band |
| ILO/LI_YU_GANG-0/after | 167056 | 16.8 | 0.652..0.688 | **0.695** | 0.73 | 0.684 (0.580..0.765) | in band |
| ILO/SHEN_CHUN_WANG-0/before | 63955 | 18.3 | 0.680..0.740 | **0.752** | 0.82 | 0.684 (0.580..0.765) | in band |
| ILO/WANG_SHU_SHENG-0/after | 82693 | 29.6 | 0.615..0.720 | **0.719** | 0.80 | 0.684 (0.580..0.765) | in band |
| ILO/XUE_YOU_TANG-0/before | 55116 | 82.0 | 0.542..0.623 | **0.606** | 0.51 | 0.684 (0.580..0.765) | in band |
| ILO/ZHANG_MAO_JIN-0/after | 95060 | 13.0 | 0.390..0.573 | **0.541** | 0.78 | 0.684 (0.580..0.765) | below p10 → inspect |
| ILO/ZHANG_WAN_ZENG-1/before | 145530 | 68.2 | 0.557..0.638 | **0.615** | 0.58 | 0.684 (0.580..0.765) | in band |

11 units: case-balanced R²_cb (ensemble) = 0.6761; per-unit ensemble R² mean 0.689 / median 0.719; verdicts {'in band': 9, 'below p10 → inspect': 2, 'far below → suspect data': 0}
per cohort mean ensemble R²: AG 0.629 (n=1), ILO 0.695 (n=10)
cv3 folded-out 136-case reference 0.7100 / test34 0.7749
