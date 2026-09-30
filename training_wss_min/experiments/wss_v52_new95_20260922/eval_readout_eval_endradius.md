# new-unit independent eval: frozen X5D_v51 (train136 stats), 5 seeds, Pa-mean ensemble — training_wss_min/experiments/wss_v52_new95_20260922/eval_endradius

| unit | n | true p99 Pa | seeds R² (min..max) | ensemble R² | top10 ratio | cohort cv3 folded-out R² mean (p10..p90) | verdict |
|---|---:|---:|---|---:|---:|---|---|
| AAA/ruputer/LI_BING_JIANG | 83987 | 19.3 | 0.786..0.820 | **0.842** | 0.83 | 0.715 (0.593..0.825) | in band |
| AAA/ruputer/WANG_FU_SHUN | 95286 | 40.5 | 0.619..0.674 | **0.676** | 0.73 | 0.715 (0.593..0.825) | in band |
| AAA/unruputer/GUO_BAO_CHUN | 65270 | 5.4 | 0.646..0.717 | **0.704** | 0.88 | 0.715 (0.593..0.825) | in band |
| AAA/unruputer/LIN_JIAN_RONG | 68653 | 55.2 | 0.823..0.839 | **0.850** | 0.82 | 0.715 (0.593..0.825) | in band |
| AG/slow/HE_SHU_ZHEN | 10244 | 16.0 | 0.734..0.772 | **0.782** | 0.83 | 0.744 (0.630..0.839) | in band |
| AG/slow/LIN_SHU_TIAN | 15088 | 29.3 | 0.754..0.798 | **0.818** | 0.96 | 0.744 (0.630..0.839) | in band |
| AG/slow/LIU_FENG | 13590 | 30.1 | 0.841..0.879 | **0.882** | 0.90 | 0.744 (0.630..0.839) | in band |
| AG/slow/LI_CHONG_ZENG | 15083 | 22.4 | 0.850..0.873 | **0.883** | 0.93 | 0.744 (0.630..0.839) | in band |
| AG/slow/LI_GUI_YING | 14932 | 52.3 | 0.812..0.839 | **0.855** | 0.86 | 0.744 (0.630..0.839) | in band |
| ILO/AN_GUANG_JIE-0/after | 70636 | 25.4 | 0.677..0.750 | **0.749** | 0.79 | 0.684 (0.580..0.765) | in band |
| ILO/BAO_EN_YUN-0/after | 125617 | 20.6 | 0.607..0.688 | **0.685** | 0.81 | 0.684 (0.580..0.765) | in band |
| ILO/GUO_AI_JUN-0/before | 98703 | 26.2 | 0.678..0.741 | **0.741** | 0.72 | 0.684 (0.580..0.765) | in band |
| ILO/GUO_YU_SHU-0/after | 85610 | 15.7 | 0.604..0.671 | **0.670** | 0.69 | 0.684 (0.580..0.765) | in band |
| ILO/LIU_BAO_JUN-0/before | 96848 | 13.0 | 0.780..0.807 | **0.823** | 0.83 | 0.684 (0.580..0.765) | in band |
| ILO/SUN_DE_QI-1/before | 70088 | 30.4 | 0.831..0.853 | **0.861** | 0.88 | 0.684 (0.580..0.765) | in band |
| ILO/SUN_YU_SHENG-0/before | 65815 | 15.6 | 0.814..0.826 | **0.848** | 0.90 | 0.684 (0.580..0.765) | in band |
| ILO/YANG_QING_REN-1/before | 64699 | 53.7 | 0.680..0.736 | **0.737** | 0.71 | 0.684 (0.580..0.765) | in band |
| ILO/ZHAO_JIAN_PING-0/after | 111166 | 39.0 | 0.551..0.660 | **0.619** | 0.82 | 0.684 (0.580..0.765) | in band |

18 units: case-balanced R²_cb (ensemble) = 0.8100; per-unit ensemble R² mean 0.779 / median 0.800; verdicts {'in band': 18, 'below p10 → inspect': 0, 'far below → suspect data': 0}
per cohort mean ensemble R²: AG 0.844 (n=5), AAA 0.768 (n=4), ILO 0.748 (n=9)
cv3 folded-out 136-case reference 0.7100 / test34 0.7749
