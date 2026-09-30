# pilot independent eval: frozen X5D_v51 (train136 stats), 5 seeds, Pa-mean ensemble

| unit | n | true p99 Pa | seeds R² (min..max) | ensemble R² | top10 ratio | cohort cv3 folded-out R² mean (p10..p90) | verdict |
|---|---:|---:|---|---:|---:|---|---|
| AAA/unruputer/CAO_DIAN_HE | 76321 | 9.1 | 0.718..0.777 | **0.776** | 0.79 | 0.715 (0.593..0.825) | in band |
| AG/slow/ZHAO_XIU_XUAN | 13442 | 16.4 | 0.801..0.837 | **0.840** | 0.83 | 0.744 (0.630..0.839) | in band |
| ILO/AN_GUANG_JIE-0/after | 70636 | 43.4 | 0.342..0.418 | **0.391** | 0.45 | 0.684 (0.580..0.765) | far below → suspect data |
| ILO/LI_BO_JUN-1/before | 85356 | 37.3 | 0.734..0.748 | **0.766** | 0.76 | 0.684 (0.580..0.765) | in band |
| ILO/ZHAO_JIAN_PING-0/before | 56376 | 40.5 | 0.769..0.810 | **0.820** | 0.77 | 0.684 (0.580..0.765) | in band |

5-unit case-balanced R²_cb (ensemble) = 0.6705; cv3 folded-out 136-case reference 0.7100 / test34 0.7749
