# v5.2 retraining readout (checkpoint last) — wss_v52_20260923

## IND: train 170 library / test 91 recovered

| recipe | seed | R²_cb (91) | jet subset R²_cb (p99>40) | n |
|---|---|---:|---:|---:|
| X5D | 1234 | 0.7286 | 0.6712 | 91 |
| X5D | 7 | 0.7250 | 0.6675 | 91 |
| X5D | 2025 | 0.7361 | 0.7081 | 91 |
| X5D | 11 | 0.7322 | 0.6834 | 91 |
| X5D | 2026 | 0.7271 | 0.6664 | 91 |
| X5Dcap | 1234 | 0.7466 | 0.6955 | 91 |
| X5Dcap | 7 | 0.7361 | 0.6701 | 91 |
| X5Dcap | 2025 | 0.7429 | 0.6827 | 91 |
| X5Dcap | 11 | 0.7535 | 0.6927 | 91 |
| X5Dcap | 2026 | 0.7381 | 0.6547 | 91 |

| recipe | seeds in ensemble | ensemble R²_cb (91) | AG | AAA | ILO | units below cohort p10 (cv3 band) |
|---|---|---:|---:|---:|---:|---|
| X5D | [7, 11, 1234, 2025, 2026] | **0.7530** | 0.771 | 0.850 | 0.745 | 3: AG/slow/WANG_DENG_FENG 0.59, ILO/YANG_WEN_TAI-0/after 0.37, ILO/ZHANG_MAO_JIN-0/after 0.51 |
| X5Dcap | [7, 11, 1234, 2025, 2026] | **0.7655** | 0.781 | 0.865 | 0.758 | 1: AG/slow/WANG_DENG_FENG 0.59 |

IND per-unit ensemble R² delta X5Dcap − X5D: mean +0.0232, median +0.0105, units improved 60/91

reference: 0.749 — frozen X5D_v51 five-seed ensemble on the 91 recovered units (wave-2/3/4/5 evals, before the end-radius rebuild)

## CV5: 261 cases, patient-grouped 5-fold (pooled out-of-fold, every case once)

| recipe | seed | folds done | pooled OOF R²_cb | jet subset | per-fold R²_cb |
|---|---|---:|---:|---:|---|
| X5D | 1234 | 5/5 | **0.7543** | 0.7023 | 0.786 / 0.766 / 0.760 / 0.758 / 0.719 |
| X5D | 7 | 5/5 | **0.7502** | 0.6961 | 0.786 / 0.747 / 0.763 / 0.753 / 0.721 |
| X5D | 2025 | 5/5 | **0.7550** | 0.7042 | 0.785 / 0.749 / 0.755 / 0.766 / 0.733 |
| X5D | mean±sd | | **0.7532 ± 0.0026** | | n_seeds=3 |
| X5Dcap | 1234 | 5/5 | **0.7603** | 0.7010 | 0.798 / 0.762 / 0.765 / 0.767 / 0.727 |
| X5Dcap | 7 | 5/5 | **0.7623** | 0.7044 | 0.805 / 0.755 / 0.769 / 0.767 / 0.735 |
| X5Dcap | 2025 | 5/5 | **0.7613** | 0.7040 | 0.792 / 0.761 / 0.770 / 0.763 / 0.735 |
| X5Dcap | mean±sd | | **0.7613 ± 0.0010** | | n_seeds=3 |

CV5 paired fold deltas X5Dcap − X5D (15 pairs): mean +0.0085, sd 0.0064, positive 13/15

reading rule: IND compares the five-seed ensemble with the frozen 0.749; CV5 compares recipes as paired (fold, seed) deltas and reports mean ± sd over seeds; no single reading is ranked.
