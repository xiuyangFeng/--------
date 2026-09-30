# v5.2 retraining readout (checkpoint best) — wss_v52_20260923

## IND: train 170 library / test 91 recovered

| recipe | seed | R²_cb (91) | jet subset R²_cb (p99>40) | n |
|---|---|---:|---:|---:|
| X5D | 1234 | 0.7294 | 0.6739 | 91 |
| X5D | 7 | 0.7240 | 0.6653 | 91 |
| X5D | 2025 | 0.7346 | 0.7022 | 91 |
| X5D | 11 | 0.7306 | 0.6719 | 91 |
| X5D | 2026 | 0.7255 | 0.6634 | 91 |
| X5Dcap | 1234 | 0.7466 | 0.6964 | 91 |
| X5Dcap | 7 | 0.7330 | 0.6645 | 91 |
| X5Dcap | 2025 | 0.7417 | 0.6806 | 91 |
| X5Dcap | 11 | 0.7527 | 0.6899 | 91 |
| X5Dcap | 2026 | 0.7380 | 0.6545 | 91 |

| recipe | seeds in ensemble | ensemble R²_cb (91) | AG | AAA | ILO | units below cohort p10 (cv3 band) |
|---|---|---:|---:|---:|---:|---|
| X5D | [7, 11, 1234, 2025, 2026] | **0.7514** | 0.772 | 0.852 | 0.743 | 3: AG/slow/WANG_DENG_FENG 0.59, ILO/YANG_WEN_TAI-0/after 0.38, ILO/ZHANG_MAO_JIN-0/after 0.51 |
| X5Dcap | [7, 11, 1234, 2025, 2026] | **0.7644** | 0.780 | 0.865 | 0.757 | 1: AG/slow/WANG_DENG_FENG 0.59 |

IND per-unit ensemble R² delta X5Dcap − X5D: mean +0.0229, median +0.0089, units improved 61/91

reference: 0.749 — frozen X5D_v51 five-seed ensemble on the 91 recovered units (wave-2/3/4/5 evals, before the end-radius rebuild)

## CV5: 261 cases, patient-grouped 5-fold (pooled out-of-fold, every case once)

| recipe | seed | folds done | pooled OOF R²_cb | jet subset | per-fold R²_cb |
|---|---|---:|---:|---:|---|
| X5D | 1234 | 5/5 | **0.7552** | 0.7043 | 0.785 / 0.766 / 0.762 / 0.755 / 0.723 |
| X5D | 7 | 5/5 | **0.7512** | 0.6980 | 0.783 / 0.747 / 0.763 / 0.755 / 0.725 |
| X5D | 2025 | 5/5 | **0.7550** | 0.7046 | 0.785 / 0.751 / 0.756 / 0.766 / 0.730 |
| X5D | mean±sd | | **0.7538 ± 0.0022** | | n_seeds=3 |
| X5Dcap | 1234 | 5/5 | **0.7607** | 0.7020 | 0.798 / 0.762 / 0.766 / 0.766 / 0.730 |
| X5Dcap | 7 | 5/5 | **0.7627** | 0.7051 | 0.805 / 0.754 / 0.768 / 0.768 / 0.737 |
| X5Dcap | 2025 | 5/5 | **0.7589** | 0.7002 | 0.791 / 0.762 / 0.769 / 0.761 / 0.728 |
| X5Dcap | mean±sd | | **0.7608 ± 0.0019** | | n_seeds=3 |

CV5 paired fold deltas X5Dcap − X5D (15 pairs): mean +0.0075, sd 0.0073, positive 12/15

reading rule: IND compares the five-seed ensemble with the frozen 0.749; CV5 compares recipes as paired (fold, seed) deltas and reports mean ± sd over seeds; no single reading is ranked.
