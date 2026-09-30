# wss_v52_phys_20260926 结果（自动生成 2026-09-27 07:36）

单 run、已暴露 test34；参照 = 同期对照 无（同配置重跑）、matrix.json 的外部参照与历史 C1；单 seed 对单 seed 的 95% 带约 ±0.034 物理 R²_cb / ±0.009 归一化。只作筛选，不作显著性或泛化结论。

| 臂 | 变化 | ckpt | Pa R²_cb | Δ vs X0 | Δ vs C1 | norm R²_cb | Δ vs X0 | MAE Pa | case P10 | high-WSS R² | top10 比 | p99 比 | IoU | 负R² |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 历史 | 参照 | best | 0.7461 | — | — | 0.8684 | — | 1.607 | 0.6537 | 0.489 | 0.754 | 0.769 | 0.5275 | 0 |
| X5Dcap_v52ind_s1234 | 参照 | best | 0.7466 | — | +0.0005 | 0.8108 | — | 1.113 | 0.6780 | 0.521 | 0.788 | 0.826 | 0.5697 | 0 |
| X5Dcap_v52ind_s7 | 参照 | best | 0.7330 | — | -0.0131 | 0.8077 | — | 1.127 | 0.6643 | 0.504 | 0.784 | 0.820 | 0.5664 | 0 |
| X5Dcap_v52ind_s2025 | 参照 | best | 0.7417 | — | -0.0044 | 0.8128 | — | 1.109 | 0.6597 | 0.517 | 0.783 | 0.820 | 0.5713 | 0 |
| X5Dcap_womfeat_s1234 | X5Dcap + 1D Womersley prior as input features (log_tau_1d_wom, log_wom_alpha), seed 1234 | best | 0.7370 | — | -0.0091 | 0.8101 | — | 1.121 | 0.6569 | 0.534 | 0.820 | 0.879 | 0.5707 | 0 |
| X5Dcap_womfeat_s1234 | X5Dcap + 1D Womersley prior as input features (log_tau_1d_wom, log_wom_alpha), seed 1234 | last | 0.7381 | — | -0.0080 | 0.8100 | — | 1.119 | 0.6575 | 0.537 | 0.820 | 0.879 | 0.5708 | 0 |
| X5Dcap_womfeat_s7 | X5Dcap + 1D Womersley prior as input features (log_tau_1d_wom, log_wom_alpha), seed 7 | best | 0.7410 | — | -0.0052 | 0.8108 | — | 1.105 | 0.6617 | 0.517 | 0.804 | 0.843 | 0.5756 | 0 |
| X5Dcap_womfeat_s7 | X5Dcap + 1D Womersley prior as input features (log_tau_1d_wom, log_wom_alpha), seed 7 | last | 0.7415 | — | -0.0046 | 0.8106 | — | 1.105 | 0.6615 | 0.518 | 0.805 | 0.842 | 0.5754 | 0 |
| X5Dcap_womfeat_s2025 | X5Dcap + 1D Womersley prior as input features (log_tau_1d_wom, log_wom_alpha), seed 2025 | best | 0.7402 | — | -0.0059 | 0.8134 | — | 1.104 | 0.6758 | 0.521 | 0.799 | 0.840 | 0.5724 | 0 |
| X5Dcap_womfeat_s2025 | X5Dcap + 1D Womersley prior as input features (log_tau_1d_wom, log_wom_alpha), seed 2025 | last | 0.7406 | — | -0.0055 | 0.8127 | — | 1.105 | 0.6745 | 0.524 | 0.803 | 0.844 | 0.5727 | 0 |
| X5Dcap_womres_s1234 | X5Dcap + log_tau_1d_wom input + residual target ln(tau/tau_1D_wom), seed 1234 | best | 0.7511 | — | +0.0050 | 0.8116 | — | 1.112 | 0.6598 | 0.550 | 0.811 | 0.857 | 0.5731 | 0 |
| X5Dcap_womres_s1234 | X5Dcap + log_tau_1d_wom input + residual target ln(tau/tau_1D_wom), seed 1234 | last | 0.7515 | — | +0.0054 | 0.8126 | — | 1.111 | 0.6605 | 0.553 | 0.817 | 0.867 | 0.5740 | 0 |
| X5Dcap_womres_s7 | X5Dcap + log_tau_1d_wom input + residual target ln(tau/tau_1D_wom), seed 7 | best | 0.7355 | — | -0.0106 | 0.8124 | — | 1.110 | 0.6640 | 0.520 | 0.809 | 0.855 | 0.5721 | 0 |
| X5Dcap_womres_s7 | X5Dcap + log_tau_1d_wom input + residual target ln(tau/tau_1D_wom), seed 7 | last | 0.7352 | — | -0.0109 | 0.8126 | — | 1.108 | 0.6671 | 0.518 | 0.804 | 0.846 | 0.5727 | 0 |
| X5Dcap_womres_s2025 | X5Dcap + log_tau_1d_wom input + residual target ln(tau/tau_1D_wom), seed 2025 | best | 0.7402 | — | -0.0059 | 0.8141 | — | 1.106 | 0.6653 | 0.508 | 0.788 | 0.822 | 0.5732 | 0 |
| X5Dcap_womres_s2025 | X5Dcap + log_tau_1d_wom input + residual target ln(tau/tau_1D_wom), seed 2025 | last | 0.7422 | — | -0.0039 | 0.8135 | — | 1.106 | 0.6664 | 0.516 | 0.796 | 0.833 | 0.5735 | 0 |
| X5Dcap_poisres_s1234 | X5Dcap + log_tau_1d_pois input + residual target ln(tau/tau_1D_pois) (Womersley ablation), seed 1234 | best | 0.7441 | — | -0.0020 | 0.8101 | — | 1.118 | 0.6615 | 0.534 | 0.804 | 0.849 | 0.5660 | 0 |
| X5Dcap_poisres_s1234 | X5Dcap + log_tau_1d_pois input + residual target ln(tau/tau_1D_pois) (Womersley ablation), seed 1234 | last | 0.7438 | — | -0.0023 | 0.8108 | — | 1.118 | 0.6430 | 0.536 | 0.810 | 0.858 | 0.5668 | 0 |
| X5Dcap_poisres_s7 | X5Dcap + log_tau_1d_pois input + residual target ln(tau/tau_1D_pois) (Womersley ablation), seed 7 | best | 0.7426 | — | -0.0035 | 0.8131 | — | 1.107 | 0.6754 | 0.519 | 0.806 | 0.846 | 0.5744 | 0 |
| X5Dcap_poisres_s7 | X5Dcap + log_tau_1d_pois input + residual target ln(tau/tau_1D_pois) (Womersley ablation), seed 7 | last | 0.7431 | — | -0.0030 | 0.8138 | — | 1.107 | 0.6733 | 0.522 | 0.811 | 0.849 | 0.5747 | 0 |
| X5Dcap_poisres_s2025 | X5Dcap + log_tau_1d_pois input + residual target ln(tau/tau_1D_pois) (Womersley ablation), seed 2025 | best | 0.7472 | — | +0.0010 | 0.8108 | — | 1.105 | 0.6791 | 0.529 | 0.785 | 0.822 | 0.5722 | 0 |
| X5Dcap_poisres_s2025 | X5Dcap + log_tau_1d_pois input + residual target ln(tau/tau_1D_pois) (Womersley ablation), seed 2025 | last | 0.7481 | — | +0.0020 | 0.8110 | — | 1.105 | 0.6762 | 0.535 | 0.793 | 0.832 | 0.5727 | 0 |
| X5Dcap_tw_s1234 | X5Dcap + target-magnitude loss weighting (loss_weight_target, alpha 2), seed 1234 | best | 0.7514 | — | +0.0053 | 0.8039 | — | 1.113 | 0.6732 | 0.543 | 0.798 | 0.832 | 0.5752 | 0 |
| X5Dcap_tw_s1234 | X5Dcap + target-magnitude loss weighting (loss_weight_target, alpha 2), seed 1234 | last | 0.7535 | — | +0.0074 | 0.8053 | — | 1.112 | 0.6757 | 0.555 | 0.811 | 0.851 | 0.5751 | 0 |
| X5Dcap_tw_s7 | X5Dcap + target-magnitude loss weighting (loss_weight_target, alpha 2), seed 7 | best | 0.7432 | — | -0.0029 | 0.8067 | — | 1.110 | 0.6688 | 0.520 | 0.805 | 0.832 | 0.5798 | 0 |
| X5Dcap_tw_s7 | X5Dcap + target-magnitude loss weighting (loss_weight_target, alpha 2), seed 7 | last | 0.7435 | — | -0.0026 | 0.8068 | — | 1.111 | 0.6687 | 0.521 | 0.804 | 0.829 | 0.5795 | 0 |
| X5Dcap_tw_s2025 | X5Dcap + target-magnitude loss weighting (loss_weight_target, alpha 2), seed 2025 | best | 0.7413 | — | -0.0048 | 0.8090 | — | 1.105 | 0.6623 | 0.527 | 0.800 | 0.827 | 0.5785 | 0 |
| X5Dcap_tw_s2025 | X5Dcap + target-magnitude loss weighting (loss_weight_target, alpha 2), seed 2025 | last | 0.7417 | — | -0.0044 | 0.8095 | — | 1.104 | 0.6656 | 0.529 | 0.802 | 0.830 | 0.5787 | 0 |
| X5Dcap_asym2_s1234 | X5Dcap + asymmetric under-prediction loss weight 2.0, seed 1234 | best | 0.7535 | — | +0.0074 | 0.7993 | — | 1.127 | 0.6494 | 0.567 | 0.845 | 0.878 | 0.5779 | 0 |
| X5Dcap_asym2_s1234 | X5Dcap + asymmetric under-prediction loss weight 2.0, seed 1234 | last | 0.7537 | — | +0.0076 | 0.7992 | — | 1.130 | 0.6558 | 0.574 | 0.857 | 0.895 | 0.5776 | 0 |
| X5Dcap_asym2_s7 | X5Dcap + asymmetric under-prediction loss weight 2.0, seed 7 | best | 0.7533 | — | +0.0072 | 0.7998 | — | 1.130 | 0.6661 | 0.574 | 0.861 | 0.904 | 0.5797 | 0 |
| X5Dcap_asym2_s7 | X5Dcap + asymmetric under-prediction loss weight 2.0, seed 7 | last | 0.7536 | — | +0.0075 | 0.7981 | — | 1.135 | 0.6620 | 0.578 | 0.867 | 0.907 | 0.5806 | 0 |
| X5Dcap_asym2_s2025 | X5Dcap + asymmetric under-prediction loss weight 2.0, seed 2025 | best | 0.7500 | — | +0.0039 | 0.7944 | — | 1.140 | 0.6662 | 0.580 | 0.857 | 0.903 | 0.5705 | 0 |
| X5Dcap_asym2_s2025 | X5Dcap + asymmetric under-prediction loss weight 2.0, seed 2025 | last | 0.7497 | — | +0.0036 | 0.7924 | — | 1.145 | 0.6676 | 0.581 | 0.860 | 0.907 | 0.5701 | 0 |
| X5Dcap_pin95_s1234 | X5Dcap + pinball quantile 0.95 (lambda 0.2 unchanged), seed 1234 | best | 0.7444 | — | -0.0017 | 0.8128 | — | 1.103 | 0.6637 | 0.545 | 0.825 | 0.869 | 0.5726 | 0 |
| X5Dcap_pin95_s1234 | X5Dcap + pinball quantile 0.95 (lambda 0.2 unchanged), seed 1234 | last | 0.7462 | — | +0.0001 | 0.8130 | — | 1.101 | 0.6663 | 0.546 | 0.824 | 0.869 | 0.5731 | 0 |
| X5Dcap_pin95_s7 | X5Dcap + pinball quantile 0.95 (lambda 0.2 unchanged), seed 7 | best | 0.7403 | — | -0.0058 | 0.8076 | — | 1.115 | 0.6632 | 0.519 | 0.808 | 0.846 | 0.5741 | 0 |
| X5Dcap_pin95_s7 | X5Dcap + pinball quantile 0.95 (lambda 0.2 unchanged), seed 7 | last | 0.7427 | — | -0.0034 | 0.8074 | — | 1.114 | 0.6612 | 0.527 | 0.814 | 0.853 | 0.5747 | 0 |
| X5Dcap_pin95_s2025 | X5Dcap + pinball quantile 0.95 (lambda 0.2 unchanged), seed 2025 | best | 0.7386 | — | -0.0075 | 0.8092 | — | 1.118 | 0.6363 | 0.525 | 0.804 | 0.836 | 0.5705 | 0 |
| X5Dcap_pin95_s2025 | X5Dcap + pinball quantile 0.95 (lambda 0.2 unchanged), seed 2025 | last | 0.7409 | — | -0.0052 | 0.8078 | — | 1.122 | 0.6339 | 0.537 | 0.820 | 0.854 | 0.5713 | 0 |

## 分域物理 R²_cb（best）

| 臂 | AG | AAA | ILO |
|---|---:|---:|---:|
| C1 历史 | 0.7855 | 0.7160 | 0.7180 |
| X5Dcap_womfeat_s1234 | 0.7389 | 0.8308 | 0.7312 |
| X5Dcap_womfeat_s7 | 0.7562 | 0.8463 | 0.7331 |
| X5Dcap_womfeat_s2025 | 0.7548 | 0.8483 | 0.7323 |
| X5Dcap_womres_s1234 | 0.7631 | 0.8504 | 0.7439 |
| X5Dcap_womres_s7 | 0.7485 | 0.8214 | 0.7285 |
| X5Dcap_womres_s2025 | 0.7657 | 0.8509 | 0.7309 |
| X5Dcap_poisres_s1234 | 0.7466 | 0.8651 | 0.7373 |
| X5Dcap_poisres_s7 | 0.7372 | 0.8349 | 0.7378 |
| X5Dcap_poisres_s2025 | 0.7545 | 0.8149 | 0.7416 |
| X5Dcap_tw_s1234 | 0.7579 | 0.8556 | 0.7448 |
| X5Dcap_tw_s7 | 0.7662 | 0.8393 | 0.7348 |
| X5Dcap_tw_s2025 | 0.7497 | 0.8536 | 0.7341 |
| X5Dcap_asym2_s1234 | 0.7581 | 0.8463 | 0.7475 |
| X5Dcap_asym2_s7 | 0.7608 | 0.8275 | 0.7476 |
| X5Dcap_asym2_s2025 | 0.7389 | 0.8243 | 0.7465 |
| X5Dcap_pin95_s1234 | 0.7297 | 0.8489 | 0.7403 |
| X5Dcap_pin95_s7 | 0.7432 | 0.8548 | 0.7337 |
| X5Dcap_pin95_s2025 | 0.7630 | 0.8260 | 0.7303 |

队列状态：complete；作业 15709；跳过的候选：
