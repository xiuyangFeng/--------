# wss_cycle_m1_20260921 结果（自动生成 2026-09-22 01:51）

单 run、已暴露 test34；参照 = 同期对照 无（同配置重跑）、matrix.json 的外部参照与历史 C1；单 seed 对单 seed 的 95% 带约 ±0.034 物理 R²_cb / ±0.009 归一化。只作筛选，不作显著性或泛化结论。

| 臂 | 变化 | ckpt | Pa R²_cb | Δ vs X0 | Δ vs C1 | norm R²_cb | Δ vs X0 | MAE Pa | case P10 | high-WSS R² | top10 比 | p99 比 | IoU | 负R² |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 历史 | 参照 | best | 0.6575 | — | — | 0.8410 | — | 1.845 | 0.5050 | 0.300 | 0.639 | 0.624 | 0.4698 | 0 |
| X5D_v51_f0_s1234 (wave 2a, peak-frame base) | 参照 | best | 0.7366 | — | +0.0791 | 0.8556 | — | 1.455 | 0.6431 | 0.478 | 0.747 | 0.770 | 0.5083 | 0 |
| X5D_v51_f1_s1234 (wave 2a, peak-frame base) | 参照 | best | 0.6701 | — | +0.0127 | 0.8537 | — | 1.664 | 0.5698 | 0.305 | 0.686 | 0.712 | 0.5049 | 0 |
| X5D_v51_f2_s1234 (wave 2a, peak-frame base) | 参照 | best | 0.7323 | — | +0.0748 | 0.8514 | — | 1.410 | 0.6109 | 0.456 | 0.788 | 0.803 | 0.5251 | 0 |
| A1_f0_s1234 (stage 1 TAWSS single head) | 参照 | best | 0.7439 | — | — | 0.8531 | — | 0.376 | 0.6226 | 0.527 | 0.772 | 0.782 | 0.5071 | 0 |
| A1_f1_s1234 (stage 1 TAWSS single head) | 参照 | best | 0.6542 | — | — | 0.8626 | — | 0.439 | 0.5220 | 0.337 | 0.691 | 0.715 | 0.5005 | 0 |
| A1_f2_s1234 (stage 1 TAWSS single head) | 参照 | best | 0.7504 | — | — | 0.8572 | — | 0.362 | 0.6228 | 0.521 | 0.801 | 0.818 | 0.5295 | 0 |
| O2_f0_s1234 (stage 1 OSI logit single head) | 参照 | best | 0.4327 | — | — | 0.4893 | — | 0.067 | 0.1954 | -12.729 | 0.623 | 0.833 | 0.2516 | 0 |
| O2_f1_s1234 (stage 1 OSI logit single head) | 参照 | best | 0.4749 | — | — | 0.5302 | — | 0.065 | 0.2588 | -12.451 | 0.621 | 0.825 | 0.2732 | 1 |
| O2_f2_s1234 (stage 1 OSI logit single head) | 参照 | best | 0.4643 | — | — | 0.5228 | — | 0.065 | 0.3184 | -12.759 | 0.641 | 0.862 | 0.2529 | 0 |
| M1_f0_s1234 | X5D_v51 fold-0 recipe with a three-channel head (out_dim 3 = peak-frame WSS log_z, TAWSS log_z, OSI logit_z | best | 0.7032 | — | +0.0457 | 0.8491 | — | 1.518 | 0.6182 | 0.409 | 0.698 | 0.736 | 0.4896 | 0 |
| M1_f0_s1234 | X5D_v51 fold-0 recipe with a three-channel head (out_dim 3 = peak-frame WSS log_z, TAWSS log_z, OSI logit_z | last | 0.7016 | — | +0.0441 | 0.8491 | — | 1.519 | 0.6141 | 0.402 | 0.692 | 0.729 | 0.4888 | 0 |
| M1_f1_s1234 | X5D_v51 fold-1 recipe with a three-channel head (out_dim 3 = peak-frame WSS log_z, TAWSS log_z, OSI logit_z | best | 0.6599 | — | +0.0024 | 0.8525 | — | 1.710 | 0.5387 | 0.292 | 0.634 | 0.655 | 0.4930 | 0 |
| M1_f1_s1234 | X5D_v51 fold-1 recipe with a three-channel head (out_dim 3 = peak-frame WSS log_z, TAWSS log_z, OSI logit_z | last | 0.6629 | — | +0.0054 | 0.8528 | — | 1.704 | 0.5342 | 0.299 | 0.640 | 0.659 | 0.4924 | 0 |
| M1_f2_s1234 | X5D_v51 fold-2 recipe with a three-channel head (out_dim 3 = peak-frame WSS log_z, TAWSS log_z, OSI logit_z | best | 0.7244 | — | +0.0669 | 0.8459 | — | 1.458 | 0.6000 | 0.406 | 0.735 | 0.765 | 0.5154 | 0 |
| M1_f2_s1234 | X5D_v51 fold-2 recipe with a three-channel head (out_dim 3 = peak-frame WSS log_z, TAWSS log_z, OSI logit_z | last | 0.7236 | — | +0.0661 | 0.8460 | — | 1.457 | 0.6031 | 0.400 | 0.734 | 0.761 | 0.5143 | 0 |

## 分域物理 R²_cb（best）

| 臂 | AG | AAA | ILO |
|---|---:|---:|---:|
| C1 历史 | 0.6937 | 0.6342 | 0.6315 |
| M1_f0_s1234 | 0.7238 | 0.6706 | 0.7419 |
| M1_f1_s1234 | 0.7295 | 0.7761 | 0.5368 |
| M1_f2_s1234 | 0.7334 | 0.7639 | 0.6518 |

队列状态：complete；作业 15452；跳过的候选：
