# wss_cycle_m1_stage3_20260922 结果（自动生成 2026-09-22 16:10）

单 run、已暴露 test34；参照 = 同期对照 无（同配置重跑）、matrix.json 的外部参照与历史 C1；单 seed 对单 seed 的 95% 带约 ±0.034 物理 R²_cb / ±0.009 归一化。只作筛选，不作显著性或泛化结论。

| 臂 | 变化 | ckpt | Pa R²_cb | Δ vs X0 | Δ vs C1 | norm R²_cb | Δ vs X0 | MAE Pa | case P10 | high-WSS R² | top10 比 | p99 比 | IoU | 负R² |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 历史 | 参照 | best | 0.6575 | — | — | 0.8410 | — | 1.845 | 0.5050 | 0.300 | 0.639 | 0.624 | 0.4698 | 0 |
| X5D_v51_s1234 (deployed peak-frame, same seed) | 参照 | best | 0.7461 | — | +0.0886 | 0.8684 | — | 1.607 | 0.6537 | 0.489 | 0.754 | 0.769 | 0.5275 | 0 |
| X5D_v51_s7 (deployed peak-frame, same seed) | 参照 | best | 0.7571 | — | +0.0996 | 0.8693 | — | 1.611 | 0.6401 | 0.504 | 0.752 | 0.775 | 0.5293 | 0 |
| X5D_v51_s2025 (deployed peak-frame, same seed) | 参照 | best | 0.7657 | — | +0.1082 | 0.8745 | — | 1.568 | 0.6741 | 0.524 | 0.764 | 0.780 | 0.5362 | 0 |
| A1_s1234 (stage 3 TAWSS single head) | 参照 | best | 0.7436 | — | — | 0.8687 | — | 0.425 | 0.6414 | 0.480 | 0.755 | 0.760 | 0.5281 | 0 |
| A1_s7 (stage 3 TAWSS single head) | 参照 | best | 0.7520 | — | — | 0.8708 | — | 0.425 | 0.6384 | 0.508 | 0.755 | 0.747 | 0.5315 | 0 |
| A1_s2025 (stage 3 TAWSS single head) | 参照 | best | 0.7506 | — | — | 0.8680 | — | 0.421 | 0.6451 | 0.522 | 0.764 | 0.782 | 0.5323 | 0 |
| O2_s1234 (stage 3 OSI logit single head) | 参照 | best | 0.4934 | — | — | 0.5402 | — | 0.064 | 0.2265 | -12.991 | 0.630 | 0.833 | 0.2781 | 0 |
| O2_s7 (stage 3 OSI logit single head) | 参照 | best | 0.4921 | — | — | 0.5439 | — | 0.064 | 0.1980 | -11.811 | 0.664 | 0.861 | 0.2885 | 0 |
| O2_s2025 (stage 3 OSI logit single head) | 参照 | best | 0.5052 | — | — | 0.5450 | — | 0.063 | 0.2409 | -11.460 | 0.666 | 0.856 | 0.2856 | 0 |
| M1_s1234 | deployed X5D_v51 seed-1234 recipe (train136) with out_dim 3 = [peak WSS log_z, TAWSS log_z, OSI logit_z] (train136 multi statistics) | best | 0.7247 | — | +0.0673 | 0.8657 | — | 1.677 | 0.5963 | 0.391 | 0.712 | 0.739 | 0.5264 | 0 |
| M1_s1234 | deployed X5D_v51 seed-1234 recipe (train136) with out_dim 3 = [peak WSS log_z, TAWSS log_z, OSI logit_z] (train136 multi statistics) | last | 0.7201 | — | +0.0626 | 0.8650 | — | 1.685 | 0.5965 | 0.377 | 0.699 | 0.720 | 0.5276 | 0 |
| M1_s7 | deployed X5D_v51 seed-7 recipe (train136) with out_dim 3 = [peak WSS log_z, TAWSS log_z, OSI logit_z] (train136 multi statistics) | best | 0.7113 | — | +0.0538 | 0.8641 | — | 1.683 | 0.6233 | 0.415 | 0.697 | 0.725 | 0.5230 | 0 |
| M1_s7 | deployed X5D_v51 seed-7 recipe (train136) with out_dim 3 = [peak WSS log_z, TAWSS log_z, OSI logit_z] (train136 multi statistics) | last | 0.7090 | — | +0.0515 | 0.8641 | — | 1.682 | 0.6212 | 0.409 | 0.692 | 0.720 | 0.5236 | 0 |
| M1_s2025 | deployed X5D_v51 seed-2025 recipe (train136) with out_dim 3 = [peak WSS log_z, TAWSS log_z, OSI logit_z] (train136 multi statistics) | best | 0.7298 | — | +0.0723 | 0.8650 | — | 1.653 | 0.6372 | 0.433 | 0.699 | 0.701 | 0.5237 | 0 |
| M1_s2025 | deployed X5D_v51 seed-2025 recipe (train136) with out_dim 3 = [peak WSS log_z, TAWSS log_z, OSI logit_z] (train136 multi statistics) | last | 0.7291 | — | +0.0716 | 0.8654 | — | 1.653 | 0.6373 | 0.431 | 0.695 | 0.697 | 0.5250 | 0 |

## 分域物理 R²_cb（best）

| 臂 | AG | AAA | ILO |
|---|---:|---:|---:|
| C1 历史 | 0.6937 | 0.6342 | 0.6315 |
| M1_s1234 | 0.7993 | 0.6795 | 0.6664 |
| M1_s7 | 0.7472 | 0.6984 | 0.6729 |
| M1_s2025 | 0.7686 | 0.7132 | 0.6911 |

队列状态：complete；作业 15480；跳过的候选：
