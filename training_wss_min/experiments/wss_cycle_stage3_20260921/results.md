# wss_cycle_stage3_20260921 结果（自动生成 2026-09-21 23:40）

单 run、已暴露 test34；参照 = 同期对照 无（同配置重跑）、matrix.json 的外部参照与历史 C1；单 seed 对单 seed 的 95% 带约 ±0.034 物理 R²_cb / ±0.009 归一化。只作筛选，不作显著性或泛化结论。

| 臂 | 变化 | ckpt | Pa R²_cb | Δ vs X0 | Δ vs C1 | norm R²_cb | Δ vs X0 | MAE Pa | case P10 | high-WSS R² | top10 比 | p99 比 | IoU | 负R² |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 历史 | 参照 | best | 0.6575 | — | — | 0.8410 | — | 1.845 | 0.5050 | 0.300 | 0.639 | 0.624 | 0.4698 | 0 |
| X5D_v51_s1234 (deployed peak-frame, same seed) | 参照 | best | 0.7461 | — | +0.0886 | 0.8684 | — | 1.607 | 0.6537 | 0.489 | 0.754 | 0.769 | 0.5275 | 0 |
| X5D_v51_s7 (deployed peak-frame, same seed) | 参照 | best | 0.7571 | — | +0.0996 | 0.8693 | — | 1.611 | 0.6401 | 0.504 | 0.752 | 0.775 | 0.5293 | 0 |
| X5D_v51_s2025 (deployed peak-frame, same seed) | 参照 | best | 0.7657 | — | +0.1082 | 0.8745 | — | 1.568 | 0.6741 | 0.524 | 0.764 | 0.780 | 0.5362 | 0 |
| A1_s1234 | deployed X5D_v51 seed-1234 recipe (train136) with target=tawss from wss_min_cycle_v1 (frames 0-79) | best | 0.7436 | — | — | 0.8687 | — | 0.425 | 0.6414 | 0.480 | 0.755 | 0.760 | 0.5281 | 0 |
| A1_s1234 | deployed X5D_v51 seed-1234 recipe (train136) with target=tawss from wss_min_cycle_v1 (frames 0-79) | last | 0.7405 | — | — | 0.8684 | — | 0.426 | 0.6375 | 0.468 | 0.741 | 0.745 | 0.5301 | 0 |
| A1_s7 | deployed X5D_v51 seed-7 recipe (train136) with target=tawss from wss_min_cycle_v1 (frames 0-79) | best | 0.7520 | — | — | 0.8708 | — | 0.425 | 0.6384 | 0.508 | 0.755 | 0.747 | 0.5315 | 0 |
| A1_s7 | deployed X5D_v51 seed-7 recipe (train136) with target=tawss from wss_min_cycle_v1 (frames 0-79) | last | 0.7494 | — | — | 0.8706 | — | 0.425 | 0.6388 | 0.501 | 0.748 | 0.740 | 0.5313 | 0 |
| A1_s2025 | deployed X5D_v51 seed-2025 recipe (train136) with target=tawss from wss_min_cycle_v1 (frames 0-79) | best | 0.7506 | — | — | 0.8680 | — | 0.421 | 0.6451 | 0.522 | 0.764 | 0.782 | 0.5323 | 0 |
| A1_s2025 | deployed X5D_v51 seed-2025 recipe (train136) with target=tawss from wss_min_cycle_v1 (frames 0-79) | last | 0.7480 | — | — | 0.8678 | — | 0.421 | 0.6460 | 0.516 | 0.757 | 0.773 | 0.5317 | 0 |
| O1_s1234 | deployed X5D_v51 seed-1234 recipe (train136) with target=osi from wss_min_cycle_v1 (frames 0-79) | best | 0.5207 | — | — | 0.5206 | — | 0.064 | 0.3322 | -11.792 | 0.645 | 0.839 | 0.2844 | 0 |
| O1_s1234 | deployed X5D_v51 seed-1234 recipe (train136) with target=osi from wss_min_cycle_v1 (frames 0-79) | last | 0.5206 | — | — | 0.5205 | — | 0.064 | 0.3298 | -11.730 | 0.645 | 0.837 | 0.2854 | 0 |
| O1_s7 | deployed X5D_v51 seed-7 recipe (train136) with target=osi from wss_min_cycle_v1 (frames 0-79) | best | 0.5219 | — | — | 0.5218 | — | 0.064 | 0.2629 | -10.867 | 0.660 | 0.840 | 0.2929 | 0 |
| O1_s7 | deployed X5D_v51 seed-7 recipe (train136) with target=osi from wss_min_cycle_v1 (frames 0-79) | last | 0.5208 | — | — | 0.5207 | — | 0.064 | 0.2671 | -10.905 | 0.660 | 0.843 | 0.2924 | 0 |
| O1_s2025 | deployed X5D_v51 seed-2025 recipe (train136) with target=osi from wss_min_cycle_v1 (frames 0-79) | best | 0.5328 | — | — | 0.5324 | — | 0.063 | 0.3147 | -11.435 | 0.651 | 0.832 | 0.2957 | 0 |
| O1_s2025 | deployed X5D_v51 seed-2025 recipe (train136) with target=osi from wss_min_cycle_v1 (frames 0-79) | last | 0.5317 | — | — | 0.5313 | — | 0.063 | 0.3101 | -11.475 | 0.649 | 0.829 | 0.2954 | 0 |
| O2_s1234 | deployed X5D_v51 seed-1234 recipe (train136) with target=osi from wss_min_cycle_v1 (frames 0-79) | best | 0.4934 | — | — | 0.5402 | — | 0.064 | 0.2265 | -12.991 | 0.630 | 0.833 | 0.2781 | 0 |
| O2_s1234 | deployed X5D_v51 seed-1234 recipe (train136) with target=osi from wss_min_cycle_v1 (frames 0-79) | last | 0.4934 | — | — | 0.5404 | — | 0.064 | 0.2214 | -12.794 | 0.634 | 0.837 | 0.2775 | 0 |
| O2_s7 | deployed X5D_v51 seed-7 recipe (train136) with target=osi from wss_min_cycle_v1 (frames 0-79) | best | 0.4921 | — | — | 0.5439 | — | 0.064 | 0.1980 | -11.811 | 0.664 | 0.861 | 0.2885 | 0 |
| O2_s7 | deployed X5D_v51 seed-7 recipe (train136) with target=osi from wss_min_cycle_v1 (frames 0-79) | last | 0.4933 | — | — | 0.5447 | — | 0.063 | 0.2038 | -11.915 | 0.661 | 0.859 | 0.2886 | 0 |
| O2_s2025 | deployed X5D_v51 seed-2025 recipe (train136) with target=osi from wss_min_cycle_v1 (frames 0-79) | best | 0.5052 | — | — | 0.5450 | — | 0.063 | 0.2409 | -11.460 | 0.666 | 0.856 | 0.2856 | 0 |
| O2_s2025 | deployed X5D_v51 seed-2025 recipe (train136) with target=osi from wss_min_cycle_v1 (frames 0-79) | last | 0.5044 | — | — | 0.5440 | — | 0.063 | 0.2383 | -11.480 | 0.665 | 0.855 | 0.2841 | 0 |

## 分域物理 R²_cb（best）

| 臂 | AG | AAA | ILO |
|---|---:|---:|---:|
| C1 历史 | 0.6937 | 0.6342 | 0.6315 |
| A1_s1234 | 0.8151 | 0.6639 | 0.7508 |
| A1_s7 | 0.8117 | 0.6662 | 0.7809 |
| A1_s2025 | 0.7978 | 0.6772 | 0.7793 |
| O1_s1234 | 0.5794 | 0.4299 | 0.5039 |
| O1_s7 | 0.5725 | 0.4312 | 0.5172 |
| O1_s2025 | 0.5863 | 0.4602 | 0.5054 |
| O2_s1234 | 0.5510 | 0.3902 | 0.4899 |
| O2_s7 | 0.5417 | 0.3829 | 0.5071 |
| O2_s2025 | 0.5566 | 0.4145 | 0.4986 |

队列状态：complete；作业 15394；跳过的候选：
