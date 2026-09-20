# wss_time_ecc_20260918 结果（自动生成 2026-09-18 20:55）

单 run、已暴露 test34；参照 = 同期对照 无（同配置重跑）、matrix.json 的外部参照与历史 C1；单 seed 对单 seed 的 95% 带约 ±0.034 物理 R²_cb / ±0.009 归一化。只作筛选，不作显著性或泛化结论。

| 臂 | 变化 | ckpt | Pa R²_cb | Δ vs X0 | Δ vs C1 | norm R²_cb | Δ vs X0 | MAE Pa | case P10 | high-WSS R² | top10 比 | p99 比 | IoU | 负R² |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 历史 | 参照 | best | 0.6575 | — | — | 0.8410 | — | 1.845 | 0.5050 | 0.300 | 0.639 | 0.624 | 0.4698 | 0 |
| X5D_v51_f0_s1234 (wave 2a, peak-frame base) | 参照 | best | 0.7366 | — | +0.0791 | 0.8556 | — | 1.455 | 0.6431 | 0.478 | 0.747 | 0.770 | 0.5083 | 0 |
| X5D_v51_f1_s1234 (wave 2a, peak-frame base) | 参照 | best | 0.6701 | — | +0.0127 | 0.8537 | — | 1.664 | 0.5698 | 0.305 | 0.686 | 0.712 | 0.5049 | 0 |
| X5D_v51_f2_s1234 (wave 2a, peak-frame base) | 参照 | best | 0.7323 | — | +0.0748 | 0.8514 | — | 1.410 | 0.6109 | 0.456 | 0.788 | 0.803 | 0.5251 | 0 |
| T0_f0_s1234 | X5D_v51 fold-0 recipe + phase condition ['q_norm', 'dq_norm', 't_sin', 't_cos'] (31 inputs, appended columns zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), frame_stats normalisation, checkpoint by EMA(train_loss, alpha 0.2) | best | 0.7224 | — | +0.0649 | 0.8432 | — | 1.510 | 0.6308 | 0.486 | 0.778 | 0.861 | 0.4879 | 0 |
| T0_f0_s1234 | X5D_v51 fold-0 recipe + phase condition ['q_norm', 'dq_norm', 't_sin', 't_cos'] (31 inputs, appended columns zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), frame_stats normalisation, checkpoint by EMA(train_loss, alpha 0.2) | last | 0.7227 | — | +0.0652 | 0.8428 | — | 1.512 | 0.6354 | 0.498 | 0.802 | 0.887 | 0.4885 | 0 |
| T0_f1_s1234 | X5D_v51 fold-1 recipe + phase condition ['q_norm', 'dq_norm', 't_sin', 't_cos'] (31 inputs, appended columns zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), frame_stats normalisation, checkpoint by EMA(train_loss, alpha 0.2) | best | 0.6687 | — | +0.0112 | 0.8445 | — | 1.706 | 0.5619 | 0.350 | 0.697 | 0.729 | 0.4849 | 0 |
| T0_f1_s1234 | X5D_v51 fold-1 recipe + phase condition ['q_norm', 'dq_norm', 't_sin', 't_cos'] (31 inputs, appended columns zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), frame_stats normalisation, checkpoint by EMA(train_loss, alpha 0.2) | last | 0.6660 | — | +0.0085 | 0.8443 | — | 1.706 | 0.5554 | 0.340 | 0.691 | 0.721 | 0.4854 | 0 |
| T0_f2_s1234 | X5D_v51 fold-2 recipe + phase condition ['q_norm', 'dq_norm', 't_sin', 't_cos'] (31 inputs, appended columns zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), frame_stats normalisation, checkpoint by EMA(train_loss, alpha 0.2) | best | 0.6990 | — | +0.0415 | 0.8444 | — | 1.477 | 0.5760 | 0.362 | 0.814 | 0.855 | 0.5023 | 0 |
| T0_f2_s1234 | X5D_v51 fold-2 recipe + phase condition ['q_norm', 'dq_norm', 't_sin', 't_cos'] (31 inputs, appended columns zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), frame_stats normalisation, checkpoint by EMA(train_loss, alpha 0.2) | last | 0.7032 | — | +0.0457 | 0.8446 | — | 1.468 | 0.5856 | 0.358 | 0.796 | 0.834 | 0.5030 | 0 |
| TB8_f0_s1234 | X5D_v51 fold-0 recipe with a time-basis output head K=8 (out_dim 9 | best | 0.6182 | — | -0.0393 | 0.8100 | — | 1.703 | 0.4675 | 0.207 | 0.623 | 0.642 | 0.4429 | 0 |
| TB8_f0_s1234 | X5D_v51 fold-0 recipe with a time-basis output head K=8 (out_dim 9 | last | 0.6190 | — | -0.0385 | 0.8099 | — | 1.703 | 0.4616 | 0.210 | 0.627 | 0.647 | 0.4408 | 0 |
| TB8_f1_s1234 | X5D_v51 fold-1 recipe with a time-basis output head K=8 (out_dim 9 | best | 0.5656 | — | -0.0919 | 0.8178 | — | 1.890 | 0.4597 | 0.127 | 0.562 | 0.588 | 0.4371 | 0 |
| TB8_f1_s1234 | X5D_v51 fold-1 recipe with a time-basis output head K=8 (out_dim 9 | last | 0.5600 | — | -0.0975 | 0.8174 | — | 1.900 | 0.4543 | 0.114 | 0.551 | 0.573 | 0.4377 | 0 |
| TB8_f2_s1234 | X5D_v51 fold-2 recipe with a time-basis output head K=8 (out_dim 9 | best | 0.6445 | — | -0.0130 | 0.8167 | — | 1.602 | 0.5265 | 0.213 | 0.639 | 0.669 | 0.4598 | 0 |
| TB8_f2_s1234 | X5D_v51 fold-2 recipe with a time-basis output head K=8 (out_dim 9 | last | 0.6453 | — | -0.0122 | 0.8172 | — | 1.599 | 0.5308 | 0.215 | 0.639 | 0.667 | 0.4600 | 0 |
| TB16_f0_s1234 | X5D_v51 fold-0 recipe with a time-basis output head K=16 (out_dim 17 | best | 0.6096 | — | -0.0479 | 0.8157 | — | 1.740 | 0.4816 | 0.172 | 0.595 | 0.609 | 0.4490 | 0 |
| TB16_f0_s1234 | X5D_v51 fold-0 recipe with a time-basis output head K=16 (out_dim 17 | last | 0.6158 | — | -0.0417 | 0.8162 | — | 1.730 | 0.4857 | 0.191 | 0.606 | 0.623 | 0.4486 | 0 |
| TB16_f1_s1234 | X5D_v51 fold-1 recipe with a time-basis output head K=16 (out_dim 17 | best | 0.5795 | — | -0.0780 | 0.8215 | — | 1.903 | 0.4096 | 0.196 | 0.598 | 0.651 | 0.4307 | 0 |
| TB16_f1_s1234 | X5D_v51 fold-1 recipe with a time-basis output head K=16 (out_dim 17 | last | 0.5756 | — | -0.0819 | 0.8213 | — | 1.907 | 0.4059 | 0.184 | 0.590 | 0.644 | 0.4314 | 0 |
| TB16_f2_s1234 | X5D_v51 fold-2 recipe with a time-basis output head K=16 (out_dim 17 | best | 0.6229 | — | -0.0346 | 0.8259 | — | 1.586 | 0.5209 | 0.202 | 0.659 | 0.678 | 0.4662 | 0 |
| TB16_f2_s1234 | X5D_v51 fold-2 recipe with a time-basis output head K=16 (out_dim 17 | last | 0.6194 | — | -0.0381 | 0.8261 | — | 1.590 | 0.5209 | 0.188 | 0.647 | 0.666 | 0.4663 | 0 |

## 分域物理 R²_cb（best）

| 臂 | AG | AAA | ILO |
|---|---:|---:|---:|
| C1 历史 | 0.6937 | 0.6342 | 0.6315 |
| T0_f0_s1234 | 0.7479 | 0.6944 | 0.7449 |
| T0_f1_s1234 | 0.7290 | 0.7650 | 0.5619 |
| T0_f2_s1234 | 0.7100 | 0.7287 | 0.6383 |
| TB8_f0_s1234 | 0.6259 | 0.5964 | 0.6510 |
| TB8_f1_s1234 | 0.5971 | 0.7240 | 0.4535 |
| TB8_f2_s1234 | 0.6331 | 0.6885 | 0.5888 |
| TB16_f0_s1234 | 0.6367 | 0.5752 | 0.6429 |
| TB16_f1_s1234 | 0.5958 | 0.7108 | 0.4919 |
| TB16_f2_s1234 | 0.6198 | 0.6450 | 0.5895 |

队列状态：complete；作业 15071；跳过的候选：
