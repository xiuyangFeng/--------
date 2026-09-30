# wss_v52p4_full265_20260930 结果（自动生成 2026-09-29 15:10）

单 run、已暴露 test34；参照 = 同期对照 无（同配置重跑）、matrix.json 的外部参照与历史 C1；单 seed 对单 seed 的 95% 带约 ±0.034 物理 R²_cb / ±0.009 归一化。只作筛选，不作显著性或泛化结论。

| 臂 | 变化 | ckpt | Pa R²_cb | Δ vs X0 | Δ vs C1 | norm R²_cb | Δ vs X0 | MAE Pa | case P10 | high-WSS R² | top10 比 | p99 比 | IoU | 负R² |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 历史 | 参照 | best | 0.7461 | — | — | 0.8684 | — | 1.607 | 0.6537 | 0.489 | 0.754 | 0.769 | 0.5275 | 0 |
| X5Dcap_asym2_full265_s1234 | X5Dcap_asym2 recipe on full265 (v5.2 261 + 4), seed 1234; test = recover8 | best | 0.8295 | — | +0.0834 | 0.8910 | — | 1.003 | 0.7288 | 0.553 | 0.908 | 0.925 | 0.5787 | 0 |
| X5Dcap_asym2_full265_s1234 | X5Dcap_asym2 recipe on full265 (v5.2 261 + 4), seed 1234; test = recover8 | last | 0.8318 | — | +0.0857 | 0.8921 | — | 0.990 | 0.7247 | 0.560 | 0.893 | 0.904 | 0.5802 | 0 |
| X5Dcap_asym2_full265_s7 | X5Dcap_asym2 recipe on full265 (v5.2 261 + 4), seed 7; test = recover8 | best | 0.8171 | — | +0.0710 | 0.8905 | — | 1.015 | 0.6957 | 0.500 | 0.882 | 0.890 | 0.5662 | 0 |
| X5Dcap_asym2_full265_s7 | X5Dcap_asym2 recipe on full265 (v5.2 261 + 4), seed 7; test = recover8 | last | 0.8182 | — | +0.0721 | 0.8894 | — | 1.019 | 0.7000 | 0.509 | 0.888 | 0.894 | 0.5670 | 0 |
| X5Dcap_asym2_full265_s2025 | X5Dcap_asym2 recipe on full265 (v5.2 261 + 4), seed 2025; test = recover8 | best | 0.8311 | — | +0.0850 | 0.8920 | — | 0.992 | 0.7052 | 0.548 | 0.873 | 0.873 | 0.5830 | 0 |
| X5Dcap_asym2_full265_s2025 | X5Dcap_asym2 recipe on full265 (v5.2 261 + 4), seed 2025; test = recover8 | last | 0.8291 | — | +0.0830 | 0.8908 | — | 1.001 | 0.7033 | 0.547 | 0.876 | 0.877 | 0.5812 | 0 |

## 分域物理 R²_cb（best）

| 臂 | AG | AAA | ILO |
|---|---:|---:|---:|
| C1 历史 | 0.7855 | 0.7160 | 0.7180 |
| X5Dcap_asym2_full265_s1234 | — | 0.8239 | 0.8482 |
| X5Dcap_asym2_full265_s7 | — | 0.8102 | 0.8420 |
| X5Dcap_asym2_full265_s2025 | — | 0.8278 | 0.8388 |

队列状态：complete；作业 16192；跳过的候选：
