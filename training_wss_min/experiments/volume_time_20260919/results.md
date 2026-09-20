# volume_time_20260919 结果（自动生成 2026-09-19 18:47）

单 run、已暴露 test34；参照 = 同期对照 无（同配置重跑）、matrix.json 的外部参照与历史 C1；单 seed 对单 seed 的 95% 带约 ±0.034 物理 R²_cb / ±0.009 归一化。只作筛选，不作显著性或泛化结论。

| 臂 | 变化 | ckpt | Pa R²_cb | Δ vs X0 | Δ vs C1 | norm R²_cb | Δ vs X0 | MAE Pa | case P10 | high-WSS R² | top10 比 | p99 比 | IoU | 负R² |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 历史 | 参照（压力 Pa） | best | 0.7843 | — | — | 0.7843 | — | 105.098 | 0.5675 | 0.550 | 0.730 | 0.711 | 0.6044 | 0 |
| PF6_s1234 (legacy old-v5 pressure base) | 参照（压力 Pa） | best | 0.7843 | — | +0.0000 | 0.7843 | — | 105.098 | 0.5675 | 0.550 | 0.730 | 0.711 | 0.6044 | 0 |
| VF6_s1234 (legacy old-v5 velocity base) | 参照（速度幅值 m/s） | best | 0.7952 | — | — | 0.7952 | — | 0.086 | 0.6200 | 0.320 | 0.834 | 0.890 | 0.5448 | 0 |
| PF6_v51_f0_s1234 | pressure peak-frame fold base（压力 Pa） | best | 0.7931 | — | +0.0088 | 0.7931 | — | 101.505 | 0.5987 | 0.548 | 0.863 | 0.993 | 0.5751 | 0 |
| PF6_v51_f1_s1234 | pressure peak-frame fold base（压力 Pa） | best | 0.6479 | — | -0.1364 | 0.6479 | — | 134.541 | 0.5785 | 0.636 | 0.654 | 0.705 | 0.6421 | 0 |
| PF6_v51_f2_s1234 | pressure peak-frame fold base（压力 Pa） | best | 0.7907 | — | +0.0064 | 0.7907 | — | 103.428 | 0.6989 | 0.782 | 0.632 | 0.706 | 0.6223 | 1 |
| VF6_v51_f0_s1234 | velocity peak-frame fold base（速度幅值 m/s） | best | 0.7661 | — | — | 0.7661 | — | 0.082 | 0.5955 | 0.089 | 0.789 | 0.836 | 0.5329 | 0 |
| VF6_v51_f1_s1234 | velocity peak-frame fold base（速度幅值 m/s） | best | 0.7857 | — | — | 0.7857 | — | 0.089 | 0.6735 | 0.244 | 0.789 | 0.818 | 0.5105 | 0 |
| VF6_v51_f2_s1234 | velocity peak-frame fold base（速度幅值 m/s） | best | 0.7838 | — | — | 0.7838 | — | 0.079 | 0.6331 | 0.284 | 0.821 | 0.851 | 0.5392 | 0 |
| PT0_f0_s1234 | PF6_v51_f0_s1234 recipe + phase condition ['q_norm', 'dq_norm', 't_sin', 't_cos'] (24 inputs, appended columns zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), per-frame linear z on the pressure target, checkpoint by EMA(train_loss, alpha 0.2)（压力 Pa） | best | 0.7844 | — | +0.0001 | 0.7844 | — | 109.075 | 0.6975 | 0.355 | 0.812 | 0.921 | 0.5730 | 0 |
| PT0_f0_s1234 | PF6_v51_f0_s1234 recipe + phase condition ['q_norm', 'dq_norm', 't_sin', 't_cos'] (24 inputs, appended columns zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), per-frame linear z on the pressure target, checkpoint by EMA(train_loss, alpha 0.2)（压力 Pa） | last | 0.7878 | — | +0.0035 | 0.7878 | — | 109.147 | 0.6798 | 0.323 | 0.738 | 0.844 | 0.5797 | 0 |
| PT0_f1_s1234 | PF6_v51_f1_s1234 recipe + phase condition ['q_norm', 'dq_norm', 't_sin', 't_cos'] (24 inputs, appended columns zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), per-frame linear z on the pressure target, checkpoint by EMA(train_loss, alpha 0.2)（压力 Pa） | best | 0.7168 | — | -0.0675 | 0.7168 | — | 131.276 | 0.6262 | 0.499 | 0.779 | 0.828 | 0.5840 | 2 |
| PT0_f1_s1234 | PF6_v51_f1_s1234 recipe + phase condition ['q_norm', 'dq_norm', 't_sin', 't_cos'] (24 inputs, appended columns zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), per-frame linear z on the pressure target, checkpoint by EMA(train_loss, alpha 0.2)（压力 Pa） | last | 0.7294 | — | -0.0549 | 0.7294 | — | 131.788 | 0.6224 | 0.493 | 0.848 | 0.887 | 0.6278 | 2 |
| PT0_f2_s1234 | PF6_v51_f2_s1234 recipe + phase condition ['q_norm', 'dq_norm', 't_sin', 't_cos'] (24 inputs, appended columns zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), per-frame linear z on the pressure target, checkpoint by EMA(train_loss, alpha 0.2)（压力 Pa） | best | 0.8182 | — | +0.0338 | 0.8182 | — | 109.260 | 0.6670 | 0.589 | 0.562 | 0.732 | 0.5401 | 0 |
| PT0_f2_s1234 | PF6_v51_f2_s1234 recipe + phase condition ['q_norm', 'dq_norm', 't_sin', 't_cos'] (24 inputs, appended columns zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), per-frame linear z on the pressure target, checkpoint by EMA(train_loss, alpha 0.2)（压力 Pa） | last | 0.8243 | — | +0.0399 | 0.8243 | — | 109.346 | 0.6663 | 0.585 | 0.557 | 0.720 | 0.5416 | 0 |
| PTB8_f0_s1234 | PF6_v51_f0_s1234 recipe with a time-basis output head K=8 (out_dim 9 = 1 component(s) × [b0, a_1..a_K] on the train-fold PCA basis（压力 Pa） | best | 0.7570 | — | -0.0273 | 0.7570 | — | 124.429 | 0.4185 | 0.431 | 1.123 | 1.417 | 0.5435 | 1 |
| PTB8_f0_s1234 | PF6_v51_f0_s1234 recipe with a time-basis output head K=8 (out_dim 9 = 1 component(s) × [b0, a_1..a_K] on the train-fold PCA basis（压力 Pa） | last | 0.7570 | — | -0.0273 | 0.7570 | — | 124.429 | 0.4185 | 0.431 | 1.123 | 1.417 | 0.5435 | 1 |
| PTB8_f1_s1234 | PF6_v51_f1_s1234 recipe with a time-basis output head K=8 (out_dim 9 = 1 component(s) × [b0, a_1..a_K] on the train-fold PCA basis（压力 Pa） | best | 0.5786 | — | -0.2057 | 0.5786 | — | 151.069 | 0.5565 | 0.635 | 0.713 | 0.689 | 0.5252 | 1 |
| PTB8_f1_s1234 | PF6_v51_f1_s1234 recipe with a time-basis output head K=8 (out_dim 9 = 1 component(s) × [b0, a_1..a_K] on the train-fold PCA basis（压力 Pa） | last | 0.5838 | — | -0.2005 | 0.5838 | — | 152.077 | 0.5251 | 0.646 | 0.723 | 0.717 | 0.5186 | 1 |
| PTB8_f2_s1234 | PF6_v51_f2_s1234 recipe with a time-basis output head K=8 (out_dim 9 = 1 component(s) × [b0, a_1..a_K] on the train-fold PCA basis（压力 Pa） | best | 0.7427 | — | -0.0416 | 0.7427 | — | 120.017 | 0.5877 | 0.577 | 0.599 | 0.693 | 0.5387 | 1 |
| PTB8_f2_s1234 | PF6_v51_f2_s1234 recipe with a time-basis output head K=8 (out_dim 9 = 1 component(s) × [b0, a_1..a_K] on the train-fold PCA basis（压力 Pa） | last | 0.7374 | — | -0.0469 | 0.7374 | — | 119.709 | 0.6173 | 0.584 | 0.610 | 0.695 | 0.5647 | 1 |
| PTB16_f0_s1234 | PF6_v51_f0_s1234 recipe with a time-basis output head K=16 (out_dim 17 = 1 component(s) × [b0, a_1..a_K] on the train-fold PCA basis（压力 Pa） | best | 0.7352 | — | -0.0491 | 0.7352 | — | 120.595 | 0.4897 | 0.431 | 0.936 | 1.083 | 0.4394 | 0 |
| PTB16_f0_s1234 | PF6_v51_f0_s1234 recipe with a time-basis output head K=16 (out_dim 17 = 1 component(s) × [b0, a_1..a_K] on the train-fold PCA basis（压力 Pa） | last | 0.7374 | — | -0.0469 | 0.7374 | — | 120.528 | 0.4730 | 0.450 | 0.941 | 1.081 | 0.4552 | 0 |
| PTB16_f1_s1234 | PF6_v51_f1_s1234 recipe with a time-basis output head K=16 (out_dim 17 = 1 component(s) × [b0, a_1..a_K] on the train-fold PCA basis（压力 Pa） | best | 0.5701 | — | -0.2143 | 0.5701 | — | 152.739 | 0.3977 | 0.447 | 0.679 | 0.603 | 0.5211 | 1 |
| PTB16_f1_s1234 | PF6_v51_f1_s1234 recipe with a time-basis output head K=16 (out_dim 17 = 1 component(s) × [b0, a_1..a_K] on the train-fold PCA basis（压力 Pa） | last | 0.5756 | — | -0.2088 | 0.5756 | — | 153.448 | 0.4008 | 0.457 | 0.707 | 0.630 | 0.4997 | 1 |
| PTB16_f2_s1234 | PF6_v51_f2_s1234 recipe with a time-basis output head K=16 (out_dim 17 = 1 component(s) × [b0, a_1..a_K] on the train-fold PCA basis（压力 Pa） | best | 0.7389 | — | -0.0455 | 0.7389 | — | 125.124 | 0.5799 | 0.218 | 0.464 | 0.528 | 0.4430 | 1 |
| PTB16_f2_s1234 | PF6_v51_f2_s1234 recipe with a time-basis output head K=16 (out_dim 17 = 1 component(s) × [b0, a_1..a_K] on the train-fold PCA basis（压力 Pa） | last | 0.7366 | — | -0.0477 | 0.7366 | — | 124.185 | 0.5928 | 0.279 | 0.480 | 0.551 | 0.4502 | 1 |
| VT0_f0_s1234 | VF6_v51_f0_s1234 recipe + phase condition ['q_norm', 'dq_norm', 't_sin', 't_cos'] (24 inputs, appended columns zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), per-frame linear z on the velocity target, checkpoint by EMA(train_loss, alpha 0.2)（速度幅值 m/s） | best | 0.7515 | — | — | 0.7515 | — | 0.086 | 0.6219 | 0.047 | 0.793 | 0.865 | 0.5161 | 0 |
| VT0_f0_s1234 | VF6_v51_f0_s1234 recipe + phase condition ['q_norm', 'dq_norm', 't_sin', 't_cos'] (24 inputs, appended columns zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), per-frame linear z on the velocity target, checkpoint by EMA(train_loss, alpha 0.2)（速度幅值 m/s） | last | 0.7547 | — | — | 0.7547 | — | 0.086 | 0.6292 | 0.073 | 0.803 | 0.874 | 0.5182 | 0 |
| VT0_f1_s1234 | VF6_v51_f1_s1234 recipe + phase condition ['q_norm', 'dq_norm', 't_sin', 't_cos'] (24 inputs, appended columns zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), per-frame linear z on the velocity target, checkpoint by EMA(train_loss, alpha 0.2)（速度幅值 m/s） | best | 0.7734 | — | — | 0.7734 | — | 0.092 | 0.6974 | 0.193 | 0.791 | 0.835 | 0.5052 | 0 |
| VT0_f1_s1234 | VF6_v51_f1_s1234 recipe + phase condition ['q_norm', 'dq_norm', 't_sin', 't_cos'] (24 inputs, appended columns zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), per-frame linear z on the velocity target, checkpoint by EMA(train_loss, alpha 0.2)（速度幅值 m/s） | last | 0.7851 | — | — | 0.7851 | — | 0.090 | 0.7072 | 0.256 | 0.812 | 0.853 | 0.5091 | 0 |
| VT0_f2_s1234 | VF6_v51_f2_s1234 recipe + phase condition ['q_norm', 'dq_norm', 't_sin', 't_cos'] (24 inputs, appended columns zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), per-frame linear z on the velocity target, checkpoint by EMA(train_loss, alpha 0.2)（速度幅值 m/s） | best | 0.7776 | — | — | 0.7776 | — | 0.083 | 0.6085 | 0.274 | 0.839 | 0.897 | 0.5291 | 0 |
| VT0_f2_s1234 | VF6_v51_f2_s1234 recipe + phase condition ['q_norm', 'dq_norm', 't_sin', 't_cos'] (24 inputs, appended columns zero-initialised), timesteps=random_frame (uniform frames, peak guarantee 1/9), per-frame linear z on the velocity target, checkpoint by EMA(train_loss, alpha 0.2)（速度幅值 m/s） | last | 0.7784 | — | — | 0.7784 | — | 0.083 | 0.6080 | 0.269 | 0.833 | 0.893 | 0.5308 | 0 |
| VTB4_f0_s1234 | VF6_v51_f0_s1234 recipe with a time-basis output head K=4 (out_dim 15 = 3 component(s) × [b0, a_1..a_K] on the train-fold PCA basis (the three components share one set of basis vectors)（速度幅值 m/s） | best | 0.6861 | — | — | 0.6861 | — | 0.103 | 0.5216 | -0.169 | 0.751 | 0.840 | 0.4937 | 1 |
| VTB4_f0_s1234 | VF6_v51_f0_s1234 recipe with a time-basis output head K=4 (out_dim 15 = 3 component(s) × [b0, a_1..a_K] on the train-fold PCA basis (the three components share one set of basis vectors)（速度幅值 m/s） | last | 0.6852 | — | — | 0.6852 | — | 0.102 | 0.5198 | -0.195 | 0.735 | 0.820 | 0.4911 | 1 |
| VTB4_f1_s1234 | VF6_v51_f1_s1234 recipe with a time-basis output head K=4 (out_dim 15 = 3 component(s) × [b0, a_1..a_K] on the train-fold PCA basis (the three components share one set of basis vectors)（速度幅值 m/s） | best | 0.7437 | — | — | 0.7437 | — | 0.106 | 0.6425 | 0.160 | 0.777 | 0.806 | 0.4869 | 0 |
| VTB4_f1_s1234 | VF6_v51_f1_s1234 recipe with a time-basis output head K=4 (out_dim 15 = 3 component(s) × [b0, a_1..a_K] on the train-fold PCA basis (the three components share one set of basis vectors)（速度幅值 m/s） | last | 0.7425 | — | — | 0.7425 | — | 0.105 | 0.6465 | 0.136 | 0.769 | 0.801 | 0.4875 | 0 |
| VTB4_f2_s1234 | VF6_v51_f2_s1234 recipe with a time-basis output head K=4 (out_dim 15 = 3 component(s) × [b0, a_1..a_K] on the train-fold PCA basis (the three components share one set of basis vectors)（速度幅值 m/s） | best | 0.7188 | — | — | 0.7188 | — | 0.098 | 0.5762 | 0.064 | 0.779 | 0.830 | 0.5039 | 0 |
| VTB4_f2_s1234 | VF6_v51_f2_s1234 recipe with a time-basis output head K=4 (out_dim 15 = 3 component(s) × [b0, a_1..a_K] on the train-fold PCA basis (the three components share one set of basis vectors)（速度幅值 m/s） | last | 0.7180 | — | — | 0.7180 | — | 0.098 | 0.5811 | 0.051 | 0.773 | 0.825 | 0.5045 | 0 |
| VTB8_f0_s1234 | VF6_v51_f0_s1234 recipe with a time-basis output head K=8 (out_dim 27 = 3 component(s) × [b0, a_1..a_K] on the train-fold PCA basis (the three components share one set of basis vectors)（速度幅值 m/s） | best | 0.6945 | — | — | 0.6945 | — | 0.101 | 0.5343 | -0.173 | 0.728 | 0.785 | 0.4869 | 0 |
| VTB8_f0_s1234 | VF6_v51_f0_s1234 recipe with a time-basis output head K=8 (out_dim 27 = 3 component(s) × [b0, a_1..a_K] on the train-fold PCA basis (the three components share one set of basis vectors)（速度幅值 m/s） | last | 0.6923 | — | — | 0.6923 | — | 0.101 | 0.5264 | -0.201 | 0.716 | 0.773 | 0.4854 | 0 |
| VTB8_f1_s1234 | VF6_v51_f1_s1234 recipe with a time-basis output head K=8 (out_dim 27 = 3 component(s) × [b0, a_1..a_K] on the train-fold PCA basis (the three components share one set of basis vectors)（速度幅值 m/s） | best | 0.7262 | — | — | 0.7262 | — | 0.107 | 0.6220 | 0.001 | 0.729 | 0.763 | 0.4804 | 0 |
| VTB8_f1_s1234 | VF6_v51_f1_s1234 recipe with a time-basis output head K=8 (out_dim 27 = 3 component(s) × [b0, a_1..a_K] on the train-fold PCA basis (the three components share one set of basis vectors)（速度幅值 m/s） | last | 0.7263 | — | — | 0.7263 | — | 0.106 | 0.6222 | -0.009 | 0.725 | 0.759 | 0.4799 | 0 |
| VTB8_f2_s1234 | VF6_v51_f2_s1234 recipe with a time-basis output head K=8 (out_dim 27 = 3 component(s) × [b0, a_1..a_K] on the train-fold PCA basis (the three components share one set of basis vectors)（速度幅值 m/s） | best | 0.7327 | — | — | 0.7327 | — | 0.094 | 0.5784 | 0.070 | 0.770 | 0.820 | 0.5156 | 0 |
| VTB8_f2_s1234 | VF6_v51_f2_s1234 recipe with a time-basis output head K=8 (out_dim 27 = 3 component(s) × [b0, a_1..a_K] on the train-fold PCA basis (the three components share one set of basis vectors)（速度幅值 m/s） | last | 0.7317 | — | — | 0.7317 | — | 0.094 | 0.5783 | 0.057 | 0.763 | 0.813 | 0.5159 | 0 |

## 分域物理 R²_cb（best）

| 臂 | AG | AAA | ILO |
|---|---:|---:|---:|
| C1 历史 | 0.8718 | 0.8651 | 0.6809 |
| PF6_v51_f0_s1234 | 0.7900 | 0.7563 | 0.8719 |
| PF6_v51_f1_s1234 | 0.8321 | 0.8484 | 0.5218 |
| PF6_v51_f2_s1234 | 0.7459 | 0.8703 | 0.7874 |
| VF6_v51_f0_s1234 | 0.7580 | 0.7223 | 0.8211 |
| VF6_v51_f1_s1234 | 0.8137 | 0.8223 | 0.6272 |
| VF6_v51_f2_s1234 | 0.7847 | 0.7903 | 0.7249 |
| PT0_f0_s1234 | 0.7733 | 0.7505 | 0.8700 |
| PT0_f1_s1234 | 0.8467 | 0.8884 | 0.6216 |
| PT0_f2_s1234 | 0.8252 | 0.8546 | 0.7721 |
| PTB8_f0_s1234 | 0.7706 | 0.7154 | 0.8191 |
| PTB8_f1_s1234 | 0.7910 | 0.8061 | 0.4337 |
| PTB8_f2_s1234 | 0.6939 | 0.8547 | 0.7155 |
| PTB16_f0_s1234 | 0.7573 | 0.6715 | 0.8288 |
| PTB16_f1_s1234 | 0.7836 | 0.8226 | 0.4198 |
| PTB16_f2_s1234 | 0.6799 | 0.8345 | 0.7429 |
| VT0_f0_s1234 | 0.7568 | 0.6890 | 0.8092 |
| VT0_f1_s1234 | 0.7995 | 0.8156 | 0.6102 |
| VT0_f2_s1234 | 0.7765 | 0.7827 | 0.7236 |
| VTB4_f0_s1234 | 0.6639 | 0.6346 | 0.7696 |
| VTB4_f1_s1234 | 0.7634 | 0.7880 | 0.5828 |
| VTB4_f2_s1234 | 0.7132 | 0.7223 | 0.6638 |
| VTB8_f0_s1234 | 0.6752 | 0.6542 | 0.7564 |
| VTB8_f1_s1234 | 0.7440 | 0.7678 | 0.5665 |
| VTB8_f2_s1234 | 0.7290 | 0.7439 | 0.6658 |

队列状态：complete；作业 15123；跳过的候选：
