# wss_time_transformer_v51_20260924_r3

状态：**complete**；验收 12/12 个正式 run。

G0：通过；最大峰值锚偏差 8.882e-16 ln Pa。

三折算术均值（Pa R²_cb 为病例等权逐帧均值；ln 与时间形态单列）：

| 臂 | cycle Pa | trough Pa | TAWSS Pa | cycle ln | peak Pa | peak p90（帧）| 时序相关 | 幅值误差 ln |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TT-warm | 0.4989 | 0.2857 | 0.6873 | 0.6517 | 0.7130 | 8.75 | 0.8533 | 0.5460 |
| TT-warm-noattn | 0.5005 | 0.2877 | 0.6852 | 0.6555 | 0.7130 | 8.72 | 0.8545 | 0.5563 |
| TT-raw | 0.5016 | 0.2923 | 0.6930 | 0.6491 | 0.7130 | 8.95 | 0.8496 | 0.5861 |
| TT-raw-noattn | 0.4999 | 0.2914 | 0.6851 | 0.6512 | 0.7130 | 8.88 | 0.8504 | 0.5701 |

同折配对差（前者减后者）：

- A1_warm_attention_minus_diagonal: cycle Pa f0=-0.0047; f1=+0.0021; f2=-0.0020；均值 -0.0015；谷底均值 -0.0020。
- A2_warm_minus_raw_attention: cycle Pa f0=-0.0001; f1=-0.0052; f2=-0.0027；均值 -0.0027；谷底均值 -0.0066。
- A3_raw_attention_minus_diagonal: cycle Pa f0=+0.0008; f1=+0.0038; f2=+0.0005；均值 +0.0017；谷底均值 +0.0009。
- A4_warm_minus_raw_diagonal: cycle Pa f0=+0.0054; f1=-0.0035; f2=-0.0003；均值 +0.0005；谷底均值 -0.0037。

已验收单 run 的预算与来源（耗时为秒，训练、评估、总耗时分别报告）：

| Run | steps | 参数量 | G0 最大差 ln Pa | 训练 | 评估 | 总耗时 | 原始指标 |
|---|---:|---:|---:|---:|---:|---:|---|
| TT-warm_f0_s1234 | 1800 | 120769 | 8.88e-16 | 19.2 | 15.0 | 36.4 | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-warm_f0_s1234/metrics.json) |
| TT-warm-noattn_f0_s1234 | 1800 | 120769 | 8.88e-16 | 22.7 | 15.6 | 40.4 | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-warm-noattn_f0_s1234/metrics.json) |
| TT-raw_f0_s1234 | 1800 | 120769 | 8.88e-16 | 21.4 | 15.4 | 38.8 | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-raw_f0_s1234/metrics.json) |
| TT-raw-noattn_f0_s1234 | 1800 | 120769 | 8.88e-16 | 22.6 | 15.5 | 40.2 | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-raw-noattn_f0_s1234/metrics.json) |
| TT-warm_f1_s1234 | 1800 | 120769 | 8.88e-16 | 21.3 | 14.4 | 37.8 | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-warm_f1_s1234/metrics.json) |
| TT-warm-noattn_f1_s1234 | 1800 | 120769 | 8.88e-16 | 22.6 | 14.3 | 39.0 | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-warm-noattn_f1_s1234/metrics.json) |
| TT-raw_f1_s1234 | 1800 | 120769 | 8.88e-16 | 21.0 | 14.4 | 37.6 | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-raw_f1_s1234/metrics.json) |
| TT-raw-noattn_f1_s1234 | 1800 | 120769 | 8.88e-16 | 22.5 | 14.4 | 39.0 | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-raw-noattn_f1_s1234/metrics.json) |
| TT-warm_f2_s1234 | 1800 | 120769 | 8.88e-16 | 21.1 | 15.7 | 39.0 | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-warm_f2_s1234/metrics.json) |
| TT-warm-noattn_f2_s1234 | 1800 | 120769 | 8.88e-16 | 22.6 | 15.8 | 40.5 | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-warm-noattn_f2_s1234/metrics.json) |
| TT-raw_f2_s1234 | 1800 | 120769 | 8.88e-16 | 21.3 | 15.8 | 39.2 | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-raw_f2_s1234/metrics.json) |
| TT-raw-noattn_f2_s1234 | 1800 | 120769 | 8.88e-16 | 22.5 | 15.9 | 40.4 | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-raw-noattn_f2_s1234/metrics.json) |

证据入口：

- [matrix](/public/newhome/cy/Digital_twin/GNN_time_transformer_frozen_20260924_r3_shared/training_wss_min/configs/wss_time_transformer_v51_20260924_r3/matrix.json)
- [queue](/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_transformer_v51_20260924_r3/queue_status.json)
- [queue_acceptance](/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_transformer_v51_20260924_r3/queue_acceptance.json)

V5.1 cv3, seed 1234: descriptive development evidence only. No statistical-significance claim or deployment decision. Historical C-raw/T0 are context only, not matched causal controls.

静态几何与协议相位输入；没有逐帧观测的速度/压力。noattn 是相同 Transformer 的对角屏蔽，比较仅识别本冻结表示和训练预算下的跨帧交互增量。
