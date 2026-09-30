# wss_time_transformer_v51_20260924_r3

状态：**incomplete**；验收 0/12 个正式 run。

G0：尚未通过完整矩阵验收。

缺失/未完成：

- /public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_transformer_v51_20260924_r3/queue_acceptance.json
- /public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-raw-noattn_f0_s1234/metrics.json
- /public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-raw-noattn_f1_s1234/metrics.json
- /public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-raw-noattn_f2_s1234/metrics.json
- /public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-raw_f0_s1234/metrics.json
- /public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-raw_f1_s1234/metrics.json
- /public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-raw_f2_s1234/metrics.json
- /public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-warm-noattn_f0_s1234/metrics.json
- /public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-warm-noattn_f1_s1234/metrics.json
- /public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-warm-noattn_f2_s1234/metrics.json
- /public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-warm_f0_s1234/metrics.json
- /public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-warm_f1_s1234/metrics.json
- /public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-warm_f2_s1234/metrics.json
- queue finished_at

完整矩阵尚未验收，不生成四臂均值或完整配对结论。

证据入口：

- [matrix](/public/newhome/cy/Digital_twin/GNN_time_transformer_frozen_20260924_r3/training_wss_min/configs/wss_time_transformer_v51_20260924_r3/matrix.json)
- [queue](/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_transformer_v51_20260924_r3/queue_status.json)
- [queue_acceptance](/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_transformer_v51_20260924_r3/queue_acceptance.json)

V5.1 cv3, seed 1234: descriptive development evidence only. No statistical-significance claim or deployment decision. Historical C-raw/T0 are context only, not matched causal controls.

静态几何与协议相位输入；没有逐帧观测的速度/压力。noattn 是相同 Transformer 的对角屏蔽，比较仅识别本冻结表示和训练预算下的跨帧交互增量。
