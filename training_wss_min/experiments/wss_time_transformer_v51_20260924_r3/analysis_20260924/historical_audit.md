# r3 与历史参照的合同与数值审计

本审计只读取既有产物，未训练、推理或重评。表中所有主指标采用原始 Pa 真值；逐折值从原始 metrics 读取，并与既有汇总表一致。三折均值仅作开发阶段描述。

TT-warm 和 TT-raw 的周期、谷底 Pa 均在三折上高于 T-null、低于 C-raw 与 T0。TT-warm 与 TL-warm 的周期均值接近，但谷底和 TAWSS 在三折上均较低。这些历史差值同时受训练预算、采样、结构和选模影响，不能作为单变量消融。

| 方法 | cycle Pa | trough Pa | TAWSS Pa | cycle ln | peak Pa | peak median 帧 | peak p90 帧 | ts corr ln | amp err ln |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TT-warm | 0.498945 | 0.285676 | 0.687322 | 0.651720 | 0.713011 | 1.147563 | 8.754282 | 0.853303 | 0.546016 |
| TT-raw | 0.501620 | 0.292283 | 0.693044 | 0.649121 | 0.713011 | 1.243412 | 8.946970 | 0.849585 | 0.586054 |
| Tnull | 0.461566 | 0.221725 | 0.696054 | 0.405298 | 0.713011 | 1.345520 | 10.338603 | 0.807559 | 0.897683 |
| C-raw | 0.510923 | 0.310775 | 0.695807 | 0.624643 | 0.713011 | 1.316535 | 9.591568 | 0.846693 | 0.576902 |
| T0 | 0.516604 | 0.321773 | 0.716379 | 0.658568 | 0.696680 | 1.199605 | 8.773353 | 0.850632 | 0.608203 |
| TL-warm | 0.498684 | 0.295595 | 0.692196 | 0.647578 | 0.713011 | 1.258564 | 8.574111 | 0.851595 | 0.556705 |

时间误差与幅值误差越低越好；其余表中指标越高越好。peak median/p90 是各病例逐点分布统计后，再做病例均值；不是合并全体点的分位数。

**逐折差值（r3 − 历史；各格为 f0 / f1 / f2，括号为三折均值）**

| 对比 | cycle Pa | trough Pa | TAWSS Pa | cycle ln |
|---|---|---|---|---|
| TT-warm minus Tnull | +0.041681 / +0.028174 / +0.042284 (+0.037379) | +0.109204 / +0.027194 / +0.055455 (+0.063951) | -0.015050 / -0.009021 / -0.002124 (-0.008732) | +0.245755 / +0.229530 / +0.263978 (+0.246421) |
| TT-warm minus C-raw | -0.010491 / -0.010278 / -0.015164 (-0.011978) | -0.020979 / -0.025634 / -0.028685 (-0.025099) | -0.014547 / -0.004766 / -0.006140 (-0.008484) | +0.029633 / +0.035133 / +0.016464 (+0.027077) |
| TT-warm minus T0 | -0.009362 / -0.022374 / -0.021239 (-0.017659) | -0.028164 / -0.034001 / -0.046128 (-0.036097) | -0.011965 / -0.045634 / -0.029571 (-0.029057) | -0.003266 / -0.004877 / -0.012402 (-0.006848) |
| TT-warm minus TL-warm | -0.008228 / +0.009158 / -0.000147 (+0.000261) | -0.020995 / -0.002352 / -0.006411 (-0.009919) | -0.006361 / -0.004071 / -0.004189 (-0.004874) | +0.004310 / +0.006697 / +0.001417 (+0.004141) |
| TT-raw minus Tnull | +0.041767 / +0.033369 / +0.045027 (+0.040054) | +0.114720 / +0.039640 / +0.057315 (+0.070558) | -0.011697 / +0.001496 / +0.001170 (-0.003010) | +0.242494 / +0.224760 / +0.264213 (+0.243823) |
| TT-raw minus C-raw | -0.010405 / -0.005083 / -0.012420 (-0.009303) | -0.015463 / -0.013188 / -0.026825 (-0.018492) | -0.011194 / +0.005750 / -0.002846 (-0.002763) | +0.026371 / +0.030364 / +0.016699 (+0.024478) |
| TT-raw minus T0 | -0.009276 / -0.017179 / -0.018496 (-0.014984) | -0.022648 / -0.021555 / -0.044268 (-0.029490) | -0.008612 / -0.035117 / -0.026277 (-0.023335) | -0.006527 / -0.009647 / -0.012167 (-0.009447) |
| TT-raw minus TL-warm | -0.008141 / +0.014354 / +0.002596 (+0.002936) | -0.015479 / +0.010093 / -0.004550 (-0.003312) | -0.003008 / +0.006445 / -0.000895 (+0.000847) | +0.001049 / +0.001927 / +0.001652 (+0.001543) |

其余时间形态、峰值指标的逐折差值保存在同目录 JSON 的 `r3_minus_historical`，每条都附两侧原始指标路径。

**合同核对**

- 同一 V5.1 cv3：train/test 为 90/46、90/46、92/44；seed 1234；81 帧为 1120:2:1280，峰值 1162；floor 0.05 Pa、eps 1e-6。Pa 真值保留原值，floor 用于训练和 ln 指标。
- T-null、C-raw、TL-warm 的历史 split/frame_stats/cache SHA 均与 r3 逐折一致。TL-warm 未保存 waveform SHA，标记为未知；其配置指向相同波形文件，帧键一致。
- T0 与 r3 的数据、split、frame stats、waveform 路径相同；T0 run 内保存 stats 的内容与对应 frame-stats 完全一致，原始评估的 test 病例数与物理帧一致。T0 运行时独立 split/stats SHA 未保存，不能用当前文件 SHA 冒充历史冻结证据。
- 原始数据 bundle 没有跨运行时期的逐文件 SHA 记录；这里确认的是已保存合同与现有文件一致，不额外宣称历史标签字节不变。

| 方法 | 训练与选模 | 相对 r3 的主要混杂 |
|---|---|---|
| T-null | 解析 B_scale，无时间头拟合；共享配置中的 4000 steps 不执行 | 无可训练残差；同一个预训练峰值锚 |
| C-raw | 4000 × 8 病例 × 2048 点，全 81 帧；66,049 参数 MLP；last | 全云缓存采样、全训练点输入统计；训练点暴露约为 r3 的 8.89 倍；结构/参数量/数值后端也不同 |
| TL-warm | 150 epoch × 12 batch = 1800 steps；8 病例 × 2048 query，末 batch 保留；FP+local wall 解冻，last | 动态 5000 support、独立 query 重采样；输入统计来自前 4 batch；只有后段解冻，未保持 r3 结构/采样/点暴露一致 |
| T0 | 400 epoch，batch 8，support/query 各 5000；每病例每 epoch 抽一帧；训练 loss EMA-best | 端到端随机初始化、无固定峰值锚、密度增强；4800 batch 不等于有效 optimizer 步数（AMP 跳步 5/5/4） |
| r3 | 1800 × 4 病例 × 1024 query，全 81 帧；120,769 参数；last | 固定 8192 query pool、4096 统计点、1024 context；编码器冻结缓存，配对 60D 投影/初始权重 |

**warm/raw 的解释边界**

r3 两臂共用预训练 X5D 峰值锚、同一初始时间模型与 60D 投影。raw 臂只在归一化后将 32 维 cached query_x 置零，保留 27 维原输入和 peak_ln。因此它检验缓存预训练表示的增量信息，不能称为“预训练 vs 随机初始化”或“从零训练”。

**历史 T-null 的口径差异**

阶段 2 报告内嵌 `tnull_*` 来自 `offline/d2_tnull.json`，其 TimeMetrics 将 log 真值还原成地板化 Pa；而本表选择 P0.10 原始 metrics 的 `eval_compatible`。D2 还使用历史峰值预测源，不能把两者差值全部归为 floor 效应。

| T-null 来源 | cycle 三折均值 | trough 三折均值 | TAWSS 三折均值 | 真值 |
|---|---:|---:|---:|---|
| 本表 P0.10 eval_compatible | 0.461565964 | 0.221724588 | 0.696054020 | 原始 Pa |
| 阶段 2 内嵌 D2 | 0.461325753 | 0.221417344 | 0.695998390 | floor 后 Pa |

**三折 split 当前 SHA256**

- f0: `0f83a219c3567270efddeb156bc241276bff6a420dc847c6d7eaa5aa2b20cd17`；原始 r3/probe/finetune 均保存相同摘要；T0 仅确认当前路径/内容合同。
- f1: `ea9d34c3e2662f36ef79edc7cb5a46a1ccd2cdf4482d9b5917bc319ed880fb94`；原始 r3/probe/finetune 均保存相同摘要；T0 仅确认当前路径/内容合同。
- f2: `02232e071a4791013d6a34d654819a32979c6b54e28ace3c4df8d0d51632302d`；原始 r3/probe/finetune 均保存相同摘要；T0 仅确认当前路径/内容合同。

**原始指标来源**

| 方法 | fold 0 | fold 1 | fold 2 |
|---|---|---|---|
| TT-warm | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-warm_f0_s1234/metrics.json) | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-warm_f1_s1234/metrics.json) | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-warm_f2_s1234/metrics.json) |
| TT-raw | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-raw_f0_s1234/metrics.json) | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-raw_f1_s1234/metrics.json) | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_transformer_v51_20260924_r3/TT-raw_f2_s1234/metrics.json) |
| Tnull | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_adapter_v51_20260923/Tnull_f0_s1234/metrics.json) | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_adapter_v51_20260923/Tnull_f1_s1234/metrics.json) | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_adapter_v51_20260923/Tnull_f2_s1234/metrics.json) |
| C-raw | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_adapter_v51_20260923/C-raw_f0_s1234/metrics.json) | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_adapter_v51_20260923/C-raw_f1_s1234/metrics.json) | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_adapter_v51_20260923/C-raw_f2_s1234/metrics.json) |
| T0 | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_ecc_20260918/T0_f0_s1234/eval/ckpt_best/metrics.json) | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_ecc_20260918/T0_f1_s1234/eval/ckpt_best/metrics.json) | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_ecc_20260918/T0_f2_s1234/eval/ckpt_best/metrics.json) |
| TL-warm | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_adapter_ft_v51_20260923/TL-warm_f0_s1234/metrics.json) | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_adapter_ft_v51_20260923/TL-warm_f1_s1234/metrics.json) | [metrics](/public/newhome/cy/Digital_twin/GNN/training_wss_min/runs/wss_time_adapter_ft_v51_20260923/TL-warm_f2_s1234/metrics.json) |

[完整审计 JSON](/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_transformer_v51_20260924_r3/analysis_20260924/historical_audit.json) 保存逐折所有指标、差值、字段路径、配置/数据摘要、合同核对和历史报告 SHA。
