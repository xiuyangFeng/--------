#!/bin/bash
# 补跑 TT 对角注意力两臂（CPU），用于分离跨帧注意力对趋势指标的贡献。
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
S=training_wss_min/experiments/wss_time_trend_20260927/trend_eval.py
LOG=training_wss_min/experiments/wss_time_trend_20260927/logs
for A in TT-warm-noattn TT-raw-noattn; do
  for f in 0 1 2; do
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nohup $PY $S metrics --arm $A --fold $f --device cpu \
      > "$LOG/${A}_f$f.log" 2>&1 < /dev/null &
  done
done
echo launched
