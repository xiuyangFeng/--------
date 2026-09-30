#!/bin/bash
# 全周期趋势指标复评：node04（Slurm 之外，ssh + nohup；用户允许的 A100 用法）。
# T0 逐帧前向（CPU 侧 query patch 构建为瓶颈）→ 每折 4 分片 × 3 折，2 张 A100 轮流；其余臂 CPU、每折一个进程。
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
S=training_wss_min/experiments/wss_time_trend_20260927/trend_eval.py
LOG=training_wss_min/experiments/wss_time_trend_20260927/logs
mkdir -p "$LOG"
NSH=4
i=0
for f in 0 1 2; do
  for s in $(seq 0 $((NSH - 1))); do
    gpu=$((i % 2)); i=$((i + 1))
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nohup $PY $S metrics --arm T0 --fold $f \
      --shard $s --nshards $NSH --device cuda > "$LOG/T0_f${f}_s${s}.log" 2>&1 < /dev/null &
  done
done
for A in Tnull C-raw TT-warm TT-raw; do
  for f in 0 1 2; do
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nohup $PY $S metrics --arm $A --fold $f --device cpu \
      > "$LOG/${A}_f$f.log" 2>&1 < /dev/null &
  done
done
echo launched
