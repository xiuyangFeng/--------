#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=voltime_accept
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
# 兼容性验收：这批体场时间改动（VolumeFrameSource / 体场 frame_stats / 体场周期评估分支 / 多通道时间基）
# 不得改变既有读数。重评三个旧 run 的 best，与已存 metrics.json 逐字段比较：
#   X5D_v51_s1234  壁面 WSS 峰值帧（上一轮 WSS 时间改动的同一把尺子）
#   PF6_s1234      体场压力峰值帧（本轮改动最密集的路径）
#   VF6_s1234      体场速度峰值帧（三分量向量目标）
set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
CMP=$PROJECT/training_wss_min/experiments/wss_time_ecc_20260918/offline/compare_metrics.py
ACC=$PROJECT/training_wss_min/experiments/volume_time_20260919/acceptance
cd "$PROJECT"
date -Is; nvidia-smi --query-gpu=index,name,memory.used --format=csv
rc=0
for spec in "wss_v51_wave1_20260916/X5D_v51_s1234" "wss_local_wave2_20260912/PF6_s1234" "wss_local_wave2_20260912/VF6_s1234"; do
  name=$(basename "$spec")
  RUN=$PROJECT/training_wss_min/runs/$spec
  OUT=$ACC/$name
  echo "=== $name ==="
  rm -rf "$OUT"; mkdir -p "$OUT"
  OMP_NUM_THREADS=4 $PY -u -m training_wss_min.evaluate --run-dir "$RUN" --partitions test --allow-test \
    --checkpoint best --no-plots --output-dir "$OUT"
  NEW=$(find "$OUT" -name metrics.json | head -1)
  $PY "$CMP" "$RUN/eval/ckpt_best/metrics.json" "$NEW" 1e-4 | tee "$OUT/compare.txt" || rc=1
done
date -Is
exit $rc
