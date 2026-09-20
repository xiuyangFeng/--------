#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=tecc_accept
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
# 兼容性验收：新代码（时间基头/frame_stats/EMA 选模）重评旧 run X5D_v51_s1234 best，输出到独立目录，与已存 metrics.json 逐字段比较。
set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
RUN=$PROJECT/training_wss_min/runs/wss_v51_wave1_20260916/X5D_v51_s1234
OUT=$PROJECT/training_wss_min/experiments/wss_time_ecc_20260918/acceptance_X5D_v51_s1234
cd "$PROJECT"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
rm -rf "$OUT"; mkdir -p "$OUT"
OMP_NUM_THREADS=4 $PY -m training_wss_min.evaluate --run-dir "$RUN" --partitions test --allow-test --checkpoint best --no-plots --output-dir "$OUT"
NEW=$(find "$OUT" -name metrics.json | head -1)
echo "new metrics: $NEW"
$PY training_wss_min/experiments/wss_time_ecc_20260918/offline/compare_metrics.py "$RUN/eval/ckpt_best/metrics.json" "$NEW" 1e-4 | tee "$OUT/compare.txt"
