#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=cycle_c1c2
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
# 周期积分量矩阵 阶段 0 / C1 自由基线 + C2 标签内分析（只读，CPU）
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_cycle_20260920/offline
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
for s in c1_baselines c2_label_analysis; do
  echo "===== $s $(date -Is) ====="
  $PY -u $s.py 2>&1 | tee ${s}_run.log
done
echo "===== done $(date -Is) ====="
