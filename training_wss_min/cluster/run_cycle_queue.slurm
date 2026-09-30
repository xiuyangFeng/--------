#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --job-name=cycle_queue
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
# 周期积分量矩阵 阶段 1 筛选队列（冻结副本执行）：A1 / O1 / O2 × cv3 三折，seed 1234；每臂 train → eval best → eval last
# 4 × 4090，每卡 1 槽（9 臂并行 4 个，约 3 轮 ≈ 1.5–2 h）；队列在每次启动前重新校验源码/配置指纹。
set -euo pipefail
FROZEN=/public/newhome/cy/Digital_twin/GNN_cycle_frozen_20260920
MAIN=/public/newhome/cy/Digital_twin/GNN
cd "$FROZEN"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname; nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.run_wss_local_wave1_queue \
  --config-dir "$MAIN/training_wss_min/configs/wss_cycle_20260920" \
  --experiment-dir "$MAIN/training_wss_min/experiments/wss_cycle_20260920" \
  --slots-per-gpu 1 --min-free-mib 7000 --launch-interval-seconds 10
echo "===== queue done $(date -Is) ====="
