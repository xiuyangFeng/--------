#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --job-name=cyclem1s3_queue
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
# 周期积分量 M1 三头三 seed 确认队列（冻结副本执行）：M1 × seed 1234/7/2025，train136 → test34 读一次；每臂 train → eval best → eval last
# 3 × 4090 各 2 槽（每臂峰值显存 2–4 GB）；留一张卡给其他用户；队列每次启动前重新校验源码/配置指纹。
set -euo pipefail
FROZEN=/public/newhome/cy/Digital_twin/GNN_cycle_m1_frozen_20260921
MAIN=/public/newhome/cy/Digital_twin/GNN
cd "$FROZEN"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname; nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.run_wss_local_wave1_queue \
  --config-dir "$MAIN/training_wss_min/configs/wss_cycle_m1_stage3_20260922" \
  --experiment-dir "$MAIN/training_wss_min/experiments/wss_cycle_m1_stage3_20260922" \
  --slots-per-gpu 2 --min-free-mib 7000 --launch-interval-seconds 20
echo "===== queue done $(date -Is) ====="
