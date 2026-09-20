#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=36:00:00
#SBATCH --job-name=tecc_queue
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
# 阶段 2 筛选队列（冻结副本执行）：T0 / TB8 / TB16 × cv3 三折，seed 1234；每臂 train → eval best → eval last
set -euo pipefail
FROZEN=/public/newhome/cy/Digital_twin/GNN_time_frozen_20260918
MAIN=/public/newhome/cy/Digital_twin/GNN
cd "$FROZEN"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname; nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.run_wss_local_wave1_queue \
  --config-dir "$MAIN/training_wss_min/configs/wss_time_ecc_20260918" \
  --experiment-dir "$MAIN/training_wss_min/experiments/wss_time_ecc_20260918" \
  --slots-per-gpu 1 --min-free-mib 7000 --launch-interval-seconds 10
