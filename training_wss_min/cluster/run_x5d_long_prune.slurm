#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --job-name=x5d_prune_ladder
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
cd /public/newhome/cy/Digital_twin/GNN_x5d_longitudinal_frozen_20260917
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.run_wss_local_wave1_queue \
  --config-dir "$ROOT/training_wss_min/configs/wss_x5d_long_prune_20260917" \
  --experiment-dir "$ROOT/training_wss_min/experiments/wss_x5d_long_prune_20260917" \
  --slots-per-gpu 1 --min-free-mib 7000 --launch-interval-seconds 10
date -Is
