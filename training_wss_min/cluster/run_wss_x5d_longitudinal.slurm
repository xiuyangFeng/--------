#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --job-name=x5d_long_3seed
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN_x5d_longitudinal_frozen_20260917
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is
hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.run_wss_local_wave1_queue \
  --config-dir /public/newhome/cy/Digital_twin/GNN/training_wss_min/configs/wss_x5d_longitudinal_20260917 \
  --experiment-dir /public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_x5d_longitudinal_20260917 \
  --slots-per-gpu 1 --min-free-mib 7000 --launch-interval-seconds 10
