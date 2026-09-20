#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=x5d_long_preflight
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN_x5d_longitudinal_frozen_20260917
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.preflight_wss_x5d_longitudinal \
  --config-dir /public/newhome/cy/Digital_twin/GNN/training_wss_min/configs/wss_x5d_longitudinal_20260917 \
  --experiment-dir /public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_x5d_longitudinal_20260917
