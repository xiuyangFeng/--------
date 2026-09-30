#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --job-name=wss_lc_queue
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=60:00:00
# Learning curve on cv3_v51 (2026-09-20): 33 trainings + 66 evaluations, 4 x RTX 4090, 2 slots per GPU.
# Runs inside the v5.1 frozen copy; the queue re-verifies source/config fingerprints before every launch.
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
NAME=wss_learning_curve_20260920
cd /public/newhome/cy/Digital_twin/GNN_v51_frozen
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.run_wss_local_wave1_queue \
  --config-dir "$ROOT/training_wss_min/configs/$NAME" \
  --experiment-dir "$ROOT/training_wss_min/experiments/$NAME" \
  --slots-per-gpu 2 --min-free-mib 7000 --launch-interval-seconds 45
date -Is
