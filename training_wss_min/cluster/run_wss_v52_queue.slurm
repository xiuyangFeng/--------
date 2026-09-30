#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --job-name=wss_v52_queue
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=72:00:00
# v5.2 retraining (2026-09-23): 40 trainings + 80 evaluations (IND 2x5 seeds, CV5 2x5 folds x3 seeds), 3 x RTX 4090, 2 slots per GPU
# (one GPU of master is held by another user's job at submission time). Runs inside the v5.2 frozen copy; the queue re-verifies fingerprints before every launch.
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
NAME=wss_v52_20260923
cd /public/newhome/cy/Digital_twin/GNN_v52_frozen_20260923
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.run_wss_local_wave1_queue \
  --config-dir "$ROOT/training_wss_min/configs/$NAME" \
  --experiment-dir "$ROOT/training_wss_min/experiments/$NAME" \
  --slots-per-gpu 2 --min-free-mib 7000 --launch-interval-seconds 45
date -Is
