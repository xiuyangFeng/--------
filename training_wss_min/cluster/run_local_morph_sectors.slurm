#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=24:00:00
#SBATCH --job-name=lmorph_sectors
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
NAME=wss_local_morph_sectors_20260918
cd /public/newhome/cy/Digital_twin/GNN_localmorph_frozen_20260918
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.prepare_local_morph_sectors \
  --config-dir "$ROOT/training_wss_min/configs/$NAME" \
  --experiment-dir "$ROOT/training_wss_min/experiments/$NAME"
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.preflight_wss_x5d_long_prune \
  --config-dir "$ROOT/training_wss_min/configs/$NAME" \
  --experiment-dir "$ROOT/training_wss_min/experiments/$NAME" \
  --control-arm S6_s1234
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.run_wss_local_wave1_queue \
  --config-dir "$ROOT/training_wss_min/configs/$NAME" \
  --experiment-dir "$ROOT/training_wss_min/experiments/$NAME" \
  --slots-per-gpu 1 --min-free-mib 7000 --launch-interval-seconds 10
date -Is
