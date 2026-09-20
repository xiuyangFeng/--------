#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=x5d_prune_prep
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
FROZEN=/public/newhome/cy/Digital_twin/GNN_x5d_longitudinal_frozen_20260917
cd "$FROZEN"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.prepare_wss_x5d_long_prune \
  --execution-root "$FROZEN" \
  --config-dir "$ROOT/training_wss_min/configs/wss_x5d_long_prune_20260917" \
  --experiment-dir "$ROOT/training_wss_min/experiments/wss_x5d_long_prune_20260917"
