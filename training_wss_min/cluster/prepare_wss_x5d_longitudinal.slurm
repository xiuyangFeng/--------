#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=x5d_long_prepare
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
FROZEN=/public/newhome/cy/Digital_twin/GNN_x5d_longitudinal_frozen_20260917
EXP=$ROOT/training_wss_min/experiments/wss_x5d_longitudinal_20260917
cd "$FROZEN"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
test ! -e "$EXP/frozen_longitudinal_features"
cp -a "$ROOT/outputs/wss_v5_longitudinal_features_20260916/coverage_features/features" "$EXP/frozen_longitudinal_features"
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.prepare_wss_x5d_longitudinal \
  --execution-root "$FROZEN" --sidecar-root "$EXP/frozen_longitudinal_features" \
  --config-dir "$ROOT/training_wss_min/configs/wss_x5d_longitudinal_20260917" --experiment-dir "$EXP"
