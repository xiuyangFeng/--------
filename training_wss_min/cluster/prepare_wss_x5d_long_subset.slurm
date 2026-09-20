#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=x5d_long_sub_prep
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
FROZEN=/public/newhome/cy/Digital_twin/GNN_x5d_longitudinal_frozen_20260917
SIDECAR=$ROOT/training_wss_min/experiments/wss_x5d_longitudinal_20260917/frozen_longitudinal_features
cd "$FROZEN"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for spec in "wss_x5d_long_ref8_20260917 ref8" "wss_x5d_long_ref5_20260917 ref5"; do
  set -- $spec
  NAME=$1; CH=$2
  echo "=== preparing $NAME (channels=$CH) ==="
  /public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.prepare_wss_x5d_longitudinal_subset \
    --experiment-name "$NAME" --channels "$CH" \
    --execution-root "$FROZEN" --sidecar-root "$SIDECAR" \
    --config-dir "$ROOT/training_wss_min/configs/$NAME" \
    --experiment-dir "$ROOT/training_wss_min/experiments/$NAME"
done
