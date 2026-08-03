#!/bin/bash
#SBATCH --job-name=vftrain
#SBATCH --partition=GPU
#SBATCH --time=0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/wss_pinn/slurm/volume_field_train_%A_%a.out
#SBATCH --error=outputs/wss_pinn/slurm/volume_field_train_%A_%a.err

set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CONFIG_LIST=${VOLUME_CONFIG_LIST:?VOLUME_CONFIG_LIST is required}
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_LIST")
if [ -z "$CONFIG" ]; then
  echo "No config for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

"$PY" -m wss_pinn.volume_field.train --config "$CONFIG"
for checkpoint in best_data best_total last; do
  "$PY" -m wss_pinn.volume_field.evaluate \
    --config "$CONFIG" --checkpoint "$checkpoint" \
    --protocol same5k \
    --output-name "evaluation_same5k_${checkpoint}.json"
  "$PY" -m wss_pinn.volume_field.evaluate \
    --config "$CONFIG" --checkpoint "$checkpoint" \
    --protocol full_volume \
    --output-name "evaluation_fullvolume_${checkpoint}.json"
done
