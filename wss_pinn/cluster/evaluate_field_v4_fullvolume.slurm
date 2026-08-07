#!/bin/bash
#SBATCH --job-name=vffullval
#SBATCH --partition=GPU
#SBATCH --time=0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/wss_pinn/slurm/volume_field_fullval_%A_%a.out
#SBATCH --error=outputs/wss_pinn/slurm/volume_field_fullval_%A_%a.err

set -euo pipefail
ulimit -n 65535
cd /public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
CONFIG_LIST=${VOLUME_CONFIG_LIST:?VOLUME_CONFIG_LIST is required}
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CONFIG_LIST")
if [ -z "$CONFIG" ]; then
  echo "No config for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

"$PY" -m wss_pinn.evaluate \
  --config "$CONFIG" \
  --checkpoint best_validation_field_cb \
  --protocol full_volume \
  --output-name evaluation_fullvolume_best_validation_field_cb.json
