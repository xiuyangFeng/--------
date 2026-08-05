#!/bin/bash
#SBATCH --job-name=vfpreflight
#SBATCH --partition=GPU
#SBATCH --time=0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/wss_pinn/slurm/volume_field_preflight_%j.out
#SBATCH --error=outputs/wss_pinn/slurm/volume_field_preflight_%j.err

set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$PY" -m unittest discover -s wss_pinn/tests -p 'test_*.py' -v
CONFIG_ROOT=${VOLUME_CONFIG_ROOT:?VOLUME_CONFIG_ROOT is required}
PREFLIGHT_OUTPUT=${VOLUME_PREFLIGHT_OUTPUT:?VOLUME_PREFLIGHT_OUTPUT is required}
"$PY" -m wss_pinn.tools.preflight \
  --config-root "$CONFIG_ROOT" --output "$PREFLIGHT_OUTPUT"
"$PY" -m wss_pinn.tools.audit_dataset

CONFIG_LIST=${VOLUME_CONFIG_LIST:?VOLUME_CONFIG_LIST is required}
while IFS= read -r config; do
  [ -z "$config" ] && continue
  "$PY" -m wss_pinn.train --config "$config" --dry-run --device cuda
done < "$CONFIG_LIST"
