#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=wsspn_matrix
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=10:00:00

# One config -> train -> best/last full-cloud train+locked-test eval -> best PostView.
# This script is an execution entrypoint only; creating it does not submit any job.

set -euo pipefail

CONFIG="${1:?usage: sbatch run_pointnet_distribution_matrix.slurm <config.json>}"
PROJECT_DIR="/public/newhome/cy/Digital_twin/GNN"
cd "$PROJECT_DIR"

if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
else
    echo "error: conda initialization script not found"; exit 1
fi
conda activate GNN
PY="${CONDA_PREFIX}/bin/python"

RUN_DIR="$($PY -c 'import sys; from training_wss_min.config import ExpConfig; print(ExpConfig.from_json(sys.argv[1]).run_dir)' "$CONFIG")"
POSTVIEW_DIR="${RUN_DIR}/postview/ckpt_best/test"

echo "config=${CONFIG}"
echo "run_dir=${RUN_DIR}"
echo "started=$(date) host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-NA}"

"$PY" -m training_wss_min.train --config "$CONFIG"

for CHECKPOINT in best last; do
    EXTRA_EVAL_ARGS=()
    if [ "$CHECKPOINT" = "last" ]; then EXTRA_EVAL_ARGS+=(--no-plots); fi
    "$PY" -m training_wss_min.evaluate \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --partitions train,test \
        --allow-test \
        "${EXTRA_EVAL_ARGS[@]}"
done

"$PY" -m training_wss_min.tools.export_wss_postview \
    --run-dir "$RUN_DIR" \
    --checkpoint best \
    --partition test \
    --allow-test \
    --output-dir "$POSTVIEW_DIR"

echo "completed=$(date)"
