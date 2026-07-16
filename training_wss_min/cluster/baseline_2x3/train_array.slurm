#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=wss2x3
#SBATCH --array=0-5%4
#SBATCH --output=training_wss_min/cluster/logs/%x_%A_%a.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%A_%a.err
#SBATCH --time=06:00:00

# One cell of the frozen 2x3 baseline. Development evaluation is val-only.
set -euo pipefail

PROJECT_DIR="/public/newhome/cy/Digital_twin/GNN"
MANIFEST="training_wss_min/configs/sweeps/baseline_2x3_simple.txt"
cd "$PROJECT_DIR"

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
test -n "$CONFIG"
test -f "$CONFIG"

if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    source /public/newapps/anaconda3/etc/profile.d/conda.sh
fi
conda activate GNN
PY="${CONDA_PREFIX}/bin/python"

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} config=${CONFIG} host=$(hostname)"
"$PY" -m training_wss_min.train --config "$CONFIG"
"$PY" -m training_wss_min.evaluate --config "$CONFIG" --partitions val --no-plots
echo "[$(date)] done config=${CONFIG}"
