#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=recomp_dual
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=2:00:00

# 用法（仓库根）:
#   RUN_DIR_REL=outputs/field/field_v3_..._20260508_001936 sbatch training/cluster/run_recompute_dual_test_metrics.slurm
#
set -euo pipefail

module purge
TRAINING_ENV=${TRAINING_ENV:-GNN}
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "$TRAINING_ENV"
PYTHON_BIN="$(command -v python)"

SCRIPT_PATH="$0"
case "$SCRIPT_PATH" in /*) ;; *) SCRIPT_PATH="$SLURM_SUBMIT_DIR/$SCRIPT_PATH" ;; esac
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
mkdir -p logs

RUN_DIR_REL="${RUN_DIR_REL:-}"
if [ -z "$RUN_DIR_REL" ]; then
    echo "错误: 请设置 RUN_DIR_REL=outputs/field/<run 目录名>"
    exit 1
fi

RUN_DIR="$PROJECT_DIR/$RUN_DIR_REL"
echo "RUN_DIR=$RUN_DIR"
"$PYTHON_BIN" -m training.scripts.recompute_dual_test_metrics --run-dir "$RUN_DIR"
