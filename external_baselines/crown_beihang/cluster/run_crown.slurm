#!/usr/bin/env bash
#SBATCH --job-name=crown_train
#SBATCH --output=external_baselines/crown_beihang/cluster/logs/crown_%j.out
#SBATCH --error=external_baselines/crown_beihang/cluster/logs/crown_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=72:00:00

set -euo pipefail

CONFIG_PATH="${1:?missing config path}"

module purge

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
fi
conda activate GNN

PROJECT_DIR="${SLURM_SUBMIT_DIR:-}"
if [ -z "$PROJECT_DIR" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    BASELINE_DIR="$(dirname "$SCRIPT_DIR")"
    EXTERNAL_DIR="$(dirname "$BASELINE_DIR")"
    PROJECT_DIR="$(dirname "$EXTERNAL_DIR")"
fi
cd "$PROJECT_DIR"

SCRIPT_DIR="$PROJECT_DIR/external_baselines/crown_beihang/cluster"
mkdir -p "$SCRIPT_DIR/logs"

CONFIG_ABS="$PROJECT_DIR/$CONFIG_PATH"
if [ ! -f "$CONFIG_ABS" ]; then
    CONFIG_ABS="$CONFIG_PATH"
fi

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python -m external_baselines.crown_beihang.train --config "$CONFIG_ABS"
