#!/usr/bin/env bash
#SBATCH --job-name=crown_pilot
#SBATCH --output=external_baselines/crown_beihang/cluster/logs/export_crown_pilot_%j.out
#SBATCH --error=external_baselines/crown_beihang/cluster/logs/export_crown_pilot_%j.err
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

CONFIG_PATH="${1:?missing export config path}"
PILOT_CASE="${2:-slow/LI_HUAN_GE}"

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

SCRIPT_PATH="${SLURM_JOB_SCRIPT:-}"
if [ -z "$SCRIPT_PATH" ] && command -v scontrol >/dev/null 2>&1; then
    SCRIPT_PATH="$(scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n' | awk -F= '/^Command=/{print $2; exit}')"
fi
if [ -z "$SCRIPT_PATH" ]; then
    SCRIPT_PATH="$0"
fi
case "$SCRIPT_PATH" in
    /*) ;;
    *) SCRIPT_PATH="$SLURM_SUBMIT_DIR/$SCRIPT_PATH" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
BASELINE_DIR="$(dirname "$SCRIPT_DIR")"
EXTERNAL_DIR="$(dirname "$BASELINE_DIR")"
PROJECT_DIR="$(dirname "$EXTERNAL_DIR")"
cd "$PROJECT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

CONFIG_ABS="$PROJECT_DIR/$CONFIG_PATH"
if [ ! -f "$CONFIG_ABS" ]; then
    CONFIG_ABS="$CONFIG_PATH"
fi

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
python -m external_baselines.crown_beihang.export_pkl \
    --config "$CONFIG_ABS" \
    --cases "$PILOT_CASE" \
    --write-partial \
    --write-pilot-timing
