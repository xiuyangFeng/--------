#!/usr/bin/env bash
#SBATCH --job-name=crown_eval
#SBATCH --output=external_baselines/crown_beihang/cluster/logs/crown_eval_%j.out
#SBATCH --error=external_baselines/crown_beihang/cluster/logs/crown_eval_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
# paper 全点 evaluate 约 4–5h；subsample 约数分钟。统一留 8h 余量。
#SBATCH --time=08:00:00

set -euo pipefail

CHECKPOINT_PATH="${1:?missing checkpoint path}"
SPLIT_NAME="${2:-test}"
EVAL_MODE="${3:-paper}"
LOG_EVERY="${4:-20}"
OUTPUT_DIR="${5:-}"

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

CHECKPOINT_ABS="$PROJECT_DIR/$CHECKPOINT_PATH"
if [ ! -f "$CHECKPOINT_ABS" ]; then
    CHECKPOINT_ABS="$CHECKPOINT_PATH"
fi

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "[crown_eval_slurm] job=${SLURM_JOB_ID:-local} checkpoint=${CHECKPOINT_ABS} split=${SPLIT_NAME} mode=${EVAL_MODE} log_every=${LOG_EVERY}"

EVAL_ARGS=(
    --checkpoint "$CHECKPOINT_ABS"
    --split "$SPLIT_NAME"
    --eval-mode "$EVAL_MODE"
    --log-every "$LOG_EVERY"
    --forward-chunk-size 65536
)
if [ -n "$OUTPUT_DIR" ]; then
    OUT_ABS="$PROJECT_DIR/$OUTPUT_DIR"
    if [ ! -d "$OUT_ABS" ] && [ -d "$OUTPUT_DIR" ]; then
        OUT_ABS="$OUTPUT_DIR"
    fi
    EVAL_ARGS+=(--output-dir "$OUT_ABS")
fi

python -m external_baselines.crown_beihang.evaluate "${EVAL_ARGS[@]}"
