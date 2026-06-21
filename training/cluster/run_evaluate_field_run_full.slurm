#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_field_full
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00

# 训练完成后：对单个 run 执行 predict_field + A3/A4 + error/regional（--wss）+ evaluate_wss_credibility，写 evaluation_summary.json。
#
# 用法（在仓库根目录）:
#   RUN_DIR_REL=outputs/field/field_v3_..._20260508_153506 sbatch training/cluster/run_evaluate_field_run_full.slurm
#
# 可选环境变量:
#   TRAINING_ENV=GNN（默认）
#   CHECKPOINT=auto（默认；与 evaluate_field_run_full --checkpoint 一致）
#   SUBSET=test（默认）
#   EVAL_EXTRA_ARGS='--force' 等，透传给 python -m training.scripts.evaluate_field_run_full
#
set -euo pipefail

module purge
TRAINING_ENV=${TRAINING_ENV:-GNN}
TRAINING_PYTHON=${TRAINING_PYTHON:-""}

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
fi
if [ -n "$TRAINING_PYTHON" ]; then
    PYTHON_BIN="$TRAINING_PYTHON"
else
    conda activate "$TRAINING_ENV"
    PYTHON_BIN="$(command -v python)"
fi

SCRIPT_PATH="${SLURM_JOB_SCRIPT:-}"
if [ -z "$SCRIPT_PATH" ] && command -v scontrol >/dev/null 2>&1; then
    SCRIPT_PATH="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | awk -F= '/^Command=/{print $2; exit}')"
fi
if [ -z "$SCRIPT_PATH" ]; then
    SCRIPT_PATH="$0"
fi
case "$SCRIPT_PATH" in
    /*) ;;
    *) SCRIPT_PATH="$SLURM_SUBMIT_DIR/$SCRIPT_PATH" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$TRAINING_DIR")"
cd "$PROJECT_DIR"

mkdir -p "$SCRIPT_DIR/logs"
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

RUN_DIR_REL="${RUN_DIR_REL:-}"
if [ -z "$RUN_DIR_REL" ]; then
    echo "错误: 请设置 RUN_DIR_REL=outputs/field/<run 目录名>"
    exit 1
fi

RUN_DIR="$PROJECT_DIR/$RUN_DIR_REL"
CHECKPOINT="${CHECKPOINT:-auto}"
SUBSET="${SUBSET:-test}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"

# shellcheck disable=SC2206
EXTRA=( $EVAL_EXTRA_ARGS )

echo "RUN_DIR=$RUN_DIR"
echo "CHECKPOINT=$CHECKPOINT SUBSET=$SUBSET"
"$PYTHON_BIN" -m training.scripts.evaluate_field_run_full \
    --run-dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --subset "$SUBSET" \
    "${EXTRA[@]}"
