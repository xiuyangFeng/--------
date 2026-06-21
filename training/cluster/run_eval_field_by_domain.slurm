#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_by_domain
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00

# V3D 分域评估：对单个 field run 按 AAA / AG / ILO 输出 metrics_by_domain.json
#
# 用法（仓库根目录）:
#   RUN_DIR_REL=outputs/field/field_v3d_..._20260521_103843 \
#   CHECKPOINT=best_model.pt \
#   sbatch training/cluster/run_eval_field_by_domain.slurm
#
# 可选:
#   TRAINING_ENV=GNN（默认）
#   SUBSET=test（默认）
#   DOMAINS="AAA AG ILO"（默认全部，透传 --domains）
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
CHECKPOINT="${CHECKPOINT:-best_model.pt}"
SUBSET="${SUBSET:-test}"
DOMAINS="${DOMAINS:-}"

if [ -z "$RUN_DIR_REL" ]; then
    echo "错误: 请设置 RUN_DIR_REL=outputs/field/<run 目录名>"
    exit 1
fi

RUN_DIR="$PROJECT_DIR/$RUN_DIR_REL"
CONFIG="$RUN_DIR/config.snapshot.json"
CKPT="$RUN_DIR/$CHECKPOINT"

if [ ! -f "$CONFIG" ]; then
    echo "错误: 缺少 config: $CONFIG"
    exit 1
fi
if [ ! -f "$CKPT" ]; then
    echo "错误: 缺少 checkpoint: $CKPT"
    exit 1
fi

DOMAIN_ARGS=()
if [ -n "$DOMAINS" ]; then
    # shellcheck disable=SC2206
    DOMAIN_ARGS=(--domains $DOMAINS)
fi

echo "RUN_DIR=$RUN_DIR"
echo "CHECKPOINT=$CKPT SUBSET=$SUBSET"
"$PYTHON_BIN" -m training.scripts.eval_field_by_domain \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --subset "$SUBSET" \
    "${DOMAIN_ARGS[@]}"
