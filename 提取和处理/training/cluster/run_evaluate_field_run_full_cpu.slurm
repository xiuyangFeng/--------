#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=eval_field_cpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00

# G5 / TODO-62 · 单 run 完整评估（CPU · node03 · 不占 GPU）
#   predict_field（device=auto → CPU，因 CUDA_VISIBLE_DEVICES 置空）
#   + WSS 可信度 + regional + --clinical-pa
#
# 用法（仓库根目录）:
#   RUN_DIR_REL=outputs/field/<run_dir> sbatch training/cluster/run_evaluate_field_run_full_cpu.slurm
#
# 可选:
#   EVAL_EXTRA_ARGS='--clinical-pa --force'
#   CHECKPOINT=auto（默认）
#   SUBSET=test（默认）

set -euo pipefail

module purge
TRAINING_ENV=${TRAINING_ENV:-GNN}
if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "$TRAINING_ENV"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=""
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
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:---clinical-pa --force}"

# shellcheck disable=SC2206
EXTRA=( $EVAL_EXTRA_ARGS )

echo "RUN_DIR=$RUN_DIR"
echo "CHECKPOINT=$CHECKPOINT SUBSET=$SUBSET"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (CPU predict)"
python -m training.scripts.evaluate_field_run_full \
    --run-dir "$RUN_DIR" \
    --checkpoint "$CHECKPOINT" \
    --subset "$SUBSET" \
    "${EXTRA[@]}"

echo ">>> eval_field_cpu 完成 $(date)"
