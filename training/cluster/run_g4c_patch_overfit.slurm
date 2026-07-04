#!/bin/bash
# G4-c Phase 1 · Geodesic Patch 单 case 过拟合（GPU）
# 用法:
#   sbatch training/cluster/run_g4c_patch_overfit.slurm overfit1c fast/CHEN_SHI_MING 1
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=g4c_p1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=8:00:00

set -eo pipefail

MODE=${1:-overfit1c}
CASE=${2:?需要 case，如 fast/CHEN_SHI_MING}
SEED=${3:-1}

PATCH_H=${G4C_PATCH_H:-12}
PATCH_W=${G4C_PATCH_W:-12}
RADIUS_MM=${G4C_RADIUS_MM:-3.0}
KNN_K=${G4C_KNN_K:-20}
EPOCHS=${G4C_EPOCHS:-200}
LR=${G4C_LR:-1e-3}
BATCH=${G4C_BATCH:-32}
EXP_ID=${G4C_EXP_ID:-V3P-M-E-G4c-PatchPhase1-post5463}

module purge
TRAINING_ENV=${TRAINING_ENV:-GNN}
PYTHON_BIN="${PYTHON_BIN:-/public/newhome/cy/.conda/envs/GNN/bin/python}"
if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$TRAINING_ENV" 2>/dev/null || true
fi
if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
fi
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python 不可用: $PYTHON_BIN" >&2
    exit 1
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "=== G4-c Phase 1 overfit $(date) ==="
echo "Job $SLURM_JOB_ID · Case $CASE · patch ${PATCH_H}x${PATCH_W} r=${RADIUS_MM}mm k=${KNN_K}"

"$PYTHON_BIN" -m training.scripts.train_field_g4c_patch \
    --mode "$MODE" \
    --case "$CASE" \
    --patch-h "$PATCH_H" \
    --patch-w "$PATCH_W" \
    --geodesic-radius-mm "$RADIUS_MM" \
    --knn-k "$KNN_K" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch-size "$BATCH" \
    --seed "$SEED" \
    --exp-id "$EXP_ID"

echo ">>> 完成 $(date) exit=$?"
