#!/bin/bash
# G4 · 2D 壁面展开 U-Net 训练（Phase 1 过拟合 / Phase 2 Probe）
# 用法（仓库根）:
#   sbatch training/cluster/run_g4_2d_unwrap.slurm probe seed1
#   sbatch training/cluster/run_g4_2d_unwrap.slurm overfit1c slow/LI_HUAN_GE seed1
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=g4_2d_unwrap
#SBATCH --output=training/cluster/logs/%x_%j.out
#SBATCH --error=training/cluster/logs/%x_%j.err
#SBATCH --time=12:00:00

MODE=${1:-probe}
CASE=${2:-}
SEED=${3:-1}
EPOCHS=${G4_EPOCHS:-150}
LR=${G4_LR:-1e-3}
BATCH=${G4_BATCH:-32}
EXP_ID=${G4_EXP_ID:-V3P-G-60-2DUnwrap}

module purge

TRAINING_ENV=${TRAINING_ENV:-GNN}
if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "$TRAINING_ENV"
PYTHON_BIN="$(command -v python)"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/training/cluster/logs"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

CMD=(
    "$PYTHON_BIN" -m training.scripts.train_field_2d_unwrap
    --mode "$MODE"
    --epochs "$EPOCHS"
    --lr "$LR"
    --batch-size "$BATCH"
    --seed "$SEED"
    --exp-id "$EXP_ID"
)
if [ -n "$CASE" ] && [ "$CASE" != "-" ]; then
    CMD+=(--case "$CASE")
fi

echo "=============================================="
echo "G4 2D Unwrap Training"
echo "Job: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Mode: $MODE  Seed: $SEED  Epochs: $EPOCHS"
echo "Workdir: $PROJECT_DIR"
echo "Start: $(date)"
echo "Cmd: ${CMD[*]}"
echo "=============================================="

"${CMD[@]}"
EXIT_CODE=$?
echo "Exit: $EXIT_CODE  End: $(date)"
exit $EXIT_CODE
