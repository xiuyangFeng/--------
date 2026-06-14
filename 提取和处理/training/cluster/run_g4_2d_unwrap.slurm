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
UNWRAP_MODE=${G4_UNWRAP_MODE:-global}

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

# 与 run_train_field.slurm 一致：从作业脚本路径解析仓库根，避免 SLURM 工作目录漂移
SCRIPT_PATH="${SLURM_JOB_SCRIPT:-}"
if [ -z "$SCRIPT_PATH" ] && command -v scontrol >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_PATH="$(scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n' | awk -F= '/^Command=/{print $2; exit}')"
fi
if [ -z "$SCRIPT_PATH" ]; then
    SCRIPT_PATH="$0"
fi
case "$SCRIPT_PATH" in
    /*) ;;
    *) SCRIPT_PATH="${SLURM_SUBMIT_DIR:-.}/$SCRIPT_PATH" ;;
esac
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$TRAINING_DIR")"
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/training/cluster/logs"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

if ! "$PYTHON_BIN" -c "import training" 2>/dev/null; then
    echo "错误: 无法 import training · cwd=$PWD · PYTHONPATH=$PYTHONPATH"
    exit 1
fi

CMD=(
    "$PYTHON_BIN" -m training.scripts.train_field_2d_unwrap
    --mode "$MODE"
    --epochs "$EPOCHS"
    --lr "$LR"
    --batch-size "$BATCH"
    --seed "$SEED"
    --exp-id "$EXP_ID"
    --unwrap-mode "$UNWRAP_MODE"
)
if [ -n "$CASE" ] && [ "$CASE" != "-" ]; then
    CMD+=(--case "$CASE")
fi

echo "=============================================="
echo "G4 2D Unwrap Training"
echo "Job: $SLURM_JOB_ID  Node: $SLURM_NODELIST"
echo "Mode: $MODE  Seed: $SEED  Epochs: $EPOCHS  Unwrap: $UNWRAP_MODE"
echo "Workdir: $PROJECT_DIR"
echo "Start: $(date)"
echo "Cmd: ${CMD[*]}"
echo "=============================================="

"${CMD[@]}"
EXIT_CODE=$?
echo "Exit: $EXIT_CODE  End: $(date)"
exit $EXIT_CODE
