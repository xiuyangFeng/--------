#!/bin/bash
#Author: XY
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=field_plan
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=72:00:00
# 可选：指定节点
# #SBATCH -w node01

# ============================================================================
# Training - 顺序执行 manifest 中的多个实验
# ============================================================================
# 使用方法:
#   sbatch run_plan.slurm training/configs/field/generated/manifest.json
#   sbatch run_plan.slurm training/configs/field/generated/manifest.json baseline
#   sbatch run_plan.slurm training/configs/field/generated/manifest.json baseline A-Main-01 2 4
# ============================================================================

MANIFEST_PATH=${1:-"training/configs/field/generated/manifest.json"}
STUDY_GROUP=${2:-""}
EXP_ID=${3:-""}
SEED=${4:-""}
LIMIT=${5:-"0"}
DRY_RUN=${DRY_RUN:-0}

module purge

TRAINING_ENV=${TRAINING_ENV:-GNN}
TRAINING_PYTHON=${TRAINING_PYTHON:-""}

source ~/.bashrc
if [ -n "$TRAINING_PYTHON" ]; then
    PYTHON_BIN="$TRAINING_PYTHON"
else
    conda activate "$TRAINING_ENV"
    PYTHON_BIN="$(command -v python)"
fi

cd "$SLURM_SUBMIT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$TRAINING_DIR")"
cd "$PROJECT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

MANIFEST_ABS="$PROJECT_DIR/$MANIFEST_PATH"
if [ ! -f "$MANIFEST_ABS" ]; then
    MANIFEST_ABS="$MANIFEST_PATH"
fi

if [ ! -f "$MANIFEST_ABS" ]; then
    echo "错误: manifest 不存在: $MANIFEST_PATH"
    exit 1
fi

echo "=============================================="
echo "Field Training - Manifest 顺序执行"
echo "=============================================="
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo "工作目录: $PROJECT_DIR"
echo "训练环境: ${TRAINING_ENV:-custom python}"
echo "Python: $PYTHON_BIN"
echo "manifest: $MANIFEST_ABS"
echo "study_group: ${STUDY_GROUP:-全部}"
echo "exp_id: ${EXP_ID:-全部}"
echo "seed: ${SEED:-全部}"
echo "limit: $LIMIT"
echo "dry_run: $DRY_RUN"
echo "=============================================="
echo ""

CMD=("$PYTHON_BIN" -m training.run_field_plan --manifest "$MANIFEST_ABS" --python "$PYTHON_BIN")
if [ -n "$STUDY_GROUP" ]; then
    CMD+=(--study-group "$STUDY_GROUP")
fi
if [ -n "$EXP_ID" ]; then
    CMD+=(--exp-id "$EXP_ID")
fi
if [ -n "$SEED" ]; then
    CMD+=(--seed "$SEED")
fi
if [ "$LIMIT" != "0" ]; then
    CMD+=(--limit "$LIMIT")
fi
if [ "$DRY_RUN" = "1" ]; then
    CMD+=(--dry-run)
fi

echo "执行命令: ${CMD[*]}"
echo ""
"${CMD[@]}"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "计划执行完成"
echo "退出码: $EXIT_CODE"
echo "结束时间: $(date)"
echo "=============================================="

exit $EXIT_CODE
