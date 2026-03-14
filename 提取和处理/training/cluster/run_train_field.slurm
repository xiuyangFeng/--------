#!/bin/bash
#Author: XY
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=field_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
# 可选：指定节点
# #SBATCH -w node01

# ============================================================================
# Training - 单配置训练脚本
# ============================================================================
# 使用方法:
#   sbatch run_train_field.slurm training/configs/field/transformer_geometry.json
#   TRAINING_ENV=GNN sbatch run_train_field.slurm training/configs/field/mlp_baseline.json
#   TRAINING_PYTHON=/public/newhome/cy/.conda/envs/GNN/bin/python \
#   sbatch run_train_field.slurm training/configs/field/transformer_geometry.json
# ============================================================================

CONFIG_PATH=${1:-""}

if [ -z "$CONFIG_PATH" ]; then
    echo "错误: 必须提供配置文件路径"
    echo "示例: sbatch run_train_field.slurm training/configs/field/transformer_geometry.json"
    exit 1
fi

module purge

TRAINING_ENV=${TRAINING_ENV:-GNN}
TRAINING_PYTHON=${TRAINING_PYTHON:-""}

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
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
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$TRAINING_DIR")"
cd "$PROJECT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

CONFIG_ABS="$PROJECT_DIR/$CONFIG_PATH"
if [ ! -f "$CONFIG_ABS" ]; then
    CONFIG_ABS="$CONFIG_PATH"
fi

if [ ! -f "$CONFIG_ABS" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

echo "=============================================="
echo "Field Training - 单配置训练"
echo "=============================================="
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo "工作目录: $PROJECT_DIR"
echo "训练环境: ${TRAINING_ENV:-custom python}"
echo "Python: $PYTHON_BIN"
echo "配置文件: $CONFIG_ABS"
echo "=============================================="
echo ""

CMD=("$PYTHON_BIN" -m training.train_field --config "$CONFIG_ABS")
echo "执行命令: ${CMD[*]}"
echo ""
"${CMD[@]}"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "训练完成"
echo "退出码: $EXIT_CODE"
echo "结束时间: $(date)"
echo "=============================================="

exit $EXIT_CODE
