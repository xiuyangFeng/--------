#!/bin/bash
#Author: XY
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=v3_diag00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=8:00:00
#
# V3P-Diag-00：仅运行诊断脚本（§9.1.1 / P0-G）
# 用法（在仓库根目录）:
#   sbatch training/cluster/run_v3_diag00.slurm training/configs/field/generated/v3_pointcloud/V3P-Diag-00_seed1.json
# 仅数据统计、不要 GPU 前向:
#   SKIP_FORWARD=1 sbatch training/cluster/run_v3_diag00.slurm .../V3P-Diag-00_seed1.json
# 覆盖诊断输出目录:
#   DIAG_OUTPUT_DIR=/path/to/out sbatch training/cluster/run_v3_diag00.slurm ...
#

CONFIG_PATH=${1:-""}

if [ -z "$CONFIG_PATH" ]; then
    echo "错误: 必须提供配置文件路径（与 train_field 相同 JSON）"
    echo "示例: sbatch training/cluster/run_v3_diag00.slurm training/configs/field/generated/v3_pointcloud/V3P-Diag-00_seed1.json"
    exit 1
fi

module purge

TRAINING_ENV=${TRAINING_ENV:-GNN}
TRAINING_PYTHON=${TRAINING_PYTHON:-""}
SKIP_FORWARD=${SKIP_FORWARD:-0}

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
echo "V3P-Diag-00 诊断脚本（run_v3_diag00）"
echo "=============================================="
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo "工作目录: $PROJECT_DIR"
echo "Python: $PYTHON_BIN"
echo "配置: $CONFIG_ABS"
echo "SKIP_FORWARD: $SKIP_FORWARD"
echo "=============================================="

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

CMD=("$PYTHON_BIN" -m training.scripts.run_v3_diag00 --config "$CONFIG_ABS")
if [ "$SKIP_FORWARD" = "1" ]; then
    CMD+=(--skip-forward)
fi
if [ -n "$DIAG_OUTPUT_DIR" ]; then
    CMD+=(--output-dir "$DIAG_OUTPUT_DIR")
fi
if [ -n "$DIAG_EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=($DIAG_EXTRA_ARGS)
    CMD+=("${EXTRA_ARR[@]}")
fi

echo "执行命令: ${CMD[*]}"
"${CMD[@]}"
exit $?
