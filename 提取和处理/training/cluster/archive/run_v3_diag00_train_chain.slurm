#!/bin/bash
#Author: XY
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=v3_diag00_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#
# V3 推荐链路：先 run_v3_diag00（完整含前向），再 train_field 同配置跑 1 epoch（或其它 epochs）
# 用法（在仓库根目录）:
#   sbatch training/cluster/run_v3_diag00_train_chain.slurm \
#     training/configs/field/generated/v3_pointcloud/V3P-Diag-00_seed1.json
#
# 环境变量:
#   SKIP_DIAG_FORWARD=1   诊断阶段加 --skip-forward（不占 GPU 前向）
#   RUN_TRAIN=0           只跑诊断、不跑 train_field（默认 1）
#   TRAINING_ENV / TRAINING_PYTHON  同 run_train_field.slurm
#

CONFIG_PATH=${1:-""}

if [ -z "$CONFIG_PATH" ]; then
    echo "错误: 必须提供配置文件路径"
    echo "示例: sbatch training/cluster/run_v3_diag00_train_chain.slurm training/configs/field/generated/v3_pointcloud/V3P-Diag-00_seed1.json"
    exit 1
fi

module purge

TRAINING_ENV=${TRAINING_ENV:-GNN}
TRAINING_PYTHON=${TRAINING_PYTHON:-""}
SKIP_DIAG_FORWARD=${SKIP_DIAG_FORWARD:-0}
RUN_TRAIN=${RUN_TRAIN:-1}

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
echo "V3：diag00 → train_field（链式）"
echo "=============================================="
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo "工作目录: $PROJECT_DIR"
echo "Python: $PYTHON_BIN"
echo "配置: $CONFIG_ABS"
echo "SKIP_DIAG_FORWARD: $SKIP_DIAG_FORWARD"
echo "RUN_TRAIN: $RUN_TRAIN"
echo "=============================================="

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# Python 日志立即写入 SLURM .out（否则长时间无输出易被误判卡死）
export PYTHONUNBUFFERED=1

DIAG_CMD=("$PYTHON_BIN" -m training.scripts.run_v3_diag00 --config "$CONFIG_ABS")
if [ "$SKIP_DIAG_FORWARD" = "1" ]; then
    DIAG_CMD+=(--skip-forward)
fi
if [ -n "$DIAG_OUTPUT_DIR" ]; then
    DIAG_CMD+=(--output-dir "$DIAG_OUTPUT_DIR")
fi
if [ -n "$DIAG_EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=($DIAG_EXTRA_ARGS)
    DIAG_CMD+=("${EXTRA_ARR[@]}")
fi

echo ">>> [1/2] 诊断: ${DIAG_CMD[*]}"
"${DIAG_CMD[@]}"
DRC=$?
if [ "$DRC" != "0" ]; then
    echo "诊断失败，退出码 $DRC；跳过 train_field"
    exit "$DRC"
fi

if [ "$RUN_TRAIN" != "1" ]; then
    echo ">>> RUN_TRAIN!=1，跳过 train_field"
    exit 0
fi

TRAIN_CMD=("$PYTHON_BIN" -m training.scripts.train_field --config "$CONFIG_ABS")
echo ">>> [2/2] 训练: ${TRAIN_CMD[*]}"
"${TRAIN_CMD[@]}"
exit $?
