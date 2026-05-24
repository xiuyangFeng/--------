#!/bin/bash
#Author: XY
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=field_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00

# ============================================================================
# Training - 单配置训练脚本（限定 GPU 0/1/3，禁用 GPU 2）
# ============================================================================
# 使用方法:
#   sbatch run_train_field_gpu013.slurm training/configs/field/generated/v3_pointcloud/V3P-Main-01-PW_seed1.json
# ============================================================================

CONFIG_PATH=${1:-""}

if [ -z "$CONFIG_PATH" ]; then
    echo "错误: 必须提供配置文件路径"
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
ARCHIVE_DIR="$(dirname "$SCRIPT_DIR")"
CLUSTER_DIR="$(dirname "$ARCHIVE_DIR")"
TRAINING_DIR="$(dirname "$CLUSTER_DIR")"
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

# ============================================================================
# GPU 选择：限定 GPU 0/1/3，禁止使用 GPU 2
# SLURM 设置的 CUDA_VISIBLE_DEVICES 可能为 0/1/2/3 之一。
# 若 SLURM 分配了 GPU 2，则改用 GPU 3（当前空闲）作为后备。
# ============================================================================
SLURM_GPU="${CUDA_VISIBLE_DEVICES:-}"
ALLOWED_GPUS="0 1 3"

echo "SLURM 分配的 GPU: CUDA_VISIBLE_DEVICES=${SLURM_GPU}"

if [ -n "$SLURM_GPU" ]; then
    # 检查是否在允许列表中（单 GPU 情况，值为单个数字）
    IS_ALLOWED=0
    for g in $ALLOWED_GPUS; do
        if [ "$SLURM_GPU" = "$g" ]; then
            IS_ALLOWED=1
            break
        fi
    done

    if [ "$IS_ALLOWED" -eq 0 ]; then
        echo "警告: SLURM 分配了受限 GPU $SLURM_GPU（GPU 2 禁止使用）"
        echo "       强制切换到 GPU 3（备用）"
        export CUDA_VISIBLE_DEVICES=3
    fi
fi

echo "实际使用 GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
# ============================================================================

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_MAX_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "=============================================="
echo "Field Training - 单配置训练 [GPU限定: 0/1/3]"
echo "=============================================="
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo "工作目录: $PROJECT_DIR"
echo "训练环境: ${TRAINING_ENV:-custom python}"
echo "Python: $PYTHON_BIN"
echo "配置文件: $CONFIG_ABS"
echo "GPU限定: 0/1/3（禁止 GPU 2）"
echo "实际 CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "=============================================="
echo ""

CMD=("$PYTHON_BIN" -m training.scripts.train_field --config "$CONFIG_ABS")
echo "执行命令: ${CMD[*]}"
echo "线程限制: OMP=$OMP_NUM_THREADS  MKL=$MKL_NUM_THREADS"
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
