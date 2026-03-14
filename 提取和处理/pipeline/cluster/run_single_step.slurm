#!/bin/bash
#Author: XY
#SBATCH --partition=CPU                  # 分区
#SBATCH --nodes=1                        # 节点数
#SBATCH --ntasks-per-node=16             # 每节点任务数
#SBATCH --job-name=gnn_step              # 作业名称
#SBATCH --output=logs/%x_%j.out          # 标准输出
#SBATCH --error=logs/%x_%j.err           # 错误输出
#SBATCH --time=02:00:00                  # 最大运行时间
# 可选：指定节点
# #SBATCH -w node01

# ============================================================================
# GNN Pipeline - 单步骤运行脚本
# ============================================================================
# 使用方法:
#   sbatch run_single_step.slurm preprocess                    # 步骤1
#   sbatch run_single_step.slurm extract_features              # 步骤2
#   sbatch run_single_step.slurm coord_normalize               # 步骤3
#   sbatch run_single_step.slurm normalize                     # 步骤4
#   sbatch run_single_step.slurm convert_to_graph              # 步骤5
#   sbatch run_single_step.slurm preprocess ZHANG_CHUN         # 指定病例
# ============================================================================

set -euo pipefail

STEP_NAME=${1:-"preprocess"}
CASE_NAME=${2:-""}
ALLOW_NEAREST_BC=${ALLOW_NEAREST_BC:-0}
PIPELINE_ENV=${PIPELINE_ENV:-GNN}
GEOMETRY_ENV=${GEOMETRY_ENV:-GNN_vmtk}

# 环境配置
module purge
TARGET_ENV="$PIPELINE_ENV"
if [ "$STEP_NAME" = "extract_features" ]; then
    TARGET_ENV="$GEOMETRY_ENV"
fi

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # 非交互式 SLURM shell 下需要显式加载 conda.sh，否则 conda activate 会失败。
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
else
    echo "错误: 找不到 conda 初始化脚本，请设置 PATH 或在脚本中指定 module load anaconda3"
    exit 1
fi
conda activate "$TARGET_ENV"

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
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$PIPELINE_DIR")"
cd "$PROJECT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

echo "=============================================="
echo "运行步骤: $STEP_NAME"
echo "病例: ${CASE_NAME:-所有病例}"
echo "环境: $TARGET_ENV"
echo "开始时间: $(date)"
echo "=============================================="

# 构建命令
CMD=(python -m "pipeline.${STEP_NAME}")
if [ -n "$CASE_NAME" ]; then
    CMD+=(--case "$CASE_NAME")
fi
if [ "$STEP_NAME" = "extract_features" ] && [ "$ALLOW_NEAREST_BC" = "1" ]; then
    CMD+=(--allow-nearest-bc)
fi

echo "执行: ${CMD[*]}"
"${CMD[@]}"

echo ""
echo "完成时间: $(date)"
