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
source ~/.bashrc
conda activate "$TARGET_ENV"

cd $SLURM_SUBMIT_DIR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
CMD="python -m pipeline.${STEP_NAME}"
if [ -n "$CASE_NAME" ]; then
    CMD="$CMD --case $CASE_NAME"
fi
if [ "$STEP_NAME" = "extract_features" ] && [ "$ALLOW_NEAREST_BC" = "1" ]; then
    CMD="$CMD --allow-nearest-bc"
fi

echo "执行: $CMD"
$CMD

echo ""
echo "完成时间: $(date)"
