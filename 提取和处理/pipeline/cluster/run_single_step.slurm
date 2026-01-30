#!/bin/bash
#SBATCH --job-name=gnn_step              # 作业名称
#SBATCH --partition=compute              # 分区
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/step_%j.out
#SBATCH --error=logs/step_%j.err

# ============================================================================
# GNN Pipeline - 单步骤运行脚本
# ============================================================================
# 使用方法:
#   sbatch run_single_step.slurm preprocess                    # 步骤1
#   sbatch run_single_step.slurm extract_features              # 步骤2
#   sbatch run_single_step.slurm normalize                     # 步骤3
#   sbatch run_single_step.slurm convert_to_graph              # 步骤4
#   sbatch run_single_step.slurm preprocess ZHANG_CHUN         # 指定病例
# ============================================================================

STEP_NAME=${1:-"preprocess"}
CASE_NAME=${2:-""}

# 环境配置
source ~/.bashrc
conda activate rag_venv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PIPELINE_DIR"

mkdir -p "$SCRIPT_DIR/logs"

echo "=============================================="
echo "运行步骤: $STEP_NAME"
echo "病例: ${CASE_NAME:-所有病例}"
echo "开始时间: $(date)"
echo "=============================================="

# 构建命令
CMD="python ${STEP_NAME}.py"
if [ -n "$CASE_NAME" ]; then
    CMD="$CMD --case $CASE_NAME"
fi

echo "执行: $CMD"
$CMD

echo ""
echo "完成时间: $(date)"
