#!/bin/bash
#Author: XY
# ============================================================================
# SLURM Array Job - 并行处理多个病例
# ============================================================================
# 每个节点192核，可同时处理多个病例，每个病例使用32核
# 理论上单节点可同时运行 192/32 = 6 个病例任务
# ============================================================================
#SBATCH --partition=CPU                  # 分区
#SBATCH --nodes=1                        # 每个任务使用1个节点
#SBATCH --ntasks-per-node=32             # 每任务32核
#SBATCH --job-name=gnn_array             # 作业名称
#SBATCH --output=logs/%x_%A_%a.out       # 标准输出 (%A=ArrayJobID, %a=ArrayTaskID)
#SBATCH --error=logs/%x_%A_%a.err        # 错误输出
#SBATCH --time=04:00:00                  # 单个任务最大运行时间
# Array Job 配置：处理 0-N 个病例，同时运行最多6个
# 使用前需要修改 --array 参数，例如: --array=0-9%6 表示10个病例，同时最多运行6个
#SBATCH --array=0-0%6

# ============================================================================
# 使用方法:
#   1. 先运行 ./generate_case_list.sh 生成病例列表
#   2. 修改上面的 --array=0-N%M，其中 N=病例数-1, M=同时运行数
#   3. sbatch run_array.slurm
#
# 示例:
#   假设有10个病例，同时最多运行6个:
#   #SBATCH --array=0-9%6
# ============================================================================

# 环境配置
module purge
PIPELINE_ENV=${PIPELINE_ENV:-GNN}
GEOMETRY_ENV=${GEOMETRY_ENV:-GNN_vmtk}
DEFAULT_GEOMETRY_PYTHON="$HOME/.conda/envs/$GEOMETRY_ENV/bin/python"
if [ -z "$GEOMETRY_PYTHON" ] && [ -x "$DEFAULT_GEOMETRY_PYTHON" ]; then
    GEOMETRY_PYTHON="$DEFAULT_GEOMETRY_PYTHON"
fi
source ~/.bashrc
conda activate "$PIPELINE_ENV"

cd $SLURM_SUBMIT_DIR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$PIPELINE_DIR")"
cd "$PROJECT_DIR"

mkdir -p "$SCRIPT_DIR/logs"

# 读取病例列表
CASE_LIST_FILE="$SCRIPT_DIR/case_list.txt"
if [ ! -f "$CASE_LIST_FILE" ]; then
    echo "错误: 病例列表文件不存在: $CASE_LIST_FILE"
    echo "请先运行: ./generate_case_list.sh"
    exit 1
fi

# 获取当前任务对应的病例名
CASE_NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CASE_LIST_FILE")
ALLOW_NEAREST_BC=${ALLOW_NEAREST_BC:-0}
GEOMETRY_PYTHON=${GEOMETRY_PYTHON:-""}

if [ -z "$CASE_NAME" ]; then
    echo "错误: 无法获取任务 $SLURM_ARRAY_TASK_ID 对应的病例名"
    exit 1
fi

echo "=============================================="
echo "GNN Pipeline - Array Job"
echo "=============================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "病例名称: $CASE_NAME"
echo "节点: $SLURM_NODELIST"
echo "CPU核数: $SLURM_NTASKS_PER_NODE"
echo "主环境: $PIPELINE_ENV"
echo "几何环境: $GEOMETRY_ENV"
if [ -n "$GEOMETRY_PYTHON" ]; then
    echo "geometry-python: $GEOMETRY_PYTHON"
fi
echo "开始时间: $(date)"
echo "=============================================="
echo ""

# 运行 pipeline
CMD="python -m pipeline.run_all --case $CASE_NAME"
if [ "$ALLOW_NEAREST_BC" = "1" ]; then
    CMD="$CMD --allow-nearest-bc"
fi
if [ -n "$GEOMETRY_PYTHON" ]; then
    CMD="$CMD --geometry-python $GEOMETRY_PYTHON"
fi
echo "执行: $CMD"
$CMD

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "任务完成"
echo "病例: $CASE_NAME"
echo "退出码: $EXIT_CODE"
echo "结束时间: $(date)"
echo "=============================================="

exit $EXIT_CODE
