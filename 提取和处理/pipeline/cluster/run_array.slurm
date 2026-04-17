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
# 大规模 CPU 任务指定 node03（与 pipeline/cluster 规范一致）
#SBATCH -w node03
#SBATCH --array=0-0%6

# ============================================================================
# 使用方法:
#   1. 先运行 ./generate_case_list.sh 生成病例列表（AG/slow: ./generate_case_list.sh --slow）
#   2. 修改上面的 --array=0-N%M，其中 N=病例数-1, M=同时运行数
#   3. AG/slow: export PIPELINE_SOURCES=AG/slow 或  ./batch_submit.sh --array --slow
#   4. sbatch run_array.slurm
#
# 示例:
#   假设有10个病例，同时最多运行6个:
#   #SBATCH --array=0-9%6
# ============================================================================

set -euo pipefail

# 环境配置
module purge
PIPELINE_ENV=${PIPELINE_ENV:-GNN}
GEOMETRY_ENV=${GEOMETRY_ENV:-GNN_vmtk}
# set -u 下须先定义，再参与下述检测（与 run_pipeline.slurm 一致）
GEOMETRY_PYTHON=${GEOMETRY_PYTHON:-""}
DEFAULT_GEOMETRY_PYTHON="$HOME/.conda/envs/$GEOMETRY_ENV/bin/python"
if [ -z "$GEOMETRY_PYTHON" ] && [ -x "$DEFAULT_GEOMETRY_PYTHON" ]; then
    GEOMETRY_PYTHON="$DEFAULT_GEOMETRY_PYTHON"
fi
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
else
    echo "错误: 找不到 conda 初始化脚本，请设置 PATH 或在脚本中指定 module load anaconda3"
    exit 1
fi
conda activate "$PIPELINE_ENV"

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
START_STEP=${START_STEP:-1}
END_STEP=${END_STEP:-5}
SAMPLING_METHOD=${SAMPLING_METHOD:-hybrid}
FPS_RATIO=${FPS_RATIO:-0.2}

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
echo "步骤范围: $START_STEP -> $END_STEP"
echo "采样方法: $SAMPLING_METHOD"
echo "FPS 占比: $FPS_RATIO"
echo "允许最近 BC 兜底: $ALLOW_NEAREST_BC"
if [ -n "$GEOMETRY_PYTHON" ]; then
    echo "geometry-python: $GEOMETRY_PYTHON"
fi
echo "开始时间: $(date)"
echo "=============================================="
echo ""

# 运行 pipeline
CMD=(python -m pipeline.run_all --case "$CASE_NAME")
CMD+=(--start-step "$START_STEP")
CMD+=(--end-step "$END_STEP")
CMD+=(--sampling-method "$SAMPLING_METHOD")
CMD+=(--fps-ratio "$FPS_RATIO")
if [ "$ALLOW_NEAREST_BC" = "1" ]; then
    CMD+=(--allow-nearest-bc)
fi
if [ -n "$GEOMETRY_PYTHON" ]; then
    CMD+=(--geometry-python "$GEOMETRY_PYTHON")
fi
if [ -n "${PIPELINE_SOURCES:-}" ]; then
    CMD+=(--sources)
    for s in $PIPELINE_SOURCES; do
        CMD+=("$s")
    done
fi
echo "执行: ${CMD[*]}"
"${CMD[@]}"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "任务完成"
echo "病例: $CASE_NAME"
echo "退出码: $EXIT_CODE"
echo "结束时间: $(date)"
echo "=============================================="

exit $EXIT_CODE
