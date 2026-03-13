#!/bin/bash
#Author: XY
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=field_array
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
# 使用前请通过 sbatch --array=0-N%M run_array.slurm 显式传入任务范围
#SBATCH --array=0-0%4

# ============================================================================
# Training - Array Job 并行执行多个配置
# ============================================================================
# 使用方法:
#   1. 先运行 ./generate_manifest_list.sh 生成配置列表
#   2. 提交: sbatch --array=0-11%4 run_array.slurm
#
# 配置列表格式:
#   exp_id<TAB>study_group<TAB>seed<TAB>config_path
# ============================================================================

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

MANIFEST_LIST_FILE="${MANIFEST_LIST_FILE:-$SCRIPT_DIR/manifest_list.tsv}"
if [ ! -f "$MANIFEST_LIST_FILE" ]; then
    echo "错误: 配置列表不存在: $MANIFEST_LIST_FILE"
    echo "请先运行: ./generate_manifest_list.sh"
    exit 1
fi

ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST_LIST_FILE")
if [ -z "$ROW" ]; then
    echo "错误: 无法获取任务 $SLURM_ARRAY_TASK_ID 对应的配置行"
    exit 1
fi

EXP_ID=$(printf '%s\n' "$ROW" | cut -f1)
STUDY_GROUP=$(printf '%s\n' "$ROW" | cut -f2)
SEED=$(printf '%s\n' "$ROW" | cut -f3)
CONFIG_PATH=$(printf '%s\n' "$ROW" | cut -f4)

CONFIG_ABS="$PROJECT_DIR/$CONFIG_PATH"
if [ ! -f "$CONFIG_ABS" ]; then
    CONFIG_ABS="$CONFIG_PATH"
fi

if [ ! -f "$CONFIG_ABS" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

echo "=============================================="
echo "Field Training - Array Job"
echo "=============================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "exp_id: $EXP_ID"
echo "study_group: $STUDY_GROUP"
echo "seed: $SEED"
echo "配置文件: $CONFIG_ABS"
echo "节点: $SLURM_NODELIST"
echo "训练环境: ${TRAINING_ENV:-custom python}"
echo "Python: $PYTHON_BIN"
echo "开始时间: $(date)"
echo "=============================================="
echo ""

CMD=("$PYTHON_BIN" -m training.train_field --config "$CONFIG_ABS")
echo "执行命令: ${CMD[*]}"
echo ""
"${CMD[@]}"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "任务完成"
echo "exp_id: $EXP_ID"
echo "seed: $SEED"
echo "退出码: $EXIT_CODE"
echo "结束时间: $(date)"
echo "=============================================="

exit $EXIT_CODE
