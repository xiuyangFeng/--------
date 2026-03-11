#!/bin/bash
#Author: XY
# ============================================================================
# 批量提交多个病例作业
# ============================================================================
# 使用方法:
#   ./batch_submit.sh                       # 提交所有病例（独立作业模式）
#   ./batch_submit.sh ZHANG_CHUN ZHANG_HAO  # 提交指定病例
#   ./batch_submit.sh --array               # 使用 Array Job 模式（推荐）
#   ./batch_submit.sh --array CASE1 CASE2   # Array Job 指定病例
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 创建日志目录
mkdir -p logs

# 检查是否使用 Array Job 模式
USE_ARRAY=false
if [ "$1" = "--array" ] || [ "$1" = "-a" ]; then
    USE_ARRAY=true
    shift  # 移除 --array 参数
fi

# 如果没有指定病例，获取所有病例
if [ $# -eq 0 ]; then
    DATA_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")/data_new/AG/fast"
    if [ -d "$DATA_DIR" ]; then
        CASES=$(ls -d "$DATA_DIR"/*/ 2>/dev/null | xargs -n1 basename)
    else
        echo "错误: 数据目录不存在: $DATA_DIR"
        exit 1
    fi
else
    CASES="$@"
fi

# 统计病例数
CASE_COUNT=$(echo "$CASES" | wc -w | tr -d ' ')

echo "=============================================="
echo "批量提交作业"
echo "=============================================="
echo "模式: $([ "$USE_ARRAY" = true ] && echo 'Array Job (推荐)' || echo '独立作业')"
echo "病例总数: $CASE_COUNT"
echo "主环境: ${PIPELINE_ENV:-GNN}"
echo "几何环境: ${GEOMETRY_ENV:-GNN_vmtk}"
if [ -n "$GEOMETRY_PYTHON" ]; then
    echo "geometry-python: $GEOMETRY_PYTHON"
fi
echo ""
echo "病例列表:"
for case in $CASES; do
    echo "  - $case"
done
echo ""

if [ "$USE_ARRAY" = true ]; then
    # Array Job 模式
    echo "生成病例列表文件..."
    ./generate_case_list.sh $CASES
    
    echo ""
    echo "提交 Array Job..."
    ARRAY_RANGE="0-$((CASE_COUNT-1))%6"  # 同时最多运行6个
    JOB_ID=$(sbatch --parsable --array=$ARRAY_RANGE run_array.slurm)
    
    echo ""
    echo "=============================================="
    echo "Array Job 已提交"
    echo "=============================================="
    echo "作业ID: $JOB_ID"
    echo "Array 范围: $ARRAY_RANGE"
    echo "同时运行: 最多6个任务"
    echo ""
    echo "查看作业状态: squeue -u $USER"
    echo "查看单个任务输出: tail -f logs/gnn_array_${JOB_ID}_<TASK_ID>.out"
    echo "取消所有任务: scancel $JOB_ID"
    echo "=============================================="
else
    # 独立作业模式
    JOB_IDS=""
    for case in $CASES; do
        echo "提交: $case"
        JOB_ID=$(sbatch --parsable run_pipeline.slurm "$case")
        echo "  作业ID: $JOB_ID"
        JOB_IDS="$JOB_IDS $JOB_ID"
    done

    echo ""
    echo "=============================================="
    echo "所有作业已提交"
    echo "=============================================="
    echo "作业ID:$JOB_IDS"
    echo ""
    echo "查看作业状态: squeue -u $USER"
    echo "查看作业输出: tail -f logs/gnn_pipeline_<JOB_ID>.out"
    echo "=============================================="
fi
