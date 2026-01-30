#!/bin/bash
# ============================================================================
# 批量提交多个病例作业
# ============================================================================
# 使用方法:
#   ./batch_submit.sh                    # 提交所有病例
#   ./batch_submit.sh ZHANG_CHUN ZHANG_HAO  # 提交指定病例
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 如果没有指定病例，获取所有病例
if [ $# -eq 0 ]; then
    # 从 data_new/AG/fast 获取所有病例目录名
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

echo "=============================================="
echo "批量提交作业"
echo "=============================================="
echo "病例列表:"
for case in $CASES; do
    echo "  - $case"
done
echo ""

# 提交作业
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
echo "作业ID:$JOB_IDS"
echo ""
echo "查看作业状态: squeue -u $USER"
echo "查看作业输出: tail -f logs/pipeline_<JOB_ID>.out"
echo "=============================================="
