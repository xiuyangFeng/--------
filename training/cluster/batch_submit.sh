#!/bin/bash
#Author: XY
# ============================================================================
# 批量提交 training 作业
# ============================================================================
# 使用方法:
#   ./batch_submit.sh
#   ./batch_submit.sh --array
#   ./batch_submit.sh --study-group baseline --limit 4
#   ./batch_submit.sh --array --manifest training/configs/field/generated/manifest.json --study-group baseline
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

USE_ARRAY=false
MANIFEST_PATH="training/configs/field/generated/manifest.json"
STUDY_GROUP=""
EXP_ID=""
SEED=""
LIMIT="0"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

while [ $# -gt 0 ]; do
    case "$1" in
        --array|-a)
            USE_ARRAY=true
            ;;
        --manifest)
            shift
            MANIFEST_PATH="${1:-}"
            ;;
        --study-group)
            shift
            STUDY_GROUP="${1:-}"
            ;;
        --exp-id)
            shift
            EXP_ID="${1:-}"
            ;;
        --seed)
            shift
            SEED="${1:-}"
            ;;
        --limit)
            shift
            LIMIT="${1:-0}"
            ;;
        *)
            echo "错误: 未知参数: $1"
            exit 1
            ;;
    esac
    shift
done

MANIFEST_LIST_FILE="${MANIFEST_LIST_FILE:-$SCRIPT_DIR/manifest_list.tsv}"
./generate_manifest_list.sh "$MANIFEST_PATH" "$STUDY_GROUP" "$EXP_ID" "$SEED" "$LIMIT"

if [ ! -f "$MANIFEST_LIST_FILE" ]; then
    echo "错误: 配置列表不存在: $MANIFEST_LIST_FILE"
    exit 1
fi

ITEM_COUNT=$(wc -l < "$MANIFEST_LIST_FILE" | tr -d ' ')
if [ "$ITEM_COUNT" -le 0 ]; then
    echo "错误: 没有可提交的实验项"
    exit 1
fi

echo "=============================================="
echo "批量提交 Training 作业"
echo "=============================================="
echo "模式: $([ "$USE_ARRAY" = true ] && echo 'Array Job' || echo '独立作业')"
echo "manifest: $MANIFEST_PATH"
echo "study_group: ${STUDY_GROUP:-全部}"
echo "exp_id: ${EXP_ID:-全部}"
echo "seed: ${SEED:-全部}"
echo "limit: $LIMIT"
echo "实验总数: $ITEM_COUNT"
echo "训练环境: ${TRAINING_ENV:-GNN}"
if [ -n "${TRAINING_PYTHON:-}" ]; then
    echo "Python: $TRAINING_PYTHON"
fi
echo "=============================================="
echo ""

if [ "$USE_ARRAY" = true ]; then
    ARRAY_RANGE="0-$((ITEM_COUNT - 1))%${MAX_PARALLEL}"
    JOB_ID=$(sbatch --parsable --array="$ARRAY_RANGE" run_array.slurm)

    echo "Array Job 已提交"
    echo "作业ID: $JOB_ID"
    echo "Array 范围: $ARRAY_RANGE"
    echo "查看输出: tail -f logs/field_array_${JOB_ID}_<TASK_ID>.out"
    echo "取消作业: scancel $JOB_ID"
else
    JOB_IDS=""
    while IFS=$'\t' read -r exp_id study_group seed config_path; do
        echo "提交: $exp_id seed=$seed group=$study_group"
        JOB_ID=$(sbatch --parsable run_train_field.slurm "$config_path")
        echo "  作业ID: $JOB_ID"
        JOB_IDS="$JOB_IDS $JOB_ID"
    done < "$MANIFEST_LIST_FILE"

    echo ""
    echo "所有作业已提交"
    echo "作业ID:$JOB_IDS"
    echo "查看输出: tail -f logs/field_train_<JOB_ID>.out"
fi
