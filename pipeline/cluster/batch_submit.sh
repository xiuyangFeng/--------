#!/bin/bash
#Author: XY
# ============================================================================
# 批量提交多个病例作业
# ============================================================================
# 使用方法:
#   ./batch_submit.sh                       # 提交 AG/fast 全部（独立作业模式）
#   ./batch_submit.sh --slow                # 仅扫描并提交 AG/slow（需配合下面模式）
#   ./batch_submit.sh --array --slow        # Array：处理 AG/slow 全部病例（推荐）
#   ./batch_submit.sh --array --all-ag      # Array：AG/fast + AG/slow 全部病例（名称不重复）
#   ./batch_submit.sh ZHANG_CHUN ZHANG_HAO  # 提交指定病例（默认仍按 AG/fast 扫描列表）
#   ./batch_submit.sh --array CASE1 CASE2   # Array Job 指定病例
# 说明: 使用 --slow 时向作业注入 PIPELINE_SOURCES=AG/slow，与 config 里 AG/slow 是否 enabled 无关。
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 创建日志目录
mkdir -p logs

USE_ARRAY=false
START_STEP=${START_STEP:-1}
END_STEP=${END_STEP:-5}
SAMPLING_METHOD=${SAMPLING_METHOD:-hybrid}
FPS_RATIO=${FPS_RATIO:-0.2}
MAX_PARALLEL=${MAX_PARALLEL:-6}
CASES=()
AG_SUBDIR="fast"
PIPELINE_SOURCES_JOB=""
ALL_AG=false

while [ $# -gt 0 ]; do
    case "$1" in
        --array|-a)
            USE_ARRAY=true
            ;;
        --all-ag)
            ALL_AG=true
            ;;
        --slow)
            AG_SUBDIR="slow"
            PIPELINE_SOURCES_JOB="AG/slow"
            ;;
        --fast-scan)
            AG_SUBDIR="fast"
            PIPELINE_SOURCES_JOB="AG/fast"
            ;;
        --start-step)
            shift
            START_STEP="${1:-}"
            ;;
        --end-step)
            shift
            END_STEP="${1:-}"
            ;;
        --sampling-method)
            shift
            SAMPLING_METHOD="${1:-}"
            ;;
        --fps-ratio)
            shift
            FPS_RATIO="${1:-}"
            ;;
        --max-parallel)
            shift
            MAX_PARALLEL="${1:-}"
            ;;
        --allow-nearest-bc)
            ALLOW_NEAREST_BC=1
            ;;
        *)
            CASES+=("$1")
            ;;
    esac
    shift
done

# 全 AG 扫描时由 config 中已启用的 AG/fast、AG/slow 共同解析病例路径，勿注入单一 PIPELINE_SOURCES
if [ "$ALL_AG" = true ]; then
    PIPELINE_SOURCES_JOB=""
fi

# 如果没有指定病例，获取所有病例
if [ ${#CASES[@]} -eq 0 ]; then
    REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
    AG_ROOT="$REPO_ROOT/data_new/AG"
    if [ "$ALL_AG" = true ]; then
        if [ ! -d "$AG_ROOT/fast" ] || [ ! -d "$AG_ROOT/slow" ]; then
            echo "错误: 需要同时存在: $AG_ROOT/fast与 $AG_ROOT/slow"
            exit 1
        fi
        mapfile -t CASES < <(
            {
                find "$AG_ROOT/fast" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;
                find "$AG_ROOT/slow" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;
            } | sort -u
        )
    else
        DATA_DIR="$AG_ROOT/$AG_SUBDIR"
        if [ -d "$DATA_DIR" ]; then
            mapfile -t CASES < <(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort)
        else
            echo "错误: 数据目录不存在: $DATA_DIR"
            exit 1
        fi
    fi
fi

# 统计病例数
CASE_COUNT=${#CASES[@]}

echo "=============================================="
echo "批量提交作业"
echo "=============================================="
echo "模式: $([ "$USE_ARRAY" = true ] && echo 'Array Job (推荐)' || echo '独立作业')"
echo "病例总数: $CASE_COUNT"
echo "主环境: ${PIPELINE_ENV:-GNN}"
echo "几何环境: ${GEOMETRY_ENV:-GNN_vmtk}"
echo "步骤范围: $START_STEP -> $END_STEP"
echo "采样方法: $SAMPLING_METHOD"
echo "FPS 占比: $FPS_RATIO"
echo "最大并发: $MAX_PARALLEL"
echo "允许最近 BC 兜底: ${ALLOW_NEAREST_BC:-0}"
if [ "$ALL_AG" = true ]; then
    echo "AG 子目录: fast + slow（全部）"
else
    echo "AG 子目录: $AG_SUBDIR"
fi
if [ -n "$PIPELINE_SOURCES_JOB" ]; then
    echo "PIPELINE_SOURCES（将传入作业）: $PIPELINE_SOURCES_JOB"
elif [ -n "${PIPELINE_SOURCES:-}" ]; then
    echo "PIPELINE_SOURCES（环境传入作业）: $PIPELINE_SOURCES"
fi
if [ -n "${GEOMETRY_PYTHON:-}" ]; then
    echo "geometry-python: $GEOMETRY_PYTHON"
fi
echo ""
echo "病例列表:"
for case in "${CASES[@]}"; do
    echo "  - $case"
done
echo ""

SBATCH_ENV=(
    START_STEP="$START_STEP"
    END_STEP="$END_STEP"
    SAMPLING_METHOD="$SAMPLING_METHOD"
    FPS_RATIO="$FPS_RATIO"
    ALLOW_NEAREST_BC="${ALLOW_NEAREST_BC:-0}"
)
if [ -n "${GEOMETRY_PYTHON:-}" ]; then
    SBATCH_ENV+=(GEOMETRY_PYTHON="$GEOMETRY_PYTHON")
fi
if [ -n "$PIPELINE_SOURCES_JOB" ]; then
    SBATCH_ENV+=(PIPELINE_SOURCES="$PIPELINE_SOURCES_JOB")
elif [ -n "${PIPELINE_SOURCES:-}" ]; then
    SBATCH_ENV+=(PIPELINE_SOURCES="$PIPELINE_SOURCES")
fi

if [ "$USE_ARRAY" = true ]; then
    # Array Job 模式（病例目录已在上方按 AG_SUBDIR 扫好，此处只写入列表文件）
    echo "生成病例列表文件..."
    bash "$SCRIPT_DIR/generate_case_list.sh" "${CASES[@]}"
    
    echo ""
    echo "提交 Array Job..."
    ARRAY_RANGE="0-$((CASE_COUNT-1))%${MAX_PARALLEL}"
    JOB_ID=$(env "${SBATCH_ENV[@]}" sbatch --parsable --array=$ARRAY_RANGE run_array.slurm)
    
    echo ""
    echo "=============================================="
    echo "Array Job 已提交"
    echo "=============================================="
    echo "作业ID: $JOB_ID"
    echo "Array 范围: $ARRAY_RANGE"
    echo "同时运行: 最多${MAX_PARALLEL}个任务"
    echo ""
    echo "查看作业状态: squeue -u $USER"
    echo "查看单个任务输出: tail -f logs/gnn_array_${JOB_ID}_<TASK_ID>.out"
    echo "取消所有任务: scancel $JOB_ID"
    echo "=============================================="
else
    # 独立作业模式
    JOB_IDS=""
    for case in "${CASES[@]}"; do
        echo "提交: $case"
        JOB_ID=$(env "${SBATCH_ENV[@]}" sbatch --parsable run_pipeline.slurm "$case" "$START_STEP" "$END_STEP" "$SAMPLING_METHOD" "$FPS_RATIO")
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
