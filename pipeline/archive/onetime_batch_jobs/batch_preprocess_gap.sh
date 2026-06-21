#!/usr/bin/env bash
# ============================================================================
# 预处理缺口队列：ILO（审计就绪）+ AAA/unruputer 未完成 5 例；不含 AG
# ============================================================================
# 病例列表由 pipeline/export_gap_preprocess_queue.py 生成（含 denylist，
#   排除项见 PREPROCESS_DENYLIST（当前 15 条，含 AG OOD + 终审 CFD/WSS 剔除）。
#
# 用法：
#   cd <repo-root>/pipeline/archive/onetime_batch_jobs
#   ./batch_preprocess_gap.sh
#
# 环境变量：MAX_PARALLEL（默认 6）、SAMPLING_METHOD、FPS_RATIO、PIPELINE_DATA_ROOT 等
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
CLUSTER_DIR="$REPO_ROOT/pipeline/cluster"
cd "$SCRIPT_DIR"
mkdir -p logs

LIST_SRC="$SCRIPT_DIR/case_list_gap_preprocess.txt"
echo "生成缺口预处理队列..."
(cd "$REPO_ROOT" && python -m pipeline.export_gap_preprocess_queue --output "$LIST_SRC")

FIRST_LINE="$(head -n 1 "$LIST_SRC" | tr -d '\r')"
if [ -z "$FIRST_LINE" ]; then
  echo "错误: 队列文件为空"
  exit 1
fi
if [[ "$FIRST_LINE" != */* ]]; then
  echo "错误: 队列首行「$FIRST_LINE」必须为相对 data_root 的路径（含 /）"
  exit 1
fi

cp -f "$LIST_SRC" "$SCRIPT_DIR/case_list.txt"

CASE_COUNT=$(wc -l < "$SCRIPT_DIR/case_list.txt" | tr -d ' ')
MAX_PARALLEL="${MAX_PARALLEL:-6}"
PREPROCESS_SOURCES="${PREPROCESS_SOURCES:-AAA/ruputer AAA/unruputer ILO}"

SBATCH_ENV=(
  START_STEP=1
  END_STEP=1
  SAMPLING_METHOD="${SAMPLING_METHOD:-hybrid}"
  FPS_RATIO="${FPS_RATIO:-0.2}"
  ALLOW_NEAREST_BC="${ALLOW_NEAREST_BC:-0}"
  PIPELINE_SOURCES="$PREPROCESS_SOURCES"
)
if [ -n "${GEOMETRY_PYTHON:-}" ]; then
  SBATCH_ENV+=(GEOMETRY_PYTHON="$GEOMETRY_PYTHON")
fi

echo ""
echo "病例数: $CASE_COUNT"
echo "PIPELINE_SOURCES: $PREPROCESS_SOURCES"
ARRAY_RANGE="0-$((CASE_COUNT - 1))%${MAX_PARALLEL}"
JOB_ID=$(cd "$CLUSTER_DIR" && env "${SBATCH_ENV[@]}" sbatch --parsable --array="$ARRAY_RANGE" run_array.slurm)
echo "作业ID: $JOB_ID  Array: $ARRAY_RANGE"
echo "日志: $SCRIPT_DIR/logs/gnn_array_${JOB_ID}_<a>.out"
