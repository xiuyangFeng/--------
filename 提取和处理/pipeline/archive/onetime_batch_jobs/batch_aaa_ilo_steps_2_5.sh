#!/usr/bin/env bash
# ============================================================================
# AAA + ILO：步骤 2–5（extract_features → convert_to_graph）Array 作业
# ============================================================================
# - 病例列表：pipeline/export_post_preprocess_queue.py（剔除 denylist、已全流程完成者）
# - run_array.slurm：主进程 conda GNN；步骤 2 经 GEOMETRY_PYTHON（默认 GNN_vmtk）子进程执行
# - 须显式 PIPELINE_SOURCES，因 config 中 AAA/ILO 默认 enabled: False
#
# 用法（在登录节点，于本目录执行）:
#   cd <repo-root>/pipeline/archive/onetime_batch_jobs
#   ./batch_aaa_ilo_steps_2_5.sh
#
# 仅包「几何+BC 就绪」可跑步骤 2 的病例（当前 179，不含缺 STL/BC 的 3 例）:
#   ONLY_FEATURE_READY=1 ./batch_aaa_ilo_steps_2_5.sh
#
# 可选环境变量:
#   MAX_PARALLEL   默认 6
#   SLURM_TIME     默认 12:00:00（步骤 2 帧多时需更长）
#   PIPELINE_DATA_ROOT
#   GEOMETRY_PYTHON  覆盖 GNN_vmtk 的 python 路径
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
CLUSTER_DIR="$REPO_ROOT/pipeline/cluster"
cd "$SCRIPT_DIR"
mkdir -p logs

LIST_SRC="$SCRIPT_DIR/case_list_AAA_ILO_steps_2_5.txt"
echo "生成步骤 2–5 队列..."
GEN_ARGS=(--output "$LIST_SRC")
if [ "${ONLY_FEATURE_READY:-0}" = "1" ]; then
  GEN_ARGS+=(--only-feature-ready)
fi
(cd "$REPO_ROOT" && python -m pipeline.export_post_preprocess_queue "${GEN_ARGS[@]}")

FIRST_LINE="$(head -n 1 "$LIST_SRC" | tr -d '\r')"
if [ -z "$FIRST_LINE" ]; then
  echo "错误: 队列为空"
  exit 1
fi
if [[ "$FIRST_LINE" != */* ]]; then
  echo "错误: 队列首行「$FIRST_LINE」须为相对 data_root 的路径（含 /）"
  exit 1
fi

CASE_COUNT=$(wc -l < "$LIST_SRC" | tr -d ' ')
MAX_PARALLEL="${MAX_PARALLEL:-6}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
POST_SOURCES="${POST_SOURCES:-AAA/ruputer AAA/unruputer ILO}"

SBATCH_ENV=(
  START_STEP=2
  END_STEP=5
  SAMPLING_METHOD="${SAMPLING_METHOD:-hybrid}"
  FPS_RATIO="${FPS_RATIO:-0.2}"
  ALLOW_NEAREST_BC="${ALLOW_NEAREST_BC:-0}"
  PIPELINE_SOURCES="$POST_SOURCES"
  CASE_LIST_FILE="$LIST_SRC"
)
if [ -n "${GEOMETRY_PYTHON:-}" ]; then
  SBATCH_ENV+=(GEOMETRY_PYTHON="$GEOMETRY_PYTHON")
fi

echo ""
echo "病例数: $CASE_COUNT"
echo "列表文件: $LIST_SRC"
echo "PIPELINE_SOURCES: $POST_SOURCES"
echo "CASE_LIST_FILE: $LIST_SRC"
ARRAY_RANGE="0-$((CASE_COUNT - 1))%${MAX_PARALLEL}"

JOB_ID=$(cd "$CLUSTER_DIR" && env "${SBATCH_ENV[@]}" sbatch --parsable \
  --time="$SLURM_TIME" \
  --array="$ARRAY_RANGE" \
  run_array.slurm)

echo "作业ID: $JOB_ID  Array: $ARRAY_RANGE  time: $SLURM_TIME"
echo "日志: $SCRIPT_DIR/logs/gnn_array_${JOB_ID}_<a>.out"
