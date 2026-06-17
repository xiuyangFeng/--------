#!/usr/bin/env bash
# 通用：提交单个 field run 的分域 test 评估（AAA / AG / ILO）
#
# 用法（仓库根目录）:
#   RUN_DIR_REL=outputs/field/<run_dir> \
#   CHECKPOINT=best_model.pt \
#   bash training/cluster/submit_eval_by_domain.sh
#
# 可选: SUBSET=test  DOMAINS="AAA AG ILO"
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

RUN_DIR_REL="${RUN_DIR_REL:-}"
CHECKPOINT="${CHECKPOINT:-best_model.pt}"
SUBSET="${SUBSET:-test}"

if [ -z "$RUN_DIR_REL" ]; then
    echo "错误: 请设置 RUN_DIR_REL=outputs/field/<run 目录名>"
    exit 1
fi
if [ ! -f "$ROOT/$RUN_DIR_REL/config.snapshot.json" ]; then
    echo "错误: 缺少 $ROOT/$RUN_DIR_REL/config.snapshot.json"
    exit 1
fi

JOB=$(sbatch --parsable \
    --job-name=eval_by_domain \
    --export=ALL,RUN_DIR_REL="$RUN_DIR_REL",CHECKPOINT="$CHECKPOINT",SUBSET="$SUBSET" \
    "$CLUSTER_DIR/run_eval_field_by_domain.slurm")
echo "分域评估已提交: job $JOB"
echo "监控: squeue -j $JOB"
echo "日志: logs/eval_by_domain_${JOB}.out 或 training/cluster/logs/eval_by_domain_${JOB}.out（视提交目录而定）"
echo "产出: $RUN_DIR_REL/eval_by_domain_${SUBSET}/metrics_by_domain.json"
