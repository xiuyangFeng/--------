#!/bin/bash
# V3P 257 池采购 Batch-1 · CPU node03 · 排名 + QA 深审（graphs 已齐，不 blind pipeline）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"

cd "$REPO_ROOT"
mkdir -p training/cluster/logs

JOB_ID=$(/public/slurm/bin/sbatch --parsable \
  --job-name=v3p_proc_b1 \
  --output="training/cluster/logs/v3p_procurement_batch1_${DATE_TAG}_%j.out" \
  --error="training/cluster/logs/v3p_procurement_batch1_${DATE_TAG}_%j.err" \
  "$SCRIPT_DIR/run_v3p_procurement_batch1.slurm")

echo "Submitted V3P procurement Batch-1 CPU job: $JOB_ID"
echo "Log: training/cluster/logs/v3p_procurement_batch1_${DATE_TAG}_${JOB_ID}.out"
