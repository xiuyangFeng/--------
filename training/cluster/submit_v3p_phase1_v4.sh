#!/bin/bash
# V3P Phase1 v4 · 并入 Batch-1 采购 + neighbor 重评
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATE_TAG="${DATE_TAG:-20260701}"
BATCH="${REPO_ROOT}/outputs/field/f0_decision/v3p_procurement_batch1_${DATE_TAG}.txt"

cd "$REPO_ROOT"
mkdir -p training/cluster/logs

JOB_ID=$(/public/slurm/bin/sbatch --parsable \
  --job-name=v3p_p1_v4 \
  --output="training/cluster/logs/v3p_phase1_v4_${DATE_TAG}_%j.out" \
  --error="training/cluster/logs/v3p_phase1_v4_${DATE_TAG}_%j.err" \
  --export=ALL,DATE_TAG="$DATE_TAG",BATCH_FILE="$BATCH" \
  "$SCRIPT_DIR/run_v3p_phase1_v4.slurm")

echo "Submitted Phase1 v4 job: $JOB_ID"
