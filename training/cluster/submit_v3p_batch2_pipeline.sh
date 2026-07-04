#!/bin/bash
# V3P Batch-2 CPU 采购 + Phase1 v5 重评（588x 链 · node03）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATE_TAG="${DATE_TAG:-20260701}"
BATCH1="${REPO_ROOT}/outputs/field/f0_decision/v3p_procurement_batch1_${DATE_TAG}.txt"

cd "$REPO_ROOT"
mkdir -p training/cluster/logs

JOB_B2=$(/public/slurm/bin/sbatch --parsable \
  --job-name=v3p_proc_b2 \
  --output="training/cluster/logs/v3p_procurement_batch2_${DATE_TAG}_%j.out" \
  --error="training/cluster/logs/v3p_procurement_batch2_${DATE_TAG}_%j.err" \
  --export=ALL,DATE_TAG="$DATE_TAG",BATCH1_FILE="$BATCH1" \
  "$SCRIPT_DIR/run_v3p_procurement_batch2.slurm")

JOB_P5=$(/public/slurm/bin/sbatch --parsable \
  --dependency=afterok:"$JOB_B2" \
  --job-name=v3p_p1_v5 \
  --output="training/cluster/logs/v3p_phase1_v5_${DATE_TAG}_%j.out" \
  --error="training/cluster/logs/v3p_phase1_v5_${DATE_TAG}_%j.err" \
  --export=ALL,DATE_TAG="$DATE_TAG" \
  "$SCRIPT_DIR/run_v3p_phase1_v5.slurm")

echo "Submitted V3P Batch-2 pipeline:"
echo "  Batch-2 procurement CPU: $JOB_B2"
echo "  Phase1 v5 (after Batch-2): $JOB_P5"
echo "Logs:"
echo "  training/cluster/logs/v3p_procurement_batch2_${DATE_TAG}_${JOB_B2}.out"
echo "  training/cluster/logs/v3p_phase1_v5_${DATE_TAG}_${JOB_P5}.out"
