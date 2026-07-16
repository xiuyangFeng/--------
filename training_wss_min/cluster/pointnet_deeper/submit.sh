#!/bin/bash
# Submit the single teacher-requested E4 deeper PointNet follow-up.
set -euo pipefail

PROJECT_DIR="/public/newhome/cy/Digital_twin/GNN"
CONFIG="training_wss_min/configs/pointnet_deeper/e4_deep_global.json"
RUNNER="training_wss_min/cluster/run_pointnet_distribution_matrix.slurm"
LOG_DIR="training_wss_min/cluster/logs"
cd "$PROJECT_DIR"

test -f "$CONFIG"
test -f "$RUNNER"
mkdir -p "$LOG_DIR"

JOB_ID=$(/public/slurm/bin/sbatch --parsable --job-name="wsspn_e4_deep" "$RUNNER" "$CONFIG")
STAMP=$(date +%Y%m%d_%H%M%S)
printf '%s\t%s\t%s\n' "$STAMP" "$JOB_ID" "$CONFIG" \
  | tee -a "$LOG_DIR/pointnet_deeper_e4_submitted.tsv"
echo "submitted E4-DEEP-GLOBAL job ${JOB_ID}"
