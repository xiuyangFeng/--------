#!/bin/bash
# Submit stage 1 (GLOBAL), stage 2 (CASE), or optional E23 interaction control.
# CASE and E23 are never chained automatically.

set -euo pipefail

STAGE="${1:?usage: submit_pointnet_distribution_stage.sh <global|case|e23>}"
PROJECT_DIR="/public/newhome/cy/Digital_twin/GNN"
cd "$PROJECT_DIR"

case "$STAGE" in
  global) MANIFEST="training_wss_min/configs/sweeps/pointnet_distribution_global_stage1.txt" ;;
  case) MANIFEST="training_wss_min/configs/sweeps/pointnet_distribution_case_stage2.txt" ;;
  e23) MANIFEST="training_wss_min/configs/sweeps/pointnet_distribution_e23_optional.txt" ;;
  *) echo "stage must be global, case, or e23"; exit 2 ;;
esac

SBATCH="/public/slurm/bin/sbatch"
STAMP="$(date +%Y%m%d_%H%M%S)"
RECORD="training_wss_min/cluster/logs/submitted_pointnet_distribution_${STAGE}_${STAMP}.txt"
mkdir -p "$(dirname "$RECORD")"

while IFS= read -r CONFIG; do
    [ -z "$CONFIG" ] && continue
    NAME="$(basename "$CONFIG" .json)"
    JOB_ID="$($SBATCH --parsable --job-name="wsspn_${NAME}" \
        training_wss_min/cluster/run_pointnet_distribution_matrix.slurm "$CONFIG")"
    echo "${JOB_ID}  ${NAME}  ${CONFIG}" | tee -a "$RECORD"
done < "$MANIFEST"

echo "submission record: $RECORD"
