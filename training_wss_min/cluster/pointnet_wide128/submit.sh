#!/bin/bash
# Submit PointNet width=128 xyz+geom capacity probe (single config).
set -euo pipefail

PROJECT_DIR="/public/newhome/cy/Digital_twin/GNN"
MANIFEST="training_wss_min/configs/sweeps/pointnet_wide128.txt"
SLURM_FILE="training_wss_min/cluster/pointnet_wide128/train_array.slurm"
LOG_DIR="training_wss_min/cluster/logs"
cd "$PROJECT_DIR"

test "$(wc -l < "$MANIFEST")" -eq 1
while IFS= read -r config; do
    test -f "$config"
done < "$MANIFEST"
mkdir -p "$LOG_DIR"

JOBID=$(/public/slurm/bin/sbatch --parsable "$SLURM_FILE")
STAMP=$(date +%Y%m%d_%H%M%S)
printf '%s\t%s\t%s\n' "$STAMP" "$JOBID" "$MANIFEST" \
  | tee -a "$LOG_DIR/pointnet_wide128_submitted.tsv"
echo "submitted array ${JOBID} (PointNet width=128 xyz+geom; FPS2000; val-only)"
