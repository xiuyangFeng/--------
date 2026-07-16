#!/bin/bash
# Submit the frozen minimal 2x3 baseline without regenerating its JSON configs.
set -euo pipefail

PROJECT_DIR="/public/newhome/cy/Digital_twin/GNN"
MANIFEST="training_wss_min/configs/sweeps/baseline_2x3_simple.txt"
SLURM_FILE="training_wss_min/cluster/baseline_2x3/train_array.slurm"
LOG_DIR="training_wss_min/cluster/logs"
cd "$PROJECT_DIR"

test "$(wc -l < "$MANIFEST")" -eq 6
while IFS= read -r config; do
    test -f "$config"
done < "$MANIFEST"
mkdir -p "$LOG_DIR"

JOBID=$(/public/slurm/bin/sbatch --parsable "$SLURM_FILE")
STAMP=$(date +%Y%m%d_%H%M%S)
printf '%s\t%s\t%s\n' "$STAMP" "$JOBID" "$MANIFEST" \
  | tee -a "$LOG_DIR/baseline_2x3_submitted.tsv"
echo "submitted array ${JOBID} (six frozen configs; val-only)"
