#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=rcroracle_pf
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=02:00:00
set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="${RCR_ORACLE_MANIFEST:?RCR_ORACLE_MANIFEST is required}"
STATIC="${RCR_ORACLE_STATIC_AUDIT:?RCR_ORACLE_STATIC_AUDIT is required}"
OUTPUT="${RCR_ORACLE_GPU_AUDIT:?RCR_ORACLE_GPU_AUDIT is required}"
cd "$PROJECT"
"$PY" -m training_wss_min.tools.smoke_rcr_oracle_matrix \
  --manifest "$MANIFEST" \
  --static-audit "$STATIC" \
  --output "$OUTPUT"
