#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=o0hottail_pf
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=02:00:00
set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="${O0_HOTTAIL_MANIFEST:?O0_HOTTAIL_MANIFEST is required}"
STATIC="${O0_HOTTAIL_STATIC_AUDIT:?O0_HOTTAIL_STATIC_AUDIT is required}"
OUTPUT="${O0_HOTTAIL_GPU_AUDIT:?O0_HOTTAIL_GPU_AUDIT is required}"
cd "$PROJECT"
"$PY" -m training_wss_min.tools.smoke_o0_hotspot_tail_matrix \
  --manifest "$MANIFEST" \
  --static-audit "$STATIC" \
  --output "$OUTPUT"
