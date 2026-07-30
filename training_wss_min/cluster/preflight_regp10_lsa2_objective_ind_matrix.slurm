#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=lsa2objind_pf
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=02:00:00
set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="${LSA2_OBJIND_MANIFEST:?LSA2_OBJIND_MANIFEST is required}"
STATIC="${LSA2_OBJIND_STATIC_AUDIT:?LSA2_OBJIND_STATIC_AUDIT is required}"
OUTPUT="${LSA2_OBJIND_GPU_AUDIT:?LSA2_OBJIND_GPU_AUDIT is required}"
cd "$PROJECT"
"$PY" -m training_wss_min.tools.smoke_regp10_lsa2_objective_ind_matrix \
  --manifest "$MANIFEST" \
  --static-audit "$STATIC" \
  --output "$OUTPUT"
