#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=v6_geom_review
#SBATCH --output=outputs/wss_v6_geometry_candidate_20260909/slurm/%x_%A_%a.out
#SBATCH --error=outputs/wss_v6_geometry_candidate_20260909/slurm/%x_%A_%a.err
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
sha256sum wss_v5/geometry_candidate.py wss_v5/section_features.py wss_v5/configs/geometry_candidate_p_v1.json
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  "$PY" -u -m wss_v5.geometry_candidate --case-index "$SLURM_ARRAY_TASK_ID" "$@"
else
  "$PY" -u -m wss_v5.geometry_candidate "$@"
fi
