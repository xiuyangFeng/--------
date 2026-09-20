#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=v6_geom_pages
#SBATCH --output=outputs/wss_v6_geometry_candidate_20260909/slurm/%x_%A_%a.out
#SBATCH --error=outputs/wss_v6_geometry_candidate_20260909/slurm/%x_%A_%a.err
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
"$PY" -u -m wss_v5.render_geometry_candidate \
    --candidate-root outputs/wss_v6_geometry_candidate_20260909 \
    --shards 16 --shard-index "$SLURM_ARRAY_TASK_ID"
