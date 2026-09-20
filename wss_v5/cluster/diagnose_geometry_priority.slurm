#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=v6_geom_priority
#SBATCH --output=outputs/wss_v6_geometry_candidate_20260909/slurm/%x_%j.out
#SBATCH --error=outputs/wss_v6_geometry_candidate_20260909/slurm/%x_%j.err
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
"$PY" -u -m wss_v5.diagnose_geometry_priority --root outputs/wss_v6_geometry_candidate_20260909 --top 6 "$@"
