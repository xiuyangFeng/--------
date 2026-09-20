#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --job-name=v6_geom_render
#SBATCH --output=outputs/wss_v6_geometry_candidate_20260909/slurm/%x_%j.out
#SBATCH --error=outputs/wss_v6_geometry_candidate_20260909/slurm/%x_%j.err
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
"$PY" -u -m wss_v5.geometry_candidate --summarize
"$PY" -u -m wss_v5.render_geometry_candidate --candidate-root outputs/wss_v6_geometry_candidate_20260909 "$@"
"$PY" -u docs/03-汇报材料/WSS_V6_新几何验收_20260909/audit_pages.py
