#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=m2_diag_finalize
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=02:00:00
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=-1
exec /public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.finalize_m2_diagnostic_capture "$@"
