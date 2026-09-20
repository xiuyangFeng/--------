#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --job-name=vf6_wss
#SBATCH --output=training_wss_min/cluster/logs/%x_%A_%a.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%A_%a.err
#SBATCH --time=01:00:00
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.vf6_velocity_to_wss "$@"
