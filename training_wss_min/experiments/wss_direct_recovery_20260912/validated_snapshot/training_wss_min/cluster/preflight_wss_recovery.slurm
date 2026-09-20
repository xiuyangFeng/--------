#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=wss_recovery_smoke
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=01:30:00
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.preflight_wss_recovery
