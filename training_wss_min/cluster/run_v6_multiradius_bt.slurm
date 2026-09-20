#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --job-name=v6_multiradius
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=12:00:00
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.run_v6_multiradius_bt_queue "$@"
