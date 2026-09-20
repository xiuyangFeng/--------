#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --job-name=wss_local_wave1
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=36:00:00
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.run_wss_local_wave1_queue \
  --slots-per-gpu 2 --min-free-mib 7000 --launch-interval-seconds 45
