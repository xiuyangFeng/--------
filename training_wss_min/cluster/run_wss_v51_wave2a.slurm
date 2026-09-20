#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=36
#SBATCH --mem=180G
#SBATCH --job-name=wss_v51_wave2a
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=48:00:00
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.run_wss_local_wave1_queue \
  --config-dir training_wss_min/configs/wss_v51_wave2a_20260916 \
  --experiment-dir training_wss_min/experiments/wss_v51_wave2a_20260916 \
  --slots-per-gpu 1 --min-free-mib 7000 --launch-interval-seconds 45
