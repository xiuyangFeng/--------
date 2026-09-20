#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=wss_local_wave3_preflight
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=04:00:00
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.preflight_wss_local_wave1 \
  --config-dir training_wss_min/configs/wss_local_wave3_20260913 \
  --experiment-dir training_wss_min/experiments/wss_local_wave3_20260913 \
  --control-arm PF6_s7 --smoke-arms PF6_s7,VF6_s2025,P02_s2025,V07_s7
