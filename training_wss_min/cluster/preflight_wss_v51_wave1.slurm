#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=wss_v51_wave1_preflight
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=04:00:00
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.preflight_wss_local_wave1 \
  --config-dir training_wss_min/configs/wss_v51_wave1_20260916 \
  --experiment-dir training_wss_min/experiments/wss_v51_wave1_20260916 \
  --control-arm X5D_v51_s1234 --smoke-arms X5Dcap_s1234,X5Ddual_s1234,X5Dnoise_s1234
