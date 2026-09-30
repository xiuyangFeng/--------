#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=wss_v52phys_preflight
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=06:00:00
# v5.2 retraining: GPU preflight of the 40 arms (anchor = X5D_v51_s1234 re-evaluated with the frozen v5.2 code; smoke arms at batch8 x 5000).
# Runs inside GNN_v52p_frozen_20260926 (configs/experiments/runs are symlinks to the main tree). Do not edit the frozen tree while this or the queue runs.
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
NAME=wss_v52_phys_20260926
cd /public/newhome/cy/Digital_twin/GNN_v52p_frozen_20260926
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.preflight_wss_local_wave1 \
  --config-dir "$ROOT/training_wss_min/configs/$NAME" \
  --experiment-dir "$ROOT/training_wss_min/experiments/$NAME" \
  --control-arm X5Dcap_womres_s1234 --smoke-arms X5Dcap_asym2_s7,X5Dcap_poisres_s2025 \
  --anchor-run "$ROOT/training_wss_min/runs/wss_v51_wave1_20260916/X5D_v51_s1234"
date -Is
