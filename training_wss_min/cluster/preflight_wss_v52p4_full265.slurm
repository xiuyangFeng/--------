#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=wss_v52p4_preflight
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=06:00:00
# v5.2p4 full265 (2026-09-30): GPU preflight of the 3 X5Dcap_asym2 full265 arms, same flow as wss_v52_phys_20260926
# (anchor = X5D_v51_s1234 re-evaluated with the frozen code; every arm 3 AMP steps at batch8 x 5000; CLI smoke = 2-epoch train + evals).
# Runs inside GNN_v52p_frozen_20260926 (configs/experiments/runs are symlinks to the main tree).
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
NAME=wss_v52p4_full265_20260930
cd /public/newhome/cy/Digital_twin/GNN_v52p_frozen_20260926
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
echo "== sha1 (configs, frozen code)"; sha1sum "$ROOT/training_wss_min/configs/$NAME"/*.json training_wss_min/*.py training_wss_min/tools/preflight_wss_local_wave1.py training_wss_min/tools/run_wss_local_wave1_queue.py training_wss_min/tools/run_next_matrix_queue.py
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.preflight_wss_local_wave1 \
  --config-dir "$ROOT/training_wss_min/configs/$NAME" \
  --experiment-dir "$ROOT/training_wss_min/experiments/$NAME" \
  --control-arm X5Dcap_asym2_full265_s1234 --smoke-arms X5Dcap_asym2_full265_s7 \
  --anchor-run "$ROOT/training_wss_min/runs/wss_v51_wave1_20260916/X5D_v51_s1234"
date -Is
