#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:3
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --job-name=wss_v52p4_queue
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=24:00:00
# v5.2p4 full265 (2026-09-30): X5Dcap_asym2 x seeds 1234/7/2025 on the 265 training cases, one arm per RTX 4090
# (train -> eval best -> eval last on recover8). Runs inside GNN_v52p_frozen_20260926; the queue re-verifies the
# preflight fingerprints before every launch. Submitted behind joint52r2 (afterany) and the preflight (afterok).
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
NAME=wss_v52p4_full265_20260930
cd /public/newhome/cy/Digital_twin/GNN_v52p_frozen_20260926
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
echo "== sha1 (configs, frozen code, preflight evidence)"; sha1sum "$ROOT/training_wss_min/configs/$NAME"/*.json training_wss_min/*.py training_wss_min/tools/run_wss_local_wave1_queue.py training_wss_min/tools/run_next_matrix_queue.py "$ROOT/training_wss_min/experiments/$NAME/runtime_preflight.json"
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.run_wss_local_wave1_queue \
  --config-dir "$ROOT/training_wss_min/configs/$NAME" \
  --experiment-dir "$ROOT/training_wss_min/experiments/$NAME" \
  --slots-per-gpu 1 --min-free-mib 7000 --launch-interval-seconds 45
date -Is
