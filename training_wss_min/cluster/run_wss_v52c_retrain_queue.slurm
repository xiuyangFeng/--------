#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=220G
#SBATCH --job-name=wss_v52c_queue
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=40:00:00
# v5.2c retrain (2026-09-30, user-confirmed matrix): 21 arms on master 4x RTX 4090, one arm per GPU (user: no doubling up;
# node04 stays free for the full-cycle time line). Order B (X5Dcap_asym2 full265 x3), C (M1cap full265 x3), A (X5Dcap_asym2
# CV5 x15); each arm train -> eval best -> eval last with --save-predictions. Runs inside GNN_v52c_frozen_20260930; the queue
# re-verifies the preflight fingerprints before every launch. Submitted afterok on the preflight.
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
NAME=wss_v52c_retrain_20260930
cd /public/newhome/cy/Digital_twin/GNN_v52c_frozen_20260930
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/public/newhome/cy/.conda/envs/GNN/bin/python -c "import training_wss_min; print('code:', training_wss_min.__file__)"
echo "== sha1 (configs, frozen code, preflight evidence)"; sha1sum "$ROOT/training_wss_min/configs/$NAME"/*.json training_wss_min/*.py training_wss_min/tools/run_wss_local_wave1_queue.py training_wss_min/tools/run_next_matrix_queue.py "$ROOT/training_wss_min/experiments/$NAME/runtime_preflight.json"
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.run_wss_local_wave1_queue \
  --config-dir "$ROOT/training_wss_min/configs/$NAME" \
  --experiment-dir "$ROOT/training_wss_min/experiments/$NAME" \
  --slots-per-gpu 1 --min-free-mib 7000 --launch-interval-seconds 45
date -Is
