#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=300G
#SBATCH --job-name=pf6vf6_v52d_queue
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=36:00:00
# PF6 / VF6 v5.2d retrain (01 block tracking §43): the master share of the 36 arms (matrix_master.json) on 4x RTX 4090,
# SLOTS arms per GPU; the node04 share (matrix_node04.json) runs on node04's 2x A100 outside Slurm
# (cluster/node04_pf6vf6_v52d_queue.sh, status queue_status_node04.json). Each arm train -> eval best -> eval last with
# --save-predictions. Runs inside GNN_pf6vf6_v52d_frozen_20261002; the queue re-verifies the preflight fingerprints before
# every launch. Submitted afterok on the preflight.
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
NAME=pf6vf6_v52d_retrain_20261002
SLOTS=${SLOTS:-1}
cd /public/newhome/cy/Digital_twin/GNN_pf6vf6_v52d_frozen_20261002
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname; echo "slots per GPU: $SLOTS"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/public/newhome/cy/.conda/envs/GNN/bin/python -c "import training_wss_min; print('code:', training_wss_min.__file__)"
echo "== sha1 (configs, frozen code, preflight evidence)"; sha1sum "$ROOT/training_wss_min/configs/$NAME"/*.json training_wss_min/*.py training_wss_min/tools/run_wss_local_wave1_queue.py training_wss_min/tools/run_next_matrix_queue.py "$ROOT/training_wss_min/experiments/$NAME/runtime_preflight.json"
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.run_wss_local_wave1_queue \
  --config-dir "$ROOT/training_wss_min/configs/$NAME" \
  --matrix "$ROOT/training_wss_min/configs/$NAME/matrix_master.json" \
  --experiment-dir "$ROOT/training_wss_min/experiments/$NAME" \
  --slots-per-gpu "$SLOTS" --min-free-mib 6000 --launch-interval-seconds 45
date -Is
