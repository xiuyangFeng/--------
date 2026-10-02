#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=wss_v52c_preflight
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=06:00:00
# v5.2c retrain (2026-09-30, user-confirmed matrix): GPU preflight of all 21 arms (X5Dcap_asym2 CV5 x3 seeds, full265 x3,
# M1cap three-head full265 x3). Anchor = the pre-label-fix full265 s1234 run repointed to v5.2c (recover8 labels/inputs
# unchanged; dry run on the login GPU 2026-09-30: 1494 fields, max diff 2.1e-6). Every arm: paired init + 3 AMP steps at
# batch8 x 5000 + full-wall inference; CLI smoke (2-epoch train + eval best/last) on one X5Dcap and one three-head arm.
# Runs inside GNN_v52c_frozen_20260930 (configs/experiments/runs are symlinks to the main tree).
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
NAME=wss_v52c_retrain_20260930
cd /public/newhome/cy/Digital_twin/GNN_v52c_frozen_20260930
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/public/newhome/cy/.conda/envs/GNN/bin/python -c "import training_wss_min; print('code:', training_wss_min.__file__)"
echo "== sha1 (configs, frozen code)"; sha1sum "$ROOT/training_wss_min/configs/$NAME"/*.json training_wss_min/*.py training_wss_min/tools/preflight_wss_local_wave1.py training_wss_min/tools/run_wss_local_wave1_queue.py training_wss_min/tools/run_next_matrix_queue.py
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.preflight_wss_local_wave1 \
  --config-dir "$ROOT/training_wss_min/configs/$NAME" \
  --experiment-dir "$ROOT/training_wss_min/experiments/$NAME" \
  --control-arm X5Dcap_asym2_full265_s1234 --smoke-arms X5Dcap_asym2_full265_s7,M1cap_full265_s7 \
  --anchor-run "$ROOT/training_wss_min/experiments/$NAME/anchor/X5Dcap_asym2_full265_s1234_v52c"
date -Is
