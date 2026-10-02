#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=wss_v52d_preflight
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=06:00:00
# v5.2d retrain (2026-10-01; the matrix the user confirmed 09-30, resubmitted on v5.2d): GPU preflight of all 21 arms
# (X5Dcap_asym2 CV5 x3 seeds, full265 x3, M1cap three-head full265 x3). Anchor = the pre-label-fix full265 s1234 run (job 16192)
# repointed to v5.2d: recover8 is not among the 18 merged units and the model's inputs are bitwise unchanged for unchanged
# units, so its stored recover8 metrics must reproduce. Every arm: paired init + 3 AMP steps at
# batch8 x 5000 + full-wall inference; CLI smoke (2-epoch train + eval best/last) on one X5Dcap and one three-head arm.
# Runs inside GNN_v52d_frozen_20261001 (configs/experiments/runs are symlinks to the main tree).
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
NAME=wss_v52d_retrain_20261001
cd /public/newhome/cy/Digital_twin/GNN_v52d_frozen_20261001
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/public/newhome/cy/.conda/envs/GNN/bin/python -c "import training_wss_min; print('code:', training_wss_min.__file__)"
echo "== sha1 (configs, frozen code)"; sha1sum "$ROOT/training_wss_min/configs/$NAME"/*.json training_wss_min/*.py training_wss_min/tools/preflight_wss_local_wave1.py training_wss_min/tools/run_wss_local_wave1_queue.py training_wss_min/tools/run_next_matrix_queue.py
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.preflight_wss_local_wave1 \
  --config-dir "$ROOT/training_wss_min/configs/$NAME" \
  --experiment-dir "$ROOT/training_wss_min/experiments/$NAME" \
  --control-arm X5Dcap_asym2_full265_s1234 --smoke-arms X5Dcap_asym2_full265_s7,M1cap_full265_s7 \
  --anchor-run "$ROOT/training_wss_min/experiments/$NAME/anchor/X5Dcap_asym2_full265_s1234_v52d"
date -Is
