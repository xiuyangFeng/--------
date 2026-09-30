#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=wss_lc_preflight
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=06:00:00
# Learning curve on cv3_v51 (2026-09-20): GPU preflight of the 33 LC arms.
# Executes inside the v5.1 frozen copy (same source that produced wss_v51_wave2a fold runs); configs/experiments/runs
# are symlinks back to the main tree.  Do not edit GNN_v51_frozen/training_wss_min/*.py while this or the queue runs.
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
NAME=wss_learning_curve_20260920
cd /public/newhome/cy/Digital_twin/GNN_v51_frozen
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.preflight_wss_local_wave1 \
  --config-dir "$ROOT/training_wss_min/configs/$NAME" \
  --experiment-dir "$ROOT/training_wss_min/experiments/$NAME" \
  --control-arm LC50_f0_s1234 --smoke-arms LC25_f1_s7,LC100_f2_s2025 \
  --anchor-run "$ROOT/training_wss_min/runs/wss_v51_wave2a_20260916/X5D_v51_f0_s1234"
date -Is
