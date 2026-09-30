#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=v52c_stats
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_v52c_labelfix_20260930/prepare_stats_%j.out
# v5.2c / v5.2p5 (2026-09-30 label fix): code-path check against the original statistics, recomputed label statistics,
# unchanged input-feature z-scores.
set -euo pipefail
G=/public/newhome/cy/Digital_twin/GNN
cd "$G"
export PYTHONPATH="$G" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
date -Is; hostname; sha1sum training_wss_min/experiments/wss_v52c_labelfix_20260930/prepare_v52c.py wss_v5/views/wss_min_view.py wss_v5/views/wall_cycle_v1.py training_wss_min/dataset.py | cut -c1-12,42-
for s in verify-codepath stats check-features; do $PY training_wss_min/experiments/wss_v52c_labelfix_20260930/prepare_v52c.py $s; done
date -Is
