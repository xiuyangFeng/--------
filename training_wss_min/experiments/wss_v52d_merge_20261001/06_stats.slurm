#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --job-name=v52d_stats
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_v52d_merge_20261001/stats_%j.out
set -euo pipefail
G=/public/newhome/cy/Digital_twin/GNN; cd "$G"
export PYTHONPATH="$G" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname; sha1sum training_wss_min/experiments/wss_v52d_merge_20261001/06_stats.py wss_v5/views/wss_min_view.py wss_v5/views/wall_cycle_v1.py training_wss_min/dataset.py | cut -c1-12,42-
/public/newhome/cy/.conda/envs/GNN/bin/python training_wss_min/experiments/wss_v52d_merge_20261001/06_stats.py
date -Is
