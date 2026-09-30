#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-8%6
#SBATCH --job-name=time_mae_eval
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_mae_20260927/logs/cpu_eval_%A_%a.log
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
families=(probe tt ft)
family=${families[$((SLURM_ARRAY_TASK_ID / 3))]}
fold=$((SLURM_ARRAY_TASK_ID % 3))
/public/newhome/cy/.conda/envs/GNN/bin/python -u training_wss_min/experiments/wss_time_mae_20260927/evaluate_saved.py --family "$family" --fold "$fold" --device cpu --threads 4
