#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --job-name=time_mae_pilot
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_mae_20260927/logs/cpu_pilot_%A_%a.log
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
SCRIPT=training_wss_min/experiments/wss_time_mae_20260927/evaluate_saved.py
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
 "$PY" -u "$SCRIPT" --family tt --fold 0 --arms TT-warm --device cpu --threads 4
else
 "$PY" -u "$SCRIPT" --family ft --fold 0 --arms TL-warm --device cpu --threads 4
fi
