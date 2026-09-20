#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --job-name=voltime_a23
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err

set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
M=training_wss_min.experiments.volume_time_20260919.offline

echo "===== a2 time basis / ceiling $(date) ====="
$PY -u -m $M.a2_time_basis 2>&1 | tee training_wss_min/experiments/volume_time_20260919/offline/a2_time_basis.txt
echo "===== a3 T-null $(date) ====="
$PY -u -m $M.a3_tnull 2>&1 | tee training_wss_min/experiments/volume_time_20260919/offline/a3_tnull.txt
echo "===== done $(date) ====="
