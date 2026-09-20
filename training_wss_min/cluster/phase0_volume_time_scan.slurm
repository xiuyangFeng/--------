#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --job-name=voltime_a0
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err

set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
# 每个 worker 单线程 BLAS：Gram 是 (81,N)x(N,81)，并行度放在病例维，避免 12 个进程各抢 192 线程
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
echo "===== a0_scan $(date) ====="
$PY -u -m training_wss_min.experiments.volume_time_20260919.offline.a0_scan --workers 12
echo "===== done $(date) ====="
