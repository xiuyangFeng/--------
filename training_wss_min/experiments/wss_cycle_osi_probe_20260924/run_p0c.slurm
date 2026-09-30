#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --job-name=osi_p0c
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_cycle_osi_probe_20260924/logs/%x_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_cycle_osi_probe_20260924/logs/%x_%j.err
# OSI 零训练探针 P0c（只读；PREREG.md §P0c）。不触碰 GPU 队列 15636 的冻结副本与指纹文件。
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_cycle_osi_probe_20260924/offline
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname
/public/newhome/cy/.conda/envs/GNN/bin/python -u p0c_ceiling.py
date -Is
