#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=joint52_probe
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/public/newhome/cy/Digital_twin/GNN
/public/newhome/cy/.conda/envs/GNN/bin/python -u training_wss_min/experiments/joint_cycle_v52_20260929/gpu_probe.py
