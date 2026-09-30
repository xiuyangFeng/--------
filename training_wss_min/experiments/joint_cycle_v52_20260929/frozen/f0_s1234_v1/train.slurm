#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export GNN_JOINT_SOURCE_ROOT=/public/newhome/cy/Digital_twin/GNN
cd /public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/joint_cycle_v52_20260929/frozen/f0_s1234_v1
date -Is
hostname
id
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/public/newhome/cy/.conda/envs/GNN/bin/python -u entry.py train
