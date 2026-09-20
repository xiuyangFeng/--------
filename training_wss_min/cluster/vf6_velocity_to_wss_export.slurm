#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --job-name=vf6_wss_export
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=01:00:00
# No --gres: master's four GPUs are held by the user's own wss_v51 queue (gres=gpu:4, launcher
# shares GPUs with --min-free-mib 7000). This short inference shares the GPU with the most free
# memory; the chosen device and its free memory are recorded in the reproduction JSON.
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1)
export CUDA_VISIBLE_DEVICES=$(echo "$best" | cut -d, -f1 | tr -d ' ')
echo "selected GPU $best (index,free MiB) at $(date)"
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.vf6_velocity_to_wss export --seed "$1"
