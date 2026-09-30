#!/bin/bash
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 WSS_GPU_UUID=GPU-d0c62ff2-6402-879c-49a1-d231608df543
exec 9>training_wss_min/experiments/wss_time_mae_20260927/ft_gpu.lock
flock -n 9
nvidia-smi -i 1 --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
for fold in 0 1 2; do
 /public/newhome/cy/.conda/envs/GNN/bin/python -u training_wss_min/experiments/wss_time_mae_20260927/evaluate_saved.py --family ft --fold "$fold" --device cuda --threads 2
 done
