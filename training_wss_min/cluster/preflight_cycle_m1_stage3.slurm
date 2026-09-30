#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=cyclem1s3_preflight
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
# 周期积分量 M1 三头三 seed 确认（train136 → test34）GPU 预检（冻结副本执行）：锚点重评 + 每臂配对构造/三步 AMP/全壁面推理 + 三臂 2-epoch 冒烟
set -euo pipefail
FROZEN=/public/newhome/cy/Digital_twin/GNN_cycle_m1_frozen_20260921
MAIN=/public/newhome/cy/Digital_twin/GNN
cd "$FROZEN"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname; nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.preflight_wss_local_wave1 \
  --config-dir "$MAIN/training_wss_min/configs/wss_cycle_m1_stage3_20260922" \
  --experiment-dir "$MAIN/training_wss_min/experiments/wss_cycle_m1_stage3_20260922" \
  --control-arm M1_s1234 --smoke-arms M1_s7 \
  --anchor-run "$MAIN/training_wss_min/runs/wss_v51_wave1_20260916/X5D_v51_s1234"
echo "===== preflight done $(date -Is) ====="
