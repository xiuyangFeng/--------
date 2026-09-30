#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=36:00:00
#SBATCH --job-name=m1_v52_cv5
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.err
# 任务 1（2026-09-25）：M1 三头 × v5.2 X5Dcap × CV5 × seed 1234/7/2025 = 15 训练（PREREG：experiments/wss_cycle_m1_v52_20260925/PREREG.md）。
# 冻结副本 GNN_m1_v52_frozen_20260925（在任务 2 改代码之前建）；先预检（锚点 X5Dcap_v52cv_f0_s1234 + M1_f0_s1234 重评到新目录、配对构造、冒烟），
# 通过后本机队列 train → eval best → eval last（--save-predictions），4 卡 × 2 槽轮转；指纹复核冻结副本 *.py + 配置 + 统计/split。
set -euo pipefail
FROZEN=/public/newhome/cy/Digital_twin/GNN_m1_v52_frozen_20260925
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
NAME=wss_cycle_m1_v52_20260925
cd "$FROZEN"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
FIRST_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f1)
CUDA_VISIBLE_DEVICES=$FIRST_GPU $PY -u -m training_wss_min.tools.preflight_local_matrix --name $NAME \
  --anchor-run "$FROZEN/training_wss_min/runs/wss_v52_20260923/X5Dcap_v52cv_f0_s1234" \
  --anchor-run "$FROZEN/training_wss_min/runs/wss_cycle_m1_20260921/M1_f0_s1234" \
  --smoke M1cap_v52cv_f0_s1234.json
$PY -u -m training_wss_min.tools.run_local_train_queue \
  --config-dir "$FROZEN/training_wss_min/configs/$NAME" \
  --gpus "$CUDA_VISIBLE_DEVICES" --slots-per-gpu 2 --min-free-mib 6000 --launch-interval-seconds 30 \
  --status-file queue_status_master.json
date -Is
