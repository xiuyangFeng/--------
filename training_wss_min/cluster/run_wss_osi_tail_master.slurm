#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --job-name=osi_tail_master
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.err
# OSI 尾部矩阵第二波（K2 / K5 × cv3）在 master 4×4090 上执行（2026-09-24 用户裁定；PREREG.md「执行变更」）。
# 先在本硬件上重跑预检（锚点 M1_f0 best 重评到新目录 + 12 臂配对构造 + 四类臂 2 epoch 冒烟），通过后本机队列：
# train → eval best → eval last（--save-predictions），轮转每卡 2 槽；指纹复核冻结副本 training_wss_min/*.py + 配置 + 统计/split。
# 第一波 M1r / K1 在 node04 另行执行，本作业只选 K2*、K5*。
set -euo pipefail
FROZEN=/public/newhome/cy/Digital_twin/GNN_osi_tail_frozen_20260924
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
cd "$FROZEN"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
FIRST_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f1)
CUDA_VISIBLE_DEVICES=$FIRST_GPU $PY -u -m training_wss_min.tools.preflight_wss_osi_tail
$PY -u -m training_wss_min.tools.run_local_train_queue \
  --config-dir "$FROZEN/training_wss_min/configs/wss_osi_tail_20260924" \
  --gpus "$CUDA_VISIBLE_DEVICES" --slots-per-gpu 2 --min-free-mib 6000 --launch-interval-seconds 30 \
  --only 'K2*,K5*' --status-file queue_status_master.json
date -Is
