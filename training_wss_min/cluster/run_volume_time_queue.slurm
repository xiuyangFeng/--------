#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --job-name=voltime_queue
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
# 体场全周期矩阵：PF6_v51/VF6_v51 折底座 6 训练 + T0/TB8/TB16 × 压力/速度 × 三折 18 训练，seed 1234。
# 每臂 train → eval best（时间臂另评 last）。配置与产物路径都指向主树，代码用冻结副本执行。
# 内存给到 320G：体场每例 wall∪interior 约 40 万点，TB16 速度的系数目标是 51 通道，
# 4 个并发 run 各自要放下 90 例训练集 + 45 例留出折。
set -euo pipefail
FROZEN=/public/newhome/cy/Digital_twin/GNN_voltime_frozen_20260919
MAIN=/public/newhome/cy/Digital_twin/GNN
cd "$FROZEN"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname; nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.run_wss_local_wave1_queue \
  --config-dir "$MAIN/training_wss_min/configs/volume_time_20260919" \
  --experiment-dir "$MAIN/training_wss_min/experiments/volume_time_20260919" \
  --slots-per-gpu 1 --min-free-mib 7000 --launch-interval-seconds 15
date -Is
