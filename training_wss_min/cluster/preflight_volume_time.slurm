#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --job-name=voltime_preflight
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
# 队列的硬门禁：24 个臂逐个做配对构造 / 三步有限 AMP / 全 query 推理 + 分块一致性，
# 外加锚点重评（X5D_v51_s1234：v5.1 上唯一还能重评的已训 run，旧 v5 体场视图已删）
# 与四个最重的臂跑 2-epoch CLI 冒烟（压力/速度 × 相位查询/时间基）。
# 内存 300G：体场每例约 46 万点，preflight 会为每种 (target, timesteps, 折) 组合各缓存 8 例。
set -euo pipefail
FROZEN=/public/newhome/cy/Digital_twin/GNN_voltime_frozen_20260919
MAIN=/public/newhome/cy/Digital_twin/GNN
cd "$FROZEN"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname; nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.preflight_wss_local_wave1 \
  --config-dir "$MAIN/training_wss_min/configs/volume_time_20260919" \
  --experiment-dir "$MAIN/training_wss_min/experiments/volume_time_20260919" \
  --control-arm PF6_v51_f0_s1234 \
  --smoke-arms PT0_f0_s1234,PTB8_f0_s1234,VT0_f0_s1234,VTB4_f0_s1234 \
  --anchor-run "$MAIN/training_wss_min/runs/wss_v51_wave1_20260916/X5D_v51_s1234"
date -Is
