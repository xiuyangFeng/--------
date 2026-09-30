#!/bin/bash
# 任务 2 在 node04 执行（Slurm 之外）：预检（GPU 1）通过后开本机队列（2×A100 每卡 3 槽）。
set -euo pipefail
FROZEN=/public/newhome/cy/Digital_twin/GNN_osi_struct_frozen_20260925
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
NAME=wss_osi_struct_20260925
cd "$FROZEN"
date -Is; hostname; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
CUDA_VISIBLE_DEVICES=1 $PY -u -m training_wss_min.tools.preflight_local_matrix --name $NAME \
  --anchor-run "$FROZEN/training_wss_min/runs/wss_cycle_m1_20260921/M1_f0_s1234" \
  --smoke V1_f0_s1234.json --smoke V2_f0_s1234.json
$PY -u -m training_wss_min.tools.run_local_train_queue --config-dir "$FROZEN/training_wss_min/configs/$NAME" \
  --gpus 0,1 --slots-per-gpu 3 --min-free-mib 6000 --launch-interval-seconds 60 --status-file queue_status_node04.json
date -Is
