#!/bin/bash
# 建立/刷新周期积分量矩阵的冻结副本（体例同 GNN_time_frozen_20260918）：
#   代码目录实拷贝：training_wss_min（含 configs、cluster、tests、tools）、wss_v5、wss_pinn、training
#   数据/产物软链回主仓库：data_new、data_wss_min、data_wss_v5、docs、outputs、tools、
#                         training_wss_min/{runs,experiments}、training_wss_min/cluster/logs
# 用法：bash training_wss_min/cluster/make_frozen_copy_cycle.sh [FROZEN_DIR]
set -euo pipefail
MAIN=/public/newhome/cy/Digital_twin/GNN
FROZEN=${1:-/public/newhome/cy/Digital_twin/GNN_cycle_frozen_20260920}
mkdir -p "$FROZEN"
for d in wss_v5 wss_pinn training; do
  rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' "$MAIN/$d/" "$FROZEN/$d/"
done
rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'runs' --exclude 'experiments' --exclude 'cluster/logs' \
  "$MAIN/training_wss_min/" "$FROZEN/training_wss_min/"
for d in data_new data_wss_min data_wss_v5 docs outputs tools; do
  [ -e "$FROZEN/$d" ] || ln -s "$MAIN/$d" "$FROZEN/$d"
done
[ -e "$FROZEN/training_wss_min/runs" ] || ln -s "$MAIN/training_wss_min/runs" "$FROZEN/training_wss_min/runs"
[ -e "$FROZEN/training_wss_min/experiments" ] || ln -s "$MAIN/training_wss_min/experiments" "$FROZEN/training_wss_min/experiments"
mkdir -p "$FROZEN/training_wss_min/cluster"
[ -e "$FROZEN/training_wss_min/cluster/logs" ] || ln -s "$MAIN/training_wss_min/cluster/logs" "$FROZEN/training_wss_min/cluster/logs"
# 逐字一致性核对（源码）
for d in wss_v5 wss_pinn training; do
  diff -rq --exclude '__pycache__' --exclude '*.pyc' "$MAIN/$d" "$FROZEN/$d" && echo "identical: $d"
done
diff -rq --exclude '__pycache__' --exclude '*.pyc' --exclude runs --exclude experiments --exclude logs \
  "$MAIN/training_wss_min" "$FROZEN/training_wss_min" && echo "identical: training_wss_min"
ls -la "$FROZEN" | sed -n 1,20p
