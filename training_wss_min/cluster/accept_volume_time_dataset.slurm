#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodelist=node03
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --job-name=voltime_acc_ds
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
# B5 的体场那一半：PF6/VF6 旧 run 的数据（旧 v5 视图）已被 v5.1 取代删除、无法重评，
# 改成与改动前的冻结代码树逐字节对拍峰值帧体场数据路径（压力 + 速度各 4 例）。
set -euo pipefail
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MAIN=/public/newhome/cy/Digital_twin/GNN
OLD=/public/newhome/cy/Digital_twin/GNN_time_frozen_20260918
OUT=$MAIN/training_wss_min/experiments/volume_time_20260919/acceptance/dataset
# 两棵树的 training_wss_min/experiments 都软链到主树，脚本本身共用一份
DUMP=training_wss_min/experiments/volume_time_20260919/offline/b5_dump.py
mkdir -p "$OUT"
export OMP_NUM_THREADS=4
date -Is; hostname
rc=0
for target in pressure_mixed velocity; do
  echo "=== $target ==="
  (cd "$OLD"  && PYTHONPATH="$OLD"  $PY -u "$DUMP" "$OUT/old_$target.npz" "$target")
  (cd "$MAIN" && PYTHONPATH="$MAIN" $PY -u "$DUMP" "$OUT/new_$target.npz" "$target")
  $PY "$MAIN/training_wss_min/experiments/volume_time_20260919/offline/b5_compare.py" \
    "$OUT/old_$target.npz" "$OUT/new_$target.npz" | tee "$OUT/compare_$target.txt" || rc=1
done
date -Is
exit $rc
