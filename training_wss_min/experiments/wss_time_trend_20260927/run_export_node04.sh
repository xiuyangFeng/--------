#!/bin/bash
# 动画选例导出（CPU）：真值 + T-null / C-raw / TT-warm 的 80 帧预测。选例见 case_selection.json。
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
S=training_wss_min/experiments/wss_time_trend_20260927/trend_eval.py
LOG=training_wss_min/experiments/wss_time_trend_20260927/logs
CASES="AAA/unruputer/LIN_JIAN_RONG,AG/fast/LI_ZHEN_SHAN,ILO/GUO_QING_SHAN-0/before,ILO/WU_JUN-0/before,AG/slow/WEI_BAO_XING"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 $PY $S truth --cases "$CASES" --device cpu > "$LOG/export_truth.log" 2>&1
for A in Tnull C-raw TT-warm; do
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nohup $PY $S export --arm $A --cases "$CASES" --device cpu > "$LOG/export_$A.log" 2>&1 < /dev/null &
done
wait
echo export-done
