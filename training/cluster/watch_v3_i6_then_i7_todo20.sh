#!/usr/bin/env bash
# 等待 V3P I6-diag 训练结束，自动跑完整 I7 + TODO-20 离线脚本。
#
# 用法（仓库根）:
#   nohup bash training/cluster/watch_v3_i6_then_i7_todo20.sh >> logs/v3p_i6_postprocess_watch.out 2>&1 &
#
# 环境变量（可选）:
#   I6_RUN_DIR   诊断 run 目录（默认 i6diag seed1 20260619）
#   I6_PID       训练进程 PID（默认 2473568；设为 0 则只等 summary.json）
#   CUDA_DEVICE  离线评测 GPU（默认 0）
#   POLL_SEC     轮询间隔秒（默认 300）

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs outputs/field/f0_decision

PYTHON="${TRAINING_PYTHON:-/public/newhome/cy/.conda/envs/GNN/bin/python}"
RUN_DIR="${I6_RUN_DIR:-$ROOT/outputs/field/field_v3_pointnext_i6diag_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260619_174001}"
I6_PID="${I6_PID:-2473568}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
POLL_SEC="${POLL_SEC:-300}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/v3p_i6_postprocess_${STAMP}.out"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== V3P I6 完成后自动 I7 + TODO-20 ==="
log "RUN_DIR=$RUN_DIR"
log "I6_PID=$I6_PID CUDA_DEVICE=$CUDA_DEVICE POLL_SEC=$POLL_SEC"
log "Python: $PYTHON"

if [[ ! -d "$RUN_DIR" ]]; then
  log "ERROR: run 目录不存在: $RUN_DIR"
  exit 1
fi

wait_for_i6() {
  while true; do
    local hist_epochs=0
    if [[ -f "$RUN_DIR/history.csv" ]]; then
      hist_epochs=$(( $(wc -l < "$RUN_DIR/history.csv") - 1 ))
    fi

    local pid_alive=0
    if [[ "$I6_PID" != "0" ]] && kill -0 "$I6_PID" 2>/dev/null; then
      pid_alive=1
    fi

    if [[ -f "$RUN_DIR/summary.json" ]] && [[ "$pid_alive" -eq 0 ]]; then
      log "I6-diag 已完成: summary.json 已落盘, PID 已退出 (history epochs=$hist_epochs)"
      return 0
    fi

    if [[ -f "$RUN_DIR/summary.json" ]] && [[ "$I6_PID" == "0" ]]; then
      log "I6-diag 已完成: summary.json 已落盘 (history epochs=$hist_epochs)"
      return 0
    fi

    log "等待 I6-diag ... pid_alive=$pid_alive history_epochs=$hist_epochs"
    sleep "$POLL_SEC"
  done
}

wait_for_i6

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

log ">>> 开始完整 I7 ckpt probe"
if ! "$PYTHON" -m training.scripts.run_v3_i7_ckpt_probe \
  --run-dir "$RUN_DIR" \
  2>&1 | tee -a "$LOG"; then
  log "ERROR: I7 probe 失败 (exit=$?)"
  exit 1
fi
log ">>> I7 完成"

log ">>> 开始 TODO-20 ckpt select / EMA"
if ! "$PYTHON" -m training.scripts.run_v3_todo20_ckpt_select \
  --run-dir "$RUN_DIR" \
  2>&1 | tee -a "$LOG"; then
  log "ERROR: TODO-20 选优失败 (exit=$?)"
  exit 1
fi
log ">>> TODO-20 完成"

log "=== 全部完成 ==="
log "产物目录: outputs/field/f0_decision/"
log "详细日志: $LOG"
