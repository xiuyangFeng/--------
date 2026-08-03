#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <run_id>" >&2
  exit 2
fi
HOST="${NODE04_HOST:-cy@node04}"
RUN_DIR="/public/newhome/cy/Digital_twin/GNN/wss_mri_calculator/experiments/pointcloud_normal_multiscale_v3/runs/$1"

ssh "$HOST" "bash -s" -- "$RUN_DIR" <<'REMOTE'
set -euo pipefail
run_dir=$1
pid_file="$run_dir/worker.pid"
if [[ ! -f "$pid_file" ]]; then
  echo "worker pid not found: $pid_file" >&2
  exit 2
fi
pid=$(cat "$pid_file")
if kill -0 "$pid" 2>/dev/null; then
  kill -TERM "$pid"
  printf '{"state":"stopped","worker_pid":%s,"stopped_at":"%s"}\n' \
    "$pid" "$(date -Iseconds)" >"$run_dir/status.json"
  echo "stopped pid=$pid"
else
  echo "pid=$pid is not running"
fi
REMOTE
