#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/newhome/cy/Digital_twin/GNN
EXPERIMENT_DIR="$ROOT/wss_mri_calculator/experiments/pointcloud_normal_multiscale_v3"
ENTRY="$ROOT/wss_mri_calculator/src/run_v3_experiment.py"
DEFAULT_PY=/public/newhome/cy/.conda/envs/GNN/bin/python

CONFIG=""
GPU="auto"
WAIT=0
WORKER=0
RUN_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --wait) WAIT=1; shift ;;
    --worker) WORKER=1; shift ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$CONFIG" || ! -f "$CONFIG" ]]; then
  echo "missing config: $CONFIG" >&2
  exit 2
fi
if [[ "$(hostname -s)" != "node04" ]]; then
  echo "this script must run on node04" >&2
  exit 20
fi
if [[ "$(id -u)" != "1006" || "$(id -g)" != "1007" ]]; then
  echo "node04 UID/GID mismatch: $(id -u)/$(id -g)" >&2
  exit 20
fi
test -x "$DEFAULT_PY"

readarray -t runtime < <(
  "$DEFAULT_PY" - "$CONFIG" <<'PY'
import json, sys
cfg=json.load(open(sys.argv[1], encoding="utf-8"))
runtime=cfg["runtime"]
print(runtime["conda_python"])
print(int(runtime.get("max_gpu_memory_used_mb_before_launch", 1024)))
print(int(runtime.get("poll_seconds", 60)))
print(cfg["experiment_name"])
PY
)
PYTHON=${runtime[0]}
MAX_USED_MB=${runtime[1]}
POLL_SECONDS=${runtime[2]}
EXPERIMENT_NAME=${runtime[3]}
test -x "$PYTHON"

if [[ "$WORKER" == "0" ]]; then
  config_sha=$(sha256sum "$CONFIG" | awk '{print substr($1,1,12)}')
  run_id="$(date +%Y%m%d_%H%M%S)_${EXPERIMENT_NAME}_${config_sha}"
  RUN_DIR="$EXPERIMENT_DIR/runs/$run_id"
  mkdir -p "$RUN_DIR"
  config_snapshot="$RUN_DIR/config.json"
  "$DEFAULT_PY" - "$CONFIG" "$config_snapshot" <<'PY'
import json, sys
from pathlib import Path
source=Path(sys.argv[1]).resolve()
target=Path(sys.argv[2])
payload=json.loads(source.read_text(encoding="utf-8"))
base=source.parent
def absolute(value):
    path=Path(value)
    return str(path if path.is_absolute() else (base/path).resolve())
payload["split"]["path"]=absolute(payload["split"]["path"])
payload["split"]["data_root"]=absolute(payload["split"]["data_root"])
payload["output"]["json_path"]=absolute(payload["output"]["json_path"])
for comparator in payload["comparators"]:
    if "config" in comparator:
        comparator["config"]=absolute(comparator["config"])
target.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
PY
  wait_arg=()
  if [[ "$WAIT" == "1" ]]; then wait_arg=(--wait); fi
  nohup "$0" --worker --config "$config_snapshot" --gpu "$GPU" \
    --run-dir "$RUN_DIR" "${wait_arg[@]}" \
    >"$RUN_DIR/run.log" 2>&1 </dev/null &
  worker_pid=$!
  echo "$worker_pid" >"$RUN_DIR/worker.pid"
  printf '{"state":"queued","worker_pid":%s,"config":"%s","requested_gpu":"%s"}\n' \
    "$worker_pid" "$config_snapshot" "$GPU" >"$RUN_DIR/status.json"
  echo "run_id=$run_id"
  echo "worker_pid=$worker_pid"
  echo "log=$RUN_DIR/run.log"
  exit 0
fi

mkdir -p "$RUN_DIR"
cd "$ROOT"

choose_and_lock_gpu() {
  local requested="$1"
  local candidates=()
  if [[ "$requested" == "auto" ]]; then
    candidates=(0 1)
  else
    candidates=("$requested")
  fi
  for candidate in "${candidates[@]}"; do
    if [[ "$candidate" != "0" && "$candidate" != "1" ]]; then
      continue
    fi
    local lock_file="/tmp/cy_wss_v3_gpu${candidate}.lock"
    exec {gpu_lock_fd}>"$lock_file"
    if ! flock -n "$gpu_lock_fd"; then
      exec {gpu_lock_fd}>&-
      continue
    fi
    local used
    used=$(nvidia-smi --id="$candidate" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used <= MAX_USED_MB )); then
      SELECTED_GPU="$candidate"
      SELECTED_GPU_USED_MB="$used"
      return 0
    fi
    flock -u "$gpu_lock_fd"
    exec {gpu_lock_fd}>&-
  done
  return 1
}

while ! choose_and_lock_gpu "$GPU"; do
  printf '{"state":"waiting_for_gpu","requested_gpu":"%s","poll_seconds":%s}\n' \
    "$GPU" "$POLL_SECONDS" >"$RUN_DIR/status.json"
  if [[ "$WAIT" != "1" ]]; then
    echo "no requested GPU is free; rerun with --wait to queue on node04" >&2
    exit 30
  fi
  sleep "$POLL_SECONDS"
done

printf '{"state":"running","worker_pid":%s,"gpu":%s,"gpu_memory_used_mb_at_start":%s,"started_at":"%s"}\n' \
  "$$" "$SELECTED_GPU" "$SELECTED_GPU_USED_MB" "$(date -Iseconds)" >"$RUN_DIR/status.json"
echo "selected_gpu=$SELECTED_GPU memory_used_mb=$SELECTED_GPU_USED_MB"
echo "config=$CONFIG"
echo "started_at=$(date -Iseconds)"

child_pid=""
forward_stop() {
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
  fi
}
trap forward_stop TERM INT
set +e
CUDA_VISIBLE_DEVICES="$SELECTED_GPU" "$PYTHON" "$ENTRY" --config "$CONFIG" &
child_pid=$!
echo "$child_pid" >"$RUN_DIR/experiment.pid"
wait "$child_pid"
exit_code=$?
set -e
trap - TERM INT

if [[ "$exit_code" == "0" ]]; then
  state=completed
else
  state=failed
fi
printf '{"state":"%s","worker_pid":%s,"gpu":%s,"exit_code":%s,"finished_at":"%s"}\n' \
  "$state" "$$" "$SELECTED_GPU" "$exit_code" "$(date -Iseconds)" >"$RUN_DIR/status.json"
echo "finished_at=$(date -Iseconds) exit_code=$exit_code"
exit "$exit_code"
