#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/newhome/cy/Digital_twin/GNN
EXPERIMENT_DIR="$ROOT/wss_mri_calculator/experiments/pointcloud_normal_multiscale_v3"
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
ENTRY="$ROOT/wss_mri_calculator/src/run_v3_experiment.py"
REMOTE_RUNNER="$EXPERIMENT_DIR/node04/run_v3_remote.sh"
PREFLIGHT="$EXPERIMENT_DIR/node04/preflight_node04.sh"
HOST="${NODE04_HOST:-cy@node04}"

CONFIG=""
GPU="auto"
EXECUTE=0
WAIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    --wait) WAIT=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "usage: $0 --config <json> [--gpu auto|0|1] [--execute] [--wait]" >&2
  exit 2
fi
CONFIG=$(realpath "$CONFIG")
"$PY" "$ENTRY" --config "$CONFIG" --validate-only >/dev/null

if [[ "$EXECUTE" == "0" ]]; then
  wait_text=""
  if [[ "$WAIT" == "1" ]]; then wait_text=" --wait"; fi
  echo "DRY RUN: configuration is valid; no remote process was started."
  echo "$PREFLIGHT"
  echo "ssh $HOST '$REMOTE_RUNNER --config $CONFIG --gpu $GPU$wait_text'"
  echo "Add --execute only when the user has authorized launch and node04 capacity is available."
  exit 0
fi

"$PREFLIGHT"
wait_arg=()
if [[ "$WAIT" == "1" ]]; then wait_arg=(--wait); fi
ssh "$HOST" "$REMOTE_RUNNER" --config "$CONFIG" --gpu "$GPU" "${wait_arg[@]}"
