#!/bin/bash
# Generic training/evaluation body shared by F0-U, F0-UP and F1.
set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
CONFIG="${WSS_PINN_CONFIG:?WSS_PINN_CONFIG is required}"
RUN_DIR="${WSS_PINN_RUN_DIR:?WSS_PINN_RUN_DIR is required}"
PY="${WSS_PINN_PYTHON:?WSS_PINN_PYTHON is required}"

cd "$PROJECT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
test -f "$RUN_DIR/preflight.json"
echo "training_start=$(date --iso-8601=seconds) config=$CONFIG run_dir=$RUN_DIR host=$(hostname)"
"$PY" -m wss_pinn.train --config "$CONFIG"
"$PY" -m wss_pinn.evaluate --config "$CONFIG" --checkpoint best
"$PY" -m wss_pinn.evaluate --config "$CONFIG" --checkpoint last
echo "training_completed=$(date --iso-8601=seconds)"
