#!/bin/bash
# Generic WSS-PINN preflight. Resources and logs are supplied by sbatch arguments.
set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
CONFIG="${WSS_PINN_CONFIG:?WSS_PINN_CONFIG is required}"
RUN_DIR="${WSS_PINN_RUN_DIR:?WSS_PINN_RUN_DIR is required}"
PY="${WSS_PINN_PYTHON:?WSS_PINN_PYTHON is required}"

cd "$PROJECT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "preflight_start=$(date --iso-8601=seconds) config=$CONFIG run_dir=$RUN_DIR host=$(hostname)"
"$PY" -m unittest discover -s wss_pinn/tests -p 'test_*.py' -v
"$PY" -m wss_pinn.tools.preflight \
  --config "$CONFIG" \
  --require-science-gate \
  --output "$RUN_DIR/preflight.json"
"$PY" -m wss_pinn.train --config "$CONFIG" --dry-run --device cuda
echo "preflight_completed=$(date --iso-8601=seconds)"
