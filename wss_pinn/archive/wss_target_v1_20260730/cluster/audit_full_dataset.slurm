#!/bin/bash
# Deep read-only audit of the frozen grouped train138/test36 source dataset.
set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
SPLIT="${WSS_PINN_SPLIT:?WSS_PINN_SPLIT is required}"
OUTPUT_DIR="${WSS_PINN_AUDIT_OUTPUT:?WSS_PINN_AUDIT_OUTPUT is required}"
PY="${WSS_PINN_PYTHON:?WSS_PINN_PYTHON is required}"

cd "$PROJECT"
echo "audit_start=$(date --iso-8601=seconds) split=$SPLIT output=$OUTPUT_DIR host=$(hostname)"
"$PY" -m wss_pinn.tools.audit_full_dataset \
  --split "$SPLIT" \
  --output-dir "$OUTPUT_DIR" \
  --mode deep
echo "audit_completed=$(date --iso-8601=seconds)"
