#!/bin/bash
# Verify the completed full sidecar dataset, including all parent/file hashes.
set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
MANIFEST="${WSS_PINN_MANIFEST:?WSS_PINN_MANIFEST is required}"
SPLIT="${WSS_PINN_SPLIT:?WSS_PINN_SPLIT is required}"
OUTPUT_DIR="${WSS_PINN_AUDIT_OUTPUT:?WSS_PINN_AUDIT_OUTPUT is required}"
PY="${WSS_PINN_PYTHON:?WSS_PINN_PYTHON is required}"

cd "$PROJECT"
echo "sidecar_audit_start=$(date --iso-8601=seconds) manifest=$MANIFEST host=$(hostname)"
"$PY" -m wss_pinn.tools.audit_sidecar_dataset \
  --manifest "$MANIFEST" \
  --split "$SPLIT" \
  --output-dir "$OUTPUT_DIR"
echo "sidecar_audit_completed=$(date --iso-8601=seconds)"
