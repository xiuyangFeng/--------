#!/bin/bash
# Aggregate all F1 diagnostic best/last evaluations against the frozen F0-UP control.
set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
CONFIG_LIST="${WSS_PINN_CONFIG_LIST:?WSS_PINN_CONFIG_LIST is required}"
OUTPUT_DIR="${WSS_PINN_MATRIX_OUTPUT:?WSS_PINN_MATRIX_OUTPUT is required}"
PY="${WSS_PINN_PYTHON:?WSS_PINN_PYTHON is required}"

cd "$PROJECT"
mapfile -t CONFIGS < <(sed '/^[[:space:]]*$/d' "$CONFIG_LIST")
echo "matrix_analysis_start=$(date --iso-8601=seconds) configs=${#CONFIGS[@]} host=$(hostname)"
"$PY" -m wss_pinn.tools.analyze_f1_matrix \
  --configs "${CONFIGS[@]}" \
  --output-dir "$OUTPUT_DIR"
echo "matrix_analysis_completed=$(date --iso-8601=seconds)"
