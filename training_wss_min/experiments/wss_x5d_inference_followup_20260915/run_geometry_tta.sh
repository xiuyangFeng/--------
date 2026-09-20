#!/bin/bash
# Inference-only follow-ups to X5D (2026-09-15): geometry-fidelity sensitivity and test-time sampling ensemble.
# Both use the deployment form = X5D five seeds (wave 6: 1234/7/2025, wave 6b: 11/2026), best checkpoint, GPU 2.
cd /public/newhome/cy/Digital_twin/GNN
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
EXP=training_wss_min/experiments/wss_x5d_inference_followup_20260915
R6=training_wss_min/runs/wss_local_wave6_20260915; R6B=training_wss_min/runs/wss_local_wave6b_20260915
RUNS="$R6/X5D_s1234 $R6/X5D_s7 $R6/X5D_s2025 $R6B/X5D_s11 $R6B/X5D_s2026"
case "$1" in
  geometry)
    $PY -m training_wss_min.tools.deployment_geometry_sensitivity --runs $RUNS --out $EXP/geometry_x5d5
    echo "=== geometry sensitivity done (exit $?) ===" ;;
  tta)
    $PY -m training_wss_min.tools.deployment_sampling_tta --runs $RUNS --out $EXP/tta_x5d5 --draws 5
    echo "=== sampling TTA done (exit $?) ===" ;;
  *) echo "usage: $0 geometry|tta"; exit 2 ;;
esac
