#!/bin/bash
cd /public/newhome/cy/Digital_twin/GNN
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
EXP=training_wss_min/experiments/wss_x5d_inference_followup_20260915
R6=training_wss_min/runs/wss_local_wave6_20260915
$PY -m training_wss_min.tools.deployment_neighbourhood_decomposition --runs $R6/X5D_s1234 $R6/X5D_s7 $R6/X5D_s2025 --out $EXP/decomposition_x5d_scale --fractions 0.25 0.1 --variants thin/thin thin/thin_rescaled thin/full_inflated thin/full
echo "=== decomposition scale diag done (exit $?) ==="
