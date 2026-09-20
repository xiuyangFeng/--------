#!/bin/bash
# B2: does deployment-side smoothing recover the boundary-noise loss?  (noise then Taubin smoothing, then 0.5 mm resampling)
cd /public/newhome/cy/Digital_twin/GNN
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
EXP=training_wss_min/experiments/wss_x5d_inference_followup_20260915
R6=training_wss_min/runs/wss_local_wave6_20260915; R6B=training_wss_min/runs/wss_local_wave6b_20260915
RUNS="$R6/X5D_s1234 $R6/X5D_s7 $R6/X5D_s2025 $R6B/X5D_s11 $R6B/X5D_s2026"
$PY -m training_wss_min.tools.deployment_geometry_sensitivity --runs $RUNS --out $EXP/geometry_x5d5_b2 --variants noise_0.2+smooth_s1.0 noise_0.4+smooth_s1.0 noise_0.4+smooth_s1.5
echo "=== geometry B2 done (exit $?) ==="
