#!/bin/bash
cd /public/newhome/cy/Digital_twin/GNN
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
OFF=training_wss_min/experiments/wss_local_wave6_20260915/offline
R6=training_wss_min/runs/wss_local_wave6_20260915; R6B=training_wss_min/runs/wss_local_wave6b_20260915
$PY -m training_wss_min.tools.deployment_stl_simulation --runs $R6/X5D_s1234 $R6/X5D_s7 $R6/X5D_s2025 $R6B/X5D_s11 $R6B/X5D_s2026 --out $OFF/stl_x5d_5seed --spacings 0.5 1.2
echo "=== stl x5d 5-seed done ==="
