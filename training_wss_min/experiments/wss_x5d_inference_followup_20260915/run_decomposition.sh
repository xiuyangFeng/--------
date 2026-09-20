#!/bin/bash
# Neighbourhood decomposition of the density loss (frozen features): X5D 3 seeds then X5 3 seeds.
cd /public/newhome/cy/Digital_twin/GNN
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
EXP=training_wss_min/experiments/wss_x5d_inference_followup_20260915
R6=training_wss_min/runs/wss_local_wave6_20260915
X5="training_wss_min/runs/wss_local_wave1_20260912/X5_s1234 training_wss_min/runs/wss_local_wave1b_20260912/X5_s7 training_wss_min/runs/wss_local_wave1b_20260912/X5_s2025"
$PY -m training_wss_min.tools.deployment_neighbourhood_decomposition --runs $R6/X5D_s1234 $R6/X5D_s7 $R6/X5D_s2025 --out $EXP/decomposition_x5d --fractions 0.5 0.25 0.1
echo "=== decomposition X5D done (exit $?) ==="
$PY -m training_wss_min.tools.deployment_neighbourhood_decomposition --runs $X5 --out $EXP/decomposition_x5 --fractions 0.5 0.25 0.1
echo "=== decomposition X5 done (exit $?) ==="
