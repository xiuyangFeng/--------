#!/bin/bash
# Acceptance of the density-augmented X5D (wave 6) once the three seeds are trained:
#   1) resample re-evaluation at 100/50/25/10 % (same tool/protocol as §21.4)
#   2) STL-resampling deployment simulation (same sweep as X5)
cd /public/newhome/cy/Digital_twin/GNN
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
OFF=training_wss_min/experiments/wss_local_wave6_20260915/offline
RUNS="training_wss_min/runs/wss_local_wave6_20260915/X5D_s1234 training_wss_min/runs/wss_local_wave6_20260915/X5D_s7 training_wss_min/runs/wss_local_wave6_20260915/X5D_s2025"
X5="training_wss_min/runs/wss_local_wave1_20260912/X5_s1234 training_wss_min/runs/wss_local_wave1b_20260912/X5_s7 training_wss_min/runs/wss_local_wave1b_20260912/X5_s2025"
$PY -m training_wss_min.tools.deployment_resample_reeval --runs $RUNS --out $OFF/deployment_reeval_x5d --fractions 1.0 0.5 0.25 0.1
$PY -m training_wss_min.tools.deployment_resample_reeval --runs $X5 --out $OFF/deployment_reeval_x5_3seed --fractions 1.0 0.5 0.25 0.1
$PY -m training_wss_min.tools.deployment_stl_simulation --runs $RUNS --out $OFF/stl_x5d --spacings 0 0.5 0.8 1.2
echo "=== X5D acceptance done ==="
