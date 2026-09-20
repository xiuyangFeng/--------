#!/bin/bash
cd /public/newhome/cy/Digital_twin/GNN
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
OFF=training_wss_min/experiments/wss_local_wave5_20260913/offline
$PY -m training_wss_min.tools.deployment_resample_reeval --out $OFF/deployment_smoke --limit-cases 1 --fractions 1.0 0.25 || exit 1
echo "=== smoke done, starting full run ==="
$PY -m training_wss_min.tools.deployment_resample_reeval --out $OFF/deployment_reeval_x5 --fractions 1.0 0.5 0.25 0.1
echo "=== full run done ==="
