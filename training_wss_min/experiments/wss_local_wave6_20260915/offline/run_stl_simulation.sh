#!/bin/bash
cd /public/newhome/cy/Digital_twin/GNN
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
OFF=training_wss_min/experiments/wss_local_wave6_20260915/offline
$PY -m training_wss_min.tools.deployment_stl_simulation --out $OFF/stl_smoke2 --limit-cases 1 --spacings 0 0.8 || exit 1
echo "=== smoke done, starting full run ==="
$PY -m training_wss_min.tools.deployment_stl_simulation --out $OFF/stl_x5 --spacings 0 0.5 0.8 1.2
echo "=== full run done ==="
