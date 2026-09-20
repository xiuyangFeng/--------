#!/bin/bash
cd /public/newhome/cy/Digital_twin/GNN
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
OFF=training_wss_min/experiments/wss_local_wave6_20260915/offline
$PY -m training_wss_min.tools.deployment_stl_simulation --out $OFF/stl_x5_fine --spacings 0.35
echo "=== fine run done ==="
