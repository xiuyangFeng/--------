#!/bin/bash
# User-authorized shared A100 execution; serial V5.1 cv3 matrix, seed 1234.
set -euo pipefail
if [[ "$(hostname -s)" != "node04" ]]; then
  echo "This experiment must run on node04." >&2
  exit 1
fi
cd /public/newhome/cy/Digital_twin/GNN_time_transformer_frozen_20260924_r3_shared
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.run_wss_time_transformer_queue \
  --config-dir /public/newhome/cy/Digital_twin/GNN_time_transformer_frozen_20260924_r3_shared/training_wss_min/configs/wss_time_transformer_v51_20260924_r3 \
  --gpu GPU-e5c48469-8122-adf2-62f0-44d68a46fc22 \
  --allow-shared-gpu --min-free-mib 12288 --poll-seconds 10
/public/newhome/cy/.conda/envs/GNN/bin/python -u -m training_wss_min.tools.report_wss_time_transformer \
  --config-dir /public/newhome/cy/Digital_twin/GNN_time_transformer_frozen_20260924_r3_shared/training_wss_min/configs/wss_time_transformer_v51_20260924_r3
