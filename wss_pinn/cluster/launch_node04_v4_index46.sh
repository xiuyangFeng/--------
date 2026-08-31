#!/bin/bash
# Direct-start V4 array index 46 on node04. Not Slurm. Fresh start only.
set -euo pipefail
ulimit -n 65535
cd /public/newhome/cy/Digital_twin/GNN
export HOME=/public/newhome/cy
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
CONFIG=/public/newhome/cy/Digital_twin/GNN/wss_pinn/configs/volume_uvwp_bc_rcr_v4/transient_autograd/v4_tr_pnpp_bc_pde_f_s3456.json
LOG=/public/newhome/cy/Digital_twin/GNN/outputs/wss_pinn/slurm/v4_train_node04_46.out
exec "$PY" -m wss_pinn.v4.train --config "$CONFIG" >"$LOG" 2>&1
