#!/bin/bash
#SBATCH --job-name=v4preflight
#SBATCH --partition=GPU
#SBATCH --time=0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/wss_pinn/slurm/v4_preflight_%j.out
#SBATCH --error=outputs/wss_pinn/slurm/v4_preflight_%j.err

set -euo pipefail
ulimit -n 65535
cd /public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$PY" -m unittest discover -s wss_pinn/tests -p 'test_volume_*.py' -v
"$PY" -c 'import json; p=json.load(open("outputs/wss_pinn/volume_uvwp_bc_rcr_v4/audits/stage0/gate_report.json")); assert p["gate_result"] == "pass"'
"$PY" -m wss_pinn.tools.preflight_bc_rcr_v4 \
  --gpu-smoke \
  --output outputs/wss_pinn/volume_uvwp_bc_rcr_v4/audits/preflight/report.json
