#!/bin/bash
#SBATCH --job-name=v4data
#SBATCH --partition=CPU
#SBATCH --time=0
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/wss_pinn/slurm/v4_build_%j.out
#SBATCH --error=outputs/wss_pinn/slurm/v4_build_%j.err

set -euo pipefail
ulimit -n 65535
cd /public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
WORKERS=${V4_BUILD_WORKERS:-8}

"$PY" -m wss_pinn.v4.build --workers "$WORKERS" --resume
"$PY" -c 'import json; p=json.load(open("outputs/wss_pinn/volume_uvwp_bc_rcr_v4/audits/stage0/gate_report.json")); assert p["gate_result"] == "pass"'
