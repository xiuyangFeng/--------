#!/bin/bash
#SBATCH --job-name=vfaudit
#SBATCH --partition=CPU
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=outputs/wss_pinn/slurm/volume_field_audit_%j.out
#SBATCH --error=outputs/wss_pinn/slurm/volume_field_audit_%j.err

set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python

"$PY" -m wss_pinn.tools.audit_dataset --deep-source
