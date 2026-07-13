#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=wssmin_r5_a0m
#SBATCH --output=training_wss_min/runs/_round5/a0_micro/slurm_%j.out
#SBATCH --error=training_wss_min/runs/_round5/a0_micro/slurm_%j.err
#SBATCH --time=01:00:00

set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then source "$HOME/.bashrc"; fi
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "/public/newapps/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/public/newapps/anaconda3/etc/profile.d/conda.sh"
else
    echo "cannot find conda initialization" >&2
    exit 1
fi
conda activate GNN

cd /public/newhome/cy/Digital_twin/GNN
PROTOCOL="${1:-training_wss_min/configs/round5/a0_micro_protocol.json}"
"${CONDA_PREFIX}/bin/python" -m training_wss_min.experiments.a0_micro_overfit \
    --mode train --protocol "$PROTOCOL"
