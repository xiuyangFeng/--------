#!/bin/bash
#SBATCH --partition=CPU
#SBATCH -w node03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=normloss_prep
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
cd "$PROJECT"

echo "stage=cpu_norm_loss_prepare started=$(date --iso-8601=seconds) host=$(hostname)"
"$PY" -m training_wss_min.tools.prepare_norm_loss_matrix --repo "$PROJECT"
echo "stage=cpu_norm_loss_prepare status=passed finished=$(date --iso-8601=seconds)"
