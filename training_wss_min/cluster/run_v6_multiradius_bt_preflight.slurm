#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --job-name=v6_mrb_pre
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=01:00:00
set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
cd "$PROJECT"
exec "$PY" -u -m training_wss_min.tools.run_v6_multiradius_bt_queue \
  --matrix "$PROJECT/training_wss_min/configs/v6_multiradius_bt_20260909/matrix.json" \
  --smoke --epochs "${V6_SMOKE_EPOCHS:-2}" --slots-per-gpu 1 --min-free-mib 7000
