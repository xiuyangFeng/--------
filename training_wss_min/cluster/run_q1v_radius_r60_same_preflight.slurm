#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=q1vr60_pf
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
CONFIG="$PROJECT/training_wss_min/configs/pointnetpp_q1v_radius_followup_20260719/q1v_radius_r60_same.json"
OUTPUT="$PROJECT/training_wss_min/preflight/q1v_radius_r60_same_gpu_preflight.json"
cd "$PROJECT"

echo "experiment=q1v_radius_r60_same started=$(date --iso-8601=seconds)"
"$PY" -m training_wss_min.tools.preflight_v4_jobs --configs "$CONFIG" --output "$OUTPUT"
echo "experiment=q1v_radius_r60_same status=passed finished=$(date --iso-8601=seconds)"
