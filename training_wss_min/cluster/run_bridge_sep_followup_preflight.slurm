#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=bridgesep_pf
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="$PROJECT/training_wss_min/preflight/bridge_sep_followup_prepared.json"
CONFIG="$PROJECT/training_wss_min/configs/pointnetpp_sa1_scale_bridge_sep_followup_20260722/bridge_rand5000_fixed500_k64_sep.json"
AUDIT="$PROJECT/training_wss_min/preflight/bridge_sep_followup_geometry_audit.json"
PREFLIGHT="$PROJECT/training_wss_min/preflight/bridge_sep_followup_gpu_preflight.json"
cd "$PROJECT"

echo "experiment=bridge_rand5000_fixed500_k64_sep started=$(date --iso-8601=seconds)"
"$PY" -m training_wss_min.tools.audit_sa1_scale_matrix \
  --manifest "$MANIFEST" --output "$AUDIT"
"$PY" -m training_wss_min.tools.preflight_v4_jobs \
  --configs "$CONFIG" --output "$PREFLIGHT"
echo "experiment=bridge_rand5000_fixed500_k64_sep status=passed finished=$(date --iso-8601=seconds)"
