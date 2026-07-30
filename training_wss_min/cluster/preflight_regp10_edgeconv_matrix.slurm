#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=regp10ec_pf
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=12:00:00
set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="${MATRIX_MANIFEST:-$PROJECT/training_wss_min/preflight/regp10_edgeconv_matrix_20260727_prepared.json}"
STATIC="$PROJECT/training_wss_min/preflight/regp10_edgeconv_matrix_20260727_static_audit.json"
GEOMETRY="$PROJECT/training_wss_min/preflight/regp10_edgeconv_matrix_20260727_geometry_audit.json"
GPU="$PROJECT/training_wss_min/preflight/regp10_edgeconv_matrix_20260727_gpu_smoke.json"
cd "$PROJECT"
"$PY" -m training_wss_min.tools.audit_s3_rootcause_static \
  --manifest "$MANIFEST" --output "$STATIC"
"$PY" -m training_wss_min.tools.audit_sa1_scale_matrix \
  --manifest "$MANIFEST" --output "$GEOMETRY"
"$PY" -m training_wss_min.tools.smoke_regp10_edgeconv_matrix \
  --manifest "$MANIFEST" --output "$GPU"
