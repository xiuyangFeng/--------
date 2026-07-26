#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=s3reg_pf
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=12:00:00
set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="${MATRIX_MANIFEST:-$PROJECT/training_wss_min/preflight/s3_regularization_matrix_prepared.json}"
STATIC="$PROJECT/training_wss_min/preflight/s3_regularization_matrix_static_audit.json"
GEOMETRY="$PROJECT/training_wss_min/preflight/s3_regularization_matrix_geometry_audit.json"
GPU="$PROJECT/training_wss_min/preflight/s3_regularization_matrix_gpu_smoke.json"
cd "$PROJECT"
"$PY" -m training_wss_min.tools.audit_s3_rootcause_static --manifest "$MANIFEST" --output "$STATIC"
"$PY" -m training_wss_min.tools.audit_sa1_scale_matrix --manifest "$MANIFEST" --output "$GEOMETRY"
"$PY" -m training_wss_min.tools.smoke_d2_k64_three_seed_confirmation --manifest "$MANIFEST" --output "$GPU"
echo "status=passed"
