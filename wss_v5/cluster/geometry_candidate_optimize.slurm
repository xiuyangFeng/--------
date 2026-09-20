#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=v6_geom_opt
#SBATCH --output=outputs/wss_v6_geometry_optimization_%A_%a.out
#SBATCH --error=outputs/wss_v6_geometry_optimization_%A_%a.err
set -euo pipefail
cd /public/newhome/cy/Digital_twin/GNN
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
GEOMETRY_PY=/public/newhome/cy/.conda/envs/GNN/bin/python

# sbatch [--output=<new>/slurm/%j.out] [--error=<new>/slurm/%j.err] \
#   wss_v5/cluster/geometry_candidate_optimize.slurm refine \
#   --source <old-candidate-root> --output <new-candidate-root> \
#   --split <train136-test34.json> --workers 8
# Modes build/audit forward their respective geometry CLI arguments unchanged.
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <build|refine|audit> [module arguments...]" >&2
  exit 2
fi
GEOMETRY_MODE=$1
shift
case "$GEOMETRY_MODE" in
  build)
    GEOMETRY_INDEX=()
    if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
      GEOMETRY_INDEX=(--case-index "$SLURM_ARRAY_TASK_ID")
    fi
    "$GEOMETRY_PY" -u -m wss_v5.geometry_candidate "${GEOMETRY_INDEX[@]}" "$@"
    ;;
  refine|audit)
    if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
      echo "$GEOMETRY_MODE must be one job without --array" >&2
      exit 2
    fi
    if [ "$GEOMETRY_MODE" = refine ]; then
      GEOMETRY_MODULE=wss_v5.refine_geometry_candidate
    else
      GEOMETRY_MODULE=wss_v5.audit_geometry_candidate
    fi
    "$GEOMETRY_PY" -u -m "$GEOMETRY_MODULE" "$@"
    ;;
  *)
    echo "Unknown mode: $GEOMETRY_MODE" >&2
    exit 2
    ;;
esac
