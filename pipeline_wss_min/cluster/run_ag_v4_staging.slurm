#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-76%8
#SBATCH --job-name=ag_v4_stage
#SBATCH --output=pipeline_wss_min/cluster/logs/%x_%A_%a.out
#SBATCH --error=pipeline_wss_min/cluster/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="/public/newhome/cy/Digital_twin/GNN"
PY="/public/newhome/cy/.conda/envs/GNN/bin/python"
OUT_ROOT="${1:?usage: run_ag_v4_staging.slurm <staging-root>}"

cd "$PROJECT_DIR"
mkdir -p pipeline_wss_min/cluster/logs "$OUT_ROOT"

"$PY" -m pipeline_wss_min.v4_cutover preprocess \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --out-root "$OUT_ROOT" \
  --split split_AG_wss_min_v1
