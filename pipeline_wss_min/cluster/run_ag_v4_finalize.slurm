#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=ag_v4_review
#SBATCH --output=pipeline_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=pipeline_wss_min/cluster/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="/public/newhome/cy/Digital_twin/GNN"
PY="/public/newhome/cy/.conda/envs/GNN/bin/python"
STAGING_ROOT="${1:?usage: run_ag_v4_finalize.slurm <ag-staging-root> <aaa-override-root> <report-dir> <visual-dir>}"
AAA_OVERRIDE_ROOT="${2:?missing aaa-override-root}"
REPORT_DIR="${3:?missing report-dir}"
VISUAL_DIR="${4:?missing visual-dir}"
ACTIVE_ROOT="$PROJECT_DIR/data_wss_min"

cd "$PROJECT_DIR"
mkdir -p pipeline_wss_min/cluster/logs "$REPORT_DIR" "$VISUAL_DIR"

"$PY" -m pipeline_wss_min.v4_cutover qa \
  --ag-root "$STAGING_ROOT" \
  --aaa-root "$AAA_OVERRIDE_ROOT" \
  --aaa-fallback-root "$ACTIVE_ROOT" \
  --report-dir "$REPORT_DIR"

"$PY" -m pipeline_wss_min.v4_cutover snapshot-manifest \
  --legacy-root "$ACTIVE_ROOT" \
  --output "$REPORT_DIR/AG_legacy_v3_snapshot_manifest.json"

"$PY" -m pipeline_wss_min.visualize_v4_cutover \
  --ag-root "$STAGING_ROOT" \
  --aaa-root "$AAA_OVERRIDE_ROOT" \
  --aaa-fallback-root "$ACTIVE_ROOT" \
  --legacy-root "$ACTIVE_ROOT" \
  --out-dir "$VISUAL_DIR"

echo "审核包已生成；promotion_authorized 仍为 false，未改动活动 AG。"
