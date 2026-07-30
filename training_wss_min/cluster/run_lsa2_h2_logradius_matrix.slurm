#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=lsa2h2log
#SBATCH --output=training_wss_min/cluster/logs/%x_%A_%a.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%A_%a.err
#SBATCH --time=48:00:00
set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="${LSA2_H2_LOG_MANIFEST:?LSA2_H2_LOG_MANIFEST is required}"
STATIC="${LSA2_H2_LOG_STATIC_AUDIT:?LSA2_H2_LOG_STATIC_AUDIT is required}"
GPU_AUDIT="${LSA2_H2_LOG_GPU_AUDIT:?LSA2_H2_LOG_GPU_AUDIT is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?array task id required}"
cd "$PROJECT"
readarray -t ROW < <("$PY" - "$MANIFEST" "$STATIC" "$GPU_AUDIT" "$TASK_ID" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, static_path, gpu_path = map(Path, sys.argv[1:4])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
static = json.loads(static_path.read_text(encoding="utf-8"))
gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if static.get("status") != "passed" or static.get("manifest_sha256") != sha(manifest_path):
    raise SystemExit("static audit failed or stale")
if gpu.get("status") != "passed" or gpu.get("manifest_sha256") != sha(manifest_path):
    raise SystemExit("GPU audit failed or stale")
row = manifest["configs"][int(sys.argv[4])]
config_path = Path(row["config"])
feature_stats_path = Path(row["feature_stats"])
if sha(config_path) != row["config_sha256"]:
    raise SystemExit("config drift")
if sha(feature_stats_path) != row["feature_stats_sha256"]:
    raise SystemExit("feature-stat drift")
if row["config"] not in {job["config"] for job in gpu["jobs"]}:
    raise SystemExit("config missing from GPU audit")
print(row["experiment_id"])
print(row["config"])
print(row["run_dir"])
PY
)
EXPERIMENT_ID="${ROW[0]}"
CONFIG="${ROW[1]}"
RUN_DIR="${ROW[2]}"
test ! -e "$RUN_DIR"
echo "experiment_id=$EXPERIMENT_ID config=$CONFIG run_dir=$RUN_DIR started=$(date --iso-8601=seconds)"
"$PY" -m training_wss_min.train --config "$CONFIG"
for CKPT in best last; do
  "$PY" -m training_wss_min.evaluate \
    --run-dir "$RUN_DIR" --partitions test --allow-test \
    --checkpoint "$CKPT" --no-plots
done
"$PY" -m training_wss_min.tools.compare_best_last --run-dir "$RUN_DIR"
