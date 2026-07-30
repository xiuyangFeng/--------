#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=rcroracle
#SBATCH --output=training_wss_min/cluster/logs/%x_%A_%a.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%A_%a.err
#SBATCH --time=48:00:00
set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="${RCR_ORACLE_MANIFEST:?RCR_ORACLE_MANIFEST is required}"
STATIC="${RCR_ORACLE_STATIC_AUDIT:?RCR_ORACLE_STATIC_AUDIT is required}"
GPU_AUDIT="${RCR_ORACLE_GPU_AUDIT:?RCR_ORACLE_GPU_AUDIT is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?array task id required}"
cd "$PROJECT"
readarray -t ROW < <("$PY" - "$MANIFEST" "$STATIC" "$GPU_AUDIT" "$TASK_ID" <<'PY'
import hashlib,json,sys
from pathlib import Path
manifest_path,static_path,gpu_path=map(Path,sys.argv[1:4])
m=json.load(open(manifest_path,encoding="utf-8"))
s=json.load(open(static_path,encoding="utf-8"))
g=json.load(open(gpu_path,encoding="utf-8"))
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if s.get("status")!="passed" or s.get("manifest_sha256")!=sha(manifest_path):
    raise SystemExit("static audit failed or stale")
if g.get("status")!="passed" or g.get("manifest_sha256")!=sha(manifest_path):
    raise SystemExit("GPU audit failed or stale")
row=m["configs"][int(sys.argv[4])]
if sha(Path(row["config"]))!=row["config_sha256"]:
    raise SystemExit("config drift")
if row["config"] not in {x["config"] for x in g["jobs"]}:
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
