#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=d2xyz
#SBATCH --output=training_wss_min/cluster/logs/%x_%A_%a.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%A_%a.err
#SBATCH --time=48:00:00

set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="${MATRIX_MANIFEST:-$PROJECT/training_wss_min/preflight/xyz_input_ablation_20260724_prepared.json}"
STATIC_AUDIT="$PROJECT/training_wss_min/preflight/xyz_input_ablation_20260724_static_audit.json"
GPU_PREFLIGHT="$PROJECT/training_wss_min/preflight/xyz_input_ablation_20260724_gpu_preflight.json"
TASK_ID="${SLURM_ARRAY_TASK_ID:?array task id required}"
cd "$PROJECT"

readarray -t ROW < <("$PY" - "$MANIFEST" "$STATIC_AUDIT" "$GPU_PREFLIGHT" "$TASK_ID" <<'PY'
import hashlib, json, sys
from pathlib import Path
m, s, p = [json.load(open(path, encoding="utf-8")) for path in sys.argv[1:4]]
if any(x.get("status") != "passed" for x in (s, p)):
    raise SystemExit("formal gates did not pass")
row = m["configs"][int(sys.argv[4])]
checked = {x["config"]: x["config_sha256"] for x in p["jobs"]}
if checked.get(row["config"]) != row["sha256"]:
    raise SystemExit("GPU preflight hash mismatch")
if hashlib.sha256(Path(row["config"]).read_bytes()).hexdigest() != row["sha256"]:
    raise SystemExit("config drift")
print(row["experiment_id"]); print(row["config"])
PY
)

EXPERIMENT_ID="${ROW[0]}"
CONFIG="${ROW[1]}"
RUN_DIR=$("$PY" - "$CONFIG" <<'PY'
import sys
from training_wss_min.config import ExpConfig
print(ExpConfig.from_json(sys.argv[1]).run_dir)
PY
)
test ! -e "$RUN_DIR"
echo "experiment_id=$EXPERIMENT_ID config=$CONFIG run_dir=$RUN_DIR"
"$PY" -m training_wss_min.train --config "$CONFIG"
for CKPT in best last; do
  "$PY" -m training_wss_min.evaluate --run-dir "$RUN_DIR" --partitions test --allow-test --checkpoint "$CKPT" --no-plots
done
"$PY" -m training_wss_min.tools.compare_best_last --run-dir "$RUN_DIR"
