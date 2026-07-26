#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=d2k64ilo
#SBATCH --output=training_wss_min/cluster/logs/%x_%A_%a.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%A_%a.err
#SBATCH --time=48:00:00

set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="${MATRIX_MANIFEST:-$PROJECT/training_wss_min/preflight/d2_k64_ilo_structure_wave1_prepared.json}"
STATIC_AUDIT="$PROJECT/training_wss_min/preflight/d2_k64_ilo_structure_wave1_static_audit.json"
GEOMETRY_AUDIT="$PROJECT/training_wss_min/preflight/d2_k64_ilo_structure_wave1_geometry_audit.json"
GPU_PREFLIGHT="$PROJECT/training_wss_min/preflight/d2_k64_ilo_structure_wave1_gpu_preflight.json"
TASK_ID="${SLURM_ARRAY_TASK_ID:?array task id required}"
cd "$PROJECT"

readarray -t ROW < <("$PY" - "$MANIFEST" "$STATIC_AUDIT" "$GEOMETRY_AUDIT" "$GPU_PREFLIGHT" "$TASK_ID" <<'PY'
import hashlib, json, sys
from pathlib import Path
m, s, g, p = [json.load(open(path, encoding="utf-8")) for path in sys.argv[1:5]]
if any(x.get("status") != "passed" for x in (s, g, p)):
    raise SystemExit("formal gates did not pass")
row = m["configs"][int(sys.argv[5])]
checked = {x["config"]: x["config_sha256"] for x in p["jobs"]}
if checked.get(row["config"]) != row["sha256"]:
    raise SystemExit("GPU preflight hash mismatch")
if hashlib.sha256(Path(row["config"]).read_bytes()).hexdigest() != row["sha256"]:
    raise SystemExit("config drift")
print(row["experiment_id"]); print(row["config"]); print(row["sha256"])
PY
)

EXPERIMENT_ID="${ROW[0]}"
CONFIG="${ROW[1]}"
EXPECTED_SHA="${ROW[2]}"
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = "$EXPECTED_SHA"
RUN_DIR=$("$PY" - "$CONFIG" <<'PY'
import sys
from training_wss_min.config import ExpConfig
print(ExpConfig.from_json(sys.argv[1]).run_dir)
PY
)
test ! -e "$RUN_DIR"
echo "experiment_id=$EXPERIMENT_ID seed=1234 config=$CONFIG run_dir=$RUN_DIR"
echo "started=$(date --iso-8601=seconds) host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-NA}"
"$PY" -m training_wss_min.train --config "$CONFIG"
for CKPT in best last; do
  "$PY" -m training_wss_min.evaluate \
    --run-dir "$RUN_DIR" --partitions test --allow-test --checkpoint "$CKPT" --no-plots
done
"$PY" -m training_wss_min.tools.compare_best_last --run-dir "$RUN_DIR"
echo "experiment_id=$EXPERIMENT_ID status=completed finished=$(date --iso-8601=seconds)"
