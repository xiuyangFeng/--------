#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=q1vr
#SBATCH --output=training_wss_min/cluster/logs/%x_%A_%a.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%A_%a.err
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="$PROJECT/training_wss_min/preflight/q1v_radius_test27_prepared.json"
PREFLIGHT="$PROJECT/training_wss_min/preflight/q1v_radius_test27_gpu_preflight.json"
SPLIT="$PROJECT/training/splits/split_AG_AAA_wss_min_v4_stratified_seed1234.json"
TASK_ID="${SLURM_ARRAY_TASK_ID:?array task id required}"
cd "$PROJECT"

readarray -t ROW < <("$PY" - "$MANIFEST" "$PREFLIGHT" "$TASK_ID" <<'PY'
import hashlib, json, sys
from pathlib import Path

prepared = json.load(open(sys.argv[1], encoding="utf-8"))
preflight = json.load(open(sys.argv[2], encoding="utf-8"))
i = int(sys.argv[3])
if prepared.get("status") != "prepared" or len(prepared.get("configs", [])) != 3:
    raise SystemExit("invalid prepared manifest")
if preflight.get("status") != "passed" or len(preflight.get("jobs", [])) != 3:
    raise SystemExit("formal GPU preflight did not pass all three configs")
row = prepared["configs"][i]
checked = {item["config"]: item["config_sha256"] for item in preflight["jobs"]}
if checked.get(row["config"]) != row["sha256"]:
    raise SystemExit("preflight config/hash does not match prepared matrix")
actual = hashlib.sha256(Path(row["config"]).read_bytes()).hexdigest()
if actual != row["sha256"]:
    raise SystemExit("config drifted after preflight")
print(row["experiment_id"])
print(row["config"])
print(row["sha256"])
PY
)

EXPERIMENT_ID="${ROW[0]}"
CONFIG="${ROW[1]}"
EXPECTED_SHA="${ROW[2]}"
ACTUAL_SHA=$(sha256sum "$CONFIG" | awk '{print $1}')
test "$ACTUAL_SHA" = "$EXPECTED_SHA"

RUN_DIR=$($PY - "$CONFIG" <<'PY'
import sys
from training_wss_min.config import ExpConfig
print(ExpConfig.from_json(sys.argv[1]).run_dir)
PY
)
test ! -e "$RUN_DIR"
echo "experiment_id=$EXPERIMENT_ID config=$CONFIG config_sha256=$ACTUAL_SHA run_dir=$RUN_DIR"
echo "test27_policy=historical_anchor_exploratory; not_independent_confirmation"
echo "started=$(date --iso-8601=seconds) host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-NA}"

"$PY" -m training_wss_min.train --config "$CONFIG"
for CKPT in best last; do
  "$PY" -m training_wss_min.evaluate \
    --run-dir "$RUN_DIR" --partitions test --allow-test --checkpoint "$CKPT" --no-plots
done
"$PY" -m training_wss_min.tools.compare_best_last --run-dir "$RUN_DIR"

POSTVIEW="$RUN_DIR/postview/ckpt_best/test"
"$PY" -m training_wss_min.tools.export_wss_postview \
  --run-dir "$RUN_DIR" --partition test --allow-test --checkpoint best \
  --output-dir "$POSTVIEW" --device cuda --resume
"$PY" -m training_wss_min.tools.verify_postview_batch \
  --split "$SPLIT" --partition test --output-dir "$POSTVIEW" \
  --report "$RUN_DIR/postview/verification.json"
echo "experiment_id=$EXPERIMENT_ID status=completed finished=$(date --iso-8601=seconds)"
