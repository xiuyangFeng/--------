#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=q2vilo_data
#SBATCH --output=training_wss_min/cluster/logs/%x_%A_%a.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%A_%a.err
#SBATCH --time=48:00:00

set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="$PROJECT/training_wss_min/preflight/q2v_ilo_arch_matrix_prepared.json"
TASK_ID="${SLURM_ARRAY_TASK_ID:?array task id required}"
cd "$PROJECT"

readarray -t ROW < <("$PY" - "$MANIFEST" "$TASK_ID" <<'PY'
import json, sys
rows = [r for r in json.load(open(sys.argv[1], encoding="utf-8"))["configs"] if r["family"] == "data"]
row = rows[int(sys.argv[2])]
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
SPLIT=$($PY - "$CONFIG" <<'PY'
import sys
from training_wss_min.config import ExpConfig
print(ExpConfig.from_json(sys.argv[1]).data.split_path)
PY
)
test ! -e "$RUN_DIR"
echo "experiment_id=$EXPERIMENT_ID config=$CONFIG config_sha256=$ACTUAL_SHA run_dir=$RUN_DIR"
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
