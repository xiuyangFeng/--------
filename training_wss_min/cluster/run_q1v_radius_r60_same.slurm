#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=q1vr60
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
CONFIG="$PROJECT/training_wss_min/configs/pointnetpp_q1v_radius_followup_20260719/q1v_radius_r60_same.json"
PREFLIGHT="$PROJECT/training_wss_min/preflight/q1v_radius_r60_same_gpu_preflight.json"
SPLIT="$PROJECT/training/splits/split_AG_AAA_wss_min_v4_stratified_seed1234.json"
cd "$PROJECT"

readarray -t ROW < <("$PY" - "$CONFIG" "$PREFLIGHT" <<'PY'
import hashlib, json, sys
from pathlib import Path

config, preflight = map(Path, sys.argv[1:])
payload = json.loads(preflight.read_text(encoding="utf-8"))
if payload.get("status") != "passed" or len(payload.get("jobs", [])) != 1:
    raise SystemExit("formal GPU preflight did not pass the single follow-up config")
job = payload["jobs"][0]
actual = hashlib.sha256(config.read_bytes()).hexdigest()
if job.get("config") != str(config.resolve()) or job.get("config_sha256") != actual:
    raise SystemExit("preflight config/hash does not match follow-up config")
print(actual)
PY
)

ACTUAL_SHA="${ROW[0]}"
RUN_DIR=$($PY - "$CONFIG" <<'PY'
import sys
from training_wss_min.config import ExpConfig
print(ExpConfig.from_json(sys.argv[1]).run_dir)
PY
)
test ! -e "$RUN_DIR"
echo "experiment_id=q1v_radius_r60_same config=$CONFIG config_sha256=$ACTUAL_SHA run_dir=$RUN_DIR"
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
echo "experiment_id=q1v_radius_r60_same status=completed finished=$(date --iso-8601=seconds)"
