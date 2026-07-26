#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=v4_sq
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=24:00:00

set -euo pipefail

CONFIG="${1:?config path required}"
EXPERIMENT_ID="${2:?experiment ID required}"
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
SPLIT="$PROJECT/training/splits/split_AG_AAA_wss_min_v4_stratified_seed1234.json"
cd "$PROJECT"

RUN_DIR=$($PY - "$CONFIG" <<'PY'
import sys
from training_wss_min.config import ExpConfig
print(ExpConfig.from_json(sys.argv[1]).run_dir)
PY
)
test ! -e "$RUN_DIR"
echo "experiment_id=$EXPERIMENT_ID"
echo "config=$CONFIG"
echo "run_dir=$RUN_DIR"
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
echo "finished=$(date --iso-8601=seconds)"
