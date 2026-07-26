#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=bridgesep
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=48:00:00

set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
CONFIG="$PROJECT/training_wss_min/configs/pointnetpp_sa1_scale_bridge_sep_followup_20260722/bridge_rand5000_fixed500_k64_sep.json"
AUDIT="$PROJECT/training_wss_min/preflight/bridge_sep_followup_geometry_audit.json"
PREFLIGHT="$PROJECT/training_wss_min/preflight/bridge_sep_followup_gpu_preflight.json"
cd "$PROJECT"

readarray -t ROW < <("$PY" - "$CONFIG" "$PREFLIGHT" "$AUDIT" <<'PY'
import hashlib, json, sys
from pathlib import Path

config, preflight, audit = map(Path, sys.argv[1:])
p = json.loads(preflight.read_text(encoding="utf-8"))
a = json.loads(audit.read_text(encoding="utf-8"))
if p.get("status") != "passed" or len(p.get("jobs", [])) != 1:
    raise SystemExit("formal GPU preflight did not pass the single follow-up config")
if a.get("status") != "passed":
    raise SystemExit("SA1 geometry audit did not pass")
job = p["jobs"][0]
actual = hashlib.sha256(config.read_bytes()).hexdigest()
if job.get("config") != str(config.resolve()) or job.get("config_sha256") != actual:
    raise SystemExit("preflight config/hash does not match follow-up config")
print(actual)
PY
)

ACTUAL_SHA="${ROW[0]}"
RUN_DIR=$("$PY" - "$CONFIG" <<'PY'
import sys
from training_wss_min.config import ExpConfig
print(ExpConfig.from_json(sys.argv[1]).run_dir)
PY
)
test ! -e "$RUN_DIR"
echo "experiment_id=bridge_rand5000_fixed500_k64_sep config=$CONFIG config_sha256=$ACTUAL_SHA run_dir=$RUN_DIR"
echo "single_change=query_mode same->independent (SAME->SEP); parent=bridge_rand5000_fixed500_k64"
echo "started=$(date --iso-8601=seconds) host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-NA}"

"$PY" -m training_wss_min.train --config "$CONFIG"
for CKPT in best last; do
  "$PY" -m training_wss_min.evaluate \
    --run-dir "$RUN_DIR" --partitions test --allow-test --checkpoint "$CKPT" --no-plots
done
"$PY" -m training_wss_min.tools.compare_best_last --run-dir "$RUN_DIR"
echo "experiment_id=bridge_rand5000_fixed500_k64_sep status=completed finished=$(date --iso-8601=seconds)"
