#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=regp10tf
#SBATCH --output=training_wss_min/cluster/logs/%x_%A_%a.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%A_%a.err
#SBATCH --time=48:00:00
set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="${MATRIX_MANIFEST:-$PROJECT/training_wss_min/preflight/regp10_transformer_matrix_20260726_prepared.json}"
STATIC="$PROJECT/training_wss_min/preflight/regp10_transformer_matrix_20260726_static_audit.json"
GEOMETRY="$PROJECT/training_wss_min/preflight/regp10_transformer_matrix_20260726_geometry_audit.json"
GPU="$PROJECT/training_wss_min/preflight/regp10_transformer_matrix_20260726_gpu_smoke.json"
TASK_ID="${SLURM_ARRAY_TASK_ID:?array task id required}"
cd "$PROJECT"
readarray -t ROW < <("$PY" - "$MANIFEST" "$STATIC" "$GEOMETRY" "$GPU" "$TASK_ID" <<'PY'
import hashlib,json,sys
from pathlib import Path
m,s,g,p=[json.load(open(x,encoding='utf-8')) for x in sys.argv[1:5]]
if any(x.get('status')!='passed' for x in (s,g,p)):
    raise SystemExit('formal gate failed')
r=m['configs'][int(sys.argv[5])]
if hashlib.sha256(Path(r['config']).read_bytes()).hexdigest()!=r['sha256']:
    raise SystemExit('config drift')
if r['config'] not in {x['config'] for x in p['jobs']}:
    raise SystemExit('GPU smoke config missing')
print(r['experiment_id']); print(r['config']); print(r['sha256'])
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
echo "experiment_id=$EXPERIMENT_ID config=$CONFIG run_dir=$RUN_DIR started=$(date --iso-8601=seconds)"
"$PY" -m training_wss_min.train --config "$CONFIG"
for CKPT in best last; do
  "$PY" -m training_wss_min.evaluate \
    --run-dir "$RUN_DIR" --partitions test --allow-test \
    --checkpoint "$CKPT" --no-plots
done
"$PY" -m training_wss_min.tools.compare_best_last --run-dir "$RUN_DIR"
