#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=sa_group
#SBATCH --output=training_wss_min/cluster/logs/%x_%A_%a.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%A_%a.err
#SBATCH --time=48:00:00

set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="$PROJECT/training_wss_min/preflight/sa_grouping_single_seed_prepared.json"
SUBSET="${MATRIX_SUBSET:?MATRIX_SUBSET must be core or fps}"
PREFLIGHT="$PROJECT/training_wss_min/preflight/sa_grouping_${SUBSET}_gpu_preflight.json"
AUDIT="$PROJECT/training_wss_min/preflight/sa_grouping_${SUBSET}_geometry_audit.json"
TASK_ID="${SLURM_ARRAY_TASK_ID:?array task id required}"
cd "$PROJECT"

readarray -t ROW < <("$PY" - "$MANIFEST" "$PREFLIGHT" "$AUDIT" "$SUBSET" "$TASK_ID" <<'PY'
import hashlib,json,sys
from pathlib import Path
m=json.load(open(sys.argv[1],encoding='utf-8'))
p=json.load(open(sys.argv[2],encoding='utf-8'))
a=json.load(open(sys.argv[3],encoding='utf-8'))
want=sys.argv[4]=='fps'
rows=[r for r in m['configs'] if bool(r['requires_fps_cache'])==want]
row=rows[int(sys.argv[5])]
if p.get('status')!='passed' or a.get('status')!='passed': raise SystemExit('formal gates did not pass')
checked={r['config']:r['config_sha256'] for r in p['jobs']}
if checked.get(row['config'])!=row['sha256']: raise SystemExit('preflight hash mismatch')
if hashlib.sha256(Path(row['config']).read_bytes()).hexdigest()!=row['sha256']: raise SystemExit('config drift')
print(row['experiment_id']); print(row['config']); print(row['sha256'])
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
echo "subset=$SUBSET experiment_id=$EXPERIMENT_ID seed=1234 config=$CONFIG run_dir=$RUN_DIR"
echo "started=$(date --iso-8601=seconds) host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-NA}"
"$PY" -m training_wss_min.train --config "$CONFIG"
for CKPT in best last; do
  "$PY" -m training_wss_min.evaluate --run-dir "$RUN_DIR" --partitions test --allow-test --checkpoint "$CKPT" --no-plots
done
"$PY" -m training_wss_min.tools.compare_best_last --run-dir "$RUN_DIR"
echo "experiment_id=$EXPERIMENT_ID status=completed finished=$(date --iso-8601=seconds)"
