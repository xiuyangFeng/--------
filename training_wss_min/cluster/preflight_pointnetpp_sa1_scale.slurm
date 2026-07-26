#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=sa1scale_pf
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=12:00:00

set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="${MATRIX_MANIFEST:-$PROJECT/training_wss_min/preflight/pointnetpp_sa1_scale_prepared.json}"
TAG="${MATRIX_TAG:-sa1_scale}"
AUDIT="$PROJECT/training_wss_min/preflight/pointnetpp_${TAG}_geometry_audit.json"
PREFLIGHT="$PROJECT/training_wss_min/preflight/pointnetpp_${TAG}_gpu_preflight.json"
cd "$PROJECT"

mapfile -t CONFIGS < <("$PY" - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding='utf-8'))
if m.get('status') != 'prepared':
    raise SystemExit('manifest is not in prepared state')
for row in m['configs']:
    print(row['config'])
PY
)

echo "tag=$TAG configs=${#CONFIGS[@]} started=$(date --iso-8601=seconds)"
"$PY" -m training_wss_min.tools.audit_sa1_scale_matrix \
  --manifest "$MANIFEST" --output "$AUDIT" --max-cases-large 4
"$PY" -m training_wss_min.tools.preflight_v4_jobs \
  --configs "${CONFIGS[@]}" --output "$PREFLIGHT" --probe-batch largest
echo "tag=$TAG status=passed finished=$(date --iso-8601=seconds)"
