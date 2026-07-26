#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=sa_group_pf
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=12:00:00

set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="$PROJECT/training_wss_min/preflight/sa_grouping_single_seed_prepared.json"
SUBSET="${MATRIX_SUBSET:?MATRIX_SUBSET must be core or fps}"
AUDIT="$PROJECT/training_wss_min/preflight/sa_grouping_${SUBSET}_geometry_audit.json"
PREFLIGHT="$PROJECT/training_wss_min/preflight/sa_grouping_${SUBSET}_gpu_preflight.json"
cd "$PROJECT"

mapfile -t CONFIGS < <("$PY" - "$MANIFEST" "$SUBSET" <<'PY'
import json, sys
m=json.load(open(sys.argv[1],encoding='utf-8'))
want=sys.argv[2]=='fps'
rows=[r for r in m['configs'] if bool(r['requires_fps_cache'])==want]
expected=7 if want else 10
if m.get('status')!='prepared' or len(rows)!=expected:
    raise SystemExit(f'invalid {sys.argv[2]} matrix: {len(rows)}')
for row in rows: print(row['config'])
PY
)

echo "subset=$SUBSET configs=${#CONFIGS[@]} started=$(date --iso-8601=seconds)"
"$PY" -m training_wss_min.tools.audit_sa1_grouping_matrix \
  --manifest "$MANIFEST" --subset "$SUBSET" --output "$AUDIT"
"$PY" -m training_wss_min.tools.preflight_v4_jobs --configs "${CONFIGS[@]}" --output "$PREFLIGHT"
echo "subset=$SUBSET status=passed finished=$(date --iso-8601=seconds)"
