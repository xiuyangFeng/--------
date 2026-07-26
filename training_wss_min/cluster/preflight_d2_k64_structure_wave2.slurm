#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=d2k64w2_pf
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=12:00:00

set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="${MATRIX_MANIFEST:-$PROJECT/training_wss_min/preflight/d2_k64_ilo_structure_wave2_prepared.json}"
STATIC_AUDIT="$PROJECT/training_wss_min/preflight/d2_k64_ilo_structure_wave2_static_audit.json"
GPU_PREFLIGHT="$PROJECT/training_wss_min/preflight/d2_k64_ilo_structure_wave2_gpu_preflight.json"
cd "$PROJECT"

mapfile -t CONFIGS < <("$PY" - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
if m.get("status") != "prepared" or len(m.get("configs", [])) != 4:
    raise SystemExit("wave2 manifest must be prepared with exactly four configs")
for row in m["configs"]:
    print(row["config"])
PY
)

echo "configs=${#CONFIGS[@]} started=$(date --iso-8601=seconds)"
"$PY" -m training_wss_min.tools.audit_d2_k64_structure_wave2 \
  --manifest "$MANIFEST" --output "$STATIC_AUDIT"
"$PY" -m training_wss_min.tools.preflight_v4_jobs \
  --configs "${CONFIGS[@]}" --output "$GPU_PREFLIGHT" --probe-batch largest
echo "status=passed finished=$(date --iso-8601=seconds)"
