#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=d2xyz_pf
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=12:00:00

set -euo pipefail
PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="${MATRIX_MANIFEST:-$PROJECT/training_wss_min/preflight/xyz_input_ablation_20260724_prepared.json}"
STATIC_AUDIT="$PROJECT/training_wss_min/preflight/xyz_input_ablation_20260724_static_audit.json"
GPU_PREFLIGHT="$PROJECT/training_wss_min/preflight/xyz_input_ablation_20260724_gpu_preflight.json"
cd "$PROJECT"

mapfile -t CONFIGS < <("$PY" - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
if m.get("status") != "prepared" or len(m.get("configs", [])) != 3:
    raise SystemExit("xyz-only manifest must contain exactly three configs")
for row in m["configs"]:
    print(row["config"])
PY
)

"$PY" -m training_wss_min.tools.audit_xyz_input_ablation --manifest "$MANIFEST" --output "$STATIC_AUDIT"
"$PY" -m training_wss_min.tools.preflight_v4_jobs --configs "${CONFIGS[@]}" --output "$GPU_PREFLIGHT" --probe-batch largest
