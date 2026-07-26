#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=qadq2v_pf
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="$PROJECT/training_wss_min/preflight/pointnetpp_qad_q2v_test27_prepared.json"
OUTPUT="$PROJECT/training_wss_min/preflight/pointnetpp_qad_q2v_test27_gpu_preflight.json"
cd "$PROJECT"

mapfile -t CONFIGS < <("$PY" - "$MANIFEST" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("status") != "prepared" or len(payload.get("configs", [])) != 5:
    raise SystemExit("prepared manifest is not the exact five-config QAD/Q2V matrix")
for row in payload["configs"]:
    print(row["config"])
PY
)

test "${#CONFIGS[@]}" = 5
echo "matrix=pointnetpp_qad_q2v_test27 configs=${#CONFIGS[@]} started=$(date --iso-8601=seconds)"
"$PY" -m training_wss_min.tools.preflight_v4_jobs --configs "${CONFIGS[@]}" --output "$OUTPUT"
echo "matrix=pointnetpp_qad_q2v_test27 status=passed finished=$(date --iso-8601=seconds)"
