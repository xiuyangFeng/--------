#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=q2vilo_pf
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
MANIFEST="$PROJECT/training_wss_min/preflight/q2v_ilo_arch_matrix_prepared.json"
REPORT="$PROJECT/training_wss_min/preflight/q2v_ilo_arch_matrix_gpu_preflight.json"
cd "$PROJECT"

mapfile -t CONFIGS < <("$PY" - "$MANIFEST" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1], encoding="utf-8"))["configs"]:
    print(row["config"])
PY
)
test "${#CONFIGS[@]}" -eq 12
echo "stage=gpu_preflight configs=${#CONFIGS[@]} started=$(date --iso-8601=seconds) host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-NA}"
"$PY" -m training_wss_min.tools.preflight_v4_jobs \
  --configs "${CONFIGS[@]}" \
  --output "$REPORT"
echo "stage=gpu_preflight status=passed finished=$(date --iso-8601=seconds) report=$REPORT"
