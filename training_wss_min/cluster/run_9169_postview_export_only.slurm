#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=pnpp9169_pvfix
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
RUN="$PROJECT/training_wss_min/runs/pointnetpp_v4/outputs/ag_aaa_v4_stratified_sa3_e2_global_fps2000"
OUT="$RUN/postview/ckpt_best/test"
SPLIT="$PROJECT/training/splits/split_AG_AAA_wss_min_v4_stratified_seed1234.json"
INTEGRITY="$RUN/postview/export_only_integrity"
mkdir -p "$INTEGRITY"
cd "$PROJECT"

sha256sum \
  "$RUN/ckpt_best.pt" \
  "$RUN/ckpt_last.pt" \
  "$RUN/eval/ckpt_best/metrics.json" \
  "$RUN/eval/ckpt_best/per_case_metrics.csv" \
  "$RUN/eval/ckpt_last/metrics.json" \
  "$RUN/eval/ckpt_last/per_case_metrics.csv" \
  > "$INTEGRITY/before.sha256"

"$PY" -m training_wss_min.tools.export_wss_postview \
  --run-dir "$RUN" \
  --partition test \
  --allow-test \
  --checkpoint best \
  --output-dir "$OUT" \
  --device cuda \
  --resume

sha256sum \
  "$RUN/ckpt_best.pt" \
  "$RUN/ckpt_last.pt" \
  "$RUN/eval/ckpt_best/metrics.json" \
  "$RUN/eval/ckpt_best/per_case_metrics.csv" \
  "$RUN/eval/ckpt_last/metrics.json" \
  "$RUN/eval/ckpt_last/per_case_metrics.csv" \
  > "$INTEGRITY/after.sha256"
diff -u "$INTEGRITY/before.sha256" "$INTEGRITY/after.sha256"

"$PY" -m training_wss_min.tools.verify_postview_batch \
  --split "$SPLIT" \
  --partition test \
  --output-dir "$OUT" \
  --report "$INTEGRITY/verification.json"
