#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=q2v_pv27
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
RUN="$PROJECT/training_wss_min/runs/pointnetpp_v4/outputs/ag_aaa_v4_stratified_sa3_random5000_fpscenter_sep"
SPLIT="$PROJECT/training/splits/split_AG_AAA_wss_min_v4_stratified_seed1234.json"
OUT="$RUN/postview/ckpt_best/test"
INTEGRITY="$RUN/postview/export_only_integrity"
cd "$PROJECT"

mkdir -p "$INTEGRITY"
sha256sum "$RUN/config.json" "$RUN/feature_stats.json" "$RUN/wss_global_stats.json" \
  "$RUN/ckpt_best.pt" "$RUN/ckpt_last.pt" \
  "$RUN/eval/ckpt_best/metrics.json" "$RUN/eval/ckpt_last/metrics.json" \
  > "$INTEGRITY/before.sha256"
"$PY" -m training_wss_min.tools.export_wss_postview \
  --run-dir "$RUN" --partition test --allow-test --checkpoint best \
  --output-dir "$OUT" --device cuda --resume
"$PY" -m training_wss_min.tools.verify_postview_batch \
  --split "$SPLIT" --partition test --output-dir "$OUT" \
  --report "$INTEGRITY/verification.json"
sha256sum "$RUN/config.json" "$RUN/feature_stats.json" "$RUN/wss_global_stats.json" \
  "$RUN/ckpt_best.pt" "$RUN/ckpt_last.pt" \
  "$RUN/eval/ckpt_best/metrics.json" "$RUN/eval/ckpt_last/metrics.json" \
  > "$INTEGRITY/after.sha256"
diff -u "$INTEGRITY/before.sha256" "$INTEGRITY/after.sha256"
echo "status=completed mode=export_only expected_cases=27 output=$OUT"
