#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=q2v_zs36
#SBATCH --output=training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=06:00:00

set -euo pipefail

PROJECT=/public/newhome/cy/Digital_twin/GNN
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
RUN="$PROJECT/training_wss_min/runs/pointnetpp_v4/outputs/ag_aaa_v4_stratified_sa3_random5000_fpscenter_sep"
SPLIT="$PROJECT/training/splits/split_AG_AAA_ILO_q2v_extended_test36_seed1234.json"
OUT="$RUN/eval_external/q2v_extended_test36/ckpt_best"
INTEGRITY="$RUN/eval_external/q2v_extended_test36/integrity"
cd "$PROJECT"

test ! -e "$OUT"
mkdir -p "$INTEGRITY"
sha256sum "$RUN/config.json" "$RUN/feature_stats.json" "$RUN/wss_global_stats.json" \
  "$RUN/ckpt_best.pt" "$RUN/ckpt_last.pt" > "$INTEGRITY/before.sha256"
"$PY" -m training_wss_min.evaluate \
  --run-dir "$RUN" --partitions test --allow-test --checkpoint best --no-plots \
  --split-path "$SPLIT" --output-dir "$OUT"
sha256sum "$RUN/config.json" "$RUN/feature_stats.json" "$RUN/wss_global_stats.json" \
  "$RUN/ckpt_best.pt" "$RUN/ckpt_last.pt" > "$INTEGRITY/after.sha256"
diff -u "$INTEGRITY/before.sha256" "$INTEGRITY/after.sha256"
echo "status=completed mode=read_only_zero_shot split=$SPLIT output=$OUT"
