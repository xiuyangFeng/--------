#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=wss_v52c_post
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=03:00:00
# v5.2c retrain post-queue evaluations (afterany on the queue; read-only on trained runs), frozen code GNN_v52c_frozen_20260930:
#   E3  the 15 X5Dcap_asym2 CV5 fold models (best) on recover8 (test partition of the full265 split)
#   E5  the deployed three-head release M1_3head_3seed_20260922 (v5.1 train136), repointed run dirs, on recover8
#   readout  training_wss_min.tools.report_wss_v52c_retrain best / last -> experiments/<exp>/readout_ckpt_*.{md,json}
# Arms that did not finish are skipped and reported as unfinished; existing output dirs are never overwritten.
set -uo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
NAME=wss_v52c_retrain_20260930
X=$ROOT/training_wss_min/experiments/$NAME
SPLIT=$ROOT/data_wss_v5/views_v5_2c_20260930/wss_min_view_v1/split_v52p4_full265_train265_test8.json
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
cd /public/newhome/cy/Digital_twin/GNN_v52c_frozen_20260930
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname; nvidia-smi --query-gpu=index,name,memory.used --format=csv
sha1sum training_wss_min/evaluate.py training_wss_min/dataset.py training_wss_min/metrics.py training_wss_min/tools/report_wss_v52c_retrain.py "$SPLIT"
ev() {  # <run dir> <output dir>
  if [ ! -f "$1/ckpt_best.pt" ]; then echo "skip (no ckpt_best): $1"; return; fi
  if [ -e "$2" ]; then echo "skip (exists): $2"; return; fi
  $PY -u -m training_wss_min.evaluate --run-dir "$1" --partitions test --allow-test --checkpoint best --no-plots \
      --save-predictions --split-path "$SPLIT" --output-dir "$2" || echo "FAILED: $1"
}
echo "== E3"
for s in 1234 7 2025; do for k in 0 1 2 3 4; do
  a=X5Dcap_asym2_v52cv_f${k}_s${s}
  ev "$ROOT/training_wss_min/runs/wss_v52c_labelfix_20260930/$a" "$X/recover8_cv5/$a"
done; done
echo "== E5"
for s in 1234 7 2025; do ev "$X/e5_deployed_m1/M1_s$s" "$X/e5_deployed_m1/M1_s$s/eval"; done
echo "== readout"
for c in best last; do $PY -u -m training_wss_min.tools.report_wss_v52c_retrain --checkpoint $c > "$X/logs/readout_$c.log" 2>&1 || echo "readout $c FAILED"; done
cat "$X/readout_ckpt_best.md"
date -Is
