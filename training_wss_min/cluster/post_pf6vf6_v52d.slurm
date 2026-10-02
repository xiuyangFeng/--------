#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=pf6vf6_v52d_post
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=24:00:00
# PF6 / VF6 v5.2d retrain post-queue (afterany on the master queue; read-only on trained runs), frozen code
# GNN_pf6vf6_v52d_frozen_20261002:
#   0  wait until the node04 share has finished (experiments/<exp>/queue_status_node04.json has finished_at; at most
#      WAIT_HOURS, then continue with whatever exists — unfinished arms are reported as unfinished)
#   E3 the 30 CV5 fold models (best) on recover8 (test partition of the full265 split)
#   E5 the deployed v5.0 release PF6_VF6_peak_3seed_20260920 (train138), repointed run dirs, on recover8
#   readout training_wss_min.tools.report_pf6vf6_v52d_retrain best / last -> experiments/<exp>/readout_ckpt_*.{md,json}
# Existing output dirs are never overwritten.
set -uo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
NAME=pf6vf6_v52d_retrain_20261002
X=$ROOT/training_wss_min/experiments/$NAME
SPLIT=$ROOT/data_wss_v5/views_v5_2d_20261001/wss_min_view_v1/split_v52p4_full265_train265_test8.json
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
WAIT_HOURS=${WAIT_HOURS:-16}
cd /public/newhome/cy/Digital_twin/GNN_pf6vf6_v52d_frozen_20261002
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname; nvidia-smi --query-gpu=index,name,memory.used --format=csv
sha1sum training_wss_min/evaluate.py training_wss_min/dataset.py training_wss_min/metrics.py training_wss_min/tools/report_pf6vf6_v52d_retrain.py "$SPLIT"
echo "== 0) waiting for the node04 share"
deadline=$(( $(date +%s) + WAIT_HOURS * 3600 ))
until [ -f "$X/queue_status_node04.json" ] && grep -q '"finished_at"' "$X/queue_status_node04.json"; do
  if [ "$(date +%s)" -ge "$deadline" ]; then echo "node04 share not finished after ${WAIT_HOURS} h; continuing"; break; fi
  sleep 120
done
date -Is
ev() {  # <run dir> <output dir>
  if [ ! -f "$1/ckpt_best.pt" ]; then echo "skip (no ckpt_best): $1"; return; fi
  if [ ! -f "$1/train.log" ] || ! grep -q "训练完成" "$1/train.log"; then echo "skip (training not finished): $1"; return; fi
  if [ -e "$2" ]; then echo "skip (exists): $2"; return; fi
  $PY -u -m training_wss_min.evaluate --run-dir "$1" --partitions test --allow-test --checkpoint best --no-plots \
      --save-predictions --split-path "$SPLIT" --output-dir "$2" || echo "FAILED: $1"
}
echo "== E3"
for t in PF6 VF6; do for s in 1234 7 2025; do for k in 0 1 2 3 4; do
  a=${t}_v52cv_f${k}_s${s}
  ev "$ROOT/training_wss_min/runs/$NAME/$a" "$X/recover8_cv5/$a"
done; done; done
echo "== E5"
for t in PF6 VF6; do for s in 1234 7 2025; do
  d=$X/e5_deployed_v50/${t}_s$s
  if [ -e "$d/eval" ]; then echo "skip (exists): $d/eval"; continue; fi
  $PY -u -m training_wss_min.evaluate --run-dir "$d" --partitions test --allow-test --checkpoint best --no-plots \
      --save-predictions --split-path "$SPLIT" --output-dir "$d/eval" || echo "FAILED: $d"
done; done
echo "== readout"
mkdir -p "$X/logs"
for c in best last; do $PY -u -m training_wss_min.tools.report_pf6vf6_v52d_retrain --checkpoint $c > "$X/logs/readout_$c.log" 2>&1 || echo "readout $c FAILED"; done
cat "$X/readout_ckpt_best.md"
date -Is
