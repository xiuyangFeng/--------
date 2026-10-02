#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=pf6vf6_v52d_preflight
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.err
#SBATCH --time=03:00:00
# PF6 / VF6 v5.2d retrain (01 block tracking §43; confirmed by the user 2026-10-02): GPU preflight of all 36 arms.
# 1) cross-version anchors: the deployed v5.0 PF6_s1234 / VF6_s1234 (release PF6_VF6_peak_3seed_20260920) repointed to the
#    v5.2d volume root are evaluated on recover8 minus its ILO "after" unit (the old split loader rejects "after"; 7 units) by the code that trained the v5.1 PF6/VF6 fold bases
#    (GNN_voltime_frozen_20260919); the frozen code must reproduce every numeric field (tolerance 5e-4): VF6 by
#    check_pf6vf6_anchor, PF6 as the --anchor-run of the preflight tool. Prediction-dependent point counts (n_*pred*) may
#    differ by <= 2 points (first attempt 16857: one point of 416k crossed the hotspot threshold, n_high_pred 41610 vs 41611;
#    the same code run twice jitters by up to 5.5e-5); every other field keeps 5e-4.
# 2) every arm: paired init + 3 AMP steps at batch8 x 5000 + full-query inference; CLI smoke (2-epoch train + eval
#    best/last) on one pressure arm and one velocity arm.
# Runs inside GNN_pf6vf6_v52d_frozen_20261002 (configs/experiments/runs are symlinks to the main tree).
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN
NAME=pf6vf6_v52d_retrain_20261002
X=$ROOT/training_wss_min/experiments/$NAME
SPLIT=$ROOT/data_wss_v5/views_v5_2d_20261001/wss_min_view_v1/split_v52p4_full265_train265_test8.json
OLD=/public/newhome/cy/Digital_twin/GNN_voltime_frozen_20260919
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
echo "== 1) old-code evaluation of the anchors (stored metrics)"
cd "$OLD"
for a in PF6 VF6; do
  A=$X/anchor/${a}_s1234_v50_on_v52d
  if [ -f "$A/eval/ckpt_best/metrics.json" ]; then echo "keep existing old-code metrics: $A"; continue; fi
  # split = the anchor config's own split (recover8 minus ILO "after", 7 units)
  $PY -u -m training_wss_min.evaluate --run-dir "$A" --partitions test --allow-test --checkpoint best --no-plots
done
cd /public/newhome/cy/Digital_twin/GNN_pf6vf6_v52d_frozen_20261002
$PY -c "import training_wss_min; print('code:', training_wss_min.__file__)"
echo "== sha1 (configs, frozen code)"; sha1sum "$ROOT/training_wss_min/configs/$NAME"/*.json training_wss_min/*.py training_wss_min/tools/preflight_wss_local_wave1.py training_wss_min/tools/run_wss_local_wave1_queue.py training_wss_min/tools/run_next_matrix_queue.py training_wss_min/tools/run_local_train_queue.py
echo "== 2) velocity anchor re-evaluation with the frozen code"
$PY -u -m training_wss_min.tools.check_pf6vf6_anchor --anchor-run "$X/anchor/VF6_s1234_v50_on_v52d" --out "$X/anchor/VF6_anchor_check_${SLURM_JOB_ID}.json" --pred-count-tolerance 2
echo "== 3) GPU preflight (pressure anchor + 36 arms + smokes)"
$PY -u -m training_wss_min.tools.preflight_wss_local_wave1 \
  --config-dir "$ROOT/training_wss_min/configs/$NAME" \
  --experiment-dir "$X" \
  --control-arm PF6_full265_s1234 --smoke-arms PF6_v52cv_f0_s7,VF6_v52cv_f0_s7 \
  --anchor-run "$X/anchor/PF6_s1234_v50_on_v52d" --anchor-pred-count-tolerance 2
date -Is
