#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --job-name=jc_v52d_cache
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_v52d_merge_20261001/joint_cache_%j.out
# v5.2d: move the joint-cycle cache, refresh signatures, rebuild the changed units, refit statistics on CV5 fold0 train.
set -euo pipefail
G=/public/newhome/cy/Digital_twin/GNN; cd "$G"
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
NEW=$G/training_wss_min/experiments/joint_cycle_v52d_20261001/data_audit_cache
export PYTHONPATH="$G" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GNN_JOINT_VIEW_ROOT=$G/data_wss_v5/views_v5_2d_20261001
export GNN_JOINT_SPLIT_PATH=$G/data_wss_v5/views_v5_2d_20261001/wss_min_view_v1/cv5_v52/fold0.json
date -Is; hostname; sha1sum training_wss_min/joint_cycle_data.py | cut -c1-12
echo "== 1 move + signatures"; $PY training_wss_min/experiments/wss_v52d_merge_20261001/07_joint_cache.py
cp "$NEW/stats.json" "$NEW/stats.v52c.json"
echo "== 2 volume views missing from the view root"; $PY -m training_wss_min.joint_cycle_data views --cache-root "$NEW" --workers 3
echo "== 3 audit"; $PY -m training_wss_min.joint_cycle_data audit --cache-root "$NEW"
echo "== 4 prepare (markers re-verified; changed units rebuilt) + stats"; $PY -m training_wss_min.joint_cycle_data prepare --cache-root "$NEW" --workers 3
echo "== 5 smoke"; $PY -m training_wss_min.joint_cycle_data smoke --cache-root "$NEW"
date -Is
