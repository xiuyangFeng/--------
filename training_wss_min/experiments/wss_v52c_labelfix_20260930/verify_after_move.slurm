#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --job-name=v52c_after_move
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_v52c_labelfix_20260930/verify_after_move_%j.out
set -euo pipefail
G=/public/newhome/cy/Digital_twin/GNN; cd $G
export PYTHONPATH=$G OMP_NUM_THREADS=4
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
E=$G/training_wss_min/experiments/wss_v52c_labelfix_20260930
echo "== 1 self-contained roots"; $PY $E/verify_after_move.py
echo "== 2 strict loading + feature z-scores"; $PY $E/verify_unified.py 2>&1 | grep -v "^\["
echo "== 3 joint-cycle cache on the unified snapshot"
export GNN_JOINT_VIEW_ROOT=$G/data_wss_v5/views_v5_2c_20260930 GNN_JOINT_SPLIT_PATH=$G/data_wss_v5/views_v5_2c_20260930/wss_min_view_v1/cv5_v52/fold0.json
JC=$G/training_wss_min/experiments/joint_cycle_v52c_20260930/data_audit_cache
$PY -m training_wss_min.joint_cycle_data audit --cache-root $JC 2>&1 | tail -2
$PY -m training_wss_min.joint_cycle_data prepare --cache-root $JC --workers 4 2>&1 | tail -2
$PY -m training_wss_min.joint_cycle_data smoke --cache-root $JC 2>&1 | tail -3
date -Is
