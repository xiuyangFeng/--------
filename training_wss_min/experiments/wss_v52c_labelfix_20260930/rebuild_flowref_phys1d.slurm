#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --job-name=v52c_flowref
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_v52c_labelfix_20260930/rebuild_flowref_phys1d_%j.out
# Unified v5.2c root: flowref (cap-area rule refitted on the corrected data, no label-derived atlas prior) and phys1d for all
# 332 units, built into a staging dir, compared array by array with the current files, then swapped in place of the links.
set -euo pipefail
G=/public/newhome/cy/Digital_twin/GNN
cd "$G"
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
E=$G/training_wss_min/experiments/wss_v52c_labelfix_20260930
C=$G/data_wss_v5/views_v5_2c_20260930
ST=$C/_staging_20260930
RULE=$C/wss_min_flowref_v1/flow_split_rule_v52c_train136.json
export PYTHONPATH="$G" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WSS_V5_SNAPSHOT_ROOT=$G/data_wss_v5/anatomy_pointcloud_v5_2c_20260930
export WSS_V5_SPLIT_PATH=$C/wss_min_view_v1/split_v52_ind_train170_test91.json
date -Is; hostname; sha1sum wss_v5/views/wall_flowref_v1.py wss_v5/views/wall_phys1d_v1.py | cut -c1-12,42-
[ -f "$RULE" ] || cp "$E/flow_split_rule_train136_v52c.json" "$RULE"
CASES=$(ls -1 $WSS_V5_SNAPSHOT_ROOT/cases | sed 's#__#/#g' | tr '\n' ' ')
echo "cases: $(echo $CASES | wc -w)"
rm -rf "$ST"; mkdir -p "$ST"
echo "== flowref"
$PY -m wss_v5.views.wall_flowref_v1 --cases $CASES --view-root "$C/wss_min_view_v1" --out "$ST/wss_min_flowref_v1" --split "$WSS_V5_SPLIT_PATH" --capfit-rule "$RULE" --skip-prior --workers 12
echo "== phys1d"
$PY -m wss_v5.views.wall_phys1d_v1 --cases $CASES --view-root "$C/wss_min_view_v1" --flow-root "$ST/wss_min_flowref_v1" --out "$ST/wss_min_phys1d_v1" --workers 12
echo "== compare + swap"
$PY "$E/swap_flowref_phys1d.py"
date -Is
