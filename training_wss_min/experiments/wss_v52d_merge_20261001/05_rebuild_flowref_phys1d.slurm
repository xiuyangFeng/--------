#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --job-name=v52d_flowref
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_v52d_merge_20261001/rebuild_flowref_phys1d_%j.out
# v5.2d: refit the cap-area rule, then flowref and phys1d for all units (staging -> array comparison -> swap).
set -euo pipefail
G=/public/newhome/cy/Digital_twin/GNN; cd "$G"
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
E=$G/training_wss_min/experiments/wss_v52d_merge_20261001
C=$G/data_wss_v5/views_v5_2d_20261001
ST=$C/_staging_20261001
RULE=$C/wss_min_flowref_v1/flow_split_rule_v52d_train136.json
export PYTHONPATH="$G" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WSS_V5_SNAPSHOT_ROOT=$G/data_wss_v5/anatomy_pointcloud_v5_2d_20261001
export WSS_V5_SPLIT_PATH=$C/wss_min_view_v1/split_v52_ind_train170_test91.json
date -Is; hostname; sha1sum wss_v5/views/wall_flowref_v1.py wss_v5/views/wall_phys1d_v1.py $E/05_refit_rule.py | cut -c1-12,42-
echo "== rule refit"; $PY $E/05_refit_rule.py
CASES=$(ls -1 $WSS_V5_SNAPSHOT_ROOT/cases | sed 's#__#/#g' | tr '\n' ' ')
echo "cases: $(echo $CASES | wc -w)"
rm -rf "$ST"; mkdir -p "$ST"
echo "== flowref"
$PY -m wss_v5.views.wall_flowref_v1 --cases $CASES --view-root "$C/wss_min_view_v1" --out "$ST/wss_min_flowref_v1" --split "$WSS_V5_SPLIT_PATH" --capfit-rule "$RULE" --skip-prior --workers 12
echo "== phys1d"
$PY -m wss_v5.views.wall_phys1d_v1 --cases $CASES --view-root "$C/wss_min_view_v1" --flow-root "$ST/wss_min_flowref_v1" --out "$ST/wss_min_phys1d_v1" --workers 12
echo "== compare + swap"
$PY "$E/05_swap_flowref_phys1d.py"
rmdir "$ST/wss_min_flowref_v1" "$ST/wss_min_phys1d_v1" "$ST" 2>/dev/null || true
date -Is
