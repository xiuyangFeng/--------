#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=170G
#SBATCH --time=08:00:00
#SBATCH --job-name=v52d_stage
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_v52d_merge_20261001/build_stage_%j.out
# 2026-10-01 merge, steps 2-3 for a group of changed units (env TAGN, TOPO, WS, VOLC, DENS, optional CLROOT):
# refresh PREP audits -> V5 build into a STAGING snapshot -> view / volume / geom_v2 / cycle / density packs into STAGING views.
# Nothing in the unified roots is touched (04_integrate.py moves the staged directories in afterwards).
set -euo pipefail
G=/public/newhome/cy/Digital_twin/GNN; cd "$G"
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
E=$G/training_wss_min/experiments/wss_v52d_merge_20261001
ST=$G/data_wss_v5/_merge_20261001
SNAP=$ST/snapshot; VW=$ST/views_$TAGN
SPLIT=$G/wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_v5_2_syn60_train287_test34_s20260926.json
VSPLIT=$G/data_wss_v5/views_v5_2c_20260930/wss_min_view_v1/legacy_splits/views_v5_1/split_V5_train136_test34.json
START_STEP=${START_STEP:-1}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$G"
CASES="$TOPO"
date -Is; hostname; echo "group $TAGN | topology: $TOPO | wall+solver: ${WS:-} | volume: ${VOLC:-} | density: ${DENS:-}"
sha1sum wss_v5/build_case.py wss_v5/conditions.py wss_pinn/v4/new_case_sources.py wss_pinn/configs/outlet_semantics_overrides_20261001.json $E/02_refresh_audits.py | cut -c1-12,42-
if [ $START_STEP -le 1 ]; then echo "== 1 refresh PREP audits"; date -Is
  $PY $E/02_refresh_audits.py --topology $TOPO ${WS:+--wall-solver $WS} ${CLROOT:+--centerline-root $CLROOT} --out $E/refresh_audits_$TAGN.json; fi
export WSS_V5_SPLIT_PATH="$SPLIT"
if [ $START_STEP -le 2 ]; then echo "== 2 V5 build -> $SNAP"; date -Is
  for c in $CASES; do rm -rf "$SNAP/cases/${c//\//__}"; done
  $PY -m wss_v5.build --cases $CASES --root "$SNAP" --workers 6 --run-name merge_20261001_$TAGN; fi
export WSS_V5_SNAPSHOT_ROOT="$SNAP"
if [ $START_STEP -le 3 ]; then echo "== 3 wss_min_view"; date -Is
  $PY -m wss_v5.views.wss_min_view --snapshot-root "$SNAP" --out-root "$VW/wss_min_view_v1" --cases $CASES --no-cohort-files; fi
if [ $START_STEP -le 4 ] && [ -n "${VOLC:-}" ]; then echo "== 4 volume view"; date -Is
  $PY -m wss_v5.views.volume_view --snapshot-root "$SNAP" --out-root "$VW/wss_min_view_v1" --cases $VOLC --split "$VSPLIT" --workers 6; fi
if [ $START_STEP -le 5 ]; then echo "== 5 wall_geom_v2"; date -Is
  $PY -m wss_v5.views.wall_geom_v2 --cases $CASES --view-root "$VW/wss_min_view_v1" --out "$VW/wss_min_geom_v2" --workers 8; fi
if [ $START_STEP -le 6 ]; then echo "== 6 wall_cycle_v1"; date -Is
  $PY -m wss_v5.views.wall_cycle_v1 --cases $CASES --view-root "$VW/wss_min_view_v1" --out "$VW/wss_min_cycle_v1" --workers 8; fi
if [ $START_STEP -le 7 ] && [ -n "${DENS:-}" ]; then echo "== 7 density sidecars"; date -Is
  $PY -m training_wss_min.tools.build_density_sidecars --cases $DENS --view-root "$VW/wss_min_view_v1" --out "$VW/wss_min_density_v1" --levels 70 50 35 25 --seed 20260915 --workers 8; fi
echo "== done"; date -Is
