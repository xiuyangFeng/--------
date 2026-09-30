#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --job-name=wss_v52p4_prep
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/training_wss_min/cluster/logs/%x_%j.err
# v5.2p4 full265 (2026-09-30): density sidecars for the 4 new training units (v5.2 parameters), density links, code-path
# reproduction of the v5.2 IND statistics, split / WSS stats / feature z-scores on 265, data checks, configs.
# Runs with the frozen v5.2 code copy (the one that trained X5Dcap_asym2); writes only new directories.
set -euo pipefail
G=/public/newhome/cy/Digital_twin/GNN
FZ=/public/newhome/cy/Digital_twin/GNN_v52p_frozen_20260926
NAME=wss_v52p4_full265_20260930
PREP=$G/training_wss_min/experiments/$NAME/prepare_full265.py
NEW=$G/data_wss_v5/views_v5_2p4_full265_20260930
DENS_NEW=$G/data_wss_v5/density_v5_2p4_new4_20260930
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
cd "$FZ"; export PYTHONPATH="$FZ" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
date -Is; hostname
echo "== sha1 (prepare script, frozen code)"; sha1sum "$PREP" "$0" training_wss_min/*.py training_wss_min/tools/build_density_sidecars.py wss_v5/views/wss_min_view.py wss_v5/sources.py
echo "== 1 density sidecars (4 new training units; levels/seed as v5.2)"; date -Is
$PY -m training_wss_min.tools.build_density_sidecars --levels 70 50 35 25 \
   --cases AAA/ruputer/YANG_BAO_KUI ILO/LI_FA_XIANG-1/before ILO/WANG_CAI-0/before ILO/XUE_YOU_TANG-0/after \
   --view-root "$NEW/wss_min_view_v1" --out "$DENS_NEW" --workers 4
for step in link-density verify-codepath split-stats check configs; do
  echo "== $step"; date -Is; $PY -u "$PREP" "$step"
done
echo "== sha1 (outputs)"; sha1sum "$NEW/wss_min_view_v1/split_v52p4_full265_train265_test8.json" "$NEW/wss_min_view_v1/wss_global_stats_full265_train265.json" \
  "$G/training_wss_min/experiments/$NAME/feature_stats/full265_X5Dcap_train_feature_stats.json" "$G/training_wss_min/configs/$NAME"/*.json
date -Is
