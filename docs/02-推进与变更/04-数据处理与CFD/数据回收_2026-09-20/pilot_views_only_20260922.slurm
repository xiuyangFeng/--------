#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --job-name=v52_pilot_views
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/pilot_views_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/pilot_views_%j.err
# pilot 后半段：视图（wss_min_view / wall_geom_v2 / wall_flowref_v1，写到独立视图根）→ 冻结 X5D_v51 五 seed 只读评估（统计量 = run 自带 train136；不刷新任何统计量、不训练）
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN; cd "$ROOT"
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
HERE="$ROOT/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20"
CASES=$(grep -v '^#' "${PILOT_CASES_FILE:-$HERE/pilot5_cases.txt}" | tr '\n' ' ')
SNAP="$ROOT/data_wss_v5/anatomy_pointcloud_v5_2_pilot_20260922"
VROOT="$ROOT/data_wss_v5/views_v5_2_pilot_20260922"
SPLIT="$ROOT/wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_v5_2_pilot5_train141_test34_s1234.json"
EVAL_SPLIT="$ROOT/wss_pinn/configs/splits/split_pilot5_eval_only_test5_20260922.json"
RULE="$ROOT/data_wss_v5/views_v5_1/wss_min_flowref_v1/flow_split_rule_train136.json"
OUT="$ROOT/training_wss_min/experiments/wss_v52_pilot_20260922"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$ROOT" WSS_V5_SPLIT_PATH="$SPLIT" WSS_V5_SNAPSHOT_ROOT="$SNAP"
date -Is; hostname; echo "cases: $CASES"
echo "== 7 wss_min_view"; date -Is
$PY -m wss_v5.views.wss_min_view --snapshot-root "$SNAP" --out-root "$VROOT/wss_min_view_v1" --cases $CASES --no-cohort-files
echo "== 8 wall_geom_v2"; date -Is
$PY -m wss_v5.views.wall_geom_v2 --cases $CASES --view-root "$VROOT/wss_min_view_v1" --out "$VROOT/wss_min_geom_v2" --workers 5
echo "== 9 wall_flowref_v1"; date -Is
$PY -m wss_v5.views.wall_flowref_v1 --cases $CASES --view-root "$VROOT/wss_min_view_v1" --out "$VROOT/wss_min_flowref_v1" --split "$SPLIT" --capfit-rule "$RULE" --skip-prior --workers 5
echo "== views done"; date -Is
