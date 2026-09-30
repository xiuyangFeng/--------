#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --job-name=v52_pilot_evalcpu
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/pilot_evalcpu_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/pilot_evalcpu_%j.err
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
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$ROOT" WSS_V5_SPLIT_PATH="$SPLIT"
date -Is; hostname; echo "cases: $CASES"
echo "== 10 frozen X5D_v51 x5 seeds, read-only eval on the 5 pilot units"; date -Is
mkdir -p "$OUT/pilot_eval"
for s in 1234 7 2025 11 2026; do
  $PY -m training_wss_min.evaluate --run-dir "$ROOT/training_wss_min/runs/wss_v51_wave1_20260916/X5D_v51_s$s" \
     --partitions test --allow-test --checkpoint best --no-plots --save-predictions \
     --split-path "$EVAL_SPLIT" --data-root "$VROOT/wss_min_view_v1" \
     --point-features-root "$VROOT/wss_min_geom_v2" "$VROOT/wss_min_flowref_v1" \
     --output-dir "$OUT/pilot_eval/X5D_v51_s$s"
done
echo "== done"; date -Is
