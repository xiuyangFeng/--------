#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --job-name=v52_views_eval
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/wave2_views_eval_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/wave2_views_eval_%j.err
# 入库链第 2 波后半段（2026-09-22）：只对 V5 build 过门的单元建视图（wss_min_view --no-cohort-files / wall_geom_v2 / wall_flowref_v1，独立视图根）
# → 冻结 X5D_v51 五 seed 只读评估（统计量 = run 自带 train136；不刷新统计量、不训练）。与 pilot 的 pilot_views_only / pilot_evalcpu 相同口径。
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN; cd "$ROOT"
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
HERE="$ROOT/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20"
SNAP="$ROOT/data_wss_v5/anatomy_pointcloud_v5_2_20260922"
VROOT="$ROOT/data_wss_v5/views_v5_2_20260922"
SPLIT="${SPLIT_PATH:-$ROOT/wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_v5_2_new91_train227_test34_s1234.json}"
TAG=${WAVE_TAG:-wave2}
EVAL_SPLIT="$ROOT/wss_pinn/configs/splits/split_new95_${TAG}_eval_only_20260922.json"
RULE="$ROOT/data_wss_v5/views_v5_1/wss_min_flowref_v1/flow_split_rule_train136.json"
OUT="$ROOT/training_wss_min/experiments/wss_v52_new95_20260922"
CF="${CASES_FILE:-$HERE/ingest_wave2_cases.txt}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$ROOT" WSS_V5_SPLIT_PATH="$SPLIT" WSS_V5_SNAPSHOT_ROOT="$SNAP"
date -Is; hostname
# 过门单元清单（只取本波清单里且快照 manifest gates.pass 的）
PASS="$HERE/ingest_${TAG}_gate_pass.txt"
$PY - "$CF" "$SNAP" "$PASS" "$EVAL_SPLIT" <<'PYEOF'
import json, sys, time
from pathlib import Path
cf, snap, out, split = sys.argv[1:]
want = [l.strip() for l in open(cf) if l.strip() and not l.startswith("#")]
ok, fail, missing = [], [], []
for c in want:
    m = Path(snap) / "cases" / c.replace("/", "__") / "manifest.json"
    if not m.is_file(): missing.append(c); continue
    d = json.loads(m.read_text()); (ok if d["gates"]["pass"] else fail).append(c)
Path(out).write_text("\n".join(ok) + "\n")
Path(split).write_text(json.dumps(dict(schema_version=1, purpose=f"2026-09-22 新单元独立评估（{Path(cf).name}）：V5 过门单元作 test，冻结 X5D_v51 五 seed 只读评估；train/val 为空（统计量取 run 自带 train136）",
    train_cases=[], val_cases=[], test_cases=ok, gate_failed=fail, not_built=missing, created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")), ensure_ascii=False, indent=1))
print(f"gate pass {len(ok)} / fail {len(fail)} / not built {len(missing)}"); print("fail:", fail); print("not built:", missing)
PYEOF
CASES=$(tr '\n' ' ' < "$PASS")
echo "== 7 wss_min_view"; date -Is
$PY -m wss_v5.views.wss_min_view --snapshot-root "$SNAP" --out-root "$VROOT/wss_min_view_v1" --cases $CASES --no-cohort-files
echo "== 8 wall_geom_v2"; date -Is
$PY -m wss_v5.views.wall_geom_v2 --cases $CASES --view-root "$VROOT/wss_min_view_v1" --out "$VROOT/wss_min_geom_v2" --workers 8
echo "== 9 wall_flowref_v1"; date -Is
$PY -m wss_v5.views.wall_flowref_v1 --cases $CASES --view-root "$VROOT/wss_min_view_v1" --out "$VROOT/wss_min_flowref_v1" --split "$SPLIT" --capfit-rule "$RULE" --skip-prior --workers 8
echo "== 10 frozen X5D_v51 x5 seeds, read-only eval"; date -Is
export OMP_NUM_THREADS=16
mkdir -p "$OUT/eval_$TAG"
for s in 1234 7 2025 11 2026; do
  $PY -m training_wss_min.evaluate --run-dir "$ROOT/training_wss_min/runs/wss_v51_wave1_20260916/X5D_v51_s$s" \
     --partitions test --allow-test --checkpoint best --no-plots --save-predictions \
     --split-path "$EVAL_SPLIT" --data-root "$VROOT/wss_min_view_v1" \
     --point-features-root "$VROOT/wss_min_geom_v2" "$VROOT/wss_min_flowref_v1" \
     --output-dir "$OUT/eval_$TAG/X5D_v51_s$s"
done
echo "== done"; date -Is
