#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --job-name=syn60_views
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/syn60_views_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/syn60_views_%j.err
# 合成子例视图（2026-09-27）：只对 V5 build 过门的子例，在 v5.2 训练视图根 views_v5_2_full_20260923 里建五种视图
# （wss_min_view --no-cohort-files / wall_geom_v2 / wall_flowref_v1 --skip-prior / density_v1 四档 / phys1d_v1），不写队列文件、不评估。
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN; cd "$ROOT"
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
HERE="$ROOT/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20"
SNAP="$ROOT/data_wss_v5/anatomy_pointcloud_v5_2_20260922"
FULL="$ROOT/data_wss_v5/views_v5_2_full_20260923"
SPLIT="$ROOT/wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_v5_2_syn60_train287_test34_s20260926.json"
RULE="$ROOT/data_wss_v5/views_v5_1/wss_min_flowref_v1/flow_split_rule_train136.json"
CF="$HERE/synth_20260926/synth_children_all60_clean.txt"
PASS="$HERE/ingest_syn60_gate_pass.txt"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$ROOT" WSS_V5_SPLIT_PATH="$SPLIT" WSS_V5_SNAPSHOT_ROOT="$SNAP"
date -Is; hostname
$PY - "$CF" "$SNAP" "$PASS" <<'PYEOF'
import json, sys
from pathlib import Path
cf, snap, out = sys.argv[1:]
want = [l.strip() for l in open(cf) if l.strip() and not l.startswith("#")]
ok, fail, missing = [], [], []
for c in want:
    m = Path(snap) / "cases" / c.replace("/", "__") / "manifest.json"
    if not m.is_file(): missing.append(c); continue
    d = json.loads(m.read_text()); (ok if d["gates"]["pass"] else fail).append(c)
Path(out).write_text("\n".join(ok) + "\n"); Path(out.replace("_gate_pass", "_gate_fail")).write_text("\n".join(fail + [f"# not built: {c}" for c in missing]) + "\n")
print(f"gate pass {len(ok)} / fail {len(fail)} / not built {len(missing)}"); print("fail:", fail); print("not built:", missing)
PYEOF
CASES=$(tr '\n' ' ' < "$PASS"); [ -n "$(echo $CASES)" ] || { echo "no gate-pass children"; exit 1; }
echo "== 7 wss_min_view"; date -Is
$PY -m wss_v5.views.wss_min_view --snapshot-root "$SNAP" --out-root "$FULL/wss_min_view_v1" --cases $CASES --no-cohort-files
echo "== 8 wall_geom_v2"; date -Is
$PY -m wss_v5.views.wall_geom_v2 --cases $CASES --view-root "$FULL/wss_min_view_v1" --out "$FULL/wss_min_geom_v2" --workers 8
echo "== 9 wall_flowref_v1"; date -Is
$PY -m wss_v5.views.wall_flowref_v1 --cases $CASES --view-root "$FULL/wss_min_view_v1" --out "$FULL/wss_min_flowref_v1" --split "$SPLIT" --capfit-rule "$RULE" --skip-prior --workers 8
echo "== 10 density sidecars"; date -Is
$PY -m training_wss_min.tools.build_density_sidecars --levels 70 50 35 25 --cases $CASES --view-root "$FULL/wss_min_view_v1" --out "$FULL/wss_min_density_v1" --workers 16
echo "== 11 phys1d sidecar"; date -Is
$PY -m wss_v5.views.wall_phys1d_v1 --view-root "$FULL/wss_min_view_v1" --flow-root "$FULL/wss_min_flowref_v1" --out "$FULL/wss_min_phys1d_v1" --cases $CASES --workers 12
echo "== done"; date -Is
