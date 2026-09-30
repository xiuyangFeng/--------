#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --ntasks-per-node=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --job-name=v52_endradius
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/rebuild_endradius_%j.out
#SBATCH --error=/public/newhome/cy/Digital_twin/GNN/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20/rebuild_endradius_%j.err
# 2026-09-23：端点半径修正（开口拟合尺度 min(端点,带状)、拟合失败退回带状半径、端区半径持到开口半径）后，
# 对开口半径有变化的 18 例（8 例 atlas 持值 + 10 例仅尺度变化；含 6 例在库病例）重建 V5 bundle 到 v5.2 快照根（覆盖）。
set -euo pipefail
ROOT=/public/newhome/cy/Digital_twin/GNN; cd "$ROOT"; PY=/public/newhome/cy/.conda/envs/GNN/bin/python
HERE="$ROOT/docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20"
SNAP="$ROOT/data_wss_v5/anatomy_pointcloud_v5_2_20260922"; SPLIT="$ROOT/wss_pinn/configs/splits/split_WSS_PINN_AG_AAA_ILO_v5_2_new91_train227_test34_s1234.json"
CASES=$(grep -v '^#' "$HERE/ingest_endradius_rebuild_cases.txt" | tr '\n' ' ')
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$ROOT"
date -Is; hostname; echo "cases: $CASES"
WSS_V5_SPLIT_PATH="$SPLIT" $PY -m wss_v5.build --cases $CASES --root "$SNAP" --workers 6 --run-name endradius_fix_20260923
$PY - <<'PYEOF'
import json
from pathlib import Path
HERE=Path("docs/02-推进与变更/04-数据处理与CFD/数据回收_2026-09-20"); SNAP=Path("data_wss_v5/anatomy_pointcloud_v5_2_20260922")
for c in [l.strip() for l in open(HERE/"ingest_endradius_rebuild_cases.txt") if l.strip() and not l.startswith("#")]:
    r=json.load(open(SNAP/"cases"/c.replace("/","__")/"report.json")); m=r.get("metrics",{})
    print(f"{c:32s} status {r['status']} gate {r['gates']['pass']} patch {list((m.get('atlas_end_radius_patch') or {}).keys())} caps {[round(v['radius_ratio_cap_over_sqrt_area'],2) for v in m.get('pointcloud',{}).get('cap_offsets',{}).values()]}")
PYEOF
echo "== done"; date -Is
