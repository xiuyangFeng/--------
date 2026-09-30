#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=v52c_cycle_fill
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_v52c_labelfix_20260930/fill_cycle_%j.out
# cycle pack (TAWSS / OSI labels from the 81 wall frames of each view bundle) for the units of the unified root that never had
# one (59 synthetic + 12 full265 extras), built in a staging dir and moved in; the pack manifest gets their reports appended.
set -euo pipefail
G=/public/newhome/cy/Digital_twin/GNN; cd $G
export PYTHONPATH=$G OMP_NUM_THREADS=2
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
C=$G/data_wss_v5/views_v5_2c_20260930
ST=$C/_staging_cycle_20260930
CASES=$($PY - <<'PYEOF'
import json
from pathlib import Path
C = Path("/public/newhome/cy/Digital_twin/GNN/data_wss_v5/views_v5_2c_20260930")
ids = [r["canonical_id"] for r in json.loads((C / "wss_min_view_v1/view_manifest.json").read_text())["reports"]]
print(" ".join(c for c in ids if not (C / "wss_min_cycle_v1" / c / "cycle.npz").exists()))
PYEOF
)
echo "cases: $(echo $CASES | wc -w)"
rm -rf "$ST"
$PY -m wss_v5.views.wall_cycle_v1 --cases $CASES --view-root "$C/wss_min_view_v1" --out "$ST" --workers 6
$PY - <<'PYEOF'
import json, shutil
from pathlib import Path
C = Path("/public/newhome/cy/Digital_twin/GNN/data_wss_v5/views_v5_2c_20260930"); ST = C / "_staging_cycle_20260930"
new = json.loads((ST / "cycle_manifest.json").read_text())
man = json.loads((C / "wss_min_cycle_v1/cycle_manifest.json").read_text())
have = {r.get("case") or r.get("canonical_id") for r in man.get("cases", [])}
moved = 0
for r in new.get("cases", []):
    cid = r.get("case") or r["canonical_id"]; dst = C / "wss_min_cycle_v1" / cid
    assert not dst.exists(), dst
    dst.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(ST / cid), str(dst)); moved += 1
    if cid not in have: man["cases"].append(r)
man["filled_20260930"] = {"units": [r.get("case") or r["canonical_id"] for r in new.get("cases", [])], "note": "synthetic + full265 extras; built from their v5.2c view bundles"}
(C / "wss_min_cycle_v1/cycle_manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
assert moved == len(new.get("cases", [])) > 0, (moved, len(new.get("cases", [])))
shutil.rmtree(ST)
print("moved", moved)
PYEOF
date -Is
