#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --exclude=node05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --job-name=jc_v52c_cache
#SBATCH --output=/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/joint_cycle_v52c_20260930/build_cache_%j.out
# Joint-cycle (full-cycle velocity / pressure / WSS) cache on the label-fixed v5.2c data (2026-09-30).
# 252 unchanged units: case / derived-volume directories linked from the original cache (their sources resolve to the same
# files, which prepare_case re-verifies); the 9 label-fixed units: rebuilt from the corrected snapshot; statistics refit on
# the v5.2c CV5 fold-0 training partition (same case list as the original fold0).
set -euo pipefail
G=/public/newhome/cy/Digital_twin/GNN
cd "$G"
PY=/public/newhome/cy/.conda/envs/GNN/bin/python
E=$G/training_wss_min/experiments/joint_cycle_v52c_20260930
OLD=$G/training_wss_min/experiments/joint_cycle_v52_20260929/data_audit_cache
NEW=$E/data_audit_cache
export PYTHONPATH="$G" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GNN_JOINT_VIEW_ROOT=$G/data_wss_v5/views_v5_2c_20260930
export GNN_JOINT_SPLIT_PATH=$G/data_wss_v5/views_v5_2c_20260930/wss_min_view_v1/cv5_v52/fold0.json
date -Is; hostname; sha1sum training_wss_min/joint_cycle_data.py | cut -c1-12
echo "== 1 link unchanged units"
$PY - <<'PYEOF'
import json, os
from pathlib import Path
G = Path("/public/newhome/cy/Digital_twin/GNN")
OLD = G / "training_wss_min/experiments/joint_cycle_v52_20260929/data_audit_cache"
NEW = G / "training_wss_min/experiments/joint_cycle_v52c_20260930/data_audit_cache"
plan = json.loads((G / "data_wss_v5/views_v5_2c_20260930/assembly_plan.json").read_text())["plan"]
old_split = json.loads((G / "data_wss_v5/views_v5_2_full_20260923/wss_min_view_v1/cv5_v52/fold0.json").read_text())
new_split = json.loads(Path(os.environ["GNN_JOINT_SPLIT_PATH"]).read_text())
assert old_split["train_cases"] == new_split["train_cases"] and old_split["test_cases"] == new_split["test_cases"]
n = {"cases": 0, "derived_volume": 0, "rebuild": []}
for cid, tag in plan.items():
    if tag == "v5.2c-labelfix":
        n["rebuild"].append(cid); continue
    for sub in ("cases", "derived_volume"):
        src = OLD / sub / cid
        if src.is_dir():
            dst = NEW / sub / cid
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.is_symlink():
                dst.symlink_to(src.resolve())
            assert dst.resolve() == src.resolve()
            n[sub] += 1
print(json.dumps(n))
(NEW / "linked_from_original.json").write_text(json.dumps({"original": str(OLD), **n}, indent=1))
PYEOF
echo "== 2 volume views missing from the view root (new v5.2 units, incl. the 3 label-fixed ones)"
$PY -m training_wss_min.joint_cycle_data views --cache-root "$NEW" --workers 3
echo "== 3 audit"
$PY -m training_wss_min.joint_cycle_data audit --cache-root "$NEW"
echo "== 4 prepare (existing markers re-verified; the 9 rebuilt) + stats"
$PY -m training_wss_min.joint_cycle_data prepare --cache-root "$NEW" --workers 3
echo "== 5 smoke"
$PY -m training_wss_min.joint_cycle_data smoke --cache-root "$NEW"
echo "== 6 compare with the original cache"
$PY - <<'PYEOF'
import json
from pathlib import Path
import numpy as np
G = Path("/public/newhome/cy/Digital_twin/GNN")
OLD = G / "training_wss_min/experiments/joint_cycle_v52_20260929/data_audit_cache"
NEW = G / "training_wss_min/experiments/joint_cycle_v52c_20260930/data_audit_cache"
link = json.loads((NEW / "linked_from_original.json").read_text())
so, sn = json.loads((OLD / "stats.json").read_text()), json.loads((NEW / "stats.json").read_text())
out = {"units": {}, "stats": {}}
for cid in link["rebuild"]:
    r = {}
    for name in ("bc", "eval_velocity", "eval_pressure", "eval_wss", "support_x", "eval_wall_rows", "eval_volume_rows"):
        a, b = np.load(OLD / "cases" / cid / f"{name}.npy"), np.load(NEW / "cases" / cid / f"{name}.npy")
        if a.shape != b.shape: r[name] = {"shape": [a.shape, b.shape]}; continue
        if name.endswith("rows") or name == "support_x":
            r[name] = "identical" if np.array_equal(a, b) else f"differ (max {float(np.abs(a.astype(float) - b.astype(float)).max()):.3g})"
        elif name == "bc":
            r[name] = {"max_abs_log_diff": float(np.abs(a - b).max())}
        else:
            r[name] = {"rel_mean_abs": float(np.abs(a - b).mean() / (np.abs(a).mean() + 1e-30))}
    out["units"][cid] = r
    print(cid, r, flush=True)
for task in ("velocity", "pressure", "wss"):
    mo, mn = np.array(so["targets"][task]["mean"]), np.array(sn["targets"][task]["mean"])
    do, dn = np.array(so["targets"][task]["std"]), np.array(sn["targets"][task]["std"])
    out["stats"][task] = {"mean_max_abs_diff_rel_std": float(np.abs(mo - mn).max() / do.mean()), "std_max_rel_diff": float(np.abs(do - dn).max() / do.mean())}
for dom in ("wall", "volume"):
    out["stats"][f"features_{dom}"] = float(np.abs(np.array(so["features"][dom]["mean"]) - np.array(sn["features"][dom]["mean"])).max())
out["stats"]["bc_mean_max_abs_diff"] = float(np.abs(np.array(so["bc"]["mean"]) - np.array(sn["bc"]["mean"])).max())
print(json.dumps(out["stats"], indent=1))
(NEW.parent / "compare_with_original.json").write_text(json.dumps(out, indent=1, default=str))
PYEOF
echo "== done"; date -Is
