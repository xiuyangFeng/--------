"""After consolidate_v52c.py move: the unified roots are self-contained (no links), every view points at a case.h5 inside the
unified snapshot, and every unit has the same set of packs."""
import json
import os
from pathlib import Path

G = Path("/public/newhome/cy/Digital_twin/GNN")
C = G / "data_wss_v5/views_v5_2c_20260930"
S = G / "data_wss_v5/anatomy_pointcloud_v5_2c_20260930"
JC = G / "training_wss_min/experiments/joint_cycle_v52c_20260930/data_audit_cache"
out = {"links": {}, "h5": {"ok": 0, "bad": []}, "packs": {}}
for r in (C, S, JC):
    n = 0   # links that leave the unified roots (the joint cache keeps derived_volume/<unit>/bundle.npz links into the unified view root)
    for dirpath, dirnames, filenames in os.walk(r):
        for x in dirnames + filenames:
            q = os.path.join(dirpath, x)
            if os.path.islink(q) and not any(os.path.realpath(q).startswith(str(u) + "/") for u in (C, S, JC)):
                n += 1
    out["links"][str(r)] = n
vm = json.loads((C / "wss_min_view_v1/view_manifest.json").read_text())
units = []
for rep in vm["reports"]:
    cid = rep["canonical_id"]; units.append(cid)
    h5 = Path(rep["source"]["case_h5"])
    vr = json.loads((C / "wss_min_view_v1" / cid / "view_report.json").read_text())
    if h5.is_file() and str(h5).startswith(str(S) + "/") and vr["source"]["case_h5"] == str(h5):
        out["h5"]["ok"] += 1
    else:
        out["h5"]["bad"].append(cid)
for pack in ["wss_min_view_v1", "wss_min_geom_v2", "wss_min_flowref_v1", "wss_min_phys1d_v1", "wss_min_cycle_v1", "wss_min_density_v1/L70"]:
    out["packs"][pack] = sum((C / pack / u).is_dir() for u in units)
out["n_units"] = len(units)
out["passed"] = all(v == 0 for v in out["links"].values()) and not out["h5"]["bad"]
(Path(__file__).resolve().parent / "verify_after_move.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
