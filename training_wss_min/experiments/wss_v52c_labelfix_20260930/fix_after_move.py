"""Follow-up to consolidate_v52c.py move (2026-09-30):
1. density sidecars: the v5.2 root held file-level links (features.npz -> old roots); the real files are moved into place
2. joint cache derived_volume/<unit>/bundle.npz: links repointed to the unified view root (the joint_cycle_data layout keeps a link here)
3. joint cache metadata: the flowref signature is refreshed where the flowref file was rebuilt with the refitted cap-area rule;
   allowed only if flowref_phys1d_compare.json shows nothing but the capfit arrays changed (and the label-derived prior removed)
   for that unit — the cached features use *_murray_cap, which is identical."""
import json
import os
import shutil
import time
from pathlib import Path

G = Path("/public/newhome/cy/Digital_twin/GNN")
C = G / "data_wss_v5/views_v5_2c_20260930"
JC = G / "training_wss_min/experiments/joint_cycle_v52c_20260930/data_audit_cache"
HERE = Path(__file__).resolve().parent
ALLOWED_DIFF = {"wall_log_q_branch_capfit", "wall_log_tau0_capfit"}
ALLOWED_REMOVED = {"wall_atlas_prior_logwss", "wall_prior_is_loo"}
log = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "density_files": 0, "bundle_links": 0, "signatures": []}
for dirpath, dirnames, filenames in os.walk(C / "wss_min_density_v1"):
    for n in filenames:
        p = Path(dirpath) / n
        if p.is_symlink():
            real = p.resolve(strict=True)
            p.unlink(); shutil.move(str(real), str(p)); log["density_files"] += 1
for p in (JC / "derived_volume").rglob("bundle.npz"):
    if p.is_symlink():
        cid = p.parent.relative_to(JC / "derived_volume").as_posix()
        target = C / "wss_min_view_v1" / cid / "bundle.npz"
        assert target.is_file(), target
        p.unlink(); p.symlink_to(target); log["bundle_links"] += 1
cmp = json.loads((HERE / "flowref_phys1d_compare.json").read_text())["wss_min_flowref_v1"]["units"]
for meta in (JC / "cases").rglob("metadata.json"):
    d = json.loads(meta.read_text())
    cid = meta.parent.relative_to(JC / "cases").as_posix()
    f = Path(d["sources"]["flowref"]["path"])
    st = f.stat()
    now = {"path": str(f.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    if now == d["sources"]["flowref"]:
        continue
    r = cmp[cid]["features.npz"]
    assert set(r["differ"]) <= ALLOWED_DIFF and set(r["only_old"]) <= ALLOWED_REMOVED and not r["only_new"], (cid, r)
    d.setdefault("signature_refreshed_20260930", []).append({"source": "flowref", "old": d["sources"]["flowref"], "new": now,
        "reason": "flowref rebuilt with the refitted cap-area rule; only the capfit arrays changed (cached features use *_murray_cap)"})
    d["sources"]["flowref"] = now
    meta.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    log["signatures"].append(cid)
log["n_signatures"] = len(log["signatures"])
(HERE / "fix_after_move.json").write_text(json.dumps(log, indent=1))
print({k: v for k, v in log.items() if k != "signatures"})
