"""Step 7 of the 2026-10-01 merge: the joint-cycle (full-cycle) cache follows the data to v5.2d.

    python 07_joint_cache.py

The cache directory moves to training_wss_min/experiments/joint_cycle_v52d_20261001/data_audit_cache (rename, nothing
copied). Entries of the units changed by the merge are removed (rebuilt by `joint_cycle_data prepare` afterwards). For every
other unit the cached arrays stay valid; only the recorded source signatures change:
  h5 / bundle / volume / geom   same file under the renamed root -> size and mtime must be unchanged
  flowref                       rebuilt with the v5.2d rule -> allowed only if 05's comparison shows nothing but the cap-area
                                arrays differ for that unit (the cached features use *_murray_cap)
"""
import json, os, shutil, time
from pathlib import Path

G = Path("/public/newhome/cy/Digital_twin/GNN")
OLD = G / "training_wss_min/experiments/joint_cycle_v52c_20260930/data_audit_cache"
NEWE = G / "training_wss_min/experiments/joint_cycle_v52d_20261001"; NEW = NEWE / "data_audit_cache"
C = G / "data_wss_v5/views_v5_2d_20261001"
HERE = Path(__file__).resolve().parent
PAIRS = [("anatomy_pointcloud_v5_2c_20260930", "anatomy_pointcloud_v5_2d_20261001"), ("views_v5_2c_20260930", "views_v5_2d_20261001"),
         ("joint_cycle_v52c_20260930/data_audit_cache", "joint_cycle_v52d_20261001/data_audit_cache")]
U = json.loads((HERE / "units.json").read_text())
CHANGED = set(U["recomputed_same_mesh"]) | set(U["rebuilt_from_stl"]) | set(U["renamed_outlets"])
ALLOWED = {"wall_log_q_branch_capfit", "wall_log_tau0_capfit"}


def sig(p):
    p = Path(p); s = p.stat(); return {"path": str(p.resolve()), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def main():
    log = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "removed": [], "signatures_refreshed": 0, "bundle_links": 0}
    if OLD.exists():
        NEWE.mkdir(parents=True, exist_ok=True); assert not NEW.exists()
        os.rename(OLD, NEW)
        (OLD.parent / "MOVED_TO_v52d.txt").write_text(f"data_audit_cache moved to {NEW} on {log['date']} (v5.2d merge)\n")
    for cid in sorted(CHANGED):
        for sub in ("cases", "derived_volume"):
            d = NEW / sub / cid
            if d.exists() or d.is_symlink():
                shutil.rmtree(d) if d.is_dir() and not d.is_symlink() else d.unlink(); log["removed"].append(f"{sub}/{cid}")
    for p in NEW.rglob("*.json"):
        t = p.read_text(); u = t
        for a, b in PAIRS: u = u.replace(a, b)
        if u != t: p.write_text(u)
    for p in (NEW / "derived_volume").rglob("bundle.npz"):
        if p.is_symlink():
            cid = p.parent.relative_to(NEW / "derived_volume").as_posix(); tgt = C / "wss_min_view_v1" / cid / "bundle.npz"
            assert tgt.is_file(), tgt
            p.unlink(); p.symlink_to(tgt); log["bundle_links"] += 1
    cmp = json.loads((HERE / "flowref_phys1d_compare.json").read_text())["wss_min_flowref_v1"]["units"]
    bad = []
    for meta in list((NEW / "cases").rglob("metadata.json")) + list((NEW / "derived_volume").rglob("source.json")):
        d = json.loads(meta.read_text()); cid = meta.parent.relative_to(NEW / ("cases" if meta.name == "metadata.json" else "derived_volume")).as_posix()
        changed = False
        for name, old in d["sources"].items():
            now = sig(old["path"])
            if now == old: continue
            if name == "flowref":
                r = cmp[cid]["features.npz"]
                if not (set(r["differ"]) <= ALLOWED and not r["only_old"] and not r["only_new"]): bad.append((cid, name, r)); continue
                d.setdefault("signature_refreshed_20261001", []).append({"source": name, "old": old, "new": now, "reason": "flowref rebuilt with the v5.2d cap-area rule; only the capfit arrays changed (cached features use *_murray_cap)"})
            elif (now["size"], now["mtime_ns"]) != (old["size"], old["mtime_ns"]):
                bad.append((cid, name, {"old": old, "now": now})); continue
            d["sources"][name] = now; changed = True
        if changed:
            meta.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n"); log["signatures_refreshed"] += 1
    log["unexpected"] = bad
    (HERE / "joint_cache_move.json").write_text(json.dumps(log, indent=1, default=str))
    print({k: (v if not isinstance(v, list) else len(v)) for k, v in log.items()}, bad[:3])
    if bad: raise SystemExit("unexpected source changes")


if __name__ == "__main__":
    main()
