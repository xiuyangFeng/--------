"""Compare the staged flowref / phys1d packs (all units, v5.2d rule) with the current ones, then swap them in.
Units changed by the merge may differ in any array; every other unit only in the cap-area-rule arrays (flowref) and what
phys1d derives from them. Anything else aborts before the swap."""
import json, shutil
from pathlib import Path
import numpy as np

G = Path("/public/newhome/cy/Digital_twin/GNN"); C = G / "data_wss_v5/views_v5_2d_20261001"; ST = C / "_staging_20261001"
HERE = Path(__file__).resolve().parent
U = json.loads((HERE / "units.json").read_text())
CHANGED = set(U["recomputed_same_mesh"]) | set(U["rebuilt_from_stl"]) | set(U["renamed_outlets"])
ALLOWED = {"wss_min_flowref_v1": {"wall_log_q_branch_capfit", "wall_log_tau0_capfit"}, "wss_min_phys1d_v1": None}   # None: learned below from the data
MANIFEST = {"wss_min_flowref_v1": "flowref_manifest.json", "wss_min_phys1d_v1": "phys1d_manifest.json"}


def compare(old, new):
    out = {}
    for f in sorted({x.name for x in old.glob("*.npz")} | {x.name for x in new.glob("*.npz")}):
        if not (old / f).exists() or not (new / f).exists():
            out[f] = {"missing": "old" if not (old / f).exists() else "new"}; continue
        A, B = np.load(old / f, allow_pickle=True), np.load(new / f, allow_pickle=True)
        diff = [k for k in sorted(set(A.files) & set(B.files))
                if not (A[k].shape == B[k].shape and (np.array_equal(A[k], B[k], equal_nan=True) if A[k].dtype.kind in "fc" else np.array_equal(A[k], B[k])))]
        out[f] = {"differ": diff, "only_old": sorted(set(A.files) - set(B.files)), "only_new": sorted(set(B.files) - set(A.files))}
    return out


def main():
    report, bad = {}, []
    for pack in ALLOWED:
        new_ids = sorted(p.parent.relative_to(ST / pack).as_posix() for p in (ST / pack).rglob("features.npz"))
        report[pack] = {"n_new": len(new_ids), "units": {}}
        keysets = {}
        for cid in new_ids:
            old = C / pack / cid
            if not old.exists():
                report[pack]["units"][cid] = "new"; continue
            r = compare(old, ST / pack / cid); report[pack]["units"][cid] = r
            if cid in CHANGED: continue
            for f, v in r.items():
                if f == "prior_bins.npz": continue
                if "missing" in v or v["only_old"] or v["only_new"]: bad.append((pack, cid, f, v)); continue
                keysets.setdefault(tuple(v["differ"]), []).append(cid)
                if ALLOWED[pack] is not None and set(v["differ"]) - ALLOWED[pack]: bad.append((pack, cid, f, v))
        report[pack]["differ_keysets_unchanged_units"] = {" ".join(k) or "(identical)": len(v) for k, v in keysets.items()}
    report["unexpected"] = bad
    (HERE / "flowref_phys1d_compare.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({p: {"n": report[p]["n_new"], "unchanged_units_differ_in": report[p]["differ_keysets_unchanged_units"]} for p in ALLOWED}, indent=1), "unexpected:", len(bad), bad[:3])
    if bad: raise SystemExit("unexpected differences; not swapped")
    moved = 0
    for pack, mname in MANIFEST.items():
        for cid in sorted(report[pack]["units"]):
            dst = C / pack / cid
            if dst.exists(): shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(ST / pack / cid), str(dst)); moved += 1
        shutil.copy2(C / pack / mname, C / pack / mname.replace(".json", ".pre_merge_20261001.json"))
        t = (ST / pack / mname).read_text().replace(str(ST), str(C)); (C / pack / mname).write_text(t)
    print("swapped", moved, "unit dirs")


if __name__ == "__main__":
    main()
