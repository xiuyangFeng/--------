"""Compare the staged flowref / phys1d packs with the current ones (array by array), then put them in place of the links.

Expected: flowref identical except the cap-area-rule arrays (wall_log_q_branch_capfit, wall_log_tau0_capfit) and the removed
label-derived atlas prior (wall_atlas_prior_logwss, wall_prior_is_loo; only the v5.1 units ever had it); phys1d identical
except fields computed from the capfit arrays. Anything else aborts before the swap.
"""
import json
import shutil
from pathlib import Path

import numpy as np

G = Path("/public/newhome/cy/Digital_twin/GNN")
C = G / "data_wss_v5/views_v5_2c_20260930"
ST = C / "_staging_20260930"
HERE = Path(__file__).resolve().parent
ALLOWED = {"wss_min_flowref_v1": {"changed": {"wall_log_q_branch_capfit", "wall_log_tau0_capfit"},
                                   "removed": {"wall_atlas_prior_logwss", "wall_prior_is_loo"}},
           "wss_min_phys1d_v1": {"changed": set(), "removed": set()}}
MANIFEST = {"wss_min_flowref_v1": "flowref_manifest.json", "wss_min_phys1d_v1": "phys1d_manifest.json"}


def ids(root: Path) -> list[str]:
    return sorted(str(p.parent.relative_to(root)) for p in root.glob("*/*/*/features.npz")) + \
        sorted(str(p.parent.relative_to(root)) for p in root.glob("ILO/*/*/*/features.npz") if len(p.parent.relative_to(root).parts) == 3)


def compare(old: Path, new: Path) -> dict:
    out = {}
    for f in sorted({x.name for x in old.glob("*.npz")} | {x.name for x in new.glob("*.npz")}):
        if not (old / f).exists() or not (new / f).exists():
            out[f] = {"missing": "old" if not (old / f).exists() else "new"}; continue
        A, B = np.load(old / f, allow_pickle=True), np.load(new / f, allow_pickle=True)
        diff = []
        for k in sorted(set(A.files) & set(B.files)):
            x, y = A[k], B[k]
            same = x.shape == y.shape and (np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc" else np.array_equal(x, y))
            if not same:
                diff.append(k)
        out[f] = {"differ": diff, "only_old": sorted(set(A.files) - set(B.files)), "only_new": sorted(set(B.files) - set(A.files))}
    return out


def main() -> None:
    report, bad = {}, []
    for pack in ALLOWED:
        new_ids = [p.parent.relative_to(ST / pack).as_posix() for p in (ST / pack).rglob("features.npz")]
        report[pack] = {"n_new": len(new_ids), "units": {}}
        for cid in sorted(new_ids):
            old = C / pack / cid
            if not old.exists():
                report[pack]["units"][cid] = "new (no previous pack dir)"; continue
            r = compare(old.resolve(), ST / pack / cid)
            report[pack]["units"][cid] = r
            for f, v in r.items():
                if f == "prior_bins.npz":     # label-derived per-case bin sums (diagnostic, never a model input)
                    continue
                if "missing" in v:
                    bad.append((pack, cid, f, v))
                    continue
                if (set(v["differ"]) - ALLOWED[pack]["changed"]) or (set(v["only_old"]) - ALLOWED[pack]["removed"]) or v["only_new"]:
                    bad.append((pack, cid, f, v))
    report["unexpected"] = bad
    (HERE / "flowref_phys1d_compare.json").write_text(json.dumps(report, indent=1, default=str))
    summary = {pack: {"n": report[pack]["n_new"],
                      "differ_keys": sorted({k for u in report[pack]["units"].values() if isinstance(u, dict) for v in u.values() if "differ" in v for k in v["differ"]}),
                      "removed_keys": sorted({k for u in report[pack]["units"].values() if isinstance(u, dict) for v in u.values() if "only_old" in v for k in v["only_old"]})}
               for pack in ALLOWED}
    print(json.dumps(summary, indent=1), "unexpected:", len(bad), bad[:5])
    if bad:
        raise SystemExit("unexpected differences; not swapped")
    moved = 0
    for pack, mname in MANIFEST.items():
        for cid in sorted(report[pack]["units"]):
            dst = C / pack / cid
            if dst.is_symlink():
                dst.unlink()
            elif dst.exists():
                shutil.rmtree(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ST / pack / cid), str(dst)); moved += 1
        if (C / pack / mname).exists():
            shutil.copy2(C / pack / mname, C / pack / mname.replace(".json", ".pre_rebuild_20260930.json"))
        shutil.copy2(ST / pack / mname, C / pack / mname)
    print("swapped", moved, "unit dirs")


if __name__ == "__main__":
    main()
