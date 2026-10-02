"""Step 9 of the 2026-10-01 merge: array-level verification of the joint-cycle cache against the v5.2d sources.

    python 09_verify_joint_cache.py [--workers 6]

Most cache entries were built on 2026-09-29/30 and carried over by refreshing source signatures (v5.2 -> v5.2c -> v5.2d).
A signature says the source files are the expected ones; it does not prove the cached arrays equal what those files give
today. Here every unit of the split is rebuilt from the current sources into a scratch cache (same seed and pool sizes; the
row pools are deterministic) and every array is compared with the live cache. Nothing in the live cache is written.
"""
import argparse, json, os, shutil, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

G = Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0, str(G))
LIVE = G / "training_wss_min/experiments/joint_cycle_v52d_20261001/data_audit_cache"
os.environ.setdefault("GNN_JOINT_VIEW_ROOT", str(G / "data_wss_v5/views_v5_2d_20261001"))
os.environ.setdefault("GNN_JOINT_SPLIT_PATH", str(G / "data_wss_v5/views_v5_2d_20261001/wss_min_view_v1/cv5_v52/fold0.json"))
from training_wss_min import joint_cycle_data as J  # noqa: E402

HERE = Path(__file__).resolve().parent
SCRATCH = HERE / "_verify_joint_scratch"


def one(cid):
    t = time.time()
    try:
        meta = json.loads((LIVE / "cases" / cid / "metadata.json").read_text())
        J.prepare_case(cid, SCRATCH, **meta["settings"])
        new, old = SCRATCH / "cases" / cid, LIVE / "cases" / cid
        names = sorted(p.name for p in old.glob("*.npy")); assert names == sorted(p.name for p in new.glob("*.npy")), "array set differs"
        worst, differ = 0.0, []
        for n in names:
            a, b = np.load(old / n), np.load(new / n)
            if a.shape != b.shape or a.dtype != b.dtype: differ.append([n, "shape/dtype", str(a.shape), str(b.shape)]); continue
            if not np.array_equal(a, b):
                d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))); worst = max(worst, d); differ.append([n, d])
        fresh = json.loads((new / "metadata.json").read_text())
        same_sources = fresh["sources"] == meta["sources"]; same_bc = fresh["boundary_conditions"] == meta["boundary_conditions"]
        shutil.rmtree(new)
        return {"unit": cid, "arrays": len(names), "differ": differ, "max_abs_diff": worst, "sources_equal": same_sources, "bc_metadata_equal": same_bc, "s": round(time.time() - t, 1)}
    except Exception as e:  # noqa: BLE001
        return {"unit": cid, "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args()
    ids = json.loads((LIVE / "manifest.json").read_text())["case_ids"]
    if SCRATCH.exists(): shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True); (SCRATCH / "derived_volume").symlink_to(LIVE / "derived_volume")
    out = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, ids), 1):
            out.append(r)
            if r.get("error") or r.get("differ") or i % 20 == 0: print(i, len(ids), {k: v for k, v in r.items() if k != "s"}, flush=True)
    shutil.rmtree(SCRATCH)
    out.sort(key=lambda r: r["unit"])
    rep = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cache": str(LIVE), "units": len(out), "errors": [r for r in out if r.get("error")],
           "units_with_any_different_array": [r for r in out if r.get("differ")],
           "units_with_metadata_mismatch": [r["unit"] for r in out if not r.get("error") and not (r["sources_equal"] and r["bc_metadata_equal"])],
           "arrays_compared": sum(r.get("arrays", 0) for r in out), "per_unit": out}
    rep["passed"] = not rep["errors"] and not rep["units_with_any_different_array"] and not rep["units_with_metadata_mismatch"]
    (HERE / "joint_cache_verify.json").write_text(json.dumps(rep, indent=1))
    print({k: (v if not isinstance(v, list) else len(v)) for k, v in rep.items() if k != "per_unit"})


if __name__ == "__main__":
    main()
