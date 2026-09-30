"""Screen (parent, seed) candidates for synthetic morphs in parallel (dry runs only). 2026-09-26.
    python tools/synth_screen_candidates.py --n-per-cohort AG=50 AAA=40 ILO=60 --workers 8 --out <json>
"""
from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))


def one(job):
    import synth_morph_case as M
    try:
        r = M.morph(job["parent"], job["tag"], int(job["seed"]), 2, (0.15, 0.40), (6.0, 14.0), True, ROOT / "data_new")
        r.pop("bumps", None); return r
    except Exception as exc:  # noqa: BLE001
        return {"parent": job["parent"], "seed": job["seed"], "valid": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--n-per-cohort", nargs="*", default=["AG=50", "AAA=40", "ILO=60"]); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--seed", type=int, default=20260926)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    view = json.loads((ROOT / "data_wss_v5/views_v5_2_full_20260923/wss_min_view_v1/view_manifest.json").read_text())
    cases = sorted(r["canonical_id"] for r in view["reports"])
    jobs = []
    for spec in a.n_per_cohort:
        fam, n = spec.split("="); pool = [c for c in cases if c.startswith(fam + "/")]
        picks = rng.choice(len(pool), size=int(n), replace=int(n) > len(pool))
        for i, k in enumerate(picks):
            jobs.append({"parent": pool[int(k)], "tag": f"m{i + 1:02d}", "seed": int(rng.integers(1, 2**31 - 1))})
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, jobs))
    ok = [r for r in res if r.get("valid")]
    a.out.write_text(json.dumps({"jobs": jobs, "results": res, "valid": len(ok), "total": len(res), "seconds": round(time.time() - t0)}, indent=1))
    by = {}
    for r in res: by.setdefault(r["parent"].split("/")[0], [0, 0]); by[r["parent"].split("/")[0]][1] += 1; by[r["parent"].split("/")[0]][0] += int(bool(r.get("valid")))
    print(f"valid {len(ok)}/{len(res)} in {time.time() - t0:.0f}s; by cohort {by}")


if __name__ == "__main__":
    main()
