"""Tier A: build_volume_case with the certified contains vs the frozen v0.15 volume_geometry, on real geometries."""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0, str(ROOT))
from wss_deploy import centerline as CL, geometry as G, geometry_cache as GC, volume_geometry as VG
from wss_deploy.pipeline import _clean_mesh, _resampled
from wss_deploy.io_utils import resolve_job_path
from wss_deploy.infer import Release
sys.path.insert(0, str(Path(__file__).parent))
from check_volume_cache import same
spec = importlib.util.spec_from_file_location("wss_deploy.volume_geometry_v015", Path(__file__).with_name("volume_geometry_v015.py"))
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)

def run(dst):
    job = json.loads((dst / "job.json").read_text()); a = json.loads((dst / "stage_a.json").read_text())
    rel = Release(ROOT / "outputs/wss_deploy_release/PF6_VF6_peak_3seed_20260920", device="cpu")
    atlas = CL.apply_mapping(CL.load_vessel_geom_atlas(dst / "centerline"), job["mapping"])
    cache = GC.GeometryCache(dst, active=True); cache.save = lambda *x, **k: False
    v, f = _clean_mesh(dst, resolve_job_path(dst, a["input_check"]["clean_stl"]), cache)
    v = np.asarray(v, np.float64); f = np.asarray(f, np.int64)
    seed = G.stable_sampling_seed(a["input_sha256"])
    sm, wall = _resampled(v, f, smooth_mm=1., spacing_mm=.5, seed=seed, cache=cache)
    kw = dict(n_internal=20000, seed=seed)
    times = {"old": [], "new": []}
    for r in range(2):
        for name, mod in (("old", OLD), ("new", VG)) if r == 0 else (("new", VG), ("old", OLD)):
            t = time.perf_counter(); res = mod.build_volume_case(wall, sm, f, atlas, rel.input_features, **kw); times[name].append(time.perf_counter() - t)
            if name == "old": ref = res
            else: new = res
        same(ref, new, dst.name)
    out = {"job": dst.name, "old_s": [round(x, 2) for x in times["old"]], "new_s": [round(x, 2) for x in times["new"]], "bit_identical": True}
    print(json.dumps(out), flush=True)
    return out

if __name__ == "__main__":
    res = [run(Path(p)) for p in sys.argv[1:]]
    Path(__file__).with_name("check_sampling_equal.json").write_text(json.dumps(res, indent=1))
