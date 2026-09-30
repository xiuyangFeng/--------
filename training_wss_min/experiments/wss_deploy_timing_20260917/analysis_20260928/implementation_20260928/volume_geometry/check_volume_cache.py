"""Tier-A check of wss_deploy.volume_cache on real volume jobs (sandbox copies)."""
import json, shutil, sys, time
from pathlib import Path
import numpy as np

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
sys.path.insert(0, str(ROOT))
from wss_deploy import centerline as CL, geometry as G, geometry_cache as GC, volume_geometry as VG, volume_cache as VC
from wss_deploy.pipeline import _clean_mesh, _resampled
from wss_deploy.io_utils import resolve_job_path

def same(a, b, path="root"):
    if type(a) is not type(b):
        raise AssertionError(f"{path}: type {type(a)} != {type(b)}")
    if isinstance(a, np.ndarray) or isinstance(a, np.generic):
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            raise AssertionError(f"{path}: array differs {a.dtype}{a.shape} vs {b.dtype}{b.shape}")
        return
    if isinstance(a, dict):
        if list(a.keys()) != list(b.keys()):
            raise AssertionError(f"{path}: keys/order differ")
        for k in a: same(a[k], b[k], f"{path}.{k}")
        return
    if isinstance(a, (list, tuple)):
        if len(a) != len(b): raise AssertionError(f"{path}: len")
        for i, (x, y) in enumerate(zip(a, b)): same(x, y, f"{path}[{i}]")
        return
    if isinstance(a, float) and np.isnan(a) and np.isnan(b): return
    if a != b: raise AssertionError(f"{path}: {a!r} != {b!r}")

def run(job_id, sandbox):
    src = ROOT / "outputs/wss_deploy_jobs" / job_id
    dst = sandbox / job_id
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src, dst)
    shutil.rmtree(dst / "geometry_cache", ignore_errors=True)
    for p in (src / "geometry_cache").glob("*.npz"):
        if p.name.split("-")[0] in {"mesh", "resample"}:
            (dst / "geometry_cache").mkdir(exist_ok=True); shutil.copy2(p, dst / "geometry_cache" / p.name)
    job = json.loads((dst / "job.json").read_text())
    a = json.loads((dst / "stage_a.json").read_text())
    from wss_deploy.infer import Release
    rel = Release(ROOT / "outputs/wss_deploy_release" / job["model_release"]["id"], device="cpu")
    atlas = CL.apply_mapping(CL.load_vessel_geom_atlas(dst / "centerline"), job["mapping"])
    cache = GC.GeometryCache(dst)
    clean = resolve_job_path(dst, a["input_check"]["clean_stl"])
    v, f = _clean_mesh(dst, clean, cache)
    v = np.asarray(v, np.float64); f = np.asarray(f, np.int64)
    seed = G.stable_sampling_seed(a["input_sha256"])
    smoothed, wall = _resampled(v, f, smooth_mm=1., spacing_mm=.5, seed=seed, cache=cache)
    kw = dict(case_name="input-" + a["input_sha256"][:24], n_internal=20000, seed=seed)
    t = time.perf_counter(); ref = VG.build_volume_case(wall, smoothed, f, atlas, rel.input_features, **kw); t_ref = time.perf_counter() - t
    c1 = GC.GeometryCache(dst)
    t = time.perf_counter(); miss = VC.build_volume_case_cached(wall, smoothed, f, atlas, rel.input_features, cache=c1, mapping=job["mapping"], **kw); t_miss = time.perf_counter() - t
    c2 = GC.GeometryCache(dst)
    t = time.perf_counter(); hit = VC.build_volume_case_cached(wall, smoothed, f, atlas, rel.input_features, cache=c2, mapping=job["mapping"], **kw); t_hit = time.perf_counter() - t
    same(ref, miss, "miss"); same(ref, hit, "hit")
    entry = [p for p in (dst / "geometry_cache").glob("volume_case-*.npz")]
    out = {"job": job_id, "n_wall": int(len(wall)), "t_direct_s": round(t_ref, 3), "t_miss_write_s": round(t_miss, 3),
           "t_hit_s": round(t_hit, 3), "miss_stats": c1.summary(), "hit_stats": c2.summary(),
           "entry_mb": round(sum(p.stat().st_size for p in entry) / 2**20, 1), "bit_identical": True}
    print(json.dumps(out, ensure_ascii=False))
    return out

if __name__ == "__main__":
    sandbox = Path(sys.argv[1]); sandbox.mkdir(exist_ok=True)
    res = [run(j, sandbox) for j in sys.argv[2:]]
    (sandbox / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
