"""Real-data Tier-A check: old build_case (full oriented cloud) vs new build_case (caps only).

Read-only on the job directories: points are recomputed exactly as stage B does on a cache miss
(load_stl -> smooth_and_resample with the stable seed) and cross-checked against the cached resample entry.
"""
from __future__ import annotations
import importlib.util, json, statistics, sys, time
from pathlib import Path
import numpy as np

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from wss_deploy import geometry as NEW, centerline as CL, pipeline as P  # noqa: E402
from wss_features import stl as STL  # noqa: E402
from wss_features.cloud import build_oriented_cloud, patch_atlas_end_radius  # noqa: E402

spec = importlib.util.spec_from_file_location("geometry_old", SCRATCH / "geometry_old.py")
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)
assert OLD.build_case is not NEW.build_case

RELEASE = ROOT / "outputs/wss_deploy_release/M1_3head_3seed_20260922"
FEATURES = json.loads((RELEASE / "models/M1_s7/config.json").read_text())["data"]["input_features"]
RULE = RELEASE / "rules/flow_split_rule_train136.json"
JOBS = [ROOT / "outputs/wss_deploy_jobs" / j for j in ("20260927_145732_67eba52ab03c", "20260927_173546_79aaf41ce5f9",
                                                     "20260922_173705_27065942b465", "20260920_174150_e0502f7ea512")]
JOBS += [ROOT / "outputs/wss_deploy_golden/20260920_baseline" / j for j in ("20260918_152113_e2e53937b25a", "20260918_154058_a72784880799")]
if len(sys.argv) > 1:
    JOBS = [j for j in JOBS if any(a in j.name for a in sys.argv[1:])]


def same(a, b, path, diffs):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)) or a.dtype != b.dtype or a.shape != b.shape \
                or a.tobytes() != b.tobytes():
            diffs.append(path)
        return
    if type(a) is not type(b):
        diffs.append(f"{path} type {type(a).__name__} vs {type(b).__name__}"); return
    if isinstance(a, dict):
        if list(a) != list(b):
            diffs.append(f"{path} keys"); return
        for k in a:
            same(a[k], b[k], f"{path}.{k}", diffs)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{path} len"); return
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]", diffs)
    elif isinstance(a, float):
        if not (a == b or (a != a and b != b)) or repr(a) != repr(b):
            diffs.append(f"{path} {a!r} vs {b!r}")
    elif a != b:
        diffs.append(f"{path} {a!r} vs {b!r}")


def median_time(fn, n=5):
    fn()
    samples = []
    for _ in range(n):
        t = time.perf_counter(); fn(); samples.append(time.perf_counter() - t)
    return statistics.median(samples), samples


results = []
for job in JOBS:
    summary = json.loads((job / "summary.json").read_text())
    a = json.loads((job / "stage_a.json").read_text())
    mapping = summary["mapping"]
    atlas_of = lambda: CL.apply_mapping(CL.load_vessel_geom_atlas(job / "centerline"), mapping)
    vertices, faces = STL.load_stl(P.resolve_job_path(job, a["input_check"]["clean_stl"]))
    vertices, faces = np.asarray(vertices, np.float64), np.asarray(faces, np.int64)
    _, pts = NEW.smooth_and_resample(vertices, faces, smooth_mm=1.0, spacing_mm=0.5, seed=NEW.stable_sampling_seed(a["input_sha256"]))
    cached = sorted((job / "geometry_cache").glob("resample-*.npz")) if (job / "geometry_cache").is_dir() else []
    cache_match = None
    if cached:
        with np.load(cached[0], allow_pickle=False) as z:
            cache_match = bool(z["points"].dtype == pts.dtype and z["points"].tobytes() == pts.tobytes())

    # Full stage-B build_case (provider None = point geometry computed inside, as on a cache miss)
    t = time.perf_counter(); case_old, aux_old = OLD.build_case(pts, atlas_of(), FEATURES, case_name="c", capfit_rule_path=RULE); t_old_full = time.perf_counter() - t
    t = time.perf_counter(); case_new, aux_new = NEW.build_case(pts, atlas_of(), FEATURES, case_name="c", capfit_rule_path=RULE); t_new_full = time.perf_counter() - t
    diffs = []
    same(case_old, case_new, "case", diffs); same(aux_old, aux_new, "aux", diffs)
    n_arrays = sum(isinstance(v, np.ndarray) for v in case_new.values()) + sum(isinstance(v, np.ndarray) for v in aux_new["geom"].values())

    # Timing with the point geometry memoised (the v0.14 cache-hit path), so the caps change is visible
    patched = atlas_of(); patch_atlas_end_radius(patched, pts)
    geo = NEW.point_geometry(pts, patched)
    provider = lambda p, at: geo
    t_old, s_old = median_time(lambda: OLD.build_case(pts, atlas_of(), FEATURES, case_name="c", capfit_rule_path=RULE, point_geometry_provider=provider))
    t_new, s_new = median_time(lambda: NEW.build_case(pts, atlas_of(), FEATURES, case_name="c", capfit_rule_path=RULE, point_geometry_provider=provider))
    # Caps step alone (old = full oriented cloud + the separate diag median_spacing; new = cap_records)
    normals = geo[0]
    from wss_features.cloud import median_spacing
    def old_caps():
        caps = build_oriented_cloud(pts, patched, normals_out=normals, calibrate=False)[1]["caps"]
        return caps, median_spacing(pts)
    tc_old, _ = median_time(old_caps)
    tc_new, _ = median_time(lambda: NEW.cap_records(pts, patched, normals))
    rec_old, sp_old = old_caps(); rec_new, sp_new = NEW.cap_records(pts, patched, normals)
    cdiffs = []; same(rec_old, rec_new, "caps", cdiffs)
    row = {"job": job.name, "n_points": int(len(pts)), "n_caps": len(rec_new), "cap_sources": [c["cap_source"] for c in rec_new],
           "resample_cache_match": cache_match, "build_case_equal": not diffs, "diffs": diffs[:20], "n_arrays_compared": n_arrays,
           "caps_equal": not cdiffs and sp_old == sp_new, "spacing_mm": sp_new,
           "end_radius_patch": sorted(aux_new["diag"]["atlas_end_radius_patch"]),
           "build_case_full_s": {"old": t_old_full, "new": t_new_full},
           "build_case_memo_geo_median_s": {"old": t_old, "new": t_new, "old_samples": s_old, "new_samples": s_new},
           "caps_step_median_s": {"old": tc_old, "new": tc_new}}
    results.append(row)
    print(json.dumps(row), flush=True)

(SCRATCH / ("realdata_" + "_".join(sys.argv[1:] or ["all"]) + ".json")).write_text(json.dumps(results, indent=2))
