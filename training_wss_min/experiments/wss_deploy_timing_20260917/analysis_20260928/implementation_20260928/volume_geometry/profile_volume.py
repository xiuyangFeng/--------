import cProfile, json, pstats, sys, time
from pathlib import Path
import numpy as np
ROOT = Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0, str(ROOT))
from wss_deploy import centerline as CL, geometry as G, geometry_cache as GC, volume_geometry as VG
from wss_deploy.pipeline import _clean_mesh, _resampled
from wss_deploy.io_utils import resolve_job_path
from wss_deploy.infer import Release
dst = Path(sys.argv[1])
job = json.loads((dst / "job.json").read_text()); a = json.loads((dst / "stage_a.json").read_text())
rel = Release(ROOT / "outputs/wss_deploy_release" / job["model_release"]["id"], device="cpu")
atlas = CL.apply_mapping(CL.load_vessel_geom_atlas(dst / "centerline"), job["mapping"])
cache = GC.GeometryCache(dst, active=True)
v, f = _clean_mesh(dst, resolve_job_path(dst, a["input_check"]["clean_stl"]), cache)
v = np.asarray(v, np.float64); f = np.asarray(f, np.int64)
seed = G.stable_sampling_seed(a["input_sha256"])
sm, wall = _resampled(v, f, smooth_mm=1., spacing_mm=.5, seed=seed, cache=cache)
VG.build_volume_case(wall, sm, f, atlas, rel.input_features, n_internal=20000, seed=seed)  # warm
pr = cProfile.Profile(); t = time.perf_counter(); pr.enable()
VG.build_volume_case(wall, sm, f, atlas, rel.input_features, n_internal=20000, seed=seed)
pr.disable(); print("total", time.perf_counter() - t)
st = pstats.Stats(pr); st.sort_stats("cumulative").print_stats(35)
