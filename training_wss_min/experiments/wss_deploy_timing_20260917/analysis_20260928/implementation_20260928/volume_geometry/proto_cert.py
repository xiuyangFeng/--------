"""Anchor-ball certificate around atlas samples: coverage, time and mask equality vs the serial VTK test."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0, str(ROOT))
from wss_deploy import centerline as CL, geometry as G, geometry_cache as GC, volume_geometry as VG
from wss_deploy.streamlines import ball_certified_inside
from wss_deploy.pipeline import _clean_mesh, _resampled
from wss_deploy.io_utils import resolve_job_path
from wss_deploy.infer import Release

def run(dst):
    job = json.loads((dst / "job.json").read_text()); a = json.loads((dst / "stage_a.json").read_text())
    rel = Release(ROOT / "outputs/wss_deploy_release/PF6_VF6_peak_3seed_20260920", device="cpu")
    atlas = CL.apply_mapping(CL.load_vessel_geom_atlas(dst / "centerline"), job["mapping"])
    cache = GC.GeometryCache(dst, active=True); cache.save = lambda *x, **k: False
    v, f = _clean_mesh(dst, resolve_job_path(dst, a["input_check"]["clean_stl"]), cache)
    v = np.asarray(v, np.float64); f = np.asarray(f, np.int64)
    seed = G.stable_sampling_seed(a["input_sha256"])
    sm, wall = _resampled(v, f, smooth_mm=1., spacing_mm=.5, seed=seed, cache=cache)
    rec = {"serial": 0.0, "cert": 0.0, "setup": 0.0, "n": 0, "vtk": 0, "diff": 0}
    original = VG.make_inside_test
    def wrapped(vertices, faces):
        serial = original(vertices, faces)
        t = time.perf_counter()
        anchors = atlas.xyz[np.unique(atlas.tree_rows)]
        anchors = anchors[serial(anchors)]
        counted = {"n": 0}
        def counting(q):
            counted["n"] += len(q); return serial(q)
        cert = ball_certified_inside(counting, anchors, vertices, faces)
        rec["setup"] += time.perf_counter() - t
        def contains(q):
            t = time.perf_counter(); m1 = serial(q); rec["serial"] += time.perf_counter() - t
            before = counted["n"]
            t = time.perf_counter(); m2 = cert(q); rec["cert"] += time.perf_counter() - t
            rec["n"] += len(q); rec["vtk"] += counted["n"] - before; rec["diff"] += int(np.sum(m1 != m2))
            return m1
        return contains
    VG.make_inside_test = wrapped
    try:
        VG.build_volume_case(wall, sm, f, atlas, rel.input_features, n_internal=20000, seed=seed)
    finally:
        VG.make_inside_test = original
    out = {"job": dst.name, "points": rec["n"], "vtk_after_cert": rec["vtk"], "certified_frac": round(1 - rec["vtk"] / rec["n"], 3),
           "serial_s": round(rec["serial"], 3), "cert_s": round(rec["cert"], 3), "setup_s": round(rec["setup"], 3), "mask_diff": rec["diff"]}
    print(json.dumps(out), flush=True)
    return out

if __name__ == "__main__":
    res = [run(Path(p)) for p in sys.argv[1:]]
    Path(__file__).with_name("proto_cert.json").write_text(json.dumps(res, indent=1))
