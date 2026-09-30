"""Record every contains() query of build_volume_case on real jobs; compare serial vs batch VTK masks."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path("/public/newhome/cy/Digital_twin/GNN"); sys.path.insert(0, str(ROOT))
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from wss_deploy import centerline as CL, geometry as G, geometry_cache as GC, volume_geometry as VG
from wss_deploy.pipeline import _clean_mesh, _resampled
from wss_deploy.io_utils import resolve_job_path
from wss_deploy.infer import Release

def batch_contains(mesh):
    def contains(q):
        poly = vtk.vtkPolyData(); pts = vtk.vtkPoints(); pts.SetData(numpy_to_vtk(np.ascontiguousarray(q, np.float64), deep=1)); poly.SetPoints(pts)
        f = vtk.vtkSelectEnclosedPoints(); f.SetInputData(poly); f.SetSurfaceData(mesh); f.SetTolerance(1e-9); f.Update()
        return vtk_to_numpy(f.GetOutput().GetPointData().GetArray("SelectedPoints")).astype(bool)
    return contains

def run(dst):
    job = json.loads((dst / "job.json").read_text()); a = json.loads((dst / "stage_a.json").read_text())
    rel = Release(ROOT / "outputs/wss_deploy_release/PF6_VF6_peak_3seed_20260920", device="cpu")
    atlas = CL.apply_mapping(CL.load_vessel_geom_atlas(dst / "centerline"), job["mapping"])
    cache = GC.GeometryCache(dst, active=True); cache.save = lambda *a, **k: False   # read-only
    v, f = _clean_mesh(dst, resolve_job_path(dst, a["input_check"]["clean_stl"]), cache)
    v = np.asarray(v, np.float64); f = np.asarray(f, np.int64)
    seed = G.stable_sampling_seed(a["input_sha256"])
    sm, wall = _resampled(v, f, smooth_mm=1., spacing_mm=.5, seed=seed, cache=cache)
    calls = []
    original = VG.make_inside_test
    def recording(vertices, faces):
        serial = original(vertices, faces)
        mesh = VG._surface(vertices, faces)
        batch = batch_contains(mesh)
        def contains(q):
            t = time.perf_counter(); m1 = serial(q); t1 = time.perf_counter() - t
            t = time.perf_counter(); m2 = batch(np.asarray(q, np.float64)); t2 = time.perf_counter() - t
            calls.append({"n": len(q), "serial_s": t1, "batch_s": t2, "equal": bool(np.array_equal(m1, m2)),
                          "n_diff": int(np.sum(m1 != m2)), "inside": int(m1.sum())})
            return m1
        return contains
    VG.make_inside_test = recording
    try:
        VG.build_volume_case(wall, sm, f, atlas, rel.input_features, n_internal=20000, seed=seed)
    finally:
        VG.make_inside_test = original
    tot = {"job": dst.name, "calls": len(calls), "points": sum(c["n"] for c in calls), "serial_s": round(sum(c["serial_s"] for c in calls), 3),
           "batch_s": round(sum(c["batch_s"] for c in calls), 3), "all_equal": all(c["equal"] for c in calls), "n_diff": sum(c["n_diff"] for c in calls)}
    print(json.dumps(tot), flush=True)
    return tot, calls

if __name__ == "__main__":
    out = [run(Path(p)) for p in sys.argv[1:]]
    Path(__file__).with_name("proto_inside.json").write_text(json.dumps(out, indent=1))
