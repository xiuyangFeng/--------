"""Does the ORIGINAL code's output depend on VTK's global RNG state? And how does the full-carry variant compare?"""
import importlib.util, json, sys, time
import numpy as np, vtk
from wss_deploy.volume_geometry import make_inside_test
S = "/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/a3bc14b7-1081-45dd-b401-3ac6128ea547/scratchpad/streamlines"
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
old = load("old", f"{S}/streamlines_orig.py"); full = load("full", f"{S}/streamlines_fullcarry.py")
z = np.load(sys.argv[1]); points, velocity, seeds = z["points"], z["velocity"], z["seeds"]
V, F = z["closed_vertices"], z["closed_faces"]
inside = old.ball_certified_inside(make_inside_test(V, F), points, V, F)
def digest(lines, info):
    import hashlib; h = hashlib.sha256(json.dumps(info, sort_keys=True).encode())
    for l in lines: h.update(l["points"].tobytes()); h.update(l["speed_m_s"].tobytes())
    return h.hexdigest()[:16]
res = {}
for mod_name, mod in (("old", old), ("full_carry", full)):
    for rs in (1, 7, 20260928, 99991):
        vtk.vtkMath.RandomSeed(rs); t = time.perf_counter()
        lines, info = mod.integrate_streamlines(points, velocity, seeds, inside)
        res[(mod_name, rs)] = (digest(lines, info), time.perf_counter() - t)
for k, v in res.items(): print(k, v[0], "%.3f s" % v[1])
