"""Old (frozen copy) vs new integrate_streamlines on a rebuilt real volume job: byte equality + timing."""
import importlib.util
import json
import sys
import time

import numpy as np
import vtk

import os
from wss_deploy.volume_geometry import make_inside_test

S = "/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/a3bc14b7-1081-45dd-b401-3ac6128ea547/scratchpad/streamlines"
spec = importlib.util.spec_from_file_location("streamlines_orig", f"{S}/streamlines_orig.py")
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
if os.environ.get("NEW_MODULE"):          # a scratch variant instead of the repo module
    spec = importlib.util.spec_from_file_location("variant", os.environ["NEW_MODULE"])
    new = importlib.util.module_from_spec(spec); spec.loader.exec_module(new)
else:
    from wss_deploy import streamlines as new
print("new =", new.__file__)

z = np.load(sys.argv[1])
repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 5
points, velocity, seeds = z["points"], z["velocity"], z["seeds"]
V, F = z["closed_vertices"], z["closed_faces"]
# Exactly the stage_b_volume construction (one certified inside test shared by all runs, as in production).
inside = old.ball_certified_inside(make_inside_test(V, F), points, V, F)
calls = []


def recording(q):
    calls.append(np.asarray(q, np.float64).tobytes())
    return inside(q)


def run(module, record=False):
    calls.clear()
    vtk.vtkMath.RandomSeed(20260928)           # same global VTK ray sequence at entry for both versions
    t = time.perf_counter()
    lines, info = module.integrate_streamlines(points, velocity, seeds, recording if record else inside)
    elapsed = time.perf_counter() - t
    rng_after = vtk.vtkMath.Random()            # position of VTK's global sequence at exit
    return lines, info, elapsed, rng_after, list(calls)


def same(a, b):
    la, ia = a; lb, ib = b
    if len(la) != len(lb) or json.dumps(ia, sort_keys=True) != json.dumps(ib, sort_keys=True):
        return False
    return all(x["points"].dtype == y["points"].dtype == np.float32 and x["speed_m_s"].dtype == y["speed_m_s"].dtype == np.float32
               and x["points"].tobytes() == y["points"].tobytes() and x["speed_m_s"].tobytes() == y["speed_m_s"].tobytes()
               and set(x) == set(y) for x, y in zip(la, lb))


lo, io, _, ro, co = run(old, record=True)
ln, in_, _, rn, cn = run(new, record=True)
print("lines", len(lo), "vertices", sum(len(l["points"]) for l in lo))
print("byte-identical lines+info:", same((lo, io), (ln, in_)))
print("identical inside() call sequence:", co == cn, f"({len(co)} calls, {sum(len(c) for c in co) // 24} points)")
print("identical VTK global RNG position at exit:", ro == rn)
thinned_o, so = old.thin_lines(lo); thinned_n, sn = new.thin_lines(ln)
print("thin_lines identical:", so == sn and same((thinned_o, {}), (thinned_n, {})))

# Production cross-check: the job's streamlines.vtp was written from thin_lines(...) in the service process.
try:
    import pyvista as pv
    job_vtp = sys.argv[3]
    saved = pv.read(job_vtp)
    xyz = np.concatenate([l["points"] for l in thinned_n])
    print("matches production streamlines.vtp points:", np.asarray(saved.points).tobytes() == xyz.tobytes(),
          "speed:", np.asarray(saved.point_data["speed_m_s"]).tobytes() == np.concatenate([l["speed_m_s"] for l in thinned_n]).tobytes())
except IndexError:
    pass

times = {"old": [], "new": []}
for _ in range(repeats):
    for name, module in (("old", old), ("new", new)):
        lines, info, elapsed, _, _ = run(module)
        assert same((lines, info), (lo, io))
        times[name].append(elapsed)
for name in times:
    print(name, "median %.3f s" % np.median(times[name]), "runs", " ".join("%.3f" % t for t in times[name]))
print("speed-up %.1f%%" % (100 * (1 - np.median(times["new"]) / np.median(times["old"]))))
