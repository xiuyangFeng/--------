"""Rebuild the exact streamline inputs of a finished PF6/VF6 volume job (on a scratch copy of the job dir).

points (float64 interior queries), velocity (float32 as exported), closed surface, seeds — the same calls
stage_b_volume makes before integrate_streamlines.  Velocity comes from field.npz (stage_b stores the
float32 prediction there verbatim); the float64 interior points and the closed lumen are recomputed with
the uncached geometry path and checked against field.npz's float32 copy.
"""
import json
import sys
from pathlib import Path

import numpy as np

from wss_deploy import centerline as CL, geometry as G
from wss_deploy import geometry_cache as GC
from wss_deploy.pipeline import _clean_mesh, _resampled
from wss_deploy.volume_geometry import build_volume_case
from wss_deploy.streamlines import centerline_seeds, volume_seeds
from wss_deploy.io_utils import resolve_job_path

job_dir = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
job = json.loads((job_dir / "job.json").read_text())
a = json.loads((job_dir / "stage_a.json").read_text())
mapping = job["mapping"]
field = np.load(job_dir / "field.npz")
atlas = CL.apply_mapping(CL.load_vessel_geom_atlas(job_dir / "centerline"), mapping)
clean_stl = resolve_job_path(job_dir, a["input_check"]["clean_stl"])
cache = GC.GeometryCache(job_dir)
vertices, faces = _clean_mesh(job_dir, clean_stl, cache)
vertices = np.asarray(vertices, np.float64); faces = np.asarray(faces, np.int64)
seed = G.stable_sampling_seed(a["input_sha256"])
smoothed, wall = _resampled(vertices, faces, smooth_mm=1., spacing_mm=.5, seed=seed, cache=cache)
# input_features only gates feature availability; interior sampling and the closed lumen do not depend on it.
case, aux = build_volume_case(wall, smoothed, faces, atlas, (), case_name="input-" + a["input_sha256"][:24],
                              n_internal=20000, seed=seed)
n_wall = int(case["n_wall"])
points = case["wall_coords_raw"][n_wall:]
assert points.dtype == np.float64
match = np.array_equal(points.astype(np.float32), field["internal_pts"])
velocity = field["velocity_m_s"]
assert velocity.dtype == np.float32 and velocity.shape == points.shape
closed = aux["closed_surface"]
seeds = np.concatenate([centerline_seeds(atlas), volume_seeds(points, n=120, seed=seed)])
np.savez(out, points=points, velocity=velocity, closed_vertices=np.asarray(closed["vertices"]),
         closed_faces=np.asarray(closed["faces"]), seeds=seeds, sampling_seed=seed)
print(job_dir.name, "points", points.shape, "matches field.npz internal_pts:", match,
      "seeds", seeds.shape, "closed", np.asarray(closed["vertices"]).shape, np.asarray(closed["faces"]).shape)
