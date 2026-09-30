"""Isolated CPU operator probe. Reads one existing deployment case; no model inference.
Run from repository root with the GNN interpreter. Output is beside this script.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '4')
import sys, json, time, statistics, platform, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
from wss_deploy.centerline import load_vessel_geom_atlas, apply_mapping
from wss_features.cloud import build_oriented_cloud, median_spacing, virtual_caps, patch_atlas_end_radius
from wss_features.atlas import map_points
job = ROOT / 'outputs/wss_deploy_jobs/20260927_145732_67eba52ab03c'
resample = next((job / 'geometry_cache').glob('resample-*.npz'))
pointgeom = next((job / 'geometry_cache').glob('pointgeom-*.npz'))
with np.load(resample, allow_pickle=False) as z:
    points = z['points']
with np.load(pointgeom, allow_pickle=False) as z:
    normals = z['normals']
atlas = apply_mapping(load_vessel_geom_atlas(job / 'centerline'), json.loads((job / 'summary.json').read_text())['mapping'])
patch_atlas_end_radius(atlas, points)
spacing = median_spacing(points)
def cap_records(caps):
    return [{"label": c["label"], "center_mm": np.asarray(c["center_mm"]).tolist(),
             "outward": np.asarray(c["outward"]).tolist(), "radius_mm": float(c["radius_mm"]),
             "atlas_radius_mm": float(c["atlas_radius_mm"]), "rim_snap_mm": c["rim_snap_mm"],
             "cap_source": c["cap_source"], "rim_fit": c["rim_fit"], "points": int(len(c["points_mm"]))} for c in caps]
def full():
    return build_oriented_cloud(points, atlas, normals_out=normals, calibrate=False)[1]['caps']
def caps_with_spacing():
    return cap_records(virtual_caps(atlas, median_spacing(points), points))
def caps_cached_spacing():
    return cap_records(virtual_caps(atlas, spacing, points))
def benchmark(fn):
    fn()
    samples=[]
    for _ in range(3):
        start=time.perf_counter(); value=fn(); samples.append(time.perf_counter()-start)
    return {'samples_s': samples, 'median_s':statistics.median(samples)}, value
old_t, old = benchmark(full)
new_t, new = benchmark(caps_with_spacing)
cached_t, cached = benchmark(caps_cached_spacing)
map_t, mapped = benchmark(lambda: map_points(points, atlas)['atlas_row'])
k1_t, nearest = benchmark(lambda: atlas.tree_rows[atlas.tree.query(points, k=1)[1]])
# k=16 includes self; the curvature query requests 128 nonself neighbours.
from scipy.spatial import cKDTree
tree=cKDTree(points)
small_t, small=benchmark(lambda: tree.query(points, k=16)[1])
large_t, large=benchmark(lambda: tree.query(points, k=129, workers=4)[1])
result={
    'case': str(job.relative_to(ROOT)), 'n_points':len(points), 'dtype':str(points.dtype),
    'environment':{'host':platform.node(),'numpy':np.__version__,'scipy':scipy.__version__, 'threads':4},
    'input_files':{str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (resample,pointgeom)},
    'full_cloud':old_t, 'caps_with_spacing_query':new_t, 'caps_with_cached_spacing':cached_t,
    'caps_exact_equal':old == new == cached,
    'full_to_caps_ratio':old_t['median_s']/new_t['median_s'],
    'map_all_features':map_t, 'nearest_only':k1_t,
    'nearest_row_mismatches':int(np.count_nonzero(mapped != nearest)),
    'knn16':small_t, 'knn129':large_t,
    'knn16_prefix_mismatches':int(np.count_nonzero(small != large[:,:16])),
    'scope':'One real cached wall cloud, warm-up + 3 CPU repetitions; no CFD/accuracy evaluation, no model inference, no production changes. kNN timing uses different worker counts and is NOT a standalone speed comparison. Ties can change order for other clouds.',
}
Path(__file__).with_name('cpu_microbench.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
