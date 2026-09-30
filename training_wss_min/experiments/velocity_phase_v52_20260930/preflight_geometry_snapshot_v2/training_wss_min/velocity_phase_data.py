"""Isolated geometry sidecars for the 261-unit velocity mechanism experiment.

The existing cycle cache is strictly read-only. Query rows, labels, weights and
sampling streams come unchanged from JointCycleDataset. New anchors use only
wall triangles, artificial opening caps and the stored centreline atlas. CFD
volume points identify original queries, but never generate graph anchors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from training_wss_min import joint_cycle_data as base

SCHEMA = 'velocity_phase_geometry_v1'
DEFAULT_GEOMETRY = base.ROOT / 'training_wss_min/experiments/velocity_phase_v52_20260930/geometry'
N_ANCHORS = 256
N_NEIGHBORS = 16


def _json(path, data):
    base._write_json(path, data)


def _array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def aligned_frames(tangent, normal, binormal, rotation):
    """Return SO(3) columns [t,n,b]; invalid source frames fall back to XYZ.

    Gram-Schmidt only removes floating-point export error. A substantially
    inconsistent source frame is marked invalid rather than silently repaired.
    """
    t, n, b = [np.asarray(a, np.float64) for a in (tangent, normal, binormal)]
    r = np.asarray(rotation, np.float64)
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-8) or np.linalg.det(r) < .999999:
        raise ValueError('atlas alignment rotation is not proper orthogonal')
    raw = np.stack((t, n, b), axis=-1)
    finite = np.isfinite(raw).all(axis=(1, 2))
    raw = np.nan_to_num(raw)
    gram_error = np.max(np.abs(np.swapaxes(raw, -1, -2) @ raw - np.eye(3)), axis=(1, 2))
    valid = finite & (gram_error < 1e-3) & (np.linalg.det(raw) > .999)
    t = np.nan_to_num(t)
    n = np.nan_to_num(n)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    n -= (n * t).sum(1, keepdims=True) * t
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    frame = np.stack((t @ r.T, n @ r.T, np.cross(t, n) @ r.T), axis=-1)
    frame[~valid] = np.eye(3)
    if not np.allclose(np.swapaxes(frame, -1, -2) @ frame, np.eye(3), atol=2e-6):
        raise ValueError('output frame is not orthonormal')
    return frame.astype(np.float32), valid.astype(np.float32)


def _read_atlas(h):
    from wss_features.atlas import Atlas
    g = h['geometry']
    table = np.asarray(g['atlas_table'], np.float64)
    columns = json.loads(g.attrs['atlas_columns'])
    tree_rows = np.flatnonzero(table[:, columns.index('is_child_duplicate')] < .5)
    xyz = table[:, [columns.index(k) for k in ('x_mm', 'y_mm', 'z_mm')]]
    return Atlas(table=table, columns=columns, segments=json.loads(g.attrs['atlas_segments']),
                 semantic_of_segment={int(k): v for k, v in json.loads(g.attrs['semantic_of_segment']).items()},
                 frame_n=np.asarray(g['atlas_frame_n'], np.float64),
                 frame_b=np.asarray(g['atlas_frame_b'], np.float64), tree_rows=tree_rows,
                 tree=cKDTree(xyz[tree_rows]), provenance=json.loads(g.attrs['atlas_provenance']))


def _closed_surface(vertices, faces, atlas):
    """Close only verified anatomical opening loops; audit residual edge count."""
    import pyvista as pv
    from wss_v5.section_features import _boundary_cycles, _remove_opposite_triangle_fins, _verified_fan_center
    triangles, cleanup = _remove_opposite_triangle_fins(np.asarray(faces, np.int64))
    loops, edge_audit = _boundary_cycles(triangles)
    endpoints = atlas.endpoints()
    if len(loops) != len(endpoints):
        raise ValueError(f'wall opening loops {len(loops)} != anatomical endpoints {len(endpoints)}')
    centers = np.asarray([vertices[loop].mean(0) for loop in loops])
    distances = np.linalg.norm(centers[:, None] - np.asarray([x['center_mm'] for x in endpoints])[None], axis=2)
    rr, cc = linear_sum_assignment(distances)
    cap_records = []
    for i, j in zip(rr, cc):
        if distances[i, j] > max(3 * float(endpoints[j]['radius_mm']), 5.):
            raise ValueError('opening is too far from its anatomical endpoint')
        cap_records.append({'loop': int(i), 'label': endpoints[j]['label'],
                            'endpoint_distance_mm': float(distances[i, j]), 'vertices': int(len(loops[i]))})
    packed = np.column_stack((np.full(len(triangles), 3), triangles))
    wall = pv.PolyData(np.asarray(vertices, np.float64), packed)
    closed = wall.fill_holes(float(np.linalg.norm(np.ptp(vertices, axis=0))) * 2).triangulate()
    remaining = int(closed.extract_feature_edges(boundary_edges=True, non_manifold_edges=True,
                                                  feature_edges=False, manifold_edges=False).n_cells)
    method = 'vtk_boundary_polygon_caps'
    fan_proofs = []
    if remaining:
        # Nonplanar opening loops can defeat VTK ear clipping. A fan is used
        # only when the existing independent geometric proof accepts it.
        extra_points, extra_faces = [], []
        for loop in loops:
            center, proof = _verified_fan_center(vertices[loop])
            fan_proofs.append(proof)
            if center is None:
                raise ValueError('nonplanar opening has no verified cap triangulation')
            index = len(vertices) + len(extra_points)
            extra_points.append(center)
            extra_faces.extend((int(a), int(b), index) for a, b in zip(loop, np.roll(loop, -1)))
        all_faces = np.concatenate((triangles, np.asarray(extra_faces, np.int64)))
        closed = pv.PolyData(np.concatenate((vertices, np.asarray(extra_points))),
                             np.column_stack((np.full(len(all_faces), 3), all_faces)))
        remaining = int(closed.extract_feature_edges(boundary_edges=True, non_manifold_edges=True,
                                                      feature_edges=False, manifold_edges=False).n_cells)
        method = 'verified_boundary_centroid_fans'
    if remaining:
        raise ValueError(f'closed wall still has {remaining} boundary/nonmanifold edges')
    return wall, closed, {'closed_reference_valid': True, 'remaining_boundary_or_nonmanifold_edges': 0,
                           'method': method, 'caps': cap_records, 'source_edge_audit': edge_audit,
                           'zero_thickness_fin_cleanup': cleanup, 'fan_proofs': fan_proofs,
                           'self_intersections': 'not_evaluated', 'source_vertices_moved': False}


def _contains(closed, points):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    cloud = vtk.vtkPolyData()
    p = vtk.vtkPoints()
    p.SetData(numpy_to_vtk(np.ascontiguousarray(points, np.float64), deep=True))
    cloud.SetPoints(p)
    enclosed = vtk.vtkSelectEnclosedPoints()
    enclosed.SetInputData(cloud)
    enclosed.SetSurfaceData(closed)
    enclosed.SetTolerance(1e-9)
    enclosed.Update()
    return vtk_to_numpy(enclosed.GetOutput().GetPointData().GetArray('SelectedPoints')).astype(bool)


def generate_anchors(vertices, faces, atlas, *, seed, count=N_ANCHORS):
    """Branch-stratified, geometry-only tube proposals followed by ray rejection."""
    from wss_features.atlas import map_points
    from wss_deploy.volume_geometry import _branch_profiles, _radius_weighted_branch_quotas, _segment_tube_candidates
    wall, closed, closure = _closed_surface(vertices, faces, atlas)
    rng = np.random.default_rng(seed)
    quotas, floor = _radius_weighted_branch_quotas(count, _branch_profiles(atlas), min_points=N_NEIGHBORS + 1)
    if floor < N_NEIGHBORS + 1:
        raise ValueError('anchor budget cannot cover every branch with 17 points')
    parts, attempts, accepted_counts = [], {}, {}
    for sid, quota in sorted(quotas.items()):
        accepted = np.empty((0, 3), np.float64)
        tried = 0
        for _ in range(12):
            proposal = _segment_tube_candidates(atlas, sid, max(256, 8 * (quota - len(accepted))), rng)
            tried += len(proposal)
            proposal = proposal[_contains(closed, proposal)]
            if len(proposal):
                mapped = map_points(proposal, atlas)
                proposal = proposal[mapped['segment_id'] == sid]
                accepted = np.concatenate((accepted, proposal))
            if len(accepted) >= quota:
                break
        if len(accepted) < quota:
            raise ValueError(f'geometry branch {sid}: only {len(accepted)} valid anchors for quota {quota}')
        parts.append(accepted[rng.choice(len(accepted), quota, replace=False)])
        attempts[sid], accepted_counts[sid] = tried, int(len(accepted))
    xyz = np.concatenate(parts)
    if len(xyz) != count or not _contains(closed, xyz).all() or len(np.unique(xyz, axis=0)) != count:
        raise ValueError('anchor count/containment/uniqueness audit failed')
    import pyvista as pv
    distance = np.abs(np.asarray(pv.PolyData(xyz).compute_implicit_distance(wall)['implicit_distance']))
    if not np.isfinite(distance).all() or distance.min() <= 1e-6:
        raise ValueError('anchor lies on the wall or has invalid wall distance')
    features = map_points(xyz, atlas)
    return xyz, features, distance, {'source': 'wall_static/xyz_mm + topology/wall_triangles + atlas; no CFD volume points or fields',
        'sampling': 'branch-stratified atlas tube proposals; exact capped triangle-surface rejection',
        'seed': int(seed), 'count': count, 'quotas': quotas, 'branch_floor': floor,
        'proposals': attempts, 'accepted_candidates': accepted_counts, 'inside_fraction': 1.,
        'wall_distance_min_mm': float(distance.min()), 'closure': closure}


def build_branch_graph(pos, segment_ids, path, parent_of, *, neighbors=N_NEIGHBORS):
    """Directed target/source neighbors with mandatory bidirectional branch bridges.

    Legal edges join one segment or a direct parent/child. For every tree link,
    its closest path pair is explicitly reserved, so a dense same-branch cloud
    cannot accidentally disconnect a bifurcation. Remaining slots minimize
    euclidean distance + 0.25 * absolute centreline path difference.
    """
    pos, sid, path = np.asarray(pos, np.float64), np.asarray(segment_ids, np.int64), np.asarray(path, np.float64)
    n = len(pos)
    same = sid[:, None] == sid[None, :]
    parent = np.asarray([parent_of[int(s)] for s in sid])
    related = (sid[:, None] == parent[None, :]) | (parent[:, None] == sid[None, :])
    legal = same | related
    np.fill_diagonal(legal, False)
    rel = pos[None, :, :] - pos[:, None, :]
    distance = np.linalg.norm(rel, axis=-1)
    delta = path[None, :] - path[:, None]
    cost = distance + .25 * np.abs(delta)
    required = [set() for _ in range(n)]
    bridges = []
    for child, par in sorted(parent_of.items()):
        if par < 0:
            continue
        ci, pi = np.flatnonzero(sid == child), np.flatnonzero(sid == par)
        if not len(ci) or not len(pi):
            raise ValueError(f'anchor graph misses branch pair {par}->{child}')
        ci0, pi0 = np.unravel_index(np.argmin(cost[np.ix_(ci, pi)]), (len(ci), len(pi)))
        c, p = int(ci[ci0]), int(pi[pi0])
        required[c].add(p)
        required[p].add(c)
        bridges.append([p, c])
    targets, sources = [], []
    for i in range(n):
        candidates = np.flatnonzero(legal[i])
        if len(candidates) < neighbors or len(required[i]) > neighbors:
            raise ValueError('branch graph cannot provide the fixed neighbor budget')
        chosen = sorted(required[i])
        for j in candidates[np.argsort(cost[i, candidates], kind='stable')]:
            if len(chosen) == neighbors:
                break
            if int(j) not in required[i]:
                chosen.append(int(j))
        targets.extend([i] * neighbors)
        sources.extend(chosen)
    ti, so = np.asarray(targets, np.int64), np.asarray(sources, np.int64)
    edge = np.stack((ti, so))
    attr = np.column_stack((rel[ti, so], distance[ti, so], delta[ti, so], same[ti, so], related[ti, so])).astype(np.float32)
    return edge, attr, {'neighbors': neighbors, 'edges': len(ti), 'parent_child_bridges': bridges,
        'parent_child_edge_count': int(related[ti, so].sum()), 'same_branch_edge_count': int(same[ti, so].sum()),
        'edge_orientation': 'target=edge_index[0], source=edge_index[1]',
        'edge_columns': ['source_minus_target_x', 'source_minus_target_y', 'source_minus_target_z',
                         'euclidean_distance', 'source_minus_target_root_path', 'same_branch', 'parent_child'],
        'length_units': 'original aligned query coord_scale (dimensionless)'}


def _anchor_features(xyz, mapped, distance, atlas, bundle, flowref):
    rotation = bundle['transform_rotation'].astype(np.float64)
    pos = (xyz - bundle['transform_centroid']) @ rotation.T / float(bundle['coord_scale'])
    cols = {k: pos[:, j] for j, k in enumerate(('x', 'y', 'z'))}
    cols.update(abscissa_norm=mapped['s_from_root_mm'] / float(bundle['wall_abscissa_mm'].max()),
                local_radius=mapped['radius_mm'], curvature=mapped['curvature_per_mm'],
                log_local_radius=np.log(np.maximum(mapped['radius_mm'], 1e-6)), rho=mapped['rho'],
                theta_sin=np.sin(mapped['theta_rad']), theta_cos=np.cos(mapped['theta_rad']),
                dr_ds=mapped['dr_ds'], dist_to_junction_mm=mapped['dist_to_junction_mm'],
                dist_to_endpoint_mm=mapped['dist_to_endpoint_mm'], end_zone=mapped['end_zone'])
    row = mapped['atlas_row'].astype(np.int64)
    d = xyz - atlas.xyz[row]
    t = atlas.tangent[row]
    d -= (d * t).sum(1, keepdims=True) * t
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    radial = np.where(norm > 1e-6, d / np.maximum(norm, 1e-12), 0.) @ rotation.T
    for j, key in enumerate(('radial_x', 'radial_y', 'radial_z')):
        cols[key] = radial[:, j]
    seg = bundle['wall_segment_id']
    shares = {int(s): float(np.median(flowref['wall_log_q_branch_murray_cap'][seg == s])) for s in np.unique(seg)}
    cols['log_q_branch_murray_cap'] = np.asarray([shares[int(s)] for s in mapped['segment_id']])
    cols['log_tau0_murray_cap'] = cols['log_q_branch_murray_cap'] - 3 * np.log(np.maximum(mapped['radius_mm'], 1e-6))
    cols['dist_to_wall_mm'] = distance
    raw = np.stack([cols[k] for k in base.VOLUME_FEATURES], axis=1)
    for j, name in enumerate(base.VOLUME_FEATURES):
        if name in base.CURVATURE_FEATURES:
            raw[:, j] = np.sign(raw[:, j]) * np.log1p(np.abs(raw[:, j]))
    if not np.isfinite(raw).all():
        raise ValueError('nonfinite anchor geometric features')
    return pos.astype(np.float32), raw.astype(np.float32)


def prepare_case(cid, cache_root, geometry_root, *, seed=1234):
    started = time.time()
    cache_root, geometry_root = Path(cache_root), Path(geometry_root)
    _, paths = base.source_paths(cid, cache_root)
    cache = cache_root / 'cases' / cid
    out = geometry_root / 'cases' / cid
    marker = out / 'metadata.json'
    settings = {'seed': int(seed), 'anchors': N_ANCHORS, 'neighbors': N_NEIGHBORS}
    original = json.loads((cache / 'metadata.json').read_text())
    signatures = {k: base._signature(paths[k]) for k in ('h5', 'bundle', 'volume', 'flowref')}
    if any(signatures[k] != original['sources'][k] for k in signatures):
        raise ValueError(f'{cid}: original geometry source changed after cycle cache preparation')
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous['schema'] != SCHEMA or previous['settings'] != settings or previous['sources'] != signatures:
            raise ValueError(f'{cid}: incompatible immutable sidecar; use a new geometry root')
        for name, digest in previous['file_sha256'].items():
            if base.sha256(out / name) != digest:
                raise ValueError(f'{cid}: sidecar changed: {name}')
        return previous
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(paths['h5'], 'r') as h, np.load(paths['bundle']) as b, np.load(paths['volume']) as v, np.load(paths['flowref']) as f:
        if str(h.attrs['canonical_id']) != cid or str(v['canonical_id']) != cid or str(b['canonical_id']) != cid:
            raise ValueError('case identity mismatch')
        atlas = _read_atlas(h)
        atlas_frames, atlas_valid = aligned_frames(atlas.tangent, atlas.frame_n, atlas.frame_b, b['transform_rotation'])
        volume_atlas_rows = h['volume_static/atlas_row'][()].astype(np.int64)
        if volume_atlas_rows.min() < 0 or volume_atlas_rows.max() >= len(atlas.table):
            raise ValueError('original volume atlas row out of bounds')
        pool_records = {}
        for mode in ('train', 'eval'):
            rows = np.load(cache / f'{mode}_volume_rows.npy')
            if not np.array_equal(rows, np.unique(rows)):
                raise ValueError('original query rows must be unique and sorted')
            # Read static coordinates only, not velocity/pressure labels.
            if not np.allclose(h['volume_static/xyz_mm'][rows], v['vol_coords_raw'][rows], atol=2e-4, rtol=0):
                raise ValueError('H5 query row identity differs from original volume view')
            if not np.array_equal(np.load(cache / f'{mode}_volume_pos.npy'), v['vol_coords_norm'][rows]):
                raise ValueError('original query positions no longer match original row identity')
            frame, valid = atlas_frames[volume_atlas_rows[rows]], atlas_valid[volume_atlas_rows[rows]]
            for name, a in (('frame', frame), ('frame_valid', valid), ('rows', rows)):
                np.save(out / f'{mode}_{name}.npy', a, allow_pickle=False)
            pool_records[mode] = {'count': len(rows), 'row_sha256': _array_hash(rows), 'frame_valid_fraction': float(valid.mean()),
                                  'invalid_frame_count': int((valid == 0).sum())}
        if np.intersect1d(np.load(out / 'train_rows.npy'), np.load(out / 'eval_rows.npy')).size:
            raise ValueError('original train and evaluation query pools overlap')
        xyz, mapped, distance, anchor_audit = generate_anchors(h['wall_static/xyz_mm'][()], h['topology/wall_triangles'][()],
                              atlas, seed=base.stable_seed(seed, cid, 'velocity-phase-geometry-anchors'))
        if not np.array_equal(b['wall_node_id_cas'], f['wall_node_id_cas']):
            raise ValueError('wall flow reference identity mismatch')
        pos, x = _anchor_features(xyz, mapped, distance, atlas, b, f)
        ar = mapped['atlas_row'].astype(np.int64)
        frame, valid = atlas_frames[ar], atlas_valid[ar]
        geom_extra = np.column_stack((frame.reshape(-1, 9), valid)).astype(np.float32)
        parent = {int(s['segment_id']): int(s['parent_id']) for s in atlas.segments}
        edge, attr, graph_audit = build_branch_graph(pos, mapped['segment_id'],
            mapped['s_from_root_mm'] / float(b['coord_scale']), parent)
        np.savez(out / 'anchors.npz', pos=pos, x=x, geom_extra=geom_extra, frame=frame, frame_valid=valid,
                 edge_index=edge, edge_attr=attr, segment_id=mapped['segment_id'], atlas_row=ar,
                 xyz_mm=xyz.astype(np.float32))
    file_hashes = {p.name: base.sha256(p) for p in sorted(out.iterdir()) if p.suffix in ('.npy', '.npz')}
    record = {'schema': SCHEMA, 'status': 'passed', 'canonical_id': cid, 'settings': settings,
              'sources': signatures, 'original_cache_metadata_sha256': base.sha256(cache / 'metadata.json'),
              'original_query_pools': pool_records, 'anchors': anchor_audit, 'graph': graph_audit,
              'anchor_frame_valid_fraction': float(valid.mean()), 'file_sha256': file_hashes,
              'seconds': time.time() - started,
              'fallback': 'invalid source RMF => identity XYZ for both A0/A1; no rows dropped'}
    _json(marker, record)
    return record


def weighted_quantile(values, weights, quantile=.8):
    values, weights = np.asarray(values, np.float64), np.asarray(weights, np.float64)
    if values.ndim != 1 or values.shape != weights.shape or not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError('invalid weighted quantile arrays')
    if np.any(weights < 0) or weights.sum() <= 0 or not 0 <= quantile <= 1:
        raise ValueError('invalid weighted quantile mass')
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    return float(values[order[min(np.searchsorted(cumulative, quantile * cumulative[-1], side='left'), len(values) - 1)]])


def fit_tail_stats(cache_root, geometry_root):
    """Fit 80 q80 values: each training case has one unit of volume-weighted mass."""
    cache_root, geometry_root = Path(cache_root), Path(geometry_root)
    split = json.loads(base.SPLIT_PATH.read_text())
    ids = split['train_cases']
    marker = geometry_root / 'tail_stats.json'
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous['train_ids'] != ids or previous['split_sha256'] != base.sha256(base.SPLIT_PATH):
            raise ValueError('existing tail thresholds belong to another split')
        return previous
    sources = []
    counts = [len(np.load(cache_root / 'cases' / cid / 'train_volume_weight.npy', mmap_mode='r')) for cid in ids]
    speeds = np.empty((80, sum(counts)), np.float64)
    weights = np.empty(sum(counts), np.float64)
    begin = 0
    for cid, count in zip(ids, counts):
        directory = cache_root / 'cases' / cid
        velocity = np.load(directory / 'train_velocity.npy', mmap_mode='r')
        w = np.asarray(np.load(directory / 'train_volume_weight.npy'), np.float64)
        if velocity.shape != (count, 80, 3) or not np.isfinite(w).all() or w.min() <= 0:
            raise ValueError('train-only velocity pool invalid')
        speeds[:, begin:begin + count] = np.linalg.norm(np.asarray(velocity, np.float64), axis=-1).T
        weights[begin:begin + count] = w / w.sum() / len(ids)
        sources.append({'canonical_id': cid, 'count': count,
            'velocity_sha256': base.sha256(directory / 'train_velocity.npy'),
            'weight_sha256': base.sha256(directory / 'train_volume_weight.npy')})
        begin += count
    q = [weighted_quantile(speeds[t], weights, .8) for t in range(80)]
    record = {'schema': SCHEMA, 'status': 'passed', 'q80_m_s': q, 'frames': list(range(80)),
              'train_ids': ids, 'complete_train_partition': len(ids) == 206,
              'split_sha256': base.sha256(base.SPLIT_PATH), 'sources': sources,
              'scope': 'only original fixed training query candidates; case-equal, within-case volume weighted',
              'quantile_definition': 'inverse CDF (left); first sorted speed reaching 0.80 cumulative mass',
              'supervision_only': True, 'heldout_labels_read': False}
    _json(marker, record)
    return record


def validate_geometry_manifest(geometry_root, cache_root=base.DEFAULT_CACHE, *, require_complete=True, verify_files=True):
    geometry_root, cache_root = Path(geometry_root), Path(cache_root)
    manifest = json.loads((geometry_root / 'manifest.json').read_text())
    split = json.loads(base.SPLIT_PATH.read_text())
    if manifest['schema'] != SCHEMA or manifest['split_sha256'] != base.sha256(base.SPLIT_PATH):
        raise ValueError('geometry schema/split mismatch')
    if Path(manifest['cache_root']).resolve() != cache_root.resolve():
        raise ValueError('geometry sidecars refer to a different original cache')
    expected = split['train_cases'] + split['test_cases']
    if require_complete and (not manifest['complete'] or manifest['case_ids'] != expected or manifest['status'] != 'passed'):
        raise ValueError('geometry sidecars do not cover all 206 train / 55 holdout units')
    if verify_files:
        for cid, record in manifest['cases'].items():
            directory = geometry_root / 'cases' / cid
            for name, digest in record['file_sha256'].items():
                if base.sha256(directory / name) != digest:
                    raise ValueError(f'sidecar fingerprint mismatch: {cid}/{name}')
            if base.sha256(directory / 'metadata.json') != record['metadata_sha256']:
                raise ValueError(f'sidecar metadata mismatch: {cid}')
    if require_complete:
        if base.sha256(geometry_root / 'tail_stats.json') != manifest['tail_stats_sha256']:
            raise ValueError('tail supervision thresholds changed')
        tail = json.loads((geometry_root / 'tail_stats.json').read_text())
        if tail['train_ids'] != split['train_cases'] or len(tail['q80_m_s']) != 80 or tail['status'] != 'passed':
            raise ValueError('tail supervision thresholds are not fitted on this 206-case training partition')
    return manifest


class VelocityPhaseDataset(base.JointCycleDataset):
    def __init__(self, cache_root=base.DEFAULT_CACHE, *args, geometry_root=DEFAULT_GEOMETRY,
                 include_anchors=False, allow_partial_geometry=False, **kwargs):
        super().__init__(cache_root, *args, **kwargs)
        self.geometry_root = Path(geometry_root)
        self.include_anchors = bool(include_anchors)
        self._sidecars = {}
        manifest = validate_geometry_manifest(self.geometry_root, self.cache_root,
            require_complete=not allow_partial_geometry, verify_files=False)
        if not set(self.ids) <= set(manifest['case_ids']):
            raise ValueError('requested units are missing geometry sidecars')
        self.geometry_manifest = manifest

    def __getstate__(self):
        state = super().__getstate__()
        state['_sidecars'] = {}
        return state

    def _sidecar(self, cid):
        if cid not in self._sidecars:
            directory = self.geometry_root / 'cases' / cid
            record = self.geometry_manifest['cases'][cid]
            data = {}
            for mode in ('train', 'eval'):
                for name in ('frame', 'frame_valid', 'rows'):
                    path = directory / f'{mode}_{name}.npy'
                    if base.sha256(path) != record['file_sha256'][path.name]:
                        raise ValueError(f'{cid}: geometry changed after preparation')
                    data[f'{mode}_{name}'] = np.load(path, mmap_mode='r')
                if not np.array_equal(data[f'{mode}_rows'], self._array(cid, f'{mode}_volume_rows')):
                    raise ValueError(f'{cid}: sidecar rows differ from original query pool')
            if self.include_anchors:
                path = directory / 'anchors.npz'
                if base.sha256(path) != record['file_sha256'][path.name]:
                    raise ValueError(f'{cid}: anchors changed after preparation')
                with np.load(path) as z:
                    data['anchors'] = {k: z[k] for k in ('pos', 'x', 'geom_extra', 'edge_index', 'edge_attr')}
            self._sidecars[cid] = data
        return self._sidecars[cid]

    def __getitem__(self, index):
        item = super().__getitem__(index)
        cid = item['unit_id']
        mode = 'train' if self.training else 'eval'
        data = self._sidecar(cid)
        query = item['queries']['velocity']
        pool_rows = data[f'{mode}_rows']
        positions = np.searchsorted(pool_rows, query['rows'])
        if np.any(positions >= len(pool_rows)) or not np.array_equal(pool_rows[positions], query['rows']):
            raise ValueError('velocity query sampling no longer agrees with original row IDs')
        frame = np.asarray(data[f'{mode}_frame'][positions], np.float32)
        valid = np.asarray(data[f'{mode}_frame_valid'][positions], np.float32)
        query.update(frame=frame, frame_valid=valid,
                     geom_extra=np.column_stack((frame.reshape(-1, 9), valid)).astype(np.float32))
        if self.include_anchors:
            item['anchors'] = {k: np.array(v, copy=True) for k, v in data['anchors'].items()}
            item['anchors']['x'] = self._x(item['anchors']['x'], 'volume')
        return item


def collate_velocity_phase(items):
    out = base.collate_joint_cycle(items)
    flags = ['anchors' in item for item in items]
    if any(flags) and not all(flags):
        raise ValueError('cannot collate mixed anchor contracts')
    if all(flags):
        parts = [item['anchors'] for item in items]
        anchors = {k: torch.from_numpy(np.concatenate([a[k] for a in parts])) for k in ('pos', 'x', 'geom_extra', 'edge_attr')}
        offsets = np.cumsum([0] + [len(a['pos']) for a in parts[:-1]])
        anchors['edge_index'] = torch.from_numpy(np.concatenate([a['edge_index'] + offset for a, offset in zip(parts, offsets)], axis=1))
        anchors['batch'] = torch.cat([torch.full((len(a['pos']),), i, dtype=torch.long) for i, a in enumerate(parts)])
        out['anchors'] = anchors
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'stats', 'verify'))
    parser.add_argument('--cache-root', type=Path, default=base.DEFAULT_CACHE)
    parser.add_argument('--geometry-root', type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument('--split-path', type=Path, default=base.SPLIT_PATH)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    if args.config:
        cfg = json.loads(args.config.read_text())
        args.cache_root = Path(cfg['data']['cache_root'])
        args.geometry_root = Path(cfg['data']['geometry_root'])
        args.split_path = Path(cfg['data']['split_path'])
        args.seed = int(cfg['seed'])
    base.SPLIT_PATH = args.split_path.resolve()
    split = json.loads(base.SPLIT_PATH.read_text())
    if len(split['train_cases']) != 206 or len(split['test_cases']) != 55 or len(set(split['train_cases'] + split['test_cases'])) != 261:
        raise ValueError('this experiment requires the frozen 206/55 (261-unit) split')
    if args.command == 'verify':
        validate_geometry_manifest(args.geometry_root, args.cache_root)
        print('velocity geometry verified: 206 train / 55 holdout', flush=True)
        return
    if args.command == 'stats':
        fit_tail_stats(args.cache_root, args.geometry_root)
        return
    ids = split['train_cases'] + split['test_cases']
    selected = ids[:args.limit] if args.limit else ids
    args.geometry_root.mkdir(parents=True, exist_ok=True)
    records = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(prepare_case, cid, args.cache_root, args.geometry_root, seed=args.seed): cid for cid in selected}
        for i, future in enumerate(as_completed(jobs)):
            cid = jobs[future]
            records[cid] = future.result()
            print(f'geometry {i + 1}/{len(selected)} {cid} passed ({records[cid]["seconds"]:.1f}s)', flush=True)
    complete = selected == ids
    if complete:
        fit_tail_stats(args.cache_root, args.geometry_root)
    manifest = {'schema': SCHEMA, 'status': 'passed' if complete else 'partial', 'complete': complete,
        'cache_root': str(args.cache_root.resolve()), 'split_path': str(base.SPLIT_PATH),
        'split_sha256': base.sha256(base.SPLIT_PATH), 'seed': args.seed, 'case_ids': selected,
        'train_ids': split['train_cases'], 'test_ids': split['test_cases'],
        'cases': {cid: {'file_sha256': records[cid]['file_sha256'],
                        'metadata_sha256': base.sha256(args.geometry_root / 'cases' / cid / 'metadata.json'),
                        'original_query_pools': records[cid]['original_query_pools'],
                        'anchor_count': records[cid]['anchors']['count']} for cid in selected},
        'tail_stats_sha256': base.sha256(args.geometry_root / 'tail_stats.json') if complete else None,
        'source_reads': 'static geometry and original query rows; train labels used only for q80 supervision thresholds',
        'original_cache_mutated': False}
    _json(args.geometry_root / 'manifest.json', manifest)
    validate_geometry_manifest(args.geometry_root, args.cache_root, require_complete=complete)
    print(json.dumps({'status': manifest['status'], 'complete': complete, 'cases': len(selected)}), flush=True)


if __name__ == '__main__':
    main()
