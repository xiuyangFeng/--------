"""New sidecars must preserve the established sampling/label contract."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from training_wss_min import joint_cycle_data as base
from training_wss_min import velocity_phase_data as data


def test_frames_rotation_handedness_and_explicit_invalid_fallback():
    r = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    t = np.array([[1., 0, 0], [1., 0, 0]])
    n = np.array([[0., 1, 0], [1., 0, 0]])
    b = np.array([[0., 0, 1], [0., 0, 0]])
    frames, valid = data.aligned_frames(t, n, b, r)
    np.testing.assert_array_equal(valid, [1, 0])
    np.testing.assert_allclose(frames[0], r)
    np.testing.assert_array_equal(frames[1], np.eye(3))
    coeff = np.array([.2, -.7, .4])
    np.testing.assert_allclose(frames[0] @ coeff, coeff @ r.T)
    assert np.linalg.det(frames[0]) == pytest.approx(1.)


def test_graph_preserves_parent_child_bridges_without_crossing_siblings():
    # Dense same-segment nearest neighbors would otherwise sever the tree.
    pos = np.array([[x, 0, 0] for x in (0., .1, .2)] +
                   [[x, 0, 0] for x in (10., 10.1, 10.2)] +
                   [[x, 1, 0] for x in (10., 10.1, 10.2)])
    sid = np.repeat([0, 1, 2], 3)
    edge, attr, audit = data.build_branch_graph(pos, sid, pos[:, 0], {0: -1, 1: 0, 2: 0}, neighbors=2)
    target, source = edge
    assert edge.shape == (2, 18)
    np.testing.assert_array_equal(np.bincount(target), np.full(9, 2))
    assert not np.any(((sid[target] == 1) & (sid[source] == 2)) | ((sid[target] == 2) & (sid[source] == 1)))
    pairs = set(map(tuple, edge.T))
    for p, c in audit['parent_child_bridges']:
        assert (p, c) in pairs and (c, p) in pairs
    np.testing.assert_allclose(attr[:, :3], pos[source] - pos[target], atol=1e-6)
    np.testing.assert_allclose(attr[:, 4], pos[source, 0] - pos[target, 0], atol=1e-6)


def test_weighted_tail_uses_case_mass_not_point_count():
    # One ten-point case and one single-point case have equal case mass.
    value = np.r_[np.zeros(10), 10.]
    mass = np.r_[np.full(10, .05), .5]
    assert data.weighted_quantile(value, mass) == 10.
    assert data.weighted_quantile(value, np.ones(11)) == 0.
    with pytest.raises(ValueError):
        data.weighted_quantile(value, -mass)


def test_closed_surface_caps_only_geometry_and_rejects_extra_holes():
    pytest.importorskip('pyvista')
    points = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                       [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    sides = np.array([[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    atlas = SimpleNamespace(endpoints=lambda: [
        {'center_mm': [0, 0, -1], 'label': 'inlet', 'radius_mm': 1.},
        {'center_mm': [0, 0, 1], 'label': 'outlet', 'radius_mm': 1.}])
    wall, closed, audit = data._closed_surface(points, sides, atlas)
    assert audit['remaining_boundary_or_nonmanifold_edges'] == 0
    np.testing.assert_array_equal(data._contains(closed, np.array([[0, 0, 0], [2, 0, 0], [0, 0, 2]])), [True, False, False])
    np.testing.assert_array_equal(wall.points, points)
    with pytest.raises(ValueError, match='opening loops'):
        data._closed_surface(points, sides, SimpleNamespace(endpoints=lambda: []))


@pytest.fixture
def contract_fixture(tmp_path, monkeypatch):
    cid = 'fixture/a'
    cache, geom = tmp_path / 'cache', tmp_path / 'geometry'
    source, side = cache / 'cases' / cid, geom / 'cases' / cid
    source.mkdir(parents=True)
    side.mkdir(parents=True)
    split_path = tmp_path / 'split.json'
    split_path.write_text(json.dumps({'train_cases': [cid], 'test_cases': []}))
    monkeypatch.setattr(base, 'SPLIT_PATH', split_path)
    stats = {'split_sha256': base.sha256(split_path), 'complete_train_partition': True,
        'bc': {'mean': [0.] * 20, 'std': [1.] * 20},
        'features': {name: {'clip': [None] * d, 'mean': [0.] * d, 'std': [1.] * d} for name, d in [('volume', 20), ('wall', 27)]},
        'targets': {t: {'mean': [[0.] * d] * 80, 'std': [[1.] * d] * 80, 'transform': 'linear'}
                    for t, d in [('velocity', 3), ('pressure', 1), ('wss', 1)]}}
    (cache / 'stats.json').write_text(json.dumps(stats))
    (source / 'metadata.json').write_text('{}')
    rng = np.random.default_rng(7)
    arrays = {'support_pos': rng.normal(size=(12, 3)).astype('f4'),
              'support_x': rng.normal(size=(12, 27)).astype('f4'), 'phase': np.zeros((80, 4), 'f4'), 'bc': np.zeros(20, 'f4')}
    for mode, offset in [('train', 0), ('eval', 100)]:
        for domain, d in [('volume', 20), ('wall', 27)]:
            arrays.update({f'{mode}_{domain}_pos': rng.normal(size=(10, 3)).astype('f4'),
                f'{mode}_{domain}_x': rng.normal(size=(10, d)).astype('f4'),
                f'{mode}_{domain}_weight': np.ones(10, 'f4'), f'{mode}_{domain}_rows': np.arange(10) + offset})
        for task, d in [('velocity', 3), ('pressure', 1), ('wss', 1)]:
            arrays[f'{mode}_{task}'] = rng.normal(size=(10, 80, d)).astype('f4')
        np.save(side / f'{mode}_rows.npy', np.arange(10) + offset)
        np.save(side / f'{mode}_frame.npy', np.tile(np.eye(3, dtype='f4'), (10, 1, 1)))
        np.save(side / f'{mode}_frame_valid.npy', np.ones(10, 'f4'))
    for name, a in arrays.items():
        np.save(source / f'{name}.npy', a)
    np.savez(side / 'anchors.npz', pos=np.zeros((2, 3), 'f4'), x=np.zeros((2, 20), 'f4'),
        geom_extra=np.zeros((2, 10), 'f4'), edge_index=np.array([[0, 1], [1, 0]]), edge_attr=np.zeros((2, 7), 'f4'))
    manifest = {'schema': data.SCHEMA, 'status': 'partial', 'complete': False, 'cache_root': str(cache),
        'split_sha256': base.sha256(split_path), 'case_ids': [cid],
        'cases': {cid: {'file_sha256': {p.name: base.sha256(p) for p in side.iterdir()}}}}
    (geom / 'manifest.json').write_text(json.dumps(manifest))
    return cache, geom


@pytest.mark.parametrize('training', [True, False])
def test_dataset_preserves_old_labels_weights_and_epoch_streams(contract_fixture, training):
    cache, geom = contract_fixture
    old = base.JointCycleDataset(cache, support_n=5, query_n=4, training=training)
    new = data.VelocityPhaseDataset(cache, support_n=5, query_n=4, training=training,
        geometry_root=geom, include_anchors=True, allow_partial_geometry=True)
    for epoch in [0, 1, 9]:
        old.set_epoch(epoch)
        new.set_epoch(epoch)
        a, b = old[0], new[0]
        for task in base.TASKS:
            for key, value in a['queries'][task].items():
                np.testing.assert_array_equal(b['queries'][task][key], value)
        for key, value in a['support'].items():
            np.testing.assert_array_equal(b['support'][key], value)
        assert b['queries']['velocity']['geom_extra'].shape == (4, 10)
    batch = data.collate_velocity_phase([new[0], new[0]])
    assert batch['queries']['velocity']['frame'].shape == (8, 3, 3)
    torch.testing.assert_close(batch['anchors']['edge_index'], torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]))
    torch.testing.assert_close(batch['anchors']['batch'], torch.tensor([0, 0, 1, 1]))


def test_dataset_rejects_modified_sidecar(contract_fixture):
    cache, geom = contract_fixture
    new = data.VelocityPhaseDataset(cache, geometry_root=geom, allow_partial_geometry=True)
    np.save(geom / 'cases/fixture/a/train_rows.npy', np.arange(10) + 1)
    with pytest.raises(ValueError, match='geometry changed'):
        new[0]
