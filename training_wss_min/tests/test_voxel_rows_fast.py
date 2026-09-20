"""The hashed-code voxel decimation must reproduce the original row-wise np.unique implementation exactly
(the §21.4 / §22 deployment tables were produced with the original)."""
import numpy as np

from training_wss_min.tools.deployment_resample_reeval import _voxel_rows_reference, voxel_rows


def _clouds():
    rng = np.random.default_rng(0)
    yield rng.standard_normal((5000, 3)) * 30.0
    t = rng.random(4000) * 2 * np.pi; z = rng.random(4000) * 120.0        # tube-like surface
    yield np.column_stack((8.0 * np.cos(t), 8.0 * np.sin(t), z)) + rng.standard_normal((4000, 3)) * 0.05
    yield rng.random((3000, 3)) * np.array([200.0, 3.0, 50.0])              # anisotropic extent


def test_voxel_rows_matches_reference():
    for xyz in _clouds():
        for frac in (0.7, 0.5, 0.25, 0.1, 0.03):
            for seed in (1, 20260913):
                a = _voxel_rows_reference(xyz, frac, seed)
                b = voxel_rows(xyz, frac, seed)
                assert np.array_equal(a, b), (frac, seed)


def test_voxel_rows_full_fraction_identity():
    xyz = np.random.default_rng(1).random((100, 3))
    assert np.array_equal(voxel_rows(xyz, 1.0, 0), np.arange(100))
