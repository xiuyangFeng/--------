from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_wss_min.config import RegistrationConfig, case_rel_path
from pipeline_wss_min.preprocess import _align_wall_fields_to_reference
from pipeline_wss_min.registration import compute_transform


def _ring(center, axis_z, radius=5.0, n=36):
    axis_z = axis_z / np.linalg.norm(axis_z)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(seed @ axis_z)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    u = seed - (seed @ axis_z) * axis_z
    u /= np.linalg.norm(u)
    v = np.cross(axis_z, u)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return center + radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)


def _synthetic_aorto_iliac():
    parts = []
    for z in np.linspace(0, 120, 40):
        parts.append(_ring(np.array([0.0, 0.0, z]), np.array([0.0, 0.0, 1.0]), 7.0))
    for sign in (-1.0, 1.0):
        direction = np.array([0.36 * sign, 0.0, -0.93])
        direction /= np.linalg.norm(direction)
        for t in np.linspace(0, 85, 34):
            parts.append(_ring(t * direction, direction, 5.0))
    return np.vstack(parts)


def _centerline(shift=None, reverse_abscissa=False):
    shift = np.zeros(3) if shift is None else np.asarray(shift, dtype=float)
    trunk = np.column_stack([np.zeros(28), np.zeros(28), np.linspace(120, 0, 28)])
    left = np.column_stack([
        np.linspace(0, -30, 18), np.zeros(18), np.linspace(0, -78, 18)
    ])
    right = np.column_stack([
        np.linspace(0, 30, 18), np.zeros(18), np.linspace(0, -78, 18)
    ])
    coords = np.vstack([trunk, left, right]) + shift
    # flow-divider 的三臂近零点必须在空间上分开，模拟 VMTK 的分叉邻域采样。
    dist = np.full(len(coords), 50.0)
    near = [26, 27, 28, 29, 45, 46, 47]
    dist[near] = 0.0
    abscissa = np.linspace(0, 1, len(coords))
    if reverse_abscissa:
        abscissa = abscissa[::-1]
    return {
        "coords": coords,
        "abscissa": abscissa,
        "radius": np.full(len(coords), 5.0),
        "curvature": np.zeros(len(coords)),
        "tangent": np.zeros_like(coords),
        "dist_to_bifurcation": dist,
    }


def test_stl_landmarks_put_trunk_up_and_iliac_branches_down():
    surface = _synthetic_aorto_iliac()
    transform = compute_transform(
        surface, surface, _centerline(reverse_abscissa=True),
        RegistrationConfig(), anatomy_pts=surface, anatomy_source="original_stl",
    )
    landmarks = transform.apply_points(np.vstack([
        transform.landmark_trunk_point,
        transform.landmark_left_point,
        transform.landmark_right_point,
    ]))
    assert landmarks[0, 2] > 0
    assert np.all(landmarks[1:, 2] < 0)
    assert transform.landmark_left_point[0] > transform.landmark_right_point[0]
    assert np.isclose(np.linalg.det(transform.rotation), 1.0, atol=1e-7)


def test_centerline_and_stl_translation_repair_is_translation_only():
    local_surface = _synthetic_aorto_iliac()
    shift = np.array([35.0, -210.0, 1250.0])
    wall_world = local_surface + shift
    transform = compute_transform(
        wall_world, wall_world, _centerline(), RegistrationConfig(),
        anatomy_pts=local_surface, anatomy_source="original_stl",
    )
    assert transform.centerline_translation_applied
    np.testing.assert_allclose(transform.centerline_translation_mm, shift, atol=8.0)
    aligned = transform.apply_points(wall_world)
    assert np.quantile(aligned[:, 2], 0.95) > 0
    assert np.quantile(aligned[:, 2], 0.05) < 0
    assert np.linalg.det(transform.rotation) > 0.999999


def test_centerline_translation_guard_rejects_shape_mismatch():
    surface = _synthetic_aorto_iliac()
    bad = _centerline()
    bad["coords"] = bad["coords"] * 3.0 + np.array([600.0, -900.0, 1500.0])
    transform = compute_transform(
        surface, surface, bad, RegistrationConfig(),
        anatomy_pts=surface, anatomy_source="original_stl",
    )
    assert transform.centerline_translation_candidate
    assert not transform.centerline_translation_applied


def test_override_key_supports_nested_and_single_segment_cohorts():
    assert case_rel_path("AG/fast", "CASE") == "fast/CASE"
    assert case_rel_path("AAA/ruputer", "CASE") == "ruputer/CASE"
    assert case_rel_path("ILO", "PATIENT-0/before") == "ILO/PATIENT-0/before"


def test_wall_alignment_drops_only_tiny_extra_set():
    ref = np.array([1, 2, 3], dtype=np.int64)
    fields = {
        "nodenumber": np.array([4, 3, 1, 2], dtype=np.int64),
        "wss": np.array([40.0, 30.0, 10.0, 20.0]),
    }
    aligned, reordered, n_extra = _align_wall_fields_to_reference(fields, ref, 1120)
    assert reordered and n_extra == 1
    np.testing.assert_array_equal(aligned["nodenumber"], ref)
    np.testing.assert_allclose(aligned["wss"], [10.0, 20.0, 30.0])


if __name__ == "__main__":
    test_stl_landmarks_put_trunk_up_and_iliac_branches_down()
    test_centerline_and_stl_translation_repair_is_translation_only()
    test_centerline_translation_guard_rejects_shape_mismatch()
    test_override_key_supports_nested_and_single_segment_cohorts()
    test_wall_alignment_drops_only_tiny_extra_set()
    print("registration_v4 synthetic tests: PASS")
