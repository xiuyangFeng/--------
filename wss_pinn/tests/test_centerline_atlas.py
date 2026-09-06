"""Synthetic regression tests for the unique-segment centerline feature atlas."""

from __future__ import annotations

import numpy as np
import pytest

from wss_pinn.v4.centerline_atlas import (
    ATLAS_SAMPLE_COLUMNS,
    CenterlineGraph,
    build_feature_atlas,
    curvature_from_derivatives,
    decompose_tree,
    smooth_and_differentiate,
)


def _chain_edges(indices: list[int]) -> list[list[int]]:
    return [[a, b] for a, b in zip(indices[:-1], indices[1:])]


def _y_tree(
    trunk: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    sub_left: tuple[np.ndarray, np.ndarray],
    sub_right: tuple[np.ndarray, np.ndarray],
    radius: float = 5.0,
) -> CenterlineGraph:
    """Build a 5-endpoint / 3-junction tree; each branch array excludes its start node."""

    points = [trunk]
    edges = []
    offset = len(trunk)
    edges += _chain_edges(list(range(len(trunk))))
    junction0 = len(trunk) - 1

    def attach(start: int, branch: np.ndarray) -> int:
        nonlocal offset
        ids = [start] + list(range(offset, offset + len(branch)))
        points.append(branch)
        edges.extend(_chain_edges(ids))
        offset += len(branch)
        return ids[-1]

    j_left = attach(junction0, left)
    j_right = attach(junction0, right)
    leaves = [
        attach(j_left, sub_left[0]),
        attach(j_left, sub_left[1]),
        attach(j_right, sub_right[0]),
        attach(j_right, sub_right[1]),
    ]
    coords = np.vstack(points)
    endpoint_opening = {0: {"opening_id": 0, "role": "inlet", "outlet_id": -1, "outlet_name": ""}}
    for name, leaf in zip(("out-le", "out-li", "out-re", "out-ri"), leaves):
        endpoint_opening[leaf] = {"opening_id": leaf, "role": "outlet", "outlet_id": 0, "outlet_name": name}
    return CenterlineGraph(
        points_mm=coords,
        radius_mm=np.full(len(coords), radius),
        edges=np.asarray(edges, dtype=np.int64),
        root=0,
        endpoint_opening=endpoint_opening,
    )


def _line(start: np.ndarray, direction: np.ndarray, length: float, step: float, include_start: bool) -> np.ndarray:
    count = int(round(length / step))
    s = np.arange(0 if include_start else 1, count + 1) * step
    return start[None, :] + s[:, None] * direction[None, :] / np.linalg.norm(direction)


def _arc(radius: float, angle: float, step: float, include_start: bool) -> np.ndarray:
    count = int(round(radius * angle / step))
    theta = np.arange(0 if include_start else 1, count + 1) * (angle / count)
    return np.column_stack([radius * np.sin(theta), radius * (1.0 - np.cos(theta)), np.zeros_like(theta)])


def test_straight_line_has_zero_curvature():
    s = np.arange(0.0, 60.0 + 1e-9, 0.5)
    coords = np.column_stack([s, 0.3 * s, -0.1 * s])
    smoothed, first, second = smooth_and_differentiate(coords, s)
    _, curvature = curvature_from_derivatives(first, second)
    assert np.max(np.abs(smoothed - coords)) < 1e-9
    assert np.max(curvature) < 1e-9


def test_circular_arc_curvature_mid_and_endpoints():
    radius = 20.0
    angle = np.pi / 2
    coords = _arc(radius, angle, 0.5, include_start=True)
    s = np.arange(len(coords)) * (radius * angle / (len(coords) - 1))
    smoothed, first, second = smooth_and_differentiate(coords, s, step=float(s[1] - s[0]))
    _, curvature = curvature_from_derivatives(first, second)
    expected = 1.0 / radius
    mid = slice(10, len(curvature) - 10)
    mid_error = float(np.max(np.abs(curvature[mid] - expected)) / expected)
    end_error = float(max(abs(curvature[0] - expected), abs(curvature[-1] - expected)) / expected)
    assert mid_error < 2e-3, mid_error
    assert end_error < 2e-2, end_error


def test_y_tree_decomposes_into_seven_unique_segments():
    trunk = _line(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), 80.0, 0.5, include_start=True)
    j0 = trunk[-1]
    left = _line(j0, np.array([-1.0, 0.0, -1.0]), 40.0, 0.5, include_start=False)
    right = _line(j0, np.array([1.0, 0.0, -1.0]), 40.0, 0.5, include_start=False)
    sub_left = (
        _line(left[-1], np.array([-1.0, 0.2, -1.0]), 30.0, 0.5, include_start=False),
        _line(left[-1], np.array([-0.2, 0.0, -1.0]), 30.0, 0.5, include_start=False),
    )
    sub_right = (
        _line(right[-1], np.array([1.0, 0.2, -1.0]), 30.0, 0.5, include_start=False),
        _line(right[-1], np.array([0.2, 0.0, -1.0]), 30.0, 0.5, include_start=False),
    )
    graph = _y_tree(trunk, left, right, sub_left, sub_right)
    segments = decompose_tree(graph)
    assert len(segments) == 7
    assert sum(segment.starts_at_root for segment in segments) == 1
    assert sum(segment.ends_at_leaf for segment in segments) == 4
    covered = sorted(node for segment in segments for node in segment.nodes[1:])
    assert covered == sorted(range(1, len(graph.points_mm)))  # every non-root node appears exactly once
    atlas = build_feature_atlas("synthetic/y", graph)
    rows = atlas.unique_rows()
    curvature = rows[:, ATLAS_SAMPLE_COLUMNS.index("curvature_per_mm")]
    assert np.isfinite(curvature).all()
    assert np.max(curvature) < 1e-6  # straight branches: junction bends never leak across segments
    root_segment = [s for s in atlas.segments if s.starts_at_root][0]
    assert abs(root_segment.length_mm - 80.0) < 1e-6
    max_s = atlas.max_s_from_root()
    assert abs(max_s - 150.0) < 1e-6
    # child duplicates are excluded from the flattened atlas
    assert int(np.sum(atlas.table[:, ATLAS_SAMPLE_COLUMNS.index("is_child_duplicate")])) == 6
    mapped = atlas.map_points(np.array([[0.0, 0.0, -10.0], [0.0, 0.0, -80.0]]))
    assert abs(mapped["abscissa_norm"][0] - 10.0 / 150.0) < 1e-9
    assert mapped["dist_to_junction_mm"][1] == pytest.approx(0.0)


def test_short_segment_uses_reduced_polynomial():
    s = np.array([0.0, 0.5, 1.0])
    coords = np.column_stack([s, np.zeros(3), np.zeros(3)])
    smoothed, first, second = smooth_and_differentiate(coords, s)
    assert np.allclose(smoothed, coords)
    assert np.allclose(first[:, 0], 1.0)
    assert np.allclose(second, 0.0)


# ---------------------------------------------------------------------------
# radius-adaptive window (frozen 2026-09-05)

from wss_pinn.v4.centerline_atlas import (  # noqa: E402
    SG_WINDOW,
    detect_kinks,
    smooth_and_differentiate_adaptive,
)


def test_adaptive_window_equals_fixed_window_for_normal_radius():
    s = np.arange(0.0, 80.0 + 1e-9, 0.5)
    coords = np.column_stack([s, 0.02 * s**2, np.sin(s / 8.0)])
    fixed = smooth_and_differentiate(coords, s)
    adaptive = smooth_and_differentiate_adaptive(coords, s, np.full(len(s), 5.0))
    for a, b in zip(fixed, adaptive[:3]):
        assert np.max(np.abs(a - b)) < 1e-12
    assert set(np.unique(adaptive[3]).tolist()) == {SG_WINDOW}


def test_adaptive_window_keeps_arc_curvature_through_a_wide_lumen():
    rho, angle = 40.0, 2.0 * np.pi / 3.0
    length = rho * angle
    count = int(np.ceil(length / 0.5)) + 1
    s = np.linspace(0.0, length, count)
    theta = s / rho
    coords = np.column_stack([rho * np.sin(theta), rho * (1.0 - np.cos(theta)), np.zeros_like(theta)])
    radius = 5.0 + 15.0 * np.exp(-(((s - length / 2.0) / 12.0) ** 2))  # 5 mm vessel with a 20 mm bulge
    _, first, second, windows = smooth_and_differentiate_adaptive(coords, s, radius, step=float(s[1] - s[0]))
    _, curvature = curvature_from_derivatives(first, second)
    error = np.abs(curvature - 1.0 / rho) * rho
    assert int(windows.min()) == SG_WINDOW and int(windows.max()) > 31
    assert float(error[10:-10].max()) < 1e-2
    assert float(error[radius > 15.0].max()) < 1e-2
    assert float(max(error[0], error[-1])) < 2e-2


def _corner_tree(wide: bool) -> CenterlineGraph:
    """Y tree whose trunk turns by 80 degrees half-way; optionally inside a 20 mm wide sac."""

    trunk_a = _line(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), 60.0, 0.5, include_start=True)
    corner = trunk_a[-1]
    turn = np.radians(80.0)
    trunk_b = _line(corner, np.array([np.sin(turn), 0.0, -np.cos(turn)]), 60.0, 0.5, include_start=False)
    trunk = np.vstack([trunk_a, trunk_b])
    j0 = trunk[-1]
    left = _line(j0, np.array([-1.0, 0.0, -1.0]), 40.0, 0.5, include_start=False)
    right = _line(j0, np.array([1.0, 0.0, -1.0]), 40.0, 0.5, include_start=False)
    sub_left = (
        _line(left[-1], np.array([-1.0, 0.2, -1.0]), 30.0, 0.5, include_start=False),
        _line(left[-1], np.array([-0.2, 0.0, -1.0]), 30.0, 0.5, include_start=False),
    )
    sub_right = (
        _line(right[-1], np.array([1.0, 0.2, -1.0]), 30.0, 0.5, include_start=False),
        _line(right[-1], np.array([0.2, 0.0, -1.0]), 30.0, 0.5, include_start=False),
    )
    graph = _y_tree(trunk, left, right, sub_left, sub_right, radius=5.0)
    if wide:
        near_corner = np.linalg.norm(graph.points_mm - corner, axis=1) < 25.0
        graph.radius_mm = np.where(near_corner, 20.0, 5.0)
    return graph


def test_wide_sac_corner_is_flattened_and_kink_gate_clears():
    narrow = build_feature_atlas("synthetic/corner-narrow", _corner_tree(wide=False))
    wide = build_feature_atlas("synthetic/corner-wide", _corner_tree(wide=True))
    trunk_narrow = [s for s in narrow.segments if s.starts_at_root][0]
    trunk_wide = [s for s in wide.segments if s.starts_at_root][0]
    # a 5 mm vessel keeps the frozen 11-sample window and the corner is a real (flagged) fold
    assert int(trunk_narrow.sg_window.max()) == SG_WINDOW
    assert float(trunk_narrow.curvature.max()) > 0.4
    assert detect_kinks(narrow)["interior"] >= 1
    # inside a 20 mm sac the window widens to ~40 samples and the fold is absorbed
    assert int(trunk_wide.sg_window.max()) >= 39
    assert float(trunk_wide.curvature.max()) < 0.25
    assert float(trunk_wide.curvature.max()) < 0.5 * float(trunk_narrow.curvature.max())
    assert detect_kinks(wide)["interior"] == 0
    # auxiliary columns are finite and describe the sac
    rows = wide.unique_rows()
    assert np.isfinite(rows).all()
    mapped = wide.map_points(np.array([[0.0, 0.0, -60.0]]))
    assert mapped["radius_over_case_median"][0] > 1.5
    assert mapped["sg_window_samples"][0] >= 39
    assert mapped["curvature_times_radius"][0] == pytest.approx(
        mapped["curvature_per_mm"][0] * mapped["local_radius_mm"][0]
    )
