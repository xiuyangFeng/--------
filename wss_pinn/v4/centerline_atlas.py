"""Unique-branch-segment centerline feature atlas (Centerline V2 -> V4 geometry).

The 2026-08-31 staging mapped volume points onto four root->outlet paths whose
derivatives were computed per path with ``np.gradient``.  The shared trunk was
therefore differentiated four times, the same trunk point received different
curvature values depending on the downstream outlet, one-sided differences
created endpoint spikes, and unsmoothed second differences amplified the
0.5 mm polyline jitter.

This module replaces that with a *feature atlas* built from the explicit V2
graph (``centerline_graph_mm.vtp``):

* the tree is split into its unique physical edges (seven segments for the
  frozen 5-endpoint / 3-junction topology); every edge is stored once;
* each segment is resampled on a strict 0.5 mm arc-length grid;
* the radius is fitted locally with a Savitzky-Golay window of 11 samples and
  polynomial order 3; the coordinates use the same cubic SG operator with a
  **radius-adaptive window** (frozen 2026-09-05): the window length in mm is
  about the local smoothed radius, bounded to 11..61 samples.  A tube cannot
  bend on a scale shorter than its own radius (``k * R <= 1``), so normal
  vessels (R <= 5.5 mm) keep the 11-sample window while an aneurysm sac with
  R = 22 mm is fitted over ~44 samples; this removes the isolated VMTK path
  folds inside wide sacs without extra smoothing of iliac bends.  Tangent and
  curvature come from the fitted first/second derivatives
  (``|r' x r''| / |r'|^3``);
* segment ends are padded by extrapolating the local cubic fit before the same
  SG operator is applied, so no one-sided difference is ever used;
* the trunk end that terminates in a junction is the *owner* of that junction
  sample; child segments inherit the junction as their first sample but that
  duplicate is excluded from the flattened atlas, so the flattened atlas has no
  overlaps and no gaps;
* every sample carries raw/smoothed coordinates, tangent, curvature, radius,
  segment/parent ids, junction/endpoint masks, along-tree distances to the
  nearest junction and endpoint, and the fit residual.

Units: the V2 products are stored in the legacy "mm" frame, which is the Fluent
metre frame scaled by ``fluent_to_mm_unit_factor`` (per-case, 810-1088 rather
than 1000).  ``load_v2_graph`` therefore accepts ``unit_rescale`` so callers can
express the atlas in true millimetres (``1000 / fluent_to_mm_unit_factor``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_coeffs
from scipy.spatial import cKDTree

PATH_STEP_MM = 0.5
SG_WINDOW = 11
SG_POLYORDER = 3
SG_HALF_WINDOW_MIN = 5  # 11 samples = 5.5 mm: the frozen fixed window
SG_HALF_WINDOW_MAX = 30  # 61 samples = 30.5 mm
SG_WINDOW_RADIUS_FACTOR = 1.0  # coordinate SG window length (mm) ~= factor x local smoothed radius
KINK_RATIO_TO_SEGMENT_MEDIAN = 8.0
KINK_CURVATURE_FLOOR_PER_MM = 0.3
# Frozen 2026-09-06: within about one local radius of an opening endpoint or of a
# child-segment start, tangent and curvature are *held* at the robust value of the
# adjacent interior band (1R..2R): VMTK paths hook towards the cap centre at
# openings and towards the parent axis where a branch departs; the hook spans
# 1-3 samples but reads as curvature 0.4-0.55 /mm.  Coordinates are never moved
# (a coordinate extrapolation was tried first and displaced samples by up to
# 36 mm where the junction sphere of an AAA sac made the zone 20 mm long).  The
# zone radius is min(radius at the end, segment median radius) and at least
# SG_HALF_WINDOW_MIN + 1 samples so the SG footprint of the hook is covered.  The
# parent's own junction end is never touched (it is shared by every path).
END_ZONE_RADIUS_FACTOR = 1.0
END_ZONE_MIN_SAMPLES = SG_HALF_WINDOW_MIN + 1
END_ZONE_MAX_RADIUS_MM = 12.0  # the hook spans 1-3 samples; beyond ~12 mm a radius-proportional zone eats real geometry (AAA sacs)
END_BAND_MIN_SAMPLES = SG_WINDOW
JUNCTION_HALF_WIDTH_MM = 2.5
ENDPOINT_HALF_WIDTH_MM = 2.5
EXPECTED_ENDPOINTS = 5
EXPECTED_JUNCTIONS = 3


# ---------------------------------------------------------------------------
# graph loading


@dataclass
class CenterlineGraph:
    points_mm: np.ndarray  # (N, 3)
    radius_mm: np.ndarray  # (N,)
    edges: np.ndarray  # (E, 2) int
    root: int
    endpoint_opening: dict[int, dict[str, Any]]  # graph node -> opening record
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def adjacency(self) -> dict[int, list[int]]:
        adjacency: dict[int, list[int]] = {index: [] for index in range(len(self.points_mm))}
        for a, b in self.edges.tolist():
            adjacency[a].append(b)
            adjacency[b].append(a)
        return adjacency

    @property
    def degrees(self) -> np.ndarray:
        degree = np.zeros(len(self.points_mm), dtype=np.int64)
        for a, b in self.edges.tolist():
            degree[a] += 1
            degree[b] += 1
        return degree


def _read_vtp_polylines(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], list[list[int]]]:
    import vtk  # local import: only the atlas builder needs VTK
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    arrays = {}
    point_data = poly.GetPointData()
    for index in range(point_data.GetNumberOfArrays()):
        name = point_data.GetArrayName(index)
        arrays[name] = vtk_to_numpy(point_data.GetArray(index))
    lines = []
    cells = poly.GetLines()
    cells.InitTraversal()
    id_list = vtk.vtkIdList()
    while cells.GetNextCell(id_list):
        lines.append([int(id_list.GetId(k)) for k in range(id_list.GetNumberOfIds())])
    return points, arrays, lines


def load_v2_graph(
    case_dir: str | Path,
    *,
    unit_rescale: float = 1.0,
    apply_cfd_translation: bool = True,
) -> CenterlineGraph:
    """Load a Centerline V2 case directory into an explicit graph.

    ``unit_rescale`` multiplies all lengths (coordinates, radii, translation).
    Use ``1000 / fluent_to_mm_unit_factor`` to obtain true millimetres in the
    Fluent frame.
    """

    case_dir = Path(case_dir)
    result = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
    selection = json.loads((case_dir / "surface_selection.json").read_text(encoding="utf-8"))
    if not bool(result.get("hard_pass")):
        raise ValueError(f"{case_dir}: Centerline V2 hard Gate did not pass")
    points, arrays, lines = _read_vtp_polylines(case_dir / "centerline_graph_mm.vtp")
    if "MaximumInscribedSphereRadius" not in arrays:
        raise KeyError(f"{case_dir}: graph VTP lacks MaximumInscribedSphereRadius")
    edges = []
    for line in lines:
        if len(line) != 2:
            raise ValueError(f"{case_dir}: graph VTP contains a non-edge cell of size {len(line)}")
        edges.append(line)
    edges_array = np.asarray(edges, dtype=np.int64)
    shift = np.asarray(selection.get("geometry_translation_to_cfd_wall_mm", [0.0, 0.0, 0.0]), dtype=np.float64)
    scale = float(unit_rescale)
    coords = points * scale
    if apply_cfd_translation:
        coords = coords + shift * scale
    radius = np.asarray(arrays["MaximumInscribedSphereRadius"], dtype=np.float64) * scale
    metrics = result["metrics"]
    root = int(metrics["root_node"])
    openings_by_id = {
        int(row["opening_id"]): row for row in selection["opening_assignment"]["openings"]
    }
    endpoint_opening = {}
    for node, opening_id in metrics["endpoint_to_opening"].items():
        row = dict(openings_by_id[int(opening_id)])
        endpoint_opening[int(node)] = {
            "opening_id": int(opening_id),
            "role": row["role"],
            "outlet_id": int(row.get("outlet_id", -1)),
            "outlet_name": str(row.get("outlet_name", "")),
        }
    if endpoint_opening.get(root, {}).get("role") != "inlet":
        raise ValueError(f"{case_dir}: root node is not the inlet endpoint")
    return CenterlineGraph(
        points_mm=coords,
        radius_mm=radius,
        edges=edges_array,
        root=root,
        endpoint_opening=endpoint_opening,
        provenance={
            "case_dir": str(case_dir.resolve()),
            "unit_rescale": scale,
            "fluent_to_mm_unit_factor": float(selection["fluent_to_mm_unit_factor"]),
            "authoritative_stl_scale_to_mm": float(selection["authoritative_stl_scale_to_mm"]),
            "geometry_translation_applied": bool(apply_cfd_translation),
            "geometry_translation_mm_legacy": shift.tolist(),
            "status": result["status"],
        },
    )


# ---------------------------------------------------------------------------
# tree decomposition


@dataclass
class RawSegment:
    segment_id: int
    parent_id: int
    nodes: list[int]  # graph nodes from start (root or junction) to end (junction or leaf)
    starts_at_root: bool
    ends_at_leaf: bool
    leaf_opening: dict[str, Any] | None
    descendant_outlets: list[str]


def decompose_tree(graph: CenterlineGraph) -> list[RawSegment]:
    """Split the tree into unique physical edges (root->junction, junction->...)."""

    adjacency = graph.adjacency
    degree = graph.degrees
    endpoints = [index for index, value in enumerate(degree) if value == 1]
    junctions = [index for index, value in enumerate(degree) if value >= 3]
    if len(endpoints) != EXPECTED_ENDPOINTS or len(junctions) != EXPECTED_JUNCTIONS:
        raise ValueError(
            f"graph has {len(endpoints)} endpoints / {len(junctions)} junctions; "
            f"expected {EXPECTED_ENDPOINTS}/{EXPECTED_JUNCTIONS}"
        )
    if np.any(degree > 3):
        raise ValueError("graph contains a node of degree > 3; not a binary tree")
    connected = int(np.sum(degree > 0))
    isolated = int(np.sum(degree == 0))
    if len(graph.edges) != connected - 1:
        raise ValueError("graph is not a tree (edge count != connected node count - 1)")

    segments: list[RawSegment] = []
    visited_nodes = {graph.root}
    stack = [(graph.root, -1)]
    while stack:
        start, parent_id = stack.pop()
        for neighbour in sorted(adjacency[start]):
            if neighbour in visited_nodes:
                continue
            chain = [start, neighbour]
            visited_nodes.add(neighbour)
            current = neighbour
            previous = start
            while degree[current] == 2:
                nxt = [n for n in adjacency[current] if n != previous][0]
                if nxt in visited_nodes:
                    raise ValueError("graph traversal revisited a node; cycle detected")
                chain.append(nxt)
                visited_nodes.add(nxt)
                previous, current = current, nxt
            segment_id = len(segments)
            ends_at_leaf = bool(degree[current] == 1)
            segments.append(
                RawSegment(
                    segment_id=segment_id,
                    parent_id=parent_id,
                    nodes=chain,
                    starts_at_root=start == graph.root,
                    ends_at_leaf=ends_at_leaf,
                    leaf_opening=graph.endpoint_opening.get(current) if ends_at_leaf else None,
                    descendant_outlets=[],
                )
            )
            if not ends_at_leaf:
                stack.append((current, segment_id))
    if len(segments) != EXPECTED_ENDPOINTS + EXPECTED_JUNCTIONS - 1:
        raise ValueError(f"tree decomposed into {len(segments)} segments; expected 7")
    if len(visited_nodes) != connected:
        raise ValueError("tree decomposition did not visit every connected graph node")
    if isolated:
        graph.provenance["isolated_graph_points_ignored"] = isolated
    # descendant outlets (semantic label of every segment)
    for segment in reversed(segments):
        names = []
        if segment.leaf_opening is not None:
            names.append(str(segment.leaf_opening["outlet_name"]))
        for child in segments:
            if child.parent_id == segment.segment_id:
                names.extend(child.descendant_outlets)
        segment.descendant_outlets = sorted(names)
    return segments


# ---------------------------------------------------------------------------
# smoothing / differentiation


def _sg_matrices(window: int, order: int, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        savgol_coeffs(window, order, deriv=0, delta=step, use="dot"),
        savgol_coeffs(window, order, deriv=1, delta=step, use="dot"),
        savgol_coeffs(window, order, deriv=2, delta=step, use="dot"),
    )


def _extrapolate_pad(values: np.ndarray, s: np.ndarray, pad: int, order: int, step: float) -> np.ndarray:
    """Pad both ends by evaluating a local polynomial fit of ``order`` outside the data."""

    count = len(values)
    fit_points = min(count, max(SG_WINDOW, order + 1))
    head_s = s[:fit_points]
    tail_s = s[-fit_points:]
    left = np.empty((pad, values.shape[1]))
    right = np.empty((pad, values.shape[1]))
    left_s = s[0] - step * np.arange(pad, 0, -1)
    right_s = s[-1] + step * np.arange(1, pad + 1)
    local_order = min(order, fit_points - 1)
    for axis in range(values.shape[1]):
        coefficients = np.polyfit(head_s - s[0], values[:fit_points, axis], local_order)
        left[:, axis] = np.polyval(coefficients, left_s - s[0])
        coefficients = np.polyfit(tail_s - s[-1], values[-fit_points:, axis], local_order)
        right[:, axis] = np.polyval(coefficients, right_s - s[-1])
    return np.vstack([left, values, right])


def smooth_and_differentiate(
    values: np.ndarray,
    s: np.ndarray,
    *,
    step: float = PATH_STEP_MM,
    window: int = SG_WINDOW,
    order: int = SG_POLYORDER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(smoothed, first derivative, second derivative)`` on a uniform grid.

    Short segments (fewer samples than the window) are fitted with a single
    polynomial of reduced order; otherwise the ends are extrapolated with a local
    cubic and the full symmetric SG window is applied everywhere.
    """

    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    count = len(values)
    if count < window:
        local_order = min(order, count - 1)
        smoothed = np.empty_like(values)
        first = np.empty_like(values)
        second = np.empty_like(values)
        for axis in range(values.shape[1]):
            coefficients = np.polyfit(s - s[0], values[:, axis], local_order)
            smoothed[:, axis] = np.polyval(coefficients, s - s[0])
            first[:, axis] = np.polyval(np.polyder(coefficients, 1), s - s[0]) if local_order >= 1 else 0.0
            second[:, axis] = np.polyval(np.polyder(coefficients, 2), s - s[0]) if local_order >= 2 else 0.0
        return smoothed, first, second
    half = window // 2
    padded = _extrapolate_pad(values, s, half, order, step)
    c0, c1, c2 = _sg_matrices(window, order, step)
    smoothed = np.empty_like(values)
    first = np.empty_like(values)
    second = np.empty_like(values)
    for axis in range(values.shape[1]):
        column = padded[:, axis]
        windows = np.lib.stride_tricks.sliding_window_view(column, window)
        smoothed[:, axis] = windows @ c0
        first[:, axis] = windows @ c1
        second[:, axis] = windows @ c2
    return smoothed, first, second


def adaptive_half_window(
    radius_mm: np.ndarray,
    step: float,
    *,
    factor: float = SG_WINDOW_RADIUS_FACTOR,
    minimum: int = SG_HALF_WINDOW_MIN,
    maximum: int = SG_HALF_WINDOW_MAX,
) -> np.ndarray:
    """Per-sample SG half window (samples) such that the window length is ~``factor x radius``."""

    half = np.rint(factor * np.asarray(radius_mm, dtype=np.float64) / (2.0 * step)).astype(np.int64)
    return np.clip(half, int(minimum), int(maximum))


def smooth_and_differentiate_adaptive(
    values: np.ndarray,
    s: np.ndarray,
    radius_mm: np.ndarray,
    *,
    step: float = PATH_STEP_MM,
    order: int = SG_POLYORDER,
    factor: float = SG_WINDOW_RADIUS_FACTOR,
    minimum: int = SG_HALF_WINDOW_MIN,
    maximum: int = SG_HALF_WINDOW_MAX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Radius-adaptive Savitzky-Golay smoothing and differentiation.

    Returns ``(smoothed, first, second, window_samples)``.  The ends are padded
    by ``minimum`` samples with the local cubic extrapolation exactly as the
    fixed-window operator, and each sample's half window is limited by the
    available support, so the operator degenerates to the fixed 11-sample
    window at segment ends and wherever ``radius <= 5.5 mm``.  Short segments
    fall back to the reduced-order polynomial of :func:`smooth_and_differentiate`.
    """

    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    count = len(values)
    fixed_window = 2 * int(minimum) + 1
    if count < fixed_window:
        smoothed, first, second = smooth_and_differentiate(values, s, step=step, window=fixed_window, order=order)
        return smoothed, first, second, np.full(count, count, dtype=np.int64)
    pad = int(minimum)
    padded = _extrapolate_pad(values, s, pad, order, step)
    index = np.arange(count)
    half = adaptive_half_window(radius_mm, step, factor=factor, minimum=minimum, maximum=maximum)
    half = np.minimum(half, np.minimum(index + pad, count - 1 - index + pad))
    smoothed = np.empty_like(values)
    first = np.empty_like(values)
    second = np.empty_like(values)
    for h in np.unique(half):
        h = int(h)
        c0, c1, c2 = _sg_matrices(2 * h + 1, order, step)
        rows = index[half == h]
        block = padded[rows[:, None] + pad + np.arange(-h, h + 1)[None, :], :]  # (m, window, axes)
        smoothed[rows] = np.tensordot(block, c0, axes=([1], [0]))
        first[rows] = np.tensordot(block, c1, axes=([1], [0]))
        second[rows] = np.tensordot(block, c2, axes=([1], [0]))
    return smoothed, first, second, (2 * half + 1).astype(np.int64)


def hold_end_features(
    tangent: np.ndarray,
    curvature: np.ndarray,
    radius_mm: np.ndarray,
    *,
    step: float,
    start: bool,
    end: bool,
    factor: float = END_ZONE_RADIUS_FACTOR,
    min_zone: int = END_ZONE_MIN_SAMPLES,
    min_band: int = END_BAND_MIN_SAMPLES,
    max_radius: float = END_ZONE_MAX_RADIUS_MM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Hold tangent/curvature inside the end zones at the robust interior value.

    Zone: ``n = max(min_zone, rint(factor * min(R_end, median R, max_radius) / step))`` samples
    at the selected ends; band: the next ``max(n, min_band)`` samples.  Inside the
    zone the curvature becomes the band median and the tangent the normalised
    band mean.  Returns ``(tangent, curvature, zone_mask, info)``; a side is
    skipped (and reported) when the segment is too short.
    """

    tangent = np.array(tangent, dtype=np.float64, copy=True)
    curvature = np.array(curvature, dtype=np.float64, copy=True)
    count = len(curvature)
    mask = np.zeros(count, dtype=bool)
    radius_zone_cap = float(np.median(radius_mm))
    info: dict[str, Any] = {}
    for side, active in (("start", start), ("end", end)):
        if not active:
            info[side] = {"applied": False, "reason": "not an opening or child start"}
            continue
        radius_end = float(radius_mm[0] if side == "start" else radius_mm[-1])
        radius_zone = min(radius_end, radius_zone_cap, float(max_radius))
        n_zone = max(int(min_zone), int(np.rint(factor * max(radius_zone, 0.0) / step)))
        n_band = max(n_zone, int(min_band))
        if n_zone + n_band > count or (int(mask.sum()) + n_zone) >= count:
            info[side] = {"applied": False, "reason": "segment too short", "zone_samples": n_zone, "band_samples": n_band}
            continue
        if side == "start":
            zone = np.arange(0, n_zone)
            band = np.arange(n_zone, n_zone + n_band)
        else:
            zone = np.arange(count - n_zone, count)
            band = np.arange(count - n_zone - n_band, count - n_zone)
        before = curvature[zone].copy()
        held_curvature = float(np.median(curvature[band]))
        held_tangent = tangent[band].mean(axis=0)
        held_tangent = held_tangent / max(float(np.linalg.norm(held_tangent)), 1.0e-12)
        curvature[zone] = held_curvature
        tangent[zone] = held_tangent
        mask[zone] = True
        info[side] = {
            "applied": True,
            "zone_samples": int(n_zone),
            "zone_mm": float(n_zone * step),
            "band_samples": int(n_band),
            "radius_end_mm": radius_end,
            "radius_zone_mm": float(radius_zone),
            "held_curvature_per_mm": held_curvature,
            "zone_curvature_max_before": float(before.max()),
            "max_curvature_change": float(np.max(np.abs(before - held_curvature))),
        }
    return tangent, curvature, mask, info


def curvature_from_derivatives(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.linalg.norm(first, axis=1)
    cross = np.cross(first, second)
    curvature = np.linalg.norm(cross, axis=1) / np.maximum(speed, 1.0e-12) ** 3
    tangent = first / np.maximum(speed, 1.0e-12)[:, None]
    return tangent, curvature


# ---------------------------------------------------------------------------
# atlas


ATLAS_SAMPLE_COLUMNS = (
    "segment_id",
    "parent_id",
    "sample_index",
    "s_local_mm",
    "s_from_root_mm",
    "x_raw_mm",
    "y_raw_mm",
    "z_raw_mm",
    "x_mm",
    "y_mm",
    "z_mm",
    "tangent_x",
    "tangent_y",
    "tangent_z",
    "curvature_per_mm",
    "radius_raw_mm",
    "radius_mm",
    "dr_ds",
    "dist_to_junction_mm",
    "dist_to_endpoint_mm",
    "junction_mask",
    "endpoint_mask",
    "is_junction_owner",
    "is_child_duplicate",
    "fit_residual_mm",
    "sg_window_samples",
    "curvature_times_radius",
    "radius_over_case_median",
    "end_zone",
)


@dataclass
class SegmentFeatures:
    segment_id: int
    parent_id: int
    starts_at_root: bool
    ends_at_leaf: bool
    outlet_name: str
    descendant_outlets: list[str]
    length_mm: float
    s_offset_mm: float  # s_from_root at the segment start
    s_local: np.ndarray
    coords_raw: np.ndarray
    coords: np.ndarray
    tangent: np.ndarray
    curvature: np.ndarray
    radius_raw: np.ndarray
    radius: np.ndarray
    dr_ds: np.ndarray
    dist_to_junction: np.ndarray
    dist_to_endpoint: np.ndarray
    fit_residual: np.ndarray
    sg_window: np.ndarray
    curvature_times_radius: np.ndarray
    end_zone: np.ndarray
    end_zone_info: dict[str, Any]
    native_length_mm: float
    native_points: int


def _resample_polyline(coords: np.ndarray, radius: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    delta = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    native_s = np.r_[0.0, np.cumsum(delta)]
    length = float(native_s[-1])
    if length <= 1.0e-9:
        raise ValueError("degenerate centerline segment")
    sample_s = np.arange(0.0, length, step, dtype=np.float64)
    if not len(sample_s) or length - sample_s[-1] > 1.0e-9:
        sample_s = np.r_[sample_s, length]
    sampled = np.column_stack([np.interp(sample_s, native_s, coords[:, axis]) for axis in range(3)])
    sampled_radius = np.interp(sample_s, native_s, radius)
    return sample_s, sampled, sampled_radius, length


def build_segment_features(
    graph: CenterlineGraph,
    segments: list[RawSegment],
    *,
    step: float = PATH_STEP_MM,
    end_hold: bool = True,
) -> list[SegmentFeatures]:
    by_id = {segment.segment_id: segment for segment in segments}
    features: dict[int, SegmentFeatures] = {}
    for segment in segments:  # parents are always created before children (DFS order)
        coords = graph.points_mm[segment.nodes]
        radius = graph.radius_mm[segment.nodes]
        s_local, sampled, sampled_radius, length = _resample_polyline(coords, radius, step)
        # uniform grid: the last sample may be closer than ``step`` to its
        # predecessor; SG assumes uniform spacing, so re-grid exactly.
        uniform_s = np.linspace(0.0, length, int(np.ceil(length / step)) + 1)
        if len(uniform_s) < 2:
            uniform_s = np.array([0.0, length])
        grid_step = float(uniform_s[1] - uniform_s[0])
        sampled = np.column_stack([np.interp(uniform_s, s_local, sampled[:, axis]) for axis in range(3)])
        sampled_radius = np.interp(uniform_s, s_local, sampled_radius)
        s_local = uniform_s
        # radius first (fixed SG11): it drives the coordinate window and must not depend on it
        radius_smooth, radius_first, _ = smooth_and_differentiate(sampled_radius, s_local, step=grid_step)
        smoothed, first, second, sg_window = smooth_and_differentiate_adaptive(
            sampled, s_local, radius_smooth[:, 0], step=grid_step
        )
        tangent, curvature = curvature_from_derivatives(first, second)
        # end hooks: opening endpoints (root start / leaf end) and child starts; never the parent's junction end
        if end_hold:
            tangent, curvature, end_mask, end_info = hold_end_features(
                tangent,
                curvature,
                radius_smooth[:, 0],
                step=grid_step,
                start=True,  # root start is the inlet opening; any other start is a child start
                end=bool(segment.ends_at_leaf),
            )
        else:
            end_mask, end_info = np.zeros(len(s_local), dtype=bool), {"disabled": True}
        parent = features.get(segment.parent_id)
        s_offset = 0.0 if parent is None else parent.s_offset_mm + parent.length_mm
        # along-tree distances are filled in after every segment length is known
        dist_junction = np.full(len(s_local), np.nan)
        dist_endpoint = np.full(len(s_local), np.nan)
        features[segment.segment_id] = SegmentFeatures(
            segment_id=segment.segment_id,
            parent_id=segment.parent_id,
            starts_at_root=segment.starts_at_root,
            ends_at_leaf=segment.ends_at_leaf,
            outlet_name=str(segment.leaf_opening["outlet_name"]) if segment.leaf_opening else "",
            descendant_outlets=list(segment.descendant_outlets),
            length_mm=length,
            s_offset_mm=s_offset,
            s_local=s_local,
            coords_raw=sampled,
            coords=smoothed,
            tangent=tangent,
            curvature=curvature,
            radius_raw=sampled_radius,
            radius=radius_smooth[:, 0],
            dr_ds=radius_first[:, 0],
            dist_to_junction=dist_junction,
            dist_to_endpoint=dist_endpoint,
            fit_residual=np.linalg.norm(smoothed - sampled, axis=1),
            sg_window=np.asarray(sg_window, dtype=np.int64),
            curvature_times_radius=curvature * radius_smooth[:, 0],
            end_zone=end_mask,
            end_zone_info=end_info,
            native_length_mm=length,
            native_points=len(segment.nodes),
        )
    _fill_tree_distances(segments, features)
    return [features[segment.segment_id] for segment in segments]


def _fill_tree_distances(segments: list[RawSegment], features: dict[int, SegmentFeatures]) -> None:
    """Along-tree distance of every sample to the nearest junction and endpoint.

    Key nodes are the root, the junctions and the leaves; segments are edges of
    a tiny graph whose all-pairs distances are computed exactly.  A sample at
    arc length ``s`` on a segment of length ``L`` from key node ``A`` to key
    node ``B`` is at ``min(s + d[A], (L - s) + d[B])`` from any target set.
    """

    # key node ids: start node of a segment is its parent's end key (or root)
    start_key: dict[int, int] = {}
    end_key: dict[int, int] = {}
    next_key = 0
    root_key = next_key
    next_key += 1
    for segment in segments:  # DFS order guarantees parents first
        start_key[segment.segment_id] = root_key if segment.starts_at_root else end_key[segment.parent_id]
        end_key[segment.segment_id] = next_key
        next_key += 1
    count = next_key
    distance = np.full((count, count), np.inf)
    np.fill_diagonal(distance, 0.0)
    for segment in segments:
        a, b = start_key[segment.segment_id], end_key[segment.segment_id]
        distance[a, b] = distance[b, a] = features[segment.segment_id].length_mm
    for k in range(count):
        distance = np.minimum(distance, distance[:, k : k + 1] + distance[k : k + 1, :])
    endpoint_keys = [root_key] + [end_key[s.segment_id] for s in segments if s.ends_at_leaf]
    junction_keys = [end_key[s.segment_id] for s in segments if not s.ends_at_leaf]
    to_endpoint = distance[:, endpoint_keys].min(axis=1)
    to_junction = distance[:, junction_keys].min(axis=1)
    for segment in segments:
        feature = features[segment.segment_id]
        a, b = start_key[segment.segment_id], end_key[segment.segment_id]
        s = feature.s_local
        remaining = feature.length_mm - s
        feature.dist_to_endpoint = np.minimum(s + to_endpoint[a], remaining + to_endpoint[b])
        feature.dist_to_junction = np.minimum(s + to_junction[a], remaining + to_junction[b])


@dataclass
class FeatureAtlas:
    canonical_id: str
    segments: list[SegmentFeatures]
    table: np.ndarray  # flattened samples, columns = ATLAS_SAMPLE_COLUMNS (child duplicates included, flagged)
    provenance: dict[str, Any]

    @property
    def unique_mask(self) -> np.ndarray:
        column = ATLAS_SAMPLE_COLUMNS.index("is_child_duplicate")
        return self.table[:, column] < 0.5

    def unique_rows(self) -> np.ndarray:
        return self.table[self.unique_mask]

    def column(self, name: str, *, unique: bool = True) -> np.ndarray:
        rows = self.unique_rows() if unique else self.table
        return rows[:, ATLAS_SAMPLE_COLUMNS.index(name)]

    def coords(self, *, unique: bool = True, raw: bool = False) -> np.ndarray:
        names = ("x_raw_mm", "y_raw_mm", "z_raw_mm") if raw else ("x_mm", "y_mm", "z_mm")
        rows = self.unique_rows() if unique else self.table
        return np.column_stack([rows[:, ATLAS_SAMPLE_COLUMNS.index(name)] for name in names])

    def max_s_from_root(self) -> float:
        return float(np.max(self.column("s_from_root_mm")))

    def map_points(self, points_mm: np.ndarray, *, workers: int = 1) -> dict[str, np.ndarray]:
        """Nearest-atlas-sample features for arbitrary points (true-mm frame)."""

        rows = self.unique_rows()
        coords = np.column_stack([rows[:, ATLAS_SAMPLE_COLUMNS.index(n)] for n in ("x_mm", "y_mm", "z_mm")])
        distance, index = cKDTree(coords).query(np.asarray(points_mm, dtype=np.float64), k=1, workers=max(int(workers), 1))
        index = np.asarray(index, dtype=np.int64)
        get = lambda name: rows[index, ATLAS_SAMPLE_COLUMNS.index(name)]  # noqa: E731
        radius = get("radius_mm")
        return {
            "atlas_index": index,
            "centerline_distance_mm": np.asarray(distance, dtype=np.float64),
            "abscissa_norm": get("s_from_root_mm") / self.max_s_from_root(),
            "local_radius_mm": radius,
            "curvature_per_mm": get("curvature_per_mm"),
            "radial_ratio": np.asarray(distance, dtype=np.float64) / np.maximum(radius, 1.0e-12),
            "segment_id": get("segment_id").astype(np.int32),
            "dist_to_junction_mm": get("dist_to_junction_mm"),
            "dist_to_endpoint_mm": get("dist_to_endpoint_mm"),
            # auxiliary (not first-round model inputs): sac / wide-lumen descriptors
            "curvature_times_radius": get("curvature_times_radius"),
            "radius_over_case_median": get("radius_over_case_median"),
            "sg_window_samples": get("sg_window_samples").astype(np.int32),
        }

    def summary(self) -> dict[str, Any]:
        rows = self.unique_rows()
        curvature = rows[:, ATLAS_SAMPLE_COLUMNS.index("curvature_per_mm")]
        residual = rows[:, ATLAS_SAMPLE_COLUMNS.index("fit_residual_mm")]
        k_times_r = rows[:, ATLAS_SAMPLE_COLUMNS.index("curvature_times_radius")]
        windows = rows[:, ATLAS_SAMPLE_COLUMNS.index("sg_window_samples")]
        order = np.argsort(-curvature)[:5]
        peaks = []
        for row_index in order:
            row = rows[row_index]
            peaks.append(
                {
                    "curvature_per_mm": float(row[ATLAS_SAMPLE_COLUMNS.index("curvature_per_mm")]),
                    "segment_id": int(row[ATLAS_SAMPLE_COLUMNS.index("segment_id")]),
                    "s_local_mm": float(row[ATLAS_SAMPLE_COLUMNS.index("s_local_mm")]),
                    "dist_to_junction_mm": float(row[ATLAS_SAMPLE_COLUMNS.index("dist_to_junction_mm")]),
                    "dist_to_endpoint_mm": float(row[ATLAS_SAMPLE_COLUMNS.index("dist_to_endpoint_mm")]),
                    "radius_mm": float(row[ATLAS_SAMPLE_COLUMNS.index("radius_mm")]),
                    "coords_mm": [float(row[ATLAS_SAMPLE_COLUMNS.index(n)]) for n in ("x_mm", "y_mm", "z_mm")],
                }
            )
        return {
            "canonical_id": self.canonical_id,
            "segments": [
                {
                    "segment_id": s.segment_id,
                    "parent_id": s.parent_id,
                    "starts_at_root": s.starts_at_root,
                    "ends_at_leaf": s.ends_at_leaf,
                    "outlet_name": s.outlet_name,
                    "descendant_outlets": s.descendant_outlets,
                    "length_mm": s.length_mm,
                    "s_offset_mm": s.s_offset_mm,
                    "samples": int(len(s.s_local)),
                    "native_points": s.native_points,
                    "curvature_max_per_mm": float(np.max(s.curvature)),
                    "fit_residual_max_mm": float(np.max(s.fit_residual)),
                }
                for s in self.segments
            ],
            "unique_samples": int(len(rows)),
            "total_length_mm": float(sum(s.length_mm for s in self.segments)),
            "max_s_from_root_mm": self.max_s_from_root(),
            "curvature_per_mm": {
                "p50": float(np.percentile(curvature, 50)),
                "p99": float(np.percentile(curvature, 99)),
                "max": float(np.max(curvature)),
                "finite": bool(np.isfinite(curvature).all()),
            },
            "fit_residual_mm": {"p50": float(np.percentile(residual, 50)), "max": float(np.max(residual))},
            "radius_mm": {
                "min": float(np.min(rows[:, ATLAS_SAMPLE_COLUMNS.index("radius_mm")])),
                "median": float(np.median(rows[:, ATLAS_SAMPLE_COLUMNS.index("radius_mm")])),
                "max": float(np.max(rows[:, ATLAS_SAMPLE_COLUMNS.index("radius_mm")])),
            },
            "curvature_times_radius": {
                "p99": float(np.percentile(k_times_r, 99)),
                "max": float(np.max(k_times_r)),
                "fraction_gt_1": float(np.mean(k_times_r > 1.0)),
            },
            "sg_window_samples": {"min": int(np.min(windows)), "p50": float(np.median(windows)), "max": int(np.max(windows))},
            "end_zone": {
                "samples": int(np.sum(rows[:, ATLAS_SAMPLE_COLUMNS.index("end_zone")] > 0.5)),
                "fraction": float(np.mean(rows[:, ATLAS_SAMPLE_COLUMNS.index("end_zone")] > 0.5)),
                "max_curvature_change": float(max(
                    (side["max_curvature_change"] for seg in self.segments for side in seg.end_zone_info.values() if isinstance(side, dict) and side.get("applied")),
                    default=0.0,
                )),
                "skipped_sides": int(sum(
                    1 for seg in self.segments for side in seg.end_zone_info.values()
                    if isinstance(side, dict) and side.get("reason") == "segment too short"
                )),
            },
            "top_curvature_peaks": peaks,
            "provenance": self.provenance,
        }


def build_feature_atlas(
    canonical_id: str,
    graph: CenterlineGraph,
    *,
    step: float = PATH_STEP_MM,
    end_hold: bool = True,
) -> FeatureAtlas:
    segments = decompose_tree(graph)
    features = build_segment_features(graph, segments, step=step, end_hold=end_hold)
    rows = []
    for feature in features:
        count = len(feature.s_local)
        for index in range(count):
            is_child_dup = (not feature.starts_at_root) and index == 0
            is_owner = feature.ends_at_leaf is False and index == count - 1
            s_local = float(feature.s_local[index])
            rows.append(
                [
                    feature.segment_id,
                    feature.parent_id,
                    index,
                    s_local,
                    feature.s_offset_mm + s_local,
                    *feature.coords_raw[index].tolist(),
                    *feature.coords[index].tolist(),
                    *feature.tangent[index].tolist(),
                    float(feature.curvature[index]),
                    float(feature.radius_raw[index]),
                    float(feature.radius[index]),
                    float(feature.dr_ds[index]),
                    float(feature.dist_to_junction[index]),
                    float(feature.dist_to_endpoint[index]),
                    float(feature.dist_to_junction[index] <= JUNCTION_HALF_WIDTH_MM),
                    float(feature.dist_to_endpoint[index] <= ENDPOINT_HALF_WIDTH_MM),
                    float(is_owner),
                    float(is_child_dup),
                    float(feature.fit_residual[index]),
                    float(feature.sg_window[index]),
                    float(feature.curvature_times_radius[index]),
                    np.nan,  # radius_over_case_median: filled once every segment is known
                    float(feature.end_zone[index]),
                ]
            )
    table = np.asarray(rows, dtype=np.float64)
    radius_column = ATLAS_SAMPLE_COLUMNS.index("radius_mm")
    unique = table[:, ATLAS_SAMPLE_COLUMNS.index("is_child_duplicate")] < 0.5
    radius_median = float(np.median(table[unique, radius_column]))
    table[:, ATLAS_SAMPLE_COLUMNS.index("radius_over_case_median")] = table[:, radius_column] / max(radius_median, 1.0e-12)
    atlas = FeatureAtlas(
        canonical_id=canonical_id,
        segments=features,
        table=table,
        provenance={
            **graph.provenance,
            "step_mm": step,
            "sg_window": SG_WINDOW,
            "sg_polyorder": SG_POLYORDER,
            "sg_window_policy": (
                "coordinates: radius-adaptive cubic SG, window length (mm) ~= "
                f"{SG_WINDOW_RADIUS_FACTOR} x local smoothed radius, half window clipped to "
                f"[{SG_HALF_WINDOW_MIN}, {SG_HALF_WINDOW_MAX}] samples and to the available support; "
                "radius and short segments: fixed 11-sample SG (frozen 2026-09-05)"
            ),
            "sg_half_window_min": SG_HALF_WINDOW_MIN,
            "sg_half_window_max": SG_HALF_WINDOW_MAX,
            "sg_window_radius_factor": SG_WINDOW_RADIUS_FACTOR,
            "radius_median_mm": radius_median,
            "end_zone_policy": (
                "opening endpoints and child-segment starts: tangent/curvature within "
                f"{END_ZONE_RADIUS_FACTOR} x min(end radius, segment median radius, {END_ZONE_MAX_RADIUS_MM} mm) (>= {END_ZONE_MIN_SAMPLES} samples) "
                "held at the median/mean of the adjacent 1R..2R band; coordinates untouched; parent junction ends "
                "untouched (frozen 2026-09-06)"
                if end_hold
                else "disabled"
            ),
            "end_zone": {str(feature.segment_id): feature.end_zone_info for feature in features},
            "junction_half_width_mm": JUNCTION_HALF_WIDTH_MM,
            "endpoint_half_width_mm": ENDPOINT_HALF_WIDTH_MM,
            "columns": list(ATLAS_SAMPLE_COLUMNS),
        },
    )
    _validate_atlas(atlas)
    return atlas


def _validate_atlas(atlas: FeatureAtlas) -> None:
    rows = atlas.unique_rows()
    if not np.isfinite(rows).all():
        raise ValueError(f"{atlas.canonical_id}: atlas contains NaN/Inf")
    if np.any(rows[:, ATLAS_SAMPLE_COLUMNS.index("radius_mm")] <= 0.0):
        raise ValueError(f"{atlas.canonical_id}: smoothed radius is not positive")
    coords = atlas.coords(raw=True)
    # no overlaps: the raw sample positions must be unique (child duplicates removed)
    distance, _ = cKDTree(coords).query(coords, k=2)
    if float(np.min(distance[:, 1])) < 1.0e-6:
        raise ValueError(f"{atlas.canonical_id}: flattened atlas contains duplicated samples")
    # no gaps: every child segment's first (duplicate) sample coincides with its parent's junction sample
    by_id = {segment.segment_id: segment for segment in atlas.segments}
    for segment in atlas.segments:
        if segment.starts_at_root:
            continue
        parent = by_id[segment.parent_id]
        gap = float(np.linalg.norm(segment.coords_raw[0] - parent.coords_raw[-1]))
        if gap > 1.0e-6:
            raise ValueError(f"{atlas.canonical_id}: segment {segment.segment_id} does not start at its parent's junction")


def detect_kinks(
    atlas: FeatureAtlas,
    *,
    ratio: float = KINK_RATIO_TO_SEGMENT_MEDIAN,
    floor: float = KINK_CURVATURE_FLOOR_PER_MM,
) -> dict[str, Any]:
    """Isolated extraction folds: curvature above both ``ratio`` x the segment median and ``floor``.

    This is the audit Gate of the 2026-09-04 review (§16.2 (b)); after the
    radius-adaptive window the interior count is expected to be zero, endpoint
    hits are reported separately (endpoint extrapolation is a distinct item).
    """

    samples = []
    for segment in atlas.segments:
        threshold = max(ratio * float(np.median(segment.curvature)), floor)
        for index in np.flatnonzero(segment.curvature > threshold):
            if index == 0 and not segment.starts_at_root:
                continue  # child duplicate of the parent's junction sample (not in the flattened atlas)
            d_end = float(segment.dist_to_endpoint[index])
            d_junction = float(segment.dist_to_junction[index])
            location = (
                "endpoint"
                if d_end <= ENDPOINT_HALF_WIDTH_MM
                else "junction" if d_junction <= JUNCTION_HALF_WIDTH_MM else "interior"
            )
            samples.append(
                {
                    "segment_id": int(segment.segment_id),
                    "s_local_mm": float(segment.s_local[index]),
                    "curvature_per_mm": float(segment.curvature[index]),
                    "segment_median_per_mm": float(np.median(segment.curvature)),
                    "radius_mm": float(segment.radius[index]),
                    "sg_window_samples": int(segment.sg_window[index]),
                    "fit_residual_mm": float(segment.fit_residual[index]),
                    "location": location,
                }
            )
    return {
        "rule": f"curvature > max({ratio} x segment median, {floor} /mm)",
        "interior": sum(1 for row in samples if row["location"] == "interior"),
        "junction": sum(1 for row in samples if row["location"] == "junction"),
        "endpoint": sum(1 for row in samples if row["location"] == "endpoint"),
        "samples": samples,
    }


def save_atlas(atlas: FeatureAtlas, path: str | Path) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        table=atlas.table,
        columns=np.asarray(ATLAS_SAMPLE_COLUMNS),
        provenance=np.asarray(json.dumps(atlas.provenance, sort_keys=True)),
    )
    return atlas.summary()
