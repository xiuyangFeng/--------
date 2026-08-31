#!/usr/bin/env python3
"""Centerline V2 extraction and audit runner.

This script intentionally writes a versioned audit package outside ``data_new``.
It freezes the authoritative STL/unit information already recorded by the WSS
bundle, uses the CFD boundary-face contract to identify inlet/outlet openings,
runs VMTK with target-level retries, rebuilds an explicit centerline graph, and
emits hard/soft gate metrics plus off-screen review renders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import vtk
from vmtk import vmtkscripts
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT / "data_wss_pinn/volume_uvwp_bc_rcr_v4_train138_test35/manifest.json"
)
DEFAULT_OUTPUT = PROJECT / "outputs/centerline_v2_pilot_10_20260828"
DEFAULT_SEED = 20260828
EXPECTED_OPENINGS = 5
PATH_STEP_MM = 0.5
JUNCTION_HALF_WIDTH_MM = 2.0
RENDER_COLORS = {
    -1: (0.10, 0.10, 0.12),
    0: (0.88, 0.18, 0.16),
    1: (0.12, 0.45, 0.86),
    2: (0.10, 0.68, 0.38),
    3: (0.84, 0.46, 0.08),
}


@dataclass
class Opening:
    opening_id: int
    center_raw: list[float]
    inward_normal_raw: list[float]
    area_raw2: float
    equivalent_radius_raw: float
    perimeter_raw: float
    boundary_points: int
    role: str = "unassigned"
    outlet_id: int = -1
    outlet_name: str = ""
    contract_distance_norm: float | None = None


@dataclass
class GraphData:
    points_raw: np.ndarray
    radii_raw: np.ndarray
    edges: list[tuple[int, int]]
    adjacency: dict[int, set[int]]
    active_nodes: list[int]
    degrees: np.ndarray
    endpoints: list[int]
    branch_nodes: list[int]
    components: list[list[int]]
    cycle_rank: int


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(array: np.lib.npyio.NpzFile, key: str):
    return np.asarray(array[key]).item()


def read_stl(path: Path) -> vtk.vtkPolyData:
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetPoints() is None or poly.GetNumberOfPolys() == 0:
        raise RuntimeError(f"Invalid STL: {path}")
    out = vtk.vtkPolyData()
    out.DeepCopy(poly)
    return out


def write_vtp(poly: vtk.vtkPolyData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToZLib()
    writer.SetInputData(poly)
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write VTP: {path}")


def write_stl(poly: vtk.vtkPolyData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetFileTypeToBinary()
    writer.SetInputData(tri.GetOutput())
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write STL: {path}")


def transform_polydata(poly: vtk.vtkPolyData, scale: float) -> vtk.vtkPolyData:
    transform = vtk.vtkTransform()
    transform.Scale(float(scale), float(scale), float(scale))
    filt = vtk.vtkTransformPolyDataFilter()
    filt.SetInputData(poly)
    filt.SetTransform(transform)
    filt.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(filt.GetOutput())
    return out


def surface_points(poly: vtk.vtkPolyData) -> np.ndarray:
    return vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)


def _feature_edge_count(surface: vtk.vtkPolyData, *, boundary=False, nonmanifold=False) -> int:
    filt = vtk.vtkFeatureEdges()
    filt.SetInputData(surface)
    filt.BoundaryEdgesOn() if boundary else filt.BoundaryEdgesOff()
    filt.NonManifoldEdgesOn() if nonmanifold else filt.NonManifoldEdgesOff()
    filt.FeatureEdgesOff()
    filt.ManifoldEdgesOff()
    filt.Update()
    return int(filt.GetOutput().GetNumberOfLines())


def _surface_regions(surface: vtk.vtkPolyData) -> int:
    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(surface)
    conn.SetExtractionModeToAllRegions()
    conn.Update()
    return int(conn.GetNumberOfExtractedRegions())


def _opening_geometry(points: np.ndarray, surface_center: np.ndarray) -> tuple:
    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(points) < 3:
        raise RuntimeError("Boundary component has fewer than three unique points")
    center = points.mean(axis=0)
    q = points - center
    covariance = q.T @ q / max(len(points), 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)
    normal = vectors[:, order[0]]
    basis_u = vectors[:, order[2]]
    basis_v = vectors[:, order[1]]
    if np.dot(normal, surface_center - center) < 0.0:
        normal = -normal
    uv = np.column_stack((q @ basis_u, q @ basis_v))
    angles = np.arctan2(uv[:, 1], uv[:, 0])
    uv = uv[np.argsort(angles)]
    x, y = uv[:, 0], uv[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    perimeter = float(np.linalg.norm(uv - np.roll(uv, -1, axis=0), axis=1).sum())
    radius = math.sqrt(max(area, 0.0) / math.pi)
    return center, normal, area, radius, perimeter, len(points)


def detect_openings(surface: vtk.vtkPolyData) -> list[Opening]:
    feature = vtk.vtkFeatureEdges()
    feature.SetInputData(surface)
    feature.BoundaryEdgesOn()
    feature.FeatureEdgesOff()
    feature.ManifoldEdgesOff()
    feature.NonManifoldEdgesOff()
    feature.Update()
    boundary = feature.GetOutput()
    if boundary.GetNumberOfLines() == 0:
        return []

    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(boundary)
    conn.SetExtractionModeToAllRegions()
    conn.ColorRegionsOn()
    conn.Update()
    n_regions = int(conn.GetNumberOfExtractedRegions())
    surface_center = surface_points(surface).mean(axis=0)
    openings = []
    for region_id in range(n_regions):
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(conn.GetOutput())
        threshold.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "RegionId"
        )
        threshold.SetLowerThreshold(region_id)
        threshold.SetUpperThreshold(region_id)
        threshold.Update()
        region = threshold.GetOutput()
        if region.GetPoints() is None or region.GetNumberOfPoints() < 3:
            continue
        pts = vtk_to_numpy(region.GetPoints().GetData())
        center, normal, area, radius, perimeter, count = _opening_geometry(
            pts, surface_center
        )
        if not np.isfinite(area) or area <= 0.0:
            continue
        openings.append(
            Opening(
                opening_id=len(openings),
                center_raw=center.tolist(),
                inward_normal_raw=normal.tolist(),
                area_raw2=area,
                equivalent_radius_raw=radius,
                perimeter_raw=perimeter,
                boundary_points=count,
            )
        )
    openings.sort(key=lambda item: tuple(np.round(item.center_raw, 8)))
    for index, opening in enumerate(openings):
        opening.opening_id = index
    return openings


def surface_precheck(surface: vtk.vtkPolyData) -> tuple[dict, list[Opening]]:
    openings = detect_openings(surface)
    report = {
        "surface_points": int(surface.GetNumberOfPoints()),
        "surface_triangles": int(surface.GetNumberOfPolys()),
        "connected_regions": _surface_regions(surface),
        "nonmanifold_edges": _feature_edge_count(surface, nonmanifold=True),
        "boundary_edges": _feature_edge_count(surface, boundary=True),
        "opening_count": len(openings),
        "openings": [asdict(item) for item in openings],
    }
    report["hard_pass"] = bool(
        report["connected_regions"] == 1
        and report["nonmanifold_edges"] == 0
        and report["opening_count"] == EXPECTED_OPENINGS
    )
    return report, openings


def _candidate_surface_rows(case_root: Path, authoritative: Path, bundle) -> list[dict]:
    wall = np.asarray(bundle["wall_coords_raw"], dtype=np.float64)
    wall_center = (wall.min(axis=0) + wall.max(axis=0)) / 2.0
    wall_span = np.ptp(wall, axis=0)
    wall_diag = float(np.linalg.norm(wall_span))
    rows = []
    for candidate in sorted(case_root.glob("*.stl")):
        try:
            poly = read_stl(candidate)
            pts = surface_points(poly)
            span = np.ptp(pts, axis=0)
            valid = (span > 1.0e-9) & (wall_span > 1.0e-9)
            inferred_scale = float(np.median(wall_span[valid] / span[valid]))
            scaled = pts * inferred_scale
            center = (scaled.min(axis=0) + scaled.max(axis=0)) / 2.0
            scaled_span = np.ptp(scaled, axis=0)
            center_error = float(np.linalg.norm(center - wall_center) / max(wall_diag, 1e-9))
            span_error = float(
                np.linalg.norm(scaled_span - wall_span) / max(wall_diag, 1e-9)
            )
            rows.append(
                {
                    "path": str(candidate.resolve()),
                    "sha256": sha256_file(candidate),
                    "is_authoritative": candidate.resolve() == authoritative.resolve(),
                    "inferred_scale_to_wall_mm": inferred_scale,
                    "bbox_center_error_diag_frac": center_error,
                    "bbox_span_error_diag_frac": span_error,
                    "screening_score": center_error + span_error,
                    "points": int(poly.GetNumberOfPoints()),
                    "triangles": int(poly.GetNumberOfPolys()),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"path": str(candidate), "error": str(exc)})
    return rows


def load_case_context(entry: dict) -> dict:
    canonical_id = entry["canonical_id"]
    case_root = PROJECT / "data_new" / canonical_id
    case_manifest = json.loads(Path(entry["manifest"]).read_text(encoding="utf-8"))
    bundle_path = Path(case_manifest["provenance"]["bundle"]["path"])
    boundary_path = Path(case_manifest["files"]["boundary_faces"]["path"])
    with np.load(bundle_path, allow_pickle=True) as bundle:
        authoritative = Path(str(scalar(bundle, "original_stl_path"))).resolve()
        context = {
            "canonical_id": canonical_id,
            "cohort": entry.get("cohort", ""),
            "role": entry.get("role", ""),
            "case_root": case_root,
            "case_manifest": Path(entry["manifest"]),
            "bundle_path": bundle_path,
            "boundary_path": boundary_path,
            "authoritative_stl": authoritative,
            "authoritative_stl_sha256": sha256_file(authoritative),
            "stl_scale_to_mm": float(scalar(bundle, "original_stl_scale_to_mm")),
            "unit_factor": float(scalar(bundle, "unit_factor")),
            "unit_override_applied": bool(scalar(bundle, "unit_override_applied"))
            if "unit_override_applied" in bundle.files
            else False,
            "unit_override_reason": str(scalar(bundle, "unit_override_reason"))
            if "unit_override_reason" in bundle.files
            else "",
            "coord_scale_mm": float(scalar(bundle, "coord_scale")),
            "transform_centroid": np.asarray(bundle["transform_centroid"], dtype=np.float64),
            "transform_rotation": np.asarray(bundle["transform_rotation"], dtype=np.float64),
            "legacy_centerline_translation_applied": bool(
                scalar(bundle, "transform_centerline_translation_applied")
            ),
            "legacy_centerline_translation_mm": np.asarray(
                bundle["transform_centerline_translation_mm"], dtype=np.float64
            ),
            "wall_coords_mm": np.asarray(bundle["wall_coords_raw"], dtype=np.float64),
            "surface_candidates": _candidate_surface_rows(case_root, authoritative, bundle),
        }
    # Newly extracted centerlines live in the authoritative STL's local frame.
    # Translation-repair cases require the STL rigid-body bbox shift (not the
    # legacy centerline-length-derived shift) before comparison with CFD wall
    # coordinates. Extraction and endpoint gates remain in the local STL frame.
    if context["legacy_centerline_translation_applied"]:
        surface_mm = surface_points(read_stl(authoritative)) * context["stl_scale_to_mm"]
        wall = context["wall_coords_mm"]
        surface_center = (surface_mm.min(axis=0) + surface_mm.max(axis=0)) / 2.0
        wall_center = (wall.min(axis=0) + wall.max(axis=0)) / 2.0
        geometry_shift = wall_center - surface_center
        geometry_shift_source = "authoritative_stl_bbox_to_cfd_wall_bbox"
    else:
        geometry_shift = np.zeros(3, dtype=np.float64)
        geometry_shift_source = "none_already_common_frame"
    context["geometry_translation_to_cfd_wall_mm"] = geometry_shift
    context["geometry_translation_source"] = geometry_shift_source
    return context


def assign_opening_roles(openings: list[Opening], context: dict) -> dict:
    scale = context["stl_scale_to_mm"]
    coord_scale = context["coord_scale_mm"]
    centroid = context["transform_centroid"]
    rotation = context["transform_rotation"]
    centers_mm = np.asarray([item.center_raw for item in openings]) * scale
    centers_norm = ((centers_mm - centroid) @ rotation) / coord_scale

    source = "largest_opening_area_fallback"
    contract = {}
    if context["boundary_path"].is_file():
        with np.load(context["boundary_path"], allow_pickle=True) as boundary:
            inlet_center = np.asarray(boundary["inlet_coords"], dtype=np.float64).mean(axis=0)
            outlet_zone = np.asarray(boundary["outlet_zone"], dtype=np.int64)
            outlet_names = [str(value) for value in np.asarray(boundary["outlet_zone_names"])]
            outlet_centers = np.vstack(
                [
                    np.asarray(boundary["outlet_coords"], dtype=np.float64)[
                        outlet_zone == zone
                    ].mean(axis=0)
                    for zone in range(len(outlet_names))
                ]
            )
        inlet_dist = np.linalg.norm(centers_norm - inlet_center, axis=1)
        inlet_index = int(np.argmin(inlet_dist))
        remaining = [index for index in range(len(openings)) if index != inlet_index]
        cost = np.linalg.norm(
            centers_norm[np.asarray(remaining), None, :] - outlet_centers[None, :, :], axis=2
        )
        rows, cols = linear_sum_assignment(cost)
        outlet_assignment = {
            remaining[int(row)]: int(col) for row, col in zip(rows.tolist(), cols.tolist())
        }
        source = "cfd_boundary_face_contract"
        contract = {
            "inlet_center_norm": inlet_center.tolist(),
            "outlet_centers_norm": outlet_centers.tolist(),
            "outlet_zone_names": outlet_names,
        }
        for index, opening in enumerate(openings):
            if index == inlet_index:
                opening.role = "inlet"
                opening.contract_distance_norm = float(inlet_dist[index])
            else:
                outlet_id = outlet_assignment[index]
                opening.role = "outlet"
                opening.outlet_id = outlet_id
                opening.outlet_name = outlet_names[outlet_id]
                opening.contract_distance_norm = float(cost[remaining.index(index), outlet_id])
    else:
        inlet_index = int(np.argmax([item.area_raw2 for item in openings]))
        remaining = [index for index in range(len(openings)) if index != inlet_index]
        source_center = centers_mm[inlet_index]
        axis = centers_mm[remaining].mean(axis=0) - source_center
        axis /= max(np.linalg.norm(axis), 1.0e-12)
        trial = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(axis, trial))) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        u = np.cross(axis, trial)
        u /= max(np.linalg.norm(u), 1.0e-12)
        v = np.cross(axis, u)
        angles = []
        for index in remaining:
            delta = centers_mm[index] - source_center
            angles.append((math.atan2(float(delta @ v), float(delta @ u)), index))
        for outlet_id, (_, index) in enumerate(sorted(angles)):
            openings[index].role = "outlet"
            openings[index].outlet_id = outlet_id
            openings[index].outlet_name = f"outlet-{outlet_id}"
        openings[inlet_index].role = "inlet"

    return {
        "assignment_source": source,
        "opening_centers_norm": centers_norm.tolist(),
        "contract": contract,
        "openings": [asdict(item) for item in openings],
    }


@contextmanager
def redirect_fds(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    with path.open("a", encoding="utf-8") as handle:
        try:
            os.dup2(handle.fileno(), 1)
            os.dup2(handle.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)


def run_vmtk(
    surface: vtk.vtkPolyData,
    source: Sequence[float],
    targets: Sequence[Sequence[float]],
    log_path: Path,
) -> vtk.vtkPolyData:
    log_path.unlink(missing_ok=True)
    with redirect_fds(log_path):
        print(
            json.dumps(
                {"source": list(source), "targets": [list(target) for target in targets]},
                ensure_ascii=False,
            ),
            flush=True,
        )
        centerlines = vmtkscripts.vmtkCenterlines()
        centerlines.Surface = surface
        centerlines.SeedSelectorName = "pointlist"
        centerlines.SourcePoints = [float(value) for value in source]
        centerlines.TargetPoints = [
            float(value) for target in targets for value in target
        ]
        # Keep VMTK's center-of-opening endpoints. Appending surface seed points
        # moves the graph endpoint toward the opening rim and makes the planned
        # center-to-endpoint soft gate report an artificial ~1R error.
        centerlines.AppendEndPoints = 0
        centerlines.Execute()
        output = centerlines.Centerlines
        if output is None or output.GetNumberOfPoints() == 0:
            raise RuntimeError("VMTK returned an empty centerline")
        out = vtk.vtkPolyData()
        out.DeepCopy(output)
        print(
            f"RESULT points={out.GetNumberOfPoints()} lines={out.GetNumberOfLines()}",
            flush=True,
        )
        return out


def append_and_clean(polys: Iterable[vtk.vtkPolyData], surface_diag: float) -> vtk.vtkPolyData:
    append = vtk.vtkAppendPolyData()
    count = 0
    for poly in polys:
        if poly is not None and poly.GetNumberOfPoints() > 0:
            append.AddInputData(poly)
            count += 1
    if count == 0:
        raise RuntimeError("No centerline paths were available for merging")
    append.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(append.GetOutput())
    clean.PointMergingOn()
    clean.ToleranceIsAbsoluteOn()
    clean.SetAbsoluteTolerance(max(surface_diag * 1.0e-7, 1.0e-8))
    clean.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(clean.GetOutput())
    return out


def lightly_clean_surface(surface: vtk.vtkPolyData) -> vtk.vtkPolyData:
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(surface)
    clean.PointMergingOn()
    clean.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(clean.GetOutput())
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(tri.GetOutput())
    return out


def graph_from_polydata(poly: vtk.vtkPolyData) -> GraphData:
    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    radius_vtk = poly.GetPointData().GetArray("MaximumInscribedSphereRadius")
    if radius_vtk is None:
        raise RuntimeError("MaximumInscribedSphereRadius is missing from VMTK output")
    radii = np.asarray(vtk_to_numpy(radius_vtk), dtype=np.float64).reshape(-1)
    edges: set[tuple[int, int]] = set()
    id_list = vtk.vtkIdList()
    lines = poly.GetLines()
    lines.InitTraversal()
    while lines.GetNextCell(id_list):
        ids = [int(id_list.GetId(i)) for i in range(id_list.GetNumberOfIds())]
        for a, b in zip(ids[:-1], ids[1:]):
            if a != b:
                edges.add((a, b) if a < b else (b, a))
    adjacency: dict[int, set[int]] = {}
    for a, b in sorted(edges):
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    active = sorted(adjacency)
    degrees = np.zeros(len(points), dtype=np.int32)
    for node, neighbors in adjacency.items():
        degrees[node] = len(neighbors)
    endpoints = [node for node in active if degrees[node] == 1]
    branch_nodes = [node for node in active if degrees[node] >= 3]
    components = []
    unseen = set(active)
    while unseen:
        root = min(unseen)
        stack = [root]
        component = []
        unseen.remove(root)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    cycle_rank = len(edges) - len(active) + len(components)
    return GraphData(
        points_raw=points,
        radii_raw=radii,
        edges=sorted(edges),
        adjacency=adjacency,
        active_nodes=active,
        degrees=degrees,
        endpoints=endpoints,
        branch_nodes=branch_nodes,
        components=components,
        cycle_rank=int(cycle_rank),
    )


def polydata_with_edges(
    source: vtk.vtkPolyData, edges: Sequence[tuple[int, int]]
) -> vtk.vtkPolyData:
    """Rebuild line cells while preserving VMTK point coordinates and arrays."""
    lines = vtk.vtkCellArray()
    for a, b in edges:
        lines.InsertNextCell(2)
        lines.InsertCellPoint(int(a))
        lines.InsertCellPoint(int(b))
    out = vtk.vtkPolyData()
    out.SetPoints(source.GetPoints())
    out.SetLines(lines)
    out.GetPointData().ShallowCopy(source.GetPointData())
    return out


def prune_terminal_spikes(
    poly: vtk.vtkPolyData,
    openings: list[Opening],
    scale_to_mm: float,
    *,
    max_edge_mm: float = 10.0,
    max_iterations: int = 8,
) -> tuple[vtk.vtkPolyData, dict]:
    """Remove only long degree-1 spikes whose parent is the valid opening endpoint.

    VMTK can occasionally append one numerically displaced terminal vertex after
    reaching the opening. The repair is deliberately conservative: the long edge
    must terminate at a leaf, its parent must have degree two, the parent must be
    close to the leaf's uniquely assigned opening, and endpoint count/connectivity
    must be preserved after pruning.
    """
    working = poly
    actions = []
    opening_points_mm = np.asarray([item.center_raw for item in openings]) * scale_to_mm
    for iteration in range(max_iterations):
        graph = graph_from_polydata(working)
        if len(graph.endpoints) != len(openings):
            break
        endpoint_points_mm = graph.points_raw[graph.endpoints] * scale_to_mm
        cost = np.linalg.norm(
            endpoint_points_mm[:, None, :] - opening_points_mm[None, :, :], axis=2
        )
        rows, cols = linear_sum_assignment(cost)
        endpoint_to_opening = {
            graph.endpoints[int(row)]: int(col) for row, col in zip(rows, cols)
        }
        candidates = []
        for a, b in graph.edges:
            length_mm = float(
                np.linalg.norm(graph.points_raw[a] - graph.points_raw[b]) * scale_to_mm
            )
            if length_mm <= max_edge_mm:
                continue
            if graph.degrees[a] == 1 and graph.degrees[b] == 2:
                leaf, parent = a, b
            elif graph.degrees[b] == 1 and graph.degrees[a] == 2:
                leaf, parent = b, a
            else:
                continue
            if leaf not in endpoint_to_opening:
                continue
            opening_id = endpoint_to_opening[leaf]
            opening = openings[opening_id]
            opening_center = opening_points_mm[opening_id]
            leaf_distance = float(
                np.linalg.norm(graph.points_raw[leaf] * scale_to_mm - opening_center)
            )
            parent_distance = float(
                np.linalg.norm(graph.points_raw[parent] * scale_to_mm - opening_center)
            )
            radius_mm = float(opening.equivalent_radius_raw * scale_to_mm)
            parent_limit = max(2.0, 0.5 * radius_mm)
            improvement = leaf_distance - parent_distance
            if parent_distance > parent_limit or improvement < 0.5 * length_mm:
                continue
            candidates.append(
                (
                    length_mm,
                    leaf,
                    parent,
                    opening_id,
                    leaf_distance,
                    parent_distance,
                    radius_mm,
                )
            )
        if not candidates:
            break

        length_mm, leaf, parent, opening_id, leaf_distance, parent_distance, radius_mm = max(
            candidates, key=lambda item: item[0]
        )
        kept_edges = [
            edge for edge in graph.edges if leaf not in edge
        ]
        candidate_poly = polydata_with_edges(working, kept_edges)
        candidate_graph = graph_from_polydata(candidate_poly)
        if not (
            len(candidate_graph.endpoints) == len(graph.endpoints)
            and len(candidate_graph.components) == 1
            and candidate_graph.cycle_rank == 0
        ):
            break
        actions.append(
            {
                "iteration": iteration + 1,
                "removed_leaf_point_id": int(leaf),
                "replacement_endpoint_point_id": int(parent),
                "opening_id": int(opening_id),
                "outlet_name": openings[opening_id].outlet_name,
                "removed_edge_length_mm": length_mm,
                "leaf_to_opening_mm_before": leaf_distance,
                "parent_to_opening_mm_after": parent_distance,
                "opening_radius_mm": radius_mm,
            }
        )
        working = candidate_poly

    final_graph = graph_from_polydata(working)
    final_lengths = [
        float(np.linalg.norm(final_graph.points_raw[a] - final_graph.points_raw[b]) * scale_to_mm)
        for a, b in final_graph.edges
    ]
    return working, {
        "applied": bool(actions),
        "n_pruned": len(actions),
        "threshold_mm": max_edge_mm,
        "actions": actions,
        "final_max_edge_mm": max(final_lengths) if final_lengths else None,
    }


def resample_safe_long_edges(
    poly: vtk.vtkPolyData,
    surface: vtk.vtkPolyData,
    scale_to_mm: float,
    *,
    max_edge_mm: float = 10.0,
    target_segment_mm: float = 2.0,
    min_alignment_cosine: float = 0.85,
    max_length_over_local_radius: float = 2.0,
) -> tuple[vtk.vtkPolyData, dict]:
    """Subdivide only smooth, intraluminal sampling gaps in the VMTK graph.

    A long graph edge is not automatically an erroneous shortcut: VMTK may
    occasionally emit a sparse but geometrically valid segment in a wide lumen.
    The repair remains conservative. Both endpoints must be degree-2 nodes, the
    neighboring directions must align with the long chord, the full chord must
    stay inside the capped surface, and its length must be plausible relative to
    the local centerline radius. Edges failing any safeguard remain unchanged so
    the hard Gate can still reject them.
    """
    graph = graph_from_polydata(poly)
    edge_lengths_mm = {
        edge: float(
            np.linalg.norm(graph.points_raw[edge[0]] - graph.points_raw[edge[1]])
            * scale_to_mm
        )
        for edge in graph.edges
    }
    long_edges = [edge for edge in graph.edges if edge_lengths_mm[edge] > max_edge_mm]
    if not long_edges:
        return poly, {
            "applied": False,
            "n_edges_resampled": 0,
            "inserted_points": 0,
            "threshold_mm": max_edge_mm,
            "target_segment_mm": target_segment_mm,
            "actions": [],
            "skipped_long_edges": [],
            "final_max_edge_mm": max(edge_lengths_mm.values()) if edge_lengths_mm else None,
        }

    approved: dict[tuple[int, int], dict] = {}
    skipped = []
    for a, b in long_edges:
        length_mm = edge_lengths_mm[(a, b)]
        reasons = []
        if graph.degrees[a] != 2 or graph.degrees[b] != 2:
            reasons.append("endpoint_degree_not_2")

        alignment_a = float("nan")
        alignment_b = float("nan")
        if not reasons:
            other_a = next(node for node in graph.adjacency[a] if node != b)
            other_b = next(node for node in graph.adjacency[b] if node != a)
            incoming = graph.points_raw[a] - graph.points_raw[other_a]
            chord = graph.points_raw[b] - graph.points_raw[a]
            outgoing = graph.points_raw[other_b] - graph.points_raw[b]
            incoming /= max(float(np.linalg.norm(incoming)), 1.0e-12)
            chord /= max(float(np.linalg.norm(chord)), 1.0e-12)
            outgoing /= max(float(np.linalg.norm(outgoing)), 1.0e-12)
            alignment_a = float(np.dot(incoming, chord))
            alignment_b = float(np.dot(chord, outgoing))
            if min(alignment_a, alignment_b) < min_alignment_cosine:
                reasons.append("local_direction_discontinuous")

        local_radius_mm = float(
            min(graph.radii_raw[a], graph.radii_raw[b]) * scale_to_mm
        )
        length_over_radius = length_mm / max(local_radius_mm, 1.0e-9)
        if length_over_radius > max_length_over_local_radius:
            reasons.append("length_exceeds_local_radius_guard")

        sample_count = max(3, int(math.ceil(length_mm / 0.5)) + 1)
        alpha = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)[:, None]
        chord_samples = (
            graph.points_raw[a][None, :] * (1.0 - alpha)
            + graph.points_raw[b][None, :] * alpha
        )
        inside_fraction, capped_closed = _inside_fraction(surface, chord_samples)
        if not capped_closed or not np.isfinite(inside_fraction) or inside_fraction < 1.0:
            reasons.append("chord_not_fully_inside_capped_surface")

        audit = {
            "point_ids": [int(a), int(b)],
            "original_length_mm": length_mm,
            "local_radius_mm": local_radius_mm,
            "length_over_local_radius": length_over_radius,
            "alignment_cosine_at_a": alignment_a,
            "alignment_cosine_at_b": alignment_b,
            "chord_inside_fraction": inside_fraction,
        }
        if reasons:
            audit["reasons"] = reasons
            skipped.append(audit)
        else:
            approved[(a, b)] = audit

    if not approved:
        return poly, {
            "applied": False,
            "n_edges_resampled": 0,
            "inserted_points": 0,
            "threshold_mm": max_edge_mm,
            "target_segment_mm": target_segment_mm,
            "actions": [],
            "skipped_long_edges": skipped,
            "final_max_edge_mm": max(edge_lengths_mm.values()) if edge_lengths_mm else None,
        }

    points = vtk.vtkPoints()
    points.DeepCopy(poly.GetPoints())
    out = vtk.vtkPolyData()
    out.SetPoints(points)
    source_point_data = poly.GetPointData()
    out_point_data = out.GetPointData()
    estimated_new_points = sum(
        int(math.ceil(edge_lengths_mm[edge] / target_segment_mm)) - 1
        for edge in approved
    )
    out_point_data.CopyAllocate(
        source_point_data, poly.GetNumberOfPoints() + estimated_new_points
    )
    for point_id in range(poly.GetNumberOfPoints()):
        out_point_data.CopyData(source_point_data, point_id, point_id)

    new_edges = []
    actions = []
    inserted_total = 0
    for a, b in graph.edges:
        audit = approved.get((a, b))
        if audit is None:
            new_edges.append((a, b))
            continue
        segments = int(math.ceil(edge_lengths_mm[(a, b)] / target_segment_mm))
        previous = a
        inserted_ids = []
        for index in range(1, segments):
            fraction = index / segments
            xyz = (
                graph.points_raw[a] * (1.0 - fraction)
                + graph.points_raw[b] * fraction
            )
            point_id = int(points.InsertNextPoint(*[float(value) for value in xyz]))
            out_point_data.InterpolateEdge(
                source_point_data, point_id, int(a), int(b), float(fraction)
            )
            new_edges.append((previous, point_id))
            previous = point_id
            inserted_ids.append(point_id)
        new_edges.append((previous, b))
        inserted_total += len(inserted_ids)
        actions.append(
            {
                **audit,
                "segments_after": segments,
                "inserted_point_ids": inserted_ids,
                "max_segment_length_mm_after": edge_lengths_mm[(a, b)] / segments,
            }
        )

    lines = vtk.vtkCellArray()
    for a, b in new_edges:
        lines.InsertNextCell(2)
        lines.InsertCellPoint(int(a))
        lines.InsertCellPoint(int(b))
    out.SetLines(lines)
    repaired_graph = graph_from_polydata(out)
    final_lengths = [
        float(
            np.linalg.norm(
                repaired_graph.points_raw[a] - repaired_graph.points_raw[b]
            )
            * scale_to_mm
        )
        for a, b in repaired_graph.edges
    ]
    return out, {
        "applied": True,
        "n_edges_resampled": len(actions),
        "inserted_points": inserted_total,
        "threshold_mm": max_edge_mm,
        "target_segment_mm": target_segment_mm,
        "actions": actions,
        "skipped_long_edges": skipped,
        "final_max_edge_mm": max(final_lengths) if final_lengths else None,
    }


def _inside_fraction(surface: vtk.vtkPolyData, points: np.ndarray) -> tuple[float, bool]:
    fill = vtk.vtkFillHolesFilter()
    fill.SetInputData(surface)
    fill.SetHoleSize(1.0e9)
    fill.Update()
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(fill.GetOutput())
    tri.Update()
    closed = tri.GetOutput()
    closed_ok = _feature_edge_count(closed, boundary=True) == 0
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.asarray(points, dtype=np.float64), deep=1))
    probes = vtk.vtkPolyData()
    probes.SetPoints(vtk_points)
    select = vtk.vtkSelectEnclosedPoints()
    select.SetInputData(probes)
    select.SetSurfaceData(closed)
    select.SetTolerance(1.0e-6)
    select.Update()
    array = select.GetOutput().GetPointData().GetArray("SelectedPoints")
    if array is None:
        return float("nan"), closed_ok
    inside = vtk_to_numpy(array).astype(bool)
    return float(np.mean(inside)) if len(inside) else float("nan"), closed_ok


def _densify_graph_edges(graph: GraphData, scale_to_mm: float, step_mm: float = 1.0) -> np.ndarray:
    """Sample every graph edge so an outside chord cannot hide between inside nodes."""
    samples = []
    for a, b in graph.edges:
        start = graph.points_raw[a]
        end = graph.points_raw[b]
        length_mm = float(np.linalg.norm(end - start) * scale_to_mm)
        count = max(2, int(math.ceil(length_mm / step_mm)) + 1)
        alpha = np.linspace(0.0, 1.0, count, dtype=np.float64)[:, None]
        samples.append(start[None, :] * (1.0 - alpha) + end[None, :] * alpha)
    return np.vstack(samples) if samples else graph.points_raw


def evaluate_centerline(
    poly: vtk.vtkPolyData,
    surface: vtk.vtkPolyData,
    openings: list[Opening],
    context: dict,
    log_paths: Sequence[Path],
) -> tuple[dict, GraphData, dict[int, int]]:
    graph = graph_from_polydata(poly)
    scale = context["stl_scale_to_mm"]
    endpoint_points_mm = graph.points_raw[graph.endpoints] * scale
    opening_points_mm = np.asarray([item.center_raw for item in openings]) * scale
    endpoint_assignment: dict[int, int] = {}
    opening_distances = np.full(len(openings), np.inf, dtype=np.float64)
    if len(graph.endpoints) and len(openings):
        cost = np.linalg.norm(
            endpoint_points_mm[:, None, :] - opening_points_mm[None, :, :], axis=2
        )
        rows, cols = linear_sum_assignment(cost)
        for row, col in zip(rows.tolist(), cols.tolist()):
            endpoint_assignment[graph.endpoints[row]] = int(col)
            opening_distances[col] = float(cost[row, col])

    inlet_opening = next(item for item in openings if item.role == "inlet")
    inlet_id = inlet_opening.opening_id
    root_candidates = [
        node for node, opening_id in endpoint_assignment.items() if opening_id == inlet_id
    ]
    root_node = root_candidates[0] if root_candidates else (
        min(
            graph.endpoints,
            key=lambda node: np.linalg.norm(
                graph.points_raw[node] - np.asarray(inlet_opening.center_raw)
            ),
        )
        if graph.endpoints
        else -1
    )

    dense_edge_points = _densify_graph_edges(graph, scale)
    inside_fraction, capped_closed = _inside_fraction(surface, dense_edge_points)
    edge_lengths_mm = np.asarray(
        [
            np.linalg.norm(graph.points_raw[a] - graph.points_raw[b]) * scale
            for a, b in graph.edges
        ],
        dtype=np.float64,
    )
    max_edge_mm = float(edge_lengths_mm.max()) if len(edge_lengths_mm) else float("inf")
    p95_edge_mm = float(np.quantile(edge_lengths_mm, 0.95)) if len(edge_lengths_mm) else float("inf")
    wall = context["wall_coords_mm"]
    if len(wall) > 10000:
        seed = int.from_bytes(
            hashlib.sha256(context["canonical_id"].encode("utf-8")).digest()[:8],
            "little",
        )
        indices = np.sort(np.random.default_rng(seed).choice(len(wall), 10000, replace=False))
        wall = wall[indices]
    coverage_shift = np.asarray(
        context["geometry_translation_to_cfd_wall_mm"], dtype=np.float64
    )
    graph_mm = graph.points_raw * scale + coverage_shift
    radii_mm = np.maximum(graph.radii_raw * scale, 1.0e-6)
    wall_distance, wall_nearest = cKDTree(graph_mm).query(wall, k=1)
    wall_ratio = wall_distance / radii_mm[wall_nearest]
    p95_ratio = float(np.quantile(wall_ratio, 0.95))
    frac_gt2 = float(np.mean(wall_ratio > 2.0))

    target_warning = False
    for path in log_paths:
        if path.is_file() and "Target not reached" in path.read_text(
            encoding="utf-8", errors="replace"
        ):
            target_warning = True
            break

    radius_mm = np.asarray(
        [item.equivalent_radius_raw * scale for item in openings], dtype=np.float64
    )
    endpoint_over_radius = opening_distances / np.maximum(radius_mm, 1.0e-9)
    max_opening_distance = float(np.max(opening_distances))
    max_opening_over_radius = float(np.max(endpoint_over_radius))
    hard_checks = {
        "surface_hash_frozen": True,
        "surface_precheck": True,
        "opening_endpoint_count_equal": len(graph.endpoints) == len(openings),
        "all_openings_uniquely_matched": len(endpoint_assignment) == len(openings),
        "graph_connected": len(graph.components) == 1,
        "graph_acyclic": graph.cycle_rank == 0,
        "no_abnormal_long_graph_edge": bool(max_edge_mm <= 10.0),
        "no_target_not_reached": not target_warning,
        "centerline_mostly_inside_capped_surface": bool(
            np.isfinite(inside_fraction) and inside_fraction >= 0.95
        ),
        "unit_independent_of_new_centerline": True,
    }
    soft_flags = {
        "opening_endpoint_gt_0p5r": bool(max_opening_over_radius > 0.5),
        "wall_distance_p95_gt_2r": bool(p95_ratio > 2.0),
        "wall_fraction_gt_2r_over_5pct": bool(frac_gt2 > 0.05),
        "opening_endpoint_gt_10mm": bool(max_opening_distance > 10.0),
        "branch_node_count_not_3": len(graph.branch_nodes) != 3,
    }
    metrics = {
        "hard_pass": all(hard_checks.values()),
        "manual_review_required": any(soft_flags.values()),
        "hard_checks": hard_checks,
        "soft_flags": soft_flags,
        "centerline_points": len(graph.active_nodes),
        "centerline_edges": len(graph.edges),
        "centerline_endpoints": len(graph.endpoints),
        "centerline_branch_nodes": len(graph.branch_nodes),
        "graph_components": len(graph.components),
        "graph_cycle_rank": graph.cycle_rank,
        "root_node": root_node,
        "inside_fraction": inside_fraction,
        "inside_sample_points": int(len(dense_edge_points)),
        "capped_surface_closed": capped_closed,
        "max_graph_edge_mm": max_edge_mm,
        "p95_graph_edge_mm": p95_edge_mm,
        "target_not_reached_in_log": target_warning,
        "opening_to_endpoint_mm": opening_distances.tolist(),
        "opening_to_endpoint_over_radius": endpoint_over_radius.tolist(),
        "max_opening_to_endpoint_mm": max_opening_distance,
        "max_opening_to_endpoint_over_radius": max_opening_over_radius,
        "wall_distance_over_local_radius_p95": p95_ratio,
        "wall_fraction_gt_2r": frac_gt2,
        "wall_coverage_geometry_translation_mm": coverage_shift.tolist(),
        "wall_coverage_translation_source": context["geometry_translation_source"],
        "endpoint_to_opening": {
            str(node): int(opening_id) for node, opening_id in endpoint_assignment.items()
        },
    }
    return metrics, graph, endpoint_assignment


def _parent_tree(graph: GraphData, root: int) -> tuple[dict[int, int], list[int]]:
    parent = {root: -1}
    order = [root]
    for node in order:
        for neighbor in sorted(graph.adjacency[node]):
            if neighbor not in parent:
                parent[neighbor] = node
                order.append(neighbor)
    return parent, order


def _node_path(parent: dict[int, int], leaf: int) -> list[int]:
    path = []
    current = leaf
    while current >= 0:
        path.append(current)
        current = parent[current]
    return path[::-1]


def _path_features(coords_mm: np.ndarray, radius_mm: np.ndarray, branch_s: np.ndarray) -> dict:
    delta = np.linalg.norm(np.diff(coords_mm, axis=0), axis=1)
    native_s = np.r_[0.0, np.cumsum(delta)]
    length = float(native_s[-1])
    if length <= 1.0e-9:
        raise RuntimeError("Degenerate centerline path")
    sample_s = np.arange(0.0, length, PATH_STEP_MM, dtype=np.float64)
    if not len(sample_s) or sample_s[-1] < length:
        sample_s = np.r_[sample_s, length]
    sampled = np.column_stack(
        [np.interp(sample_s, native_s, coords_mm[:, axis]) for axis in range(3)]
    )
    sampled_radius = np.interp(sample_s, native_s, radius_mm)
    edge_order = 2 if len(sample_s) >= 3 else 1
    velocity = np.gradient(sampled, sample_s, axis=0, edge_order=edge_order)
    speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    tangent = np.divide(velocity, speed, out=np.zeros_like(velocity), where=speed > 1.0e-12)
    dt_ds = np.gradient(tangent, sample_s, axis=0, edge_order=edge_order)
    curvature = np.linalg.norm(dt_ds, axis=1)
    acceleration = np.gradient(velocity, sample_s, axis=0, edge_order=edge_order)
    jerk = np.gradient(acceleration, sample_s, axis=0, edge_order=edge_order)
    cross_va = np.cross(velocity, acceleration)
    denom = np.sum(cross_va * cross_va, axis=1)
    torsion = np.divide(
        np.sum(cross_va * jerk, axis=1),
        denom,
        out=np.zeros(len(sample_s), dtype=np.float64),
        where=denom > 1.0e-20,
    )
    dr_ds = np.gradient(sampled_radius, sample_s, edge_order=edge_order)
    if len(branch_s):
        dist_to_bif = np.min(abs(sample_s[:, None] - branch_s[None, :]), axis=1)
    else:
        dist_to_bif = np.zeros(len(sample_s), dtype=np.float64)
    return {
        "coords": sampled,
        "abscissa": sample_s,
        "radius": sampled_radius,
        "tangent": tangent,
        "curvature": curvature,
        "torsion": torsion,
        "dr_ds": dr_ds,
        "dist_to_bifurcation": dist_to_bif,
        "junction_mask": (dist_to_bif <= JUNCTION_HALF_WIDTH_MM).astype(np.uint8),
    }


def build_outputs(
    graph: GraphData,
    metrics: dict,
    openings: list[Opening],
    endpoint_assignment: dict[int, int],
    scale: float,
) -> tuple[vtk.vtkPolyData, vtk.vtkPolyData, vtk.vtkPolyData, pd.DataFrame]:
    root = int(metrics["root_node"])
    if root < 0:
        raise RuntimeError("Cannot build paths without a root endpoint")
    parent, order = _parent_tree(graph, root)
    opening_by_id = {item.opening_id: item for item in openings}
    outlet_leaf: dict[int, int] = {}
    for endpoint, opening_id in endpoint_assignment.items():
        opening = opening_by_id[opening_id]
        if opening.role == "outlet":
            outlet_leaf[opening.outlet_id] = endpoint
    if len(outlet_leaf) != EXPECTED_OPENINGS - 1:
        raise RuntimeError(f"Expected four outlet leaves, got {len(outlet_leaf)}")

    children = {node: [] for node in order}
    for node, par in parent.items():
        if par >= 0:
            children[par].append(node)
    leaf_to_outlet = {leaf: outlet_id for outlet_id, leaf in outlet_leaf.items()}
    descendant_outlets: dict[int, set[int]] = {}
    for node in reversed(order):
        values = set()
        if node in leaf_to_outlet:
            values.add(leaf_to_outlet[node])
        for child in children.get(node, []):
            values.update(descendant_outlets[child])
        descendant_outlets[node] = values

    graph_points = vtk.vtkPoints()
    graph_points.SetData(numpy_to_vtk(graph.points_raw * scale, deep=1))
    graph_lines = vtk.vtkCellArray()
    edge_color_ids = []
    for a, b in graph.edges:
        if parent.get(b) == a:
            child = b
        elif parent.get(a) == b:
            child = a
        else:
            child = b
        values = descendant_outlets.get(child, set())
        edge_color_ids.append(next(iter(values)) if len(values) == 1 else -1)
        graph_lines.InsertNextCell(2)
        graph_lines.InsertCellPoint(a)
        graph_lines.InsertCellPoint(b)
    graph_poly = vtk.vtkPolyData()
    graph_poly.SetPoints(graph_points)
    graph_poly.SetLines(graph_lines)
    for name, values in (
        ("GraphDegree", graph.degrees.astype(np.int32)),
        ("MaximumInscribedSphereRadius", graph.radii_raw * scale),
    ):
        array = numpy_to_vtk(np.asarray(values), deep=1)
        array.SetName(name)
        graph_poly.GetPointData().AddArray(array)
    color_arr = numpy_to_vtk(np.asarray(edge_color_ids, dtype=np.int32), deep=1)
    color_arr.SetName("BranchColorId")
    graph_poly.GetCellData().AddArray(color_arr)

    path_points = vtk.vtkPoints()
    path_lines = vtk.vtkCellArray()
    point_arrays: dict[str, list] = {
        "PathId": [],
        "OutletId": [],
        "Abscissas": [],
        "MaximumInscribedSphereRadius": [],
        "Curvature": [],
        "Torsion": [],
        "dR_ds": [],
        "DistToBifurcation": [],
        "JunctionMask": [],
        "FrenetTangent": [],
    }
    cell_path_ids, cell_outlet_ids, cell_names = [], [], []
    csv_rows = []
    for path_id, outlet_id in enumerate(sorted(outlet_leaf)):
        leaf = outlet_leaf[outlet_id]
        nodes = _node_path(parent, leaf)
        coords_mm = graph.points_raw[nodes] * scale
        radii_mm = graph.radii_raw[nodes] * scale
        native_s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(coords_mm, axis=0), axis=1))]
        branch_s = np.asarray(
            [native_s[index] for index, node in enumerate(nodes) if graph.degrees[node] >= 3],
            dtype=np.float64,
        )
        feats = _path_features(coords_mm, radii_mm, branch_s)
        start = path_points.GetNumberOfPoints()
        path_lines.InsertNextCell(len(feats["coords"]))
        opening = next(item for item in openings if item.outlet_id == outlet_id)
        for local_index, xyz in enumerate(feats["coords"]):
            point_id = path_points.InsertNextPoint(*[float(value) for value in xyz])
            path_lines.InsertCellPoint(point_id)
            point_arrays["PathId"].append(path_id)
            point_arrays["OutletId"].append(outlet_id)
            point_arrays["Abscissas"].append(feats["abscissa"][local_index])
            point_arrays["MaximumInscribedSphereRadius"].append(feats["radius"][local_index])
            point_arrays["Curvature"].append(feats["curvature"][local_index])
            point_arrays["Torsion"].append(feats["torsion"][local_index])
            point_arrays["dR_ds"].append(feats["dr_ds"][local_index])
            point_arrays["DistToBifurcation"].append(
                feats["dist_to_bifurcation"][local_index]
            )
            point_arrays["JunctionMask"].append(feats["junction_mask"][local_index])
            point_arrays["FrenetTangent"].append(feats["tangent"][local_index])
            csv_rows.append(
                {
                    "PathId": path_id,
                    "OutletId": outlet_id,
                    "OutletName": opening.outlet_name,
                    "PointIndex": local_index,
                    "X_mm": xyz[0],
                    "Y_mm": xyz[1],
                    "Z_mm": xyz[2],
                    "Abscissas_mm": feats["abscissa"][local_index],
                    "MaximumInscribedSphereRadius_mm": feats["radius"][local_index],
                    "Tangent_X": feats["tangent"][local_index, 0],
                    "Tangent_Y": feats["tangent"][local_index, 1],
                    "Tangent_Z": feats["tangent"][local_index, 2],
                    "Curvature_per_mm": feats["curvature"][local_index],
                    "Torsion_per_mm": feats["torsion"][local_index],
                    "dR_ds": feats["dr_ds"][local_index],
                    "DistToBifurcation_mm": feats["dist_to_bifurcation"][local_index],
                    "JunctionMask": int(feats["junction_mask"][local_index]),
                }
            )
        cell_path_ids.append(path_id)
        cell_outlet_ids.append(outlet_id)
        cell_names.append(opening.outlet_name)
        assert path_points.GetNumberOfPoints() > start

    paths_poly = vtk.vtkPolyData()
    paths_poly.SetPoints(path_points)
    paths_poly.SetLines(path_lines)
    for name, values in point_arrays.items():
        array = numpy_to_vtk(np.asarray(values), deep=1)
        array.SetName(name)
        paths_poly.GetPointData().AddArray(array)
    for name, values in (
        ("PathId", np.asarray(cell_path_ids, dtype=np.int32)),
        ("OutletId", np.asarray(cell_outlet_ids, dtype=np.int32)),
    ):
        array = numpy_to_vtk(values, deep=1)
        array.SetName(name)
        paths_poly.GetCellData().AddArray(array)
    names = vtk.vtkStringArray()
    names.SetName("OutletName")
    for name in cell_names:
        names.InsertNextValue(name)
    paths_poly.GetCellData().AddArray(names)

    opening_points = vtk.vtkPoints()
    opening_vertices = vtk.vtkCellArray()
    for opening in openings:
        point_id = opening_points.InsertNextPoint(
            *[float(value) * scale for value in opening.center_raw]
        )
        opening_vertices.InsertNextCell(1)
        opening_vertices.InsertCellPoint(point_id)
    opening_poly = vtk.vtkPolyData()
    opening_poly.SetPoints(opening_points)
    opening_poly.SetVerts(opening_vertices)
    for name, values in (
        ("OpeningId", [item.opening_id for item in openings]),
        ("IsInlet", np.asarray([item.role == "inlet" for item in openings], dtype=np.uint8)),
        ("OutletId", [item.outlet_id for item in openings]),
        ("EquivalentRadiusMm", [item.equivalent_radius_raw * scale for item in openings]),
    ):
        array = numpy_to_vtk(np.asarray(values), deep=1)
        array.SetName(name)
        opening_poly.GetPointData().AddArray(array)

    return graph_poly, paths_poly, opening_poly, pd.DataFrame(csv_rows)


def build_basic_outputs(
    graph: GraphData, openings: list[Opening], scale: float
) -> tuple[vtk.vtkPolyData, vtk.vtkPolyData]:
    """Build reviewable graph/opening outputs when a failed graph has <4 outlets."""
    graph_points = vtk.vtkPoints()
    graph_points.SetData(numpy_to_vtk(graph.points_raw * scale, deep=1))
    graph_lines = vtk.vtkCellArray()
    for a, b in graph.edges:
        graph_lines.InsertNextCell(2)
        graph_lines.InsertCellPoint(a)
        graph_lines.InsertCellPoint(b)
    graph_poly = vtk.vtkPolyData()
    graph_poly.SetPoints(graph_points)
    graph_poly.SetLines(graph_lines)
    for name, values in (
        ("GraphDegree", graph.degrees.astype(np.int32)),
        ("MaximumInscribedSphereRadius", graph.radii_raw * scale),
    ):
        array = numpy_to_vtk(np.asarray(values), deep=1)
        array.SetName(name)
        graph_poly.GetPointData().AddArray(array)
    colors = numpy_to_vtk(
        np.full(len(graph.edges), -1, dtype=np.int32), deep=1
    )
    colors.SetName("BranchColorId")
    graph_poly.GetCellData().AddArray(colors)

    opening_points = vtk.vtkPoints()
    opening_vertices = vtk.vtkCellArray()
    for opening in openings:
        point_id = opening_points.InsertNextPoint(
            *[float(value) * scale for value in opening.center_raw]
        )
        opening_vertices.InsertNextCell(1)
        opening_vertices.InsertCellPoint(point_id)
    opening_poly = vtk.vtkPolyData()
    opening_poly.SetPoints(opening_points)
    opening_poly.SetVerts(opening_vertices)
    for name, values in (
        ("OpeningId", [item.opening_id for item in openings]),
        ("IsInlet", np.asarray([item.role == "inlet" for item in openings], dtype=np.uint8)),
        ("OutletId", [item.outlet_id for item in openings]),
        ("EquivalentRadiusMm", [item.equivalent_radius_raw * scale for item in openings]),
    ):
        array = numpy_to_vtk(np.asarray(values), deep=1)
        array.SetName(name)
        opening_poly.GetPointData().AddArray(array)
    return graph_poly, opening_poly


def _attempt(
    surface: vtk.vtkPolyData,
    openings: list[Opening],
    context: dict,
    attempt_dir: Path,
    mode: str,
    seed_shift_frac: float,
) -> tuple[vtk.vtkPolyData, list[Path]]:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    inlet = next(item for item in openings if item.role == "inlet")
    outlets = sorted(
        [item for item in openings if item.role == "outlet"], key=lambda item: item.outlet_id
    )

    def seed(opening: Opening) -> np.ndarray:
        return np.asarray(opening.center_raw) + seed_shift_frac * opening.equivalent_radius_raw * np.asarray(
            opening.inward_normal_raw
        )

    source = seed(inlet)
    outputs = []
    logs = []
    if mode == "all_targets":
        log_path = attempt_dir / "vmtk_all_targets.log"
        logs.append(log_path)
        outputs.append(run_vmtk(surface, source, [seed(item) for item in outlets], log_path))
    elif mode == "per_target":
        for outlet in outlets:
            log_path = attempt_dir / f"vmtk_target_{outlet.outlet_id}_{outlet.outlet_name}.log"
            logs.append(log_path)
            try:
                outputs.append(run_vmtk(surface, source, [seed(outlet)], log_path))
            except Exception:
                with log_path.open("a", encoding="utf-8") as handle:
                    traceback.print_exc(file=handle)
    else:
        raise ValueError(mode)
    diag = float(np.linalg.norm(np.ptp(surface_points(surface), axis=0)))
    merged = append_and_clean(outputs, diag)
    write_vtp(merged, attempt_dir / "centerline_raw.vtp")
    return merged, logs


def process_case(entry: dict, output_root: Path) -> dict:
    started = time.time()
    context = load_case_context(entry)
    canonical_id = context["canonical_id"]
    case_dir = output_root / "cases" / canonical_id
    case_dir.mkdir(parents=True, exist_ok=True)
    surface = read_stl(context["authoritative_stl"])
    precheck, openings = surface_precheck(surface)
    assignment = assign_opening_roles(openings, context) if openings else {}
    surface_manifest = {
        "canonical_id": canonical_id,
        "selection_source": "existing_bundle_authoritative_stl",
        "authoritative_stl": str(context["authoritative_stl"]),
        "authoritative_stl_sha256": context["authoritative_stl_sha256"],
        "authoritative_stl_scale_to_mm": context["stl_scale_to_mm"],
        "fluent_to_mm_unit_factor": context["unit_factor"],
        "unit_source": "existing_bundle_pre_centerline_freeze",
        "unit_override_applied": context["unit_override_applied"],
        "unit_override_reason": context["unit_override_reason"],
        "legacy_centerline_translation_applied": context[
            "legacy_centerline_translation_applied"
        ],
        "legacy_centerline_translation_mm": context[
            "legacy_centerline_translation_mm"
        ],
        "geometry_translation_to_cfd_wall_mm": context[
            "geometry_translation_to_cfd_wall_mm"
        ],
        "geometry_translation_source": context["geometry_translation_source"],
        "candidates": context["surface_candidates"],
        "precheck": precheck,
        "opening_assignment": assignment,
    }
    write_json(case_dir / "surface_selection.json", surface_manifest)

    if not precheck["hard_pass"]:
        result = {
            "canonical_id": canonical_id,
            "status": "surface_precheck_failed",
            "hard_pass": False,
            "manual_review_required": True,
            "selected_attempt": "",
            "elapsed_s": time.time() - started,
            "precheck": precheck,
        }
        write_json(case_dir / "result.json", result)
        return result

    attempts = [
        ("attempt_1_standard", surface, "all_targets", 0.0),
        ("attempt_2_per_target", surface, "per_target", 0.0),
        ("attempt_3_clean_shifted", lightly_clean_surface(surface), "per_target", 0.35),
    ]
    selected = None
    best = None
    attempt_rows = []
    for name, attempt_surface, mode, shift_frac in attempts:
        attempt_dir = case_dir / "attempts" / name
        try:
            poly, logs = _attempt(
                attempt_surface, openings, context, attempt_dir, mode, shift_frac
            )
            poly, spike_pruning = prune_terminal_spikes(
                poly, openings, context["stl_scale_to_mm"]
            )
            if spike_pruning["applied"]:
                write_vtp(poly, attempt_dir / "centerline_terminal_spikes_pruned.vtp")
            poly, long_edge_resampling = resample_safe_long_edges(
                poly,
                attempt_surface,
                context["stl_scale_to_mm"],
            )
            if long_edge_resampling["applied"]:
                write_vtp(poly, attempt_dir / "centerline_long_edges_resampled.vtp")
            metrics, graph, endpoint_assignment = evaluate_centerline(
                poly, attempt_surface, openings, context, logs
            )
            metrics["terminal_spike_pruning_applied"] = spike_pruning["applied"]
            metrics["terminal_spikes_pruned"] = spike_pruning["n_pruned"]
            metrics["long_edge_resampling_applied"] = long_edge_resampling["applied"]
            metrics["long_edges_resampled"] = long_edge_resampling["n_edges_resampled"]
            metrics["long_edge_inserted_points"] = long_edge_resampling[
                "inserted_points"
            ]
            row = {
                "name": name,
                "mode": mode,
                "seed_shift_fraction_of_opening_radius": shift_frac,
                "terminal_spike_pruning": spike_pruning,
                "long_edge_resampling": long_edge_resampling,
                "metrics": metrics,
                "logs": [str(path.relative_to(case_dir)) for path in logs],
            }
            write_json(attempt_dir / "gate_metrics.json", row)
            attempt_rows.append(row)
            score = (
                int(metrics["hard_pass"]),
                -sum(metrics["soft_flags"].values()),
                -metrics["max_opening_to_endpoint_over_radius"],
            )
            candidate = (score, name, attempt_surface, poly, metrics, graph, endpoint_assignment)
            if best is None or candidate[0] > best[0]:
                best = candidate
            if metrics["hard_pass"]:
                selected = candidate
                break
        except Exception as exc:  # noqa: BLE001
            error = {
                "name": name,
                "mode": mode,
                "seed_shift_fraction_of_opening_radius": shift_frac,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(attempt_dir / "error.json", error)
            attempt_rows.append(error)

    if selected is None:
        selected = best
    if selected is None:
        result = {
            "canonical_id": canonical_id,
            "status": "all_attempts_failed",
            "hard_pass": False,
            "manual_review_required": True,
            "selected_attempt": "",
            "elapsed_s": time.time() - started,
            "attempts": attempt_rows,
        }
        write_json(case_dir / "result.json", result)
        return result

    _, selected_name, selected_surface, poly, metrics, graph, endpoint_assignment = selected
    surface_mm = transform_polydata(selected_surface, context["stl_scale_to_mm"])
    write_stl(surface_mm, case_dir / "surface_authoritative_mm.stl")
    path_build_error = ""
    try:
        graph_poly, paths_poly, opening_poly, path_frame = build_outputs(
            graph,
            metrics,
            openings,
            endpoint_assignment,
            context["stl_scale_to_mm"],
        )
    except Exception as exc:  # failed cases still need graph/opening review files
        if metrics["hard_pass"]:
            raise
        graph_poly, opening_poly = build_basic_outputs(
            graph, openings, context["stl_scale_to_mm"]
        )
        paths_poly = None
        path_frame = None
        path_build_error = str(exc)
    write_vtp(graph_poly, case_dir / "centerline_graph_mm.vtp")
    write_vtp(opening_poly, case_dir / "openings_mm.vtp")
    outputs = {
        "surface": "surface_authoritative_mm.stl",
        "graph": "centerline_graph_mm.vtp",
        "openings": "openings_mm.vtp",
    }
    if paths_poly is not None and path_frame is not None:
        write_vtp(paths_poly, case_dir / "centerline_paths_mm.vtp")
        path_frame.to_csv(case_dir / "centerline_paths_mm.csv", index=False)
        outputs.update(
            {
                "paths": "centerline_paths_mm.vtp",
                "path_csv": "centerline_paths_mm.csv",
            }
        )
    result = {
        "canonical_id": canonical_id,
        "status": "pass_review" if metrics["hard_pass"] and metrics["manual_review_required"] else (
            "pass" if metrics["hard_pass"] else "hard_gate_failed"
        ),
        "hard_pass": metrics["hard_pass"],
        "manual_review_required": metrics["manual_review_required"],
        "selected_attempt": selected_name,
        "elapsed_s": time.time() - started,
        "metrics": metrics,
        "attempts": attempt_rows,
        "path_build_error": path_build_error,
        "outputs": outputs,
    }
    write_json(case_dir / "result.json", result)
    return result


def _threshold_cells(poly: vtk.vtkPolyData, value: int) -> vtk.vtkPolyData:
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(poly)
    threshold.SetInputArrayToProcess(
        0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "BranchColorId"
    )
    threshold.SetLowerThreshold(value)
    threshold.SetUpperThreshold(value)
    threshold.Update()
    geom = vtk.vtkGeometryFilter()
    geom.SetInputData(threshold.GetOutput())
    geom.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(geom.GetOutput())
    return out


def _sphere_actor(center, radius, color):
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(*[float(value) for value in center])
    sphere.SetRadius(float(radius))
    sphere.SetThetaResolution(18)
    sphere.SetPhiResolution(18)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    return actor


def render_case(case_dir: Path, result: dict) -> tuple[Path, Path]:
    surface = read_stl(case_dir / "surface_authoritative_mm.stl")
    graph_reader = vtk.vtkXMLPolyDataReader()
    graph_reader.SetFileName(str(case_dir / "centerline_graph_mm.vtp"))
    graph_reader.Update()
    graph = graph_reader.GetOutput()
    opening_reader = vtk.vtkXMLPolyDataReader()
    opening_reader.SetFileName(str(case_dir / "openings_mm.vtp"))
    opening_reader.Update()
    openings = opening_reader.GetOutput()
    pts = surface_points(surface)
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    diag = float(np.linalg.norm(np.ptp(pts, axis=0)))
    _, _, vt = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
    long_axis, side_axis, depth_axis = vt[0], vt[1], vt[2]

    def render(direction: np.ndarray, path: Path, label: str) -> None:
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(1.0, 1.0, 1.0)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(surface)
        surface_actor = vtk.vtkActor()
        surface_actor.SetMapper(mapper)
        surface_actor.GetProperty().SetColor(0.72, 0.76, 0.80)
        surface_actor.GetProperty().SetOpacity(0.20)
        renderer.AddActor(surface_actor)
        for color_id, color in RENDER_COLORS.items():
            part = _threshold_cells(graph, color_id)
            if part.GetNumberOfLines() == 0:
                continue
            tube = vtk.vtkTubeFilter()
            tube.SetInputData(part)
            tube.SetRadius(max(diag * 0.004, 0.35))
            tube.SetNumberOfSides(12)
            tube.CappingOn()
            tube.Update()
            line_mapper = vtk.vtkPolyDataMapper()
            line_mapper.SetInputData(tube.GetOutput())
            actor = vtk.vtkActor()
            actor.SetMapper(line_mapper)
            actor.GetProperty().SetColor(*color)
            renderer.AddActor(actor)
        opening_xyz = surface_points(openings)
        inlet_arr = vtk_to_numpy(openings.GetPointData().GetArray("IsInlet")).astype(bool)
        outlet_arr = vtk_to_numpy(openings.GetPointData().GetArray("OutletId")).astype(int)
        radii = vtk_to_numpy(openings.GetPointData().GetArray("EquivalentRadiusMm"))
        for index, xyz in enumerate(opening_xyz):
            if inlet_arr[index]:
                color = (0.55, 0.05, 0.75)
            else:
                color = RENDER_COLORS.get(int(outlet_arr[index]), (0.2, 0.2, 0.2))
            renderer.AddActor(_sphere_actor(xyz, max(float(radii[index]) * 0.22, diag * 0.006), color))
        camera = renderer.GetActiveCamera()
        direction = direction / max(np.linalg.norm(direction), 1.0e-12)
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(center + direction * diag * 1.8))
        camera.SetViewUp(*long_axis)
        camera.ParallelProjectionOn()
        renderer.ResetCamera()
        camera.Zoom(1.15)
        title = vtk.vtkTextActor()
        title.SetInput(
            f"{result['canonical_id']} | {result['status']} | {label}\n"
            f"endpoints={result['metrics']['centerline_endpoints']}  "
            f"branches={result['metrics']['centerline_branch_nodes']}  "
            f"max end-open={result['metrics']['max_opening_to_endpoint_mm']:.2f} mm"
        )
        title.GetTextProperty().SetFontSize(19)
        title.GetTextProperty().SetColor(0.05, 0.05, 0.05)
        title.SetPosition(18, 18)
        renderer.AddActor2D(title)
        window = vtk.vtkRenderWindow()
        window.SetOffScreenRendering(1)
        window.SetSize(720, 900)
        window.SetMultiSamples(0)
        window.AddRenderer(renderer)
        window.Render()
        image_filter = vtk.vtkWindowToImageFilter()
        image_filter.SetInput(window)
        image_filter.SetScale(1)
        image_filter.ReadFrontBufferOff()
        image_filter.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(image_filter.GetOutputPort())
        writer.Write()
        window.Finalize()

    iso_path = case_dir / "review_iso.png"
    render(side_axis + 0.55 * depth_axis, iso_path, "ISO")
    view_paths = []
    for name, direction in (("front", side_axis), ("side", depth_axis), ("iso", side_axis + depth_axis)):
        path = case_dir / f"review_{name}.png"
        render(direction, path, name.upper())
        view_paths.append(path)

    tri_path = case_dir / "review_three_views.png"
    append = vtk.vtkImageAppend()
    append.SetAppendAxis(0)
    readers = []
    for path in view_paths:
        reader = vtk.vtkPNGReader()
        reader.SetFileName(str(path))
        reader.Update()
        readers.append(reader)
        append.AddInputData(reader.GetOutput())
    append.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(tri_path))
    writer.SetInputData(append.GetOutput())
    writer.Write()
    return iso_path, tri_path


def build_contact_sheets(
    output_root: Path,
    results: list[dict],
    page_size: int,
    *,
    force_render: bool = False,
) -> list[Path]:
    rendered = []
    for result in results:
        if not result.get("selected_attempt") or "metrics" not in result:
            continue
        case_dir = output_root / "cases" / result["canonical_id"]
        iso = case_dir / "review_iso.png"
        tri = case_dir / "review_three_views.png"
        recorded_outputs = result.get("outputs", {})
        has_current_render_record = bool(
            recorded_outputs.get("review_iso")
            and recorded_outputs.get("review_three_views")
        )
        if force_render or not has_current_render_record or not iso.is_file() or not tri.is_file():
            iso, tri = render_case(case_dir, result)
        result.setdefault("outputs", {})["review_iso"] = str(iso.relative_to(case_dir))
        result["outputs"]["review_three_views"] = str(tri.relative_to(case_dir))
        write_json(case_dir / "result.json", result)
        rendered.append((result, iso))
    if not rendered:
        return []

    cols = 5
    page_size = max(cols, int(page_size))
    page_size = int(math.ceil(page_size / cols) * cols)
    pages = []
    for page_start in range(0, len(rendered), page_size):
        page_items = rendered[page_start : page_start + page_size]
        rows = math.ceil(len(page_items) / cols)
        png_readers = []
        image_rows = []
        for row_index in range(rows):
            row_append = vtk.vtkImageAppend()
            row_append.SetAppendAxis(0)
            for column_index in range(cols):
                index = row_index * cols + column_index
                if index < len(page_items):
                    reader = vtk.vtkPNGReader()
                    reader.SetFileName(str(page_items[index][1]))
                    reader.Update()
                    png_readers.append(reader)
                    row_append.AddInputData(reader.GetOutput())
                else:
                    blank = vtk.vtkImageCanvasSource2D()
                    blank.SetScalarTypeToUnsignedChar()
                    blank.SetNumberOfScalarComponents(3)
                    blank.SetExtent(0, 719, 0, 899, 0, 0)
                    blank.SetDrawColor(255, 255, 255)
                    blank.FillBox(0, 719, 0, 899)
                    blank.Update()
                    png_readers.append(blank)
                    row_append.AddInputData(blank.GetOutput())
            row_append.Update()
            row_image = vtk.vtkImageData()
            row_image.DeepCopy(row_append.GetOutput())
            image_rows.append(row_image)
        vertical = vtk.vtkImageAppend()
        vertical.SetAppendAxis(1)
        for row_image in reversed(image_rows):
            vertical.AddInputData(row_image)
        vertical.Update()
        if len(rendered) <= page_size:
            out = output_root / "centerline_v2_contact_sheet.png"
        else:
            page_number = len(pages) + 1
            out = output_root / f"centerline_v2_contact_sheet_page_{page_number:03d}.png"
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(out))
        writer.SetInputData(vertical.GetOutput())
        writer.Write()
        pages.append(out)
    write_json(
        output_root / "contact_sheet_index.json",
        {
            "page_size": page_size,
            "pages": [str(path.relative_to(output_root)) for path in pages],
        },
    )
    return pages


def write_summary(output_root: Path, results: list[dict]) -> None:
    rows = []
    for result in results:
        metrics = result.get("metrics", {})
        rows.append(
            {
                "canonical_id": result["canonical_id"],
                "status": result["status"],
                "hard_pass": result.get("hard_pass", False),
                "manual_review_required": result.get("manual_review_required", True),
                "selected_attempt": result.get("selected_attempt", ""),
                "endpoints": metrics.get("centerline_endpoints"),
                "branch_nodes": metrics.get("centerline_branch_nodes"),
                "max_opening_to_endpoint_mm": metrics.get("max_opening_to_endpoint_mm"),
                "max_opening_to_endpoint_over_radius": metrics.get(
                    "max_opening_to_endpoint_over_radius"
                ),
                "wall_distance_over_local_radius_p95": metrics.get(
                    "wall_distance_over_local_radius_p95"
                ),
                "wall_fraction_gt_2r": metrics.get("wall_fraction_gt_2r"),
                "inside_fraction": metrics.get("inside_fraction"),
                "terminal_spike_pruning_applied": metrics.get(
                    "terminal_spike_pruning_applied", False
                ),
                "terminal_spikes_pruned": metrics.get("terminal_spikes_pruned", 0),
                "long_edge_resampling_applied": metrics.get(
                    "long_edge_resampling_applied", False
                ),
                "long_edges_resampled": metrics.get("long_edges_resampled", 0),
                "long_edge_inserted_points": metrics.get(
                    "long_edge_inserted_points", 0
                ),
                "hard_fail_reasons": ";".join(
                    key
                    for key, value in metrics.get("hard_checks", {}).items()
                    if not value
                ),
                "soft_flags": ";".join(
                    key
                    for key, value in metrics.get("soft_flags", {}).items()
                    if value
                ),
                "elapsed_s": result.get("elapsed_s"),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "pilot_summary.csv", index=False, encoding="utf-8-sig")
    write_json(output_root / "pilot_summary.json", rows)


def normalize_result_schema(result: dict) -> dict:
    """Backfill explicit no-op repair fields when reusing an older case result."""
    metrics = result.get("metrics")
    if metrics is None:
        return result
    metrics.setdefault("long_edge_resampling_applied", False)
    metrics.setdefault("long_edges_resampled", 0)
    metrics.setdefault("long_edge_inserted_points", 0)
    for attempt in result.get("attempts", []):
        attempt_metrics = attempt.get("metrics")
        if attempt_metrics is None:
            continue
        attempt_metrics.setdefault("long_edge_resampling_applied", False)
        attempt_metrics.setdefault("long_edges_resampled", 0)
        attempt_metrics.setdefault("long_edge_inserted_points", 0)
        attempt.setdefault(
            "long_edge_resampling",
            {
                "applied": False,
                "n_edges_resampled": 0,
                "inserted_points": 0,
                "threshold_mm": 10.0,
                "target_segment_mm": 2.0,
                "actions": [],
                "skipped_long_edges": [],
                "final_max_edge_mm": attempt_metrics.get("max_graph_edge_mm"),
            },
        )
    return result


def select_entries(manifest: dict, count: int, seed: int, explicit: list[str]) -> list[dict]:
    entries = manifest["cases"]
    if explicit:
        by_id = {entry["canonical_id"]: entry for entry in entries}
        missing = [case_id for case_id in explicit if case_id not in by_id]
        if missing:
            raise ValueError(f"Unknown cases: {missing}")
        return [by_id[case_id] for case_id in explicit]
    if count > len(entries):
        raise ValueError(f"Requested {count} cases from a {len(entries)}-case manifest")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(entries), size=count, replace=False))
    return [entries[int(index)] for index in indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--contact-page-size", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = select_entries(manifest, args.count, args.seed, args.case)
    selection = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "source_manifest": str(args.manifest.resolve()),
        "source_case_count": len(manifest["cases"]),
        "random_seed": args.seed,
        "requested_count": len(entries),
        "cases": [entry["canonical_id"] for entry in entries],
        "write_policy": "versioned_output_only_do_not_overwrite_data_new",
    }
    write_json(output_root / "pilot_manifest.json", selection)
    print(json.dumps(selection, ensure_ascii=False), flush=True)

    results = []
    for index, entry in enumerate(entries, start=1):
        case_result_path = output_root / "cases" / entry["canonical_id"] / "result.json"
        if case_result_path.is_file() and not args.force:
            result = normalize_result_schema(
                json.loads(case_result_path.read_text(encoding="utf-8"))
            )
            print(
                f"[{index:02d}/{len(entries):02d}] reuse {entry['canonical_id']} {result['status']}",
                flush=True,
            )
        else:
            print(f"[{index:02d}/{len(entries):02d}] process {entry['canonical_id']}", flush=True)
            try:
                result = process_case(entry, output_root)
            except Exception as exc:  # noqa: BLE001
                result = {
                    "canonical_id": entry["canonical_id"],
                    "status": "processing_error",
                    "hard_pass": False,
                    "manual_review_required": True,
                    "selected_attempt": "",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                write_json(case_result_path, result)
            print(
                f"[{index:02d}/{len(entries):02d}] done {entry['canonical_id']} "
                f"status={result['status']} attempt={result.get('selected_attempt', '')}",
                flush=True,
            )
        result = normalize_result_schema(result)
        results.append(result)
        write_summary(output_root, results)

    contact_sheets = build_contact_sheets(
        output_root,
        results,
        page_size=args.contact_page_size,
        force_render=args.force,
    )
    write_summary(output_root, results)
    final = {
        "output": str(output_root),
        "cases": len(results),
        "hard_pass": sum(bool(item.get("hard_pass")) for item in results),
        "manual_review": sum(bool(item.get("manual_review_required")) for item in results),
        "errors": sum(item.get("status") in {"processing_error", "all_attempts_failed"} for item in results),
        "contact_sheet": str(contact_sheets[0]) if contact_sheets else "",
        "contact_sheets": [str(path) for path in contact_sheets],
    }
    write_json(output_root / "run_result.json", final)
    print(json.dumps(final, ensure_ascii=False), flush=True)
    return 0 if contact_sheets else 2


if __name__ == "__main__":
    raise SystemExit(main())
