#!/usr/bin/env python3
"""Centerline V2 extraction from the Fluent case's own anatomy wall.

For cases whose authoritative STL is not the surface the CFD mesh was built
from (2026-09-04: ``AAA/ruputer/ZHOU_KE_XUN``, ``AAA/unruputer/LIU_WEN_QI``,
``ILO/YU_XIANG_SHENG-1/before``), this driver

* rebuilds the anatomy wall surface directly from the ``.cas`` topology
  (wall faces adjacent to the ``blood`` cell zone, polygons fan-triangulated,
  coordinates in true millimetres = Fluent metres x 1000);
* assigns the five openings to ``inlet / out-le / out-li / out-ri / out-re``
  from the ``blood <-> bloodN`` interface topology instead of the legacy
  bbox/boundary heuristics;
* runs the unchanged Centerline V2 attempt/gate/output machinery of
  ``tools/centerline_v2_pilot.py`` and writes the same case layout
  (``surface_selection.json``, ``result.json``, ``centerline_graph_mm.vtp``,
  ``centerline_paths_mm.csv``, ``openings_mm.vtp``, review renders) into a
  new dated root so the feature atlas can consume it with ``unit_rescale=1``.

Run inside the ``GNN_vmtk`` environment:

    /public/newhome/cy/.conda/envs/GNN_vmtk/bin/python tools/centerline_v2_meshwall.py \
        --output outputs/centerline_v2_meshwall_20260904 \
        --cases AAA/ruputer/ZHOU_KE_XUN AAA/unruputer/LIU_WEN_QI ILO/YU_XIANG_SHENG-1/before
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import vtk
from scipy.optimize import linear_sum_assignment
from vtk.util.numpy_support import numpy_to_vtk

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tools"))

import centerline_v2_pilot as pilot  # noqa: E402

# ``wss_pinn.v4`` imports torch at package level, which the vmtk environment
# lacks; load the dependency-free topology module straight from its file.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "fluent_topology", PROJECT / "wss_pinn" / "v4" / "fluent_topology.py"
)
fluent_topology = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["fluent_topology"] = fluent_topology  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(fluent_topology)
anatomy_topology = fluent_topology.anatomy_topology
anatomy_wall_faces = fluent_topology.anatomy_wall_faces
read_fluent_mesh = fluent_topology.read_fluent_mesh

OUTLET_ORDER = ("out-le", "out-li", "out-ri", "out-re")
BOUNDARY_ROOT = PROJECT / "data_wss_pinn/volume_uvwp_peak_qs_smooth_v3_train123_val15_test35/cases"
DEFAULT_OUTPUT = PROJECT / "outputs/centerline_v2_meshwall_20260904"
FLUENT_TO_MM = 1000.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _outlet_semantics(canonical_id: str) -> dict[int, str]:
    manifest = json.loads((BOUNDARY_ROOT / canonical_id / "manifest.json").read_text(encoding="utf-8"))
    return {int(row["mesh_zone_id"]): label for label, row in manifest["outlets"].items()}


def _case_file(canonical_id: str) -> Path:
    manifest = json.loads((BOUNDARY_ROOT / canonical_id / "manifest.json").read_text(encoding="utf-8"))
    return Path(manifest["provenance"]["fluent_case"]["path"])


def build_wall_surface(mesh, wall: dict) -> tuple[vtk.vtkPolyData, dict]:
    """Fan-triangulated, outward-oriented anatomy wall in true millimetres."""

    offsets = wall["face_offsets"]
    nodes = wall["face_nodes"]
    counts = np.diff(offsets)
    points = mesh.nodes_m * FLUENT_TO_MM  # index 0 unused
    centres = wall["centres_m"] * FLUENT_TO_MM
    outward = mesh.c0_sign  # +1: stored node order already gives outward RHR normals
    triangles = []
    extra_points = []
    next_id = len(points)
    for face in range(len(counts)):
        ids = nodes[offsets[face] : offsets[face + 1]]
        if outward < 0:
            ids = ids[::-1]
        if len(ids) == 3:
            triangles.append([int(ids[0]), int(ids[1]), int(ids[2])])
        else:
            centre_id = next_id
            next_id += 1
            extra_points.append(centres[face])
            for k in range(len(ids)):
                triangles.append([centre_id, int(ids[k]), int(ids[(k + 1) % len(ids)])])
    all_points = np.vstack([points, np.asarray(extra_points).reshape(-1, 3)]) if extra_points else points
    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(all_points, dtype=np.float64), deep=1))
    poly.SetPoints(vtk_points)
    cells = vtk.vtkCellArray()
    tri = np.asarray(triangles, dtype=np.int64)
    connectivity = np.column_stack([np.full(len(tri), 3, dtype=np.int64), tri]).reshape(-1)
    cells.SetCells(len(tri), numpy_to_vtk(connectivity, deep=1, array_type=vtk.VTK_ID_TYPE))
    poly.SetPolys(cells)
    # drop unused points (index 0 and non-wall nodes) and merge exactly coincident ones
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.PointMergingOn()
    clean.SetTolerance(0.0)
    clean.ConvertLinesToPointsOff()
    clean.ConvertPolysToLinesOff()
    clean.ConvertStripsToPolysOff()
    clean.Update()
    cleaned = vtk.vtkPolyData()
    cleaned.DeepCopy(clean.GetOutput())
    info = {
        "wall_faces": int(len(counts)),
        "polygon_faces_fanned": int(np.sum(counts > 3)),
        "triangles": int(cleaned.GetNumberOfPolys()),
        "points": int(cleaned.GetNumberOfPoints()),
        "unique_wall_nodes": int(len(wall["node_ids"])),
        "c0_sign": float(outward),
    }
    return cleaned, info


def assign_roles_from_topology(openings, interfaces, semantics: dict[int, str]) -> dict:
    """Match detected boundary loops to the topological interfaces (true mm)."""

    interface_centres = []
    labels = []
    for interface in interfaces:
        weights = np.linalg.norm(interface.area_vectors_m2, axis=1)
        interface_centres.append(np.average(interface.centres_m, axis=0, weights=weights) * FLUENT_TO_MM)
        if interface.distal_bc_type in (4, 10, 20):
            labels.append("inlet")
        else:
            labels.append(semantics.get(int(interface.distal_bc_zone_id), ""))
    interface_centres = np.asarray(interface_centres)
    opening_centres = np.asarray([item.center_raw for item in openings], dtype=np.float64)
    cost = np.linalg.norm(opening_centres[:, None, :] - interface_centres[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    rows_out = []
    for row, col in zip(rows.tolist(), cols.tolist()):
        opening = openings[row]
        label = labels[col]
        opening.contract_distance_norm = float(cost[row, col])
        if label == "inlet":
            opening.role = "inlet"
            opening.outlet_id = -1
            opening.outlet_name = ""
        else:
            opening.role = "outlet"
            opening.outlet_id = OUTLET_ORDER.index(label)
            opening.outlet_name = label
        rows_out.append(
            {
                "opening_id": opening.opening_id,
                "interface_extension_zone": interfaces[col].extension_zone_name,
                "interface_distal_bc": interfaces[col].distal_bc_name,
                "semantic_label": label,
                "centre_distance_mm": float(cost[row, col]),
                "opening_area_mm2": opening.area_raw2,
                "interface_area_mm2": interfaces[col].area_m2 * FLUENT_TO_MM**2,
            }
        )
    assigned = sorted(item.role + (item.outlet_name or "") for item in openings)
    expected = sorted(["inlet", *["outlet" + name for name in OUTLET_ORDER]])
    return {
        "assignment_source": "fluent_interface_topology",
        "complete": assigned == expected,
        "max_centre_distance_mm": float(np.max(cost[rows, cols])),
        "matches": rows_out,
        "openings": [asdict(item) for item in openings],
    }


def process_meshwall_case(canonical_id: str, output_root: Path) -> dict:
    started = time.time()
    case_dir = output_root / "cases" / canonical_id
    case_dir.mkdir(parents=True, exist_ok=True)
    case_path = _case_file(canonical_id)
    mesh = read_fluent_mesh(case_path)
    topology = anatomy_topology(mesh)
    wall = anatomy_wall_faces(mesh, topology)
    surface, surface_info = build_wall_surface(mesh, wall)
    stl_path = case_dir / "surface_meshwall_true_mm.stl"
    pilot.write_stl(surface, stl_path)
    surface = pilot.read_stl(stl_path)  # identical round trip to the pilot's reader
    precheck, openings = pilot.surface_precheck(surface)
    assignment = (
        assign_roles_from_topology(openings, topology["interfaces"], _outlet_semantics(canonical_id))
        if len(openings) == len(topology["interfaces"])
        else {"assignment_source": "fluent_interface_topology", "complete": False, "openings": [asdict(item) for item in openings]}
    )
    wall_nodes_mm = mesh.nodes_m[wall["node_ids"]] * FLUENT_TO_MM
    context = {
        "canonical_id": canonical_id,
        "stl_scale_to_mm": 1.0,
        "unit_factor": FLUENT_TO_MM,
        "wall_coords_mm": wall_nodes_mm,
        "geometry_translation_to_cfd_wall_mm": np.zeros(3),
        "geometry_translation_source": "none_mesh_wall_is_cfd_frame",
        "boundary_path": Path("/nonexistent"),
    }
    surface_manifest = {
        "canonical_id": canonical_id,
        "selection_source": "fluent_case_anatomy_wall",
        "authoritative_stl": str(stl_path.resolve()),
        "authoritative_stl_sha256": sha256_file(stl_path),
        "authoritative_stl_scale_to_mm": 1.0,
        "fluent_to_mm_unit_factor": FLUENT_TO_MM,
        "unit_source": "fluent_si_metres_x1000_mesh_wall",
        "unit_override_applied": False,
        "unit_override_reason": "",
        "legacy_centerline_translation_applied": False,
        "legacy_centerline_translation_mm": [0.0, 0.0, 0.0],
        "geometry_translation_to_cfd_wall_mm": [0.0, 0.0, 0.0],
        "geometry_translation_source": "none_mesh_wall_is_cfd_frame",
        "fluent_case": {"path": str(case_path), "sha256": sha256_file(case_path)},
        "wall_surface": surface_info,
        "interfaces": [
            {
                "extension_zone": interface.extension_zone_name,
                "distal_bc": interface.distal_bc_name,
                "faces": int(len(interface.anatomy_cells)),
                "area_mm2": interface.area_m2 * FLUENT_TO_MM**2,
            }
            for interface in topology["interfaces"]
        ],
        "candidates": [],
        "precheck": precheck,
        "opening_assignment": assignment,
    }
    pilot.write_json(case_dir / "surface_selection.json", surface_manifest)
    if not precheck["hard_pass"] or not assignment.get("complete"):
        result = {
            "canonical_id": canonical_id,
            "status": "surface_precheck_failed" if not precheck["hard_pass"] else "opening_assignment_failed",
            "hard_pass": False,
            "manual_review_required": True,
            "selected_attempt": "",
            "elapsed_s": time.time() - started,
            "precheck": precheck,
            "opening_assignment": assignment,
        }
        pilot.write_json(case_dir / "result.json", result)
        return result

    attempts = [
        ("attempt_1_standard", surface, "all_targets", 0.0),
        ("attempt_2_per_target", surface, "per_target", 0.0),
        ("attempt_3_clean_shifted", pilot.lightly_clean_surface(surface), "per_target", 0.35),
    ]
    selected = None
    best = None
    attempt_rows = []
    for name, attempt_surface, mode, shift_frac in attempts:
        attempt_dir = case_dir / "attempts" / name
        try:
            poly, logs = pilot._attempt(attempt_surface, openings, context, attempt_dir, mode, shift_frac)
            poly, spike_pruning = pilot.prune_terminal_spikes(poly, openings, 1.0)
            if spike_pruning["applied"]:
                pilot.write_vtp(poly, attempt_dir / "centerline_terminal_spikes_pruned.vtp")
            poly, long_edge_resampling = pilot.resample_safe_long_edges(poly, attempt_surface, 1.0)
            if long_edge_resampling["applied"]:
                pilot.write_vtp(poly, attempt_dir / "centerline_long_edges_resampled.vtp")
            metrics, graph, endpoint_assignment = pilot.evaluate_centerline(poly, attempt_surface, openings, context, logs)
            metrics["terminal_spike_pruning_applied"] = spike_pruning["applied"]
            metrics["terminal_spikes_pruned"] = spike_pruning["n_pruned"]
            metrics["long_edge_resampling_applied"] = long_edge_resampling["applied"]
            metrics["long_edges_resampled"] = long_edge_resampling["n_edges_resampled"]
            metrics["long_edge_inserted_points"] = long_edge_resampling["inserted_points"]
            row = {
                "name": name,
                "mode": mode,
                "seed_shift_fraction_of_opening_radius": shift_frac,
                "terminal_spike_pruning": spike_pruning,
                "long_edge_resampling": long_edge_resampling,
                "metrics": metrics,
                "logs": [str(path.relative_to(case_dir)) for path in logs],
            }
            pilot.write_json(attempt_dir / "gate_metrics.json", row)
            attempt_rows.append(row)
            score = (int(metrics["hard_pass"]), -sum(metrics["soft_flags"].values()), -metrics["max_opening_to_endpoint_over_radius"])
            candidate = (score, name, attempt_surface, poly, metrics, graph, endpoint_assignment)
            if best is None or candidate[0] > best[0]:
                best = candidate
            if metrics["hard_pass"]:
                selected = candidate
                break
        except Exception as exc:  # noqa: BLE001
            error = {"name": name, "mode": mode, "seed_shift_fraction_of_opening_radius": shift_frac, "error": str(exc), "traceback": traceback.format_exc()}
            pilot.write_json(attempt_dir / "error.json", error)
            attempt_rows.append(error)
    if selected is None:
        selected = best
    if selected is None:
        result = {"canonical_id": canonical_id, "status": "all_attempts_failed", "hard_pass": False, "manual_review_required": True, "selected_attempt": "", "elapsed_s": time.time() - started, "attempts": attempt_rows}
        pilot.write_json(case_dir / "result.json", result)
        return result

    _, selected_name, selected_surface, poly, metrics, graph, endpoint_assignment = selected
    pilot.write_stl(selected_surface, case_dir / "surface_authoritative_mm.stl")
    path_build_error = ""
    try:
        graph_poly, paths_poly, opening_poly, path_frame = pilot.build_outputs(graph, metrics, openings, endpoint_assignment, 1.0)
    except Exception as exc:  # noqa: BLE001
        if metrics["hard_pass"]:
            raise
        graph_poly, opening_poly = pilot.build_basic_outputs(graph, openings, 1.0)
        paths_poly = None
        path_frame = None
        path_build_error = str(exc)
    pilot.write_vtp(graph_poly, case_dir / "centerline_graph_mm.vtp")
    pilot.write_vtp(opening_poly, case_dir / "openings_mm.vtp")
    outputs = {"surface": "surface_authoritative_mm.stl", "graph": "centerline_graph_mm.vtp", "openings": "openings_mm.vtp"}
    if paths_poly is not None and path_frame is not None:
        pilot.write_vtp(paths_poly, case_dir / "centerline_paths_mm.vtp")
        path_frame.to_csv(case_dir / "centerline_paths_mm.csv", index=False)
        outputs.update({"paths": "centerline_paths_mm.vtp", "path_csv": "centerline_paths_mm.csv"})
    result = {
        "canonical_id": canonical_id,
        "status": "pass_review" if metrics["hard_pass"] and metrics["manual_review_required"] else ("pass" if metrics["hard_pass"] else "hard_gate_failed"),
        "hard_pass": metrics["hard_pass"],
        "manual_review_required": metrics["manual_review_required"],
        "selected_attempt": selected_name,
        "elapsed_s": time.time() - started,
        "metrics": metrics,
        "attempts": attempt_rows,
        "path_build_error": path_build_error,
        "outputs": outputs,
        "surface_source": "fluent_case_anatomy_wall_true_mm",
    }
    pilot.write_json(case_dir / "result.json", result)
    try:
        pilot.render_case(case_dir, result)
    except Exception as exc:  # noqa: BLE001
        result["render_error"] = str(exc)
        pilot.write_json(case_dir / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for canonical_id in args.cases:
        try:
            result = process_meshwall_case(canonical_id, args.output)
        except Exception as exc:  # noqa: BLE001
            result = {"canonical_id": canonical_id, "status": "error", "hard_pass": False, "error": repr(exc), "traceback": traceback.format_exc()}
        summary.append(result)
        metrics = result.get("metrics", {})
        print(
            f"{canonical_id}: {result['status']} attempt={result.get('selected_attempt', '')} "
            f"endpoints={metrics.get('centerline_endpoints')} branch={metrics.get('centerline_branch_nodes')} "
            f"max_end_open={metrics.get('max_opening_to_endpoint_mm')} wall_p95/R={metrics.get('wall_distance_over_local_radius_p95')} "
            f"frac>2R={metrics.get('wall_fraction_gt_2r')} elapsed={result.get('elapsed_s', 0):.0f}s",
            flush=True,
        )
        if result.get("error"):
            print("   ", result["error"], flush=True)
    pilot.write_json(args.output / "run_result.json", {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cases": summary})
    return 0 if all(row.get("hard_pass") for row in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
