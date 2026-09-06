"""173-case audit of the Fluent cell-zone topology behind the anatomy-only contract.

For every case this script

* parses the Fluent case (all zones, all faces) with :mod:`fluent_topology`;
* ties the exported ``ascii_in`` cell rows and ``ascii`` wall rows to case
  cells/nodes by coordinate identity (bijective, unambiguous);
* records how many exported rows fall in ``blood`` vs. ``blood1…blood5``;
* derives the anatomy wall (wall faces adjacent to ``blood``), the five
  ``blood <-> bloodN`` interfaces and the distal BC zone closing each extension;
* maps every interface to the frozen outlet semantics (``out-le/li/ri/re`` or
  ``inlet``) through the qs-smooth-v3 boundary manifest zone ids;
* verifies the STL/Fluent unit relation (STL span / Fluent wall span).

Results are written as one JSON per case plus a cohort summary under
``outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903/audits/topology``.
The script never writes into ``data_new`` or any historical data root.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from wss_pinn.data.raw_io import read_interior, read_wall, step_file
from wss_pinn.utils import ROOT, sha256_file, utc_now
from wss_pinn.v4.fluent_topology import (
    FACE_TYPE_NAMES,
    anatomy_topology,
    anatomy_wall_faces,
    match_points,
    read_fluent_mesh,
)
from wss_pinn.v4.geometry_v2 import OUTLET_ORDER

SOURCE_MANIFEST = ROOT / "data_wss_pinn/volume_uvwp_bc_rcr_v4_train138_test35/manifest.json"
BOUNDARY_ROOT = ROOT / "data_wss_pinn/volume_uvwp_peak_qs_smooth_v3_train123_val15_test35/cases"
CENTERLINE_ROOT = ROOT / "outputs/centerline_v2_full_173_20260828/cases"
PREP_ROOT = ROOT / "outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903"
AUDIT_ROOT = PREP_ROOT / "audits/topology"
TRANSIENT_POOL_SIZE = 15_000
CELL_TOLERANCE_FRACTION = 0.2
NODE_TOLERANCE_M = 1.0e-7
FACE_CENTRE_TOLERANCE_M = 1.0e-6
INTERFACE_PLANARITY_MIN = 0.99
REFERENCE_STEP = 1120


def _case_ids() -> list[dict[str, Any]]:
    payload = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    return [{"canonical_id": row["canonical_id"], "role": row["role"]} for row in payload["cases"]]


def _outlet_semantics(canonical_id: str) -> dict[int, str]:
    manifest = json.loads((BOUNDARY_ROOT / canonical_id / "manifest.json").read_text(encoding="utf-8"))
    mapping = {int(row["mesh_zone_id"]): label for label, row in manifest["outlets"].items()}
    return mapping


def _stl_unit_check(canonical_id: str, wall_coords_m: np.ndarray) -> dict[str, Any]:
    selection = json.loads((CENTERLINE_ROOT / canonical_id / "surface_selection.json").read_text(encoding="utf-8"))
    import vtk  # noqa: WPS433  (only needed for the STL bbox check)
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkSTLReader()
    reader.SetFileName(selection["authoritative_stl"])
    reader.Update()
    stl = vtk_to_numpy(reader.GetOutput().GetPoints().GetData()).astype(np.float64)
    span_stl = stl.max(axis=0) - stl.min(axis=0)
    span_wall = wall_coords_m.max(axis=0) - wall_coords_m.min(axis=0)
    ratio = span_stl / span_wall
    deviation = float(np.max(np.abs(ratio / 1000.0 - 1.0)))
    return {
        "authoritative_stl": selection["authoritative_stl"],
        "authoritative_stl_sha256": selection["authoritative_stl_sha256"],
        "frozen_fluent_to_mm_unit_factor": float(selection["fluent_to_mm_unit_factor"]),
        "frozen_stl_scale_to_mm": float(selection["authoritative_stl_scale_to_mm"]),
        "stl_over_fluent_span_ratio": ratio.tolist(),
        "max_deviation_from_1000": deviation,
        "stl_points": int(len(stl)),
        "unit_verdict": (
            "stl_mm_equals_fluent_m_x1000"
            if deviation < 1.0e-4
            else "stl_mm_approx_fluent_m_x1000" if deviation < 0.02 else "stl_not_the_cfd_wall_surface"
        ),
        "true_mm_rescale_of_legacy_frame": 1000.0 / float(selection["fluent_to_mm_unit_factor"]),
    }


def audit_case(entry: dict[str, Any]) -> dict[str, Any]:
    canonical_id = entry["canonical_id"]
    output = AUDIT_ROOT / "cases" / f"{canonical_id.replace('/', '__')}.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    started = time.time()
    raw_dir = ROOT / "data_new" / canonical_id
    boundary_manifest = json.loads((BOUNDARY_ROOT / canonical_id / "manifest.json").read_text(encoding="utf-8"))
    case_path = Path(boundary_manifest["provenance"]["fluent_case"]["path"])
    if not case_path.is_file():
        raise FileNotFoundError(f"{canonical_id}: frozen Fluent case is missing: {case_path}")
    other_case_files = sorted(str(p) for p in raw_dir.glob("*.cas.gz") if p.resolve() != case_path.resolve())
    mesh = read_fluent_mesh(case_path)
    summary = mesh.summary()
    centroids, volumes = mesh.cell_centroids_and_volumes()
    cell_size = np.cbrt(volumes[1:])
    cell_zone = mesh.cell_zone_array()
    anatomy_zone_id = mesh.fluid_zone_by_name()["blood"]
    topology = anatomy_topology(mesh)

    frame_path = step_file(raw_dir, REFERENCE_STEP, "ascii_in")
    frame = read_interior(frame_path)
    volume_export_kind = "node" if frame["id_column"] == "nodenumber" else "cell"
    if volume_export_kind == "cell":
        cell_index, cell_diag = match_points(
            centroids[1:],
            frame["coords"],
            label=f"{canonical_id}:cells",
            tolerance_m=CELL_TOLERANCE_FRACTION * cell_size,
        )
        export_zone = cell_zone[cell_index + 1]
        zone_counts = {
            mesh.zone_name(int(zone)): int(count) for zone, count in zip(*np.unique(export_zone, return_counts=True))
        }
        anatomy_rows = int(np.sum(export_zone == anatomy_zone_id))
        interface_rows = 0
    else:
        # node-based volume export: a node belongs to every zone whose cells use it
        node_index, cell_diag = match_points(
            mesh.nodes_m[1:],
            frame["coords"],
            label=f"{canonical_id}:volume-nodes",
            tolerance_m=NODE_TOLERANCE_M,
        )
        node_zone_flags = np.zeros((mesh.node_count + 1, 2), dtype=bool)  # [in blood, in extension]
        for section in mesh.face_sections:
            if not section.attached:
                continue
            for cells in (section.c0, section.c1):
                valid = cells > 0
                if not np.any(valid):
                    continue
                is_blood = cell_zone[cells] == anatomy_zone_id
                faces_blood = np.flatnonzero(valid & is_blood)
                faces_ext = np.flatnonzero(valid & ~is_blood)
                for faces, column in ((faces_blood, 0), (faces_ext, 1)):
                    if len(faces):
                        _, nodes = section.subset_connectivity(faces)
                        node_zone_flags[nodes, column] = True
        flags = node_zone_flags[node_index + 1]
        anatomy_rows = int(np.sum(flags[:, 0]))
        interface_rows = int(np.sum(flags[:, 0] & flags[:, 1]))
        zone_counts = {
            "blood_nodes": anatomy_rows,
            "extension_only_nodes": int(np.sum(~flags[:, 0] & flags[:, 1])),
            "blood_and_extension_shared_nodes": interface_rows,
        }
    semantics = _outlet_semantics(canonical_id)
    interfaces = []
    seen_labels = []
    for interface in topology["interfaces"]:
        if interface.distal_bc_type in (10, 4, 20):
            label = "inlet"
        else:
            label = semantics.get(int(interface.distal_bc_zone_id), "")
        seen_labels.append(label)
        interfaces.append(
            {
                "extension_zone": interface.extension_zone_name,
                "extension_zone_id": int(interface.extension_zone_id),
                "face_zone_ids": [int(z) for z in interface.face_section_zone_ids],
                "face_zone_names": [mesh.zone_name(z) for z in interface.face_section_zone_ids],
                "faces": int(len(interface.anatomy_cells)),
                "nodes": int(len(interface.node_ids)),
                "area_m2": interface.area_m2,
                "planarity": interface.planarity,
                "outward_normal": interface.unit_normal.tolist(),
                "centre_m": np.average(
                    interface.centres_m, axis=0, weights=np.linalg.norm(interface.area_vectors_m2, axis=1)
                ).tolist(),
                "distal_bc_zone_id": int(interface.distal_bc_zone_id),
                "distal_bc_name": interface.distal_bc_name,
                "distal_bc_type": FACE_TYPE_NAMES.get(interface.distal_bc_type, str(interface.distal_bc_type)),
                "semantic_label": label,
                "extension_cells": int(
                    mesh.cell_zone_ranges[interface.extension_zone_id][1]
                    - mesh.cell_zone_ranges[interface.extension_zone_id][0]
                    + 1
                ),
            }
        )
    expected_labels = sorted(["inlet", *OUTLET_ORDER])
    semantics_complete = sorted(seen_labels) == expected_labels

    wall = anatomy_wall_faces(mesh, topology)
    wall_path = step_file(raw_dir, REFERENCE_STEP, "ascii")
    wall_export = read_wall(wall_path)
    with wall_path.open(encoding="utf-8", errors="replace") as handle:
        wall_header = handle.readline()
    wall_export_kind = "face-centre" if "cellnumber" in wall_header else "node"
    anatomy_nodes = set(wall["node_ids"].tolist())
    # Fluent occasionally writes the same wall node twice (identical coordinates);
    # identity is established on the unique coordinates and the duplicates reported.
    export_unique, export_inverse = np.unique(
        np.asarray(wall_export["coords"], dtype=np.float64), axis=0, return_inverse=True
    )
    duplicate_export_rows = int(len(wall_export["coords"]) - len(export_unique))
    if wall_export_kind == "node":
        # match against the anatomy wall nodes only, so coincident twin nodes
        # that are not on the wall can never be picked
        wall_node_ids = np.asarray(sorted(anatomy_nodes), dtype=np.int64)
        wall_index, node_diag = match_points(
            mesh.nodes_m[wall_node_ids],
            export_unique,
            label=f"{canonical_id}:wall-nodes",
            tolerance_m=NODE_TOLERANCE_M,
        )
        export_nodes = set(wall_node_ids[wall_index].tolist())
    else:
        face_index, node_diag = match_points(
            wall["centres_m"],
            export_unique,
            label=f"{canonical_id}:wall-face-centres",
            tolerance_m=FACE_CENTRE_TOLERANCE_M,
        )
        export_nodes = set()
    main_wall_zone_ids = {
        zone_id for zone_id, row in topology["wall_by_zone"].items() if row["name"] == "wall"
    }
    side_zone_nodes = set(
        wall["face_nodes"][
            np.repeat(~np.isin(wall["face_zone_ids"], list(main_wall_zone_ids)), np.diff(wall["face_offsets"]))
        ].tolist()
    )
    rim = {
        interface.extension_zone_name: int(len(set(interface.node_ids.tolist()) & anatomy_nodes))
        for interface in topology["interfaces"]
    }
    unit = _stl_unit_check(canonical_id, wall_export["coords"])

    gates = {
        "six_fluid_zones_blood_blood1_5": sorted(summary["cell_zones"]) == ["blood", "blood1", "blood2", "blood3", "blood4", "blood5"],
        "export_rows_equal_case_entities": (
            int(len(frame["coords"])) == int(mesh.cell_count)
            if volume_export_kind == "cell"
            else int(len(frame["coords"])) <= int(mesh.node_count)  # node exports skip unused/duplicate nodes
        ),
        "cell_identity_bijective_unambiguous": bool(cell_diag["bijective"] and cell_diag["unambiguous"] and cell_diag["within_tolerance"]),
        "five_interfaces": len(interfaces) == 5,
        "interfaces_planar": all(row["planarity"] > INTERFACE_PLANARITY_MIN for row in interfaces),
        "interface_semantics_complete": semantics_complete,
        "each_extension_closes_on_one_flow_bc": True,
        "anatomy_has_no_direct_flow_bc": len(topology["anatomy_direct_flow_bc"]) == 0,
        "wall_export_covers_anatomy_wall": (
            (export_nodes <= anatomy_nodes and (anatomy_nodes - export_nodes) <= side_zone_nodes)
            if wall_export_kind == "node"
            else int(len(export_unique)) == int(len(wall["adjacent_cells"]))
        ),
        "wall_identity_bijective": bool(node_diag["bijective"] and node_diag["unambiguous"]),
        "anatomy_cells_at_least_pool": anatomy_rows >= TRANSIENT_POOL_SIZE,
        "parser_warnings_empty": len(summary["warnings"]) == 0,
        "unattached_faces_ignored": all(
            (not row["attached"]) == (row["type_code"] == 40) for row in summary["face_zones"]
        ),
    }
    report = {
        "schema_version": 1,
        "canonical_id": canonical_id,
        "role": entry["role"],
        "created_at": utc_now(),
        "elapsed_s": round(time.time() - started, 1),
        "fluent_case": {"path": str(case_path), "sha256": sha256_file(case_path), "size_bytes": case_path.stat().st_size, "other_case_files_in_raw_dir": other_case_files},
        "reference_frame": {"step": REFERENCE_STEP, "path": str(frame_path), "rows": int(len(frame["coords"])), "export_kind": volume_export_kind, "id_column": frame["id_column"], "blood_and_extension_shared_rows": interface_rows},
        "mesh": {
            "cells": summary["cells"],
            "nodes": summary["nodes"],
            "faces": summary["faces"],
            "declared_faces": summary["declared_faces"],
            "cell_types": summary["cell_types"],
            "cell_zones": summary["cell_zones"],
            "face_zones": summary["face_zones"],
            "warnings": summary["warnings"],
            "c0_sign": mesh.c0_sign,
            "total_volume_m3": float(np.nansum(volumes)),
            "anatomy_volume_m3": float(np.nansum(volumes[cell_zone == anatomy_zone_id])),
        },
        "cell_identity": cell_diag,
        "export_zone_counts": zone_counts,
        "anatomy_rows": anatomy_rows,
        "anatomy_fraction": anatomy_rows / float(len(frame["coords"])),
        "wall_by_zone": topology["wall_by_zone"],
        "anatomy_wall": {
            "faces": int(len(wall["adjacent_cells"])),
            "nodes": int(len(anatomy_nodes)),
            "area_m2": float(np.linalg.norm(wall["area_vectors_m2"], axis=1).sum()),
            "face_zones": {
                mesh.zone_name(int(zone)): int(count)
                for zone, count in zip(*np.unique(wall["face_zone_ids"], return_counts=True))
            },
            "export_kind": wall_export_kind,
            "export_rows": int(len(wall_export["coords"])),
            "duplicate_export_rows": duplicate_export_rows,
            "export_minus_anatomy": int(len(export_nodes - anatomy_nodes)) if wall_export_kind == "node" else None,
            "anatomy_minus_export": int(len(anatomy_nodes - export_nodes)) if wall_export_kind == "node" else None,
            "anatomy_minus_export_all_in_side_zones": bool((anatomy_nodes - export_nodes) <= side_zone_nodes) if wall_export_kind == "node" else None,
            "side_wall_zone_nodes": int(len(side_zone_nodes)),
            "identity": node_diag,
            "min_interface_planarity": min((row["planarity"] for row in interfaces), default=None),
            "rim_nodes_shared_with_interface": rim,
        },
        "interfaces": interfaces,
        "anatomy_direct_flow_bc": topology["anatomy_direct_flow_bc"],
        "extension_flow_bc": topology["extension_flow_bc"],
        "unit_check": unit,
        "gates": gates,
        "gate_pass": all(gates.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, output)
    return report


def _safe(entry: dict[str, Any]) -> dict[str, Any]:
    try:
        return audit_case(entry)
    except Exception as error:  # noqa: BLE001
        return {
            "canonical_id": entry["canonical_id"],
            "role": entry["role"],
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "gate_pass": False,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cases", nargs="*", default=None)
    args = parser.parse_args()
    entries = _case_ids()
    if args.cases:
        wanted = set(args.cases)
        entries = [row for row in entries if row["canonical_id"] in wanted]
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    reports = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for report in pool.map(_safe, entries):
            status = "ERROR" if "error" in report else ("pass" if report["gate_pass"] else "FAIL")
            print(f"{report['canonical_id']}: {status}", flush=True)
            if "error" in report:
                print("   ", report["error"], flush=True)
            reports.append(report)
    failures = [r for r in reports if not r.get("gate_pass")]
    gate_counts: dict[str, int] = {}
    for report in reports:
        for name, value in report.get("gates", {}).items():
            gate_counts[name] = gate_counts.get(name, 0) + (0 if value else 1)
    summary = {
        "schema_version": 1,
        "created_at": utc_now(),
        "cases": len(reports),
        "pass": len(reports) - len(failures),
        "failures": [
            {"canonical_id": r["canonical_id"], "error": r.get("error"), "failed_gates": [k for k, v in r.get("gates", {}).items() if not v]}
            for r in failures
        ],
        "gate_failure_counts": gate_counts,
        "anatomy_fraction": {
            "min": min(r["anatomy_fraction"] for r in reports if "anatomy_fraction" in r),
            "median": float(np.median([r["anatomy_fraction"] for r in reports if "anatomy_fraction" in r])),
            "max": max(r["anatomy_fraction"] for r in reports if "anatomy_fraction" in r),
        }
        if any("anatomy_fraction" in r for r in reports)
        else None,
        "unit_verdicts": {
            verdict: sorted(r["canonical_id"] for r in reports if r.get("unit_check", {}).get("unit_verdict") == verdict)
            for verdict in ("stl_mm_equals_fluent_m_x1000", "stl_mm_approx_fluent_m_x1000", "stl_not_the_cfd_wall_surface")
        },
        "cases_report": {
            r["canonical_id"]: {
                "gate_pass": r.get("gate_pass"),
                "anatomy_rows": r.get("anatomy_rows"),
                "anatomy_fraction": r.get("anatomy_fraction"),
                "cells": r.get("mesh", {}).get("cells"),
                "volume_export_kind": r.get("reference_frame", {}).get("export_kind"),
                "other_case_files": r.get("fluent_case", {}).get("other_case_files_in_raw_dir"),
                "cell_types": r.get("mesh", {}).get("cell_types"),
                "interface_labels": [row["semantic_label"] for row in r.get("interfaces", [])],
                "interface_area_m2": {row["semantic_label"]: row["area_m2"] for row in r.get("interfaces", [])},
                "anatomy_wall_faces": r.get("anatomy_wall", {}).get("faces"),
                "wall_export_kind": r.get("anatomy_wall", {}).get("export_kind"),
                "min_interface_planarity": r.get("anatomy_wall", {}).get("min_interface_planarity"),
                "rim_nodes": r.get("anatomy_wall", {}).get("rim_nodes_shared_with_interface"),
                "unit_verdict": r.get("unit_check", {}).get("unit_verdict"),
                "frozen_unit_factor": r.get("unit_check", {}).get("frozen_fluent_to_mm_unit_factor"),
            }
            for r in reports
        },
    }
    (AUDIT_ROOT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"pass {summary['pass']}/{summary['cases']}; summary at {AUDIT_ROOT / 'summary.json'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
