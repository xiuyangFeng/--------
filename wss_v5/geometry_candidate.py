"""Build P1--P7 geometry candidates for visual review, with training disabled.

This command reads only geometry from the frozen V5 HDF5 bundle. It never
changes that bundle, the atlas, or legacy views. Reference surface sections and
bare-point-cloud sections are separate outputs; neither is user-approved.
Heavy case processing is intentionally restricted to a Slurm allocation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from . import contract as C
from .anatomy_overrides import correct_topology, correct_openings, correct_report
from .pointcloud import _rim_plane
from .section_features import (SurfaceSlicer, contains_origin, frame, log_area_slopes,
                               pointcloud_section, polygon_metrics, project_wall, same_segment_nonlocal_crossings, unit)

SCHEMA = "wss_v6_geometry_review_candidate_v1.3"
DEFAULT_OUTPUT = C.ROOT / "outputs/wss_v6_geometry_candidate_20260909"
DEFAULT_CONFIG = Path(__file__).parent / "configs/geometry_candidate_p_v1.json"


def plain(obj):
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, np.generic):
        return plain(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(plain(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temp.replace(path)


def digest_arrays(arrays: dict) -> str:
    h = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def code_fingerprints() -> dict:
    modules = [Path(__file__), Path(__file__).with_name("section_features.py"), Path(__file__).with_name("pointcloud.py"), Path(__file__).with_name("anatomy_overrides.py")]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in modules}
    return {"files": hashes, "combined_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def topology_semantics(segments: list[dict], xyz: np.ndarray, sid: np.ndarray,
                       s: np.ndarray) -> tuple[list[dict], dict]:
    """Separate topological entities from unverified CFD-zone anatomy names."""
    by = {int(a["segment_id"]): a for a in segments}
    children = {i: [] for i in by}
    roots = []
    errors = []
    for i, seg in by.items():
        parent = int(seg["parent_id"])
        if parent == -1:
            roots.append(i)
        elif parent in by:
            children[parent].append(i)
        else:
            errors.append(f"missing_parent:{i}")
    if len(by) != 7 or len(roots) != 1:
        errors.append("expected_seven_segments_one_root")
    if any(len(c) not in (0, 2) for c in children.values()):
        errors.append("not_binary_tree")
    root = roots[0] if roots else -999
    visited = set()

    def leaves(i, ancestors=()):
        if i in ancestors:
            errors.append(f"cycle:{i}")
            return []
        visited.add(i)
        if not children[i]:
            return [by[i].get("outlet_name", "unknown")]
        return sum((leaves(j, (*ancestors, i)) for j in children[i]), [])

    if root in by:
        leaves(root)
    if visited != set(by):
        errors.append("disconnected_segments")
    generic_tree_valid = not errors
    root_children = children.get(root, [])
    expected_pattern = (len(root_children) == 2 and all(len(children[c]) == 2 for c in root_children)
                        and all(not children[leaf] for c in root_children for leaf in children[c]))
    if not expected_pattern:
        errors.append("expected_trunk_two_CIA_four_terminal_pattern_missing")
    result = []
    mixed = []
    for i, seg in by.items():
        rows = np.flatnonzero(sid == i)
        rows = rows[np.argsort(s[rows])]
        if len(rows) < 2:
            errors.append(f"short_segment:{i}")
            continue
        pts = xyz[rows]
        length = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        chord = float(np.linalg.norm(pts[-1] - pts[0]))
        role = "trunk" if i == root else ("cia" if int(seg["parent_id"]) == root and children[i] else "terminal")
        names = leaves(i)
        sides = {"left" if n in ("out-le", "out-li") else "right" if n in ("out-re", "out-ri") else "unknown" for n in names}
        consistent = len(sides) == 1 and "unknown" not in sides
        label = "trunk" if role == "trunk" else "unknown"
        if role == "cia" and consistent:
            label = f"{next(iter(sides))}_cia"
        if role == "terminal":
            label = {"out-le": "left_external", "out-li": "left_internal", "out-re": "right_external", "out-ri": "right_internal"}.get(names[0], "unknown")
        if role == "cia" and not consistent:
            mixed.append(i)
        result.append({
            "segment_id": i, "parent_id": int(seg["parent_id"]), "children": children[i], "role": role,
            "anatomy_label": label, "anatomy_label_user_confirmed": False,
            "cfd_zone_label": seg.get("outlet_name", "") if role == "terminal" else "",
            "cfd_descendant_zone_labels": names,
            "semantic_valid": role == "trunk" or consistent,
            "semantic_reason": "topological_trunk" if role == "trunk" else "CFD_name_tree_consistent_but_anatomy_unconfirmed" if consistent else "mixed_or_unknown_CFD_sides_anatomy_unknown",
            "label_provenance": "existing_CFD_interface_correspondence_not_independent_anatomy",
            "start_mm": pts[0], "end_mm": pts[-1], "length_mm": length, "tortuosity": length / max(chord, 1e-9),
            "start_s_mm": float(s[rows[0]]), "end_s_mm": float(s[rows[-1]]),
            "definition": "aortic_bifurcation_to_ipsilateral_internal_external_bifurcation" if role == "cia" else role,
            "start_direction": unit(pts[min(10, len(pts) - 1)] - pts[0]),
            "end_direction": unit(pts[-1] - pts[max(0, len(pts) - 11)]),
        })
    result_by = {r["segment_id"]: r for r in result}
    for row in result:
        parent = result_by.get(row["parent_id"])
        if row["parent_id"] in mixed:
            row["anatomy_label"] = "unknown"
            row["semantic_valid"] = False
            row["semantic_reason"] = "parent_CIA_has_mixed_CFD_sides_anatomy_unknown"
        row["parent_branch_angle_deg"] = None if parent is None else float(np.degrees(np.arccos(np.clip(np.dot(parent["end_direction"], row["start_direction"]), -1, 1))))
        row["daughter_angle_deg"] = None
        if len(row["children"]) == 2 and all(c in result_by for c in row["children"]):
            a, b = [result_by[c]["start_direction"] for c in row["children"]]
            row["daughter_angle_deg"] = float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))
    return result, {"valid": not errors, "generic_tree_valid": generic_tree_valid,
                    "expected_aortoiliac_pattern_valid": expected_pattern, "errors": sorted(set(errors)), "root_segment_id": root,
                    "segments": len(by), "leaf_segments": sum(not v for v in children.values()),
                    "junctions": sum(bool(v) for v in children.values()), "mixed_side_cia_segments": mixed,
                    "anatomy_names_user_confirmed": False}


def extract_openings(slicer: SurfaceSlicer, segments: list[dict]) -> tuple[list[dict], dict]:
    loops = slicer.boundary_loops()
    recipes = []
    for seg in segments:
        if seg["role"] == "trunk":
            recipes.append({"segment_id": seg["segment_id"], "point": seg["start_mm"], "outward": -seg["start_direction"], "label": "inlet"})
        if not seg["children"]:
            recipes.append({"segment_id": seg["segment_id"], "point": seg["end_mm"], "outward": seg["end_direction"], "label": seg["cfd_zone_label"]})
    if not loops or not recipes:
        return [], {"valid": False, "reason": "no_boundary_loops_or_endpoints", "boundary_loops": len(loops)}
    centers = np.array([p.mean(axis=0) for p in loops])
    distance = np.linalg.norm(np.array([r["point"] for r in recipes])[:, None] - centers[None], axis=2)
    ri, li = linear_sum_assignment(distance)
    out = []
    for r, loop_i in zip(ri, li):
        poly = loops[loop_i]
        c = centers[loop_i]
        _, _, vh = np.linalg.svd(poly - c, full_matrices=False)
        normal = unit(vh[-1])
        if np.dot(normal, recipes[r]["outward"]) < 0:
            normal = -normal
        metrics = polygon_metrics(poly, c, normal)
        residual = np.abs((poly - c) @ normal)
        out.append({**metrics, "segment_id": recipes[r]["segment_id"], "center_mm": c, "normal": normal,
                    "label_provisional": recipes[r]["label"], "label_role": "CFD_correspondence_except_topological_inlet",
                    "endpoint_gap_mm": float(distance[r, loop_i]), "plane_residual_p95_mm": float(np.percentile(residual, 95)),
                    "source": "closed_boundary_loop_of_reference_anatomy_wall_surface"})
    return out, {"valid": len(loops) == 5 and len(out) == 5, "boundary_loops": len(loops),
                 "assigned_openings": len(out), "unassigned_loops": sorted(set(range(len(loops))) - set(li))}


def interpolate_valid_field(sq: np.ndarray, s: np.ndarray, value: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both adjacent section samples must be valid; never bridge a masked gap."""
    out = np.full(len(sq), np.nan)
    good = np.zeros(len(sq), bool)
    if len(s) < 2:
        return out, good
    hi = np.searchsorted(s, sq, side="right")
    hi = np.clip(hi, 1, len(s) - 1)
    lo = hi - 1
    good = (sq >= s[0]) & (sq <= s[-1]) & valid[lo] & valid[hi]
    alpha = (sq - s[lo]) / np.maximum(s[hi] - s[lo], 1e-12)
    out[good] = ((1 - alpha) * value[lo] + alpha * value[hi])[good]
    return out, good


def packed_polygons(rows: list[dict], key: str, shape=(3,)) -> tuple[np.ndarray, np.ndarray]:
    values = [np.asarray(r.get(key, []), float).reshape(-1, *shape) for r in rows]
    offsets = np.r_[0, np.cumsum([len(v) for v in values])].astype(np.int64)
    return (np.concatenate(values) if offsets[-1] else np.empty((0, *shape))), offsets


def field_array(rows: list[dict], key: str, default=np.nan, dtype=float):
    return np.asarray([r.get(key, default) for r in rows], dtype=dtype)


def other_branch_in_contour(result: dict, center: np.ndarray, normal: np.ndarray,
                            xyz: np.ndarray, sid: np.ndarray, segment_id: int) -> bool:
    """Each source checks its own polygon, without consuming the other source."""
    if not result.get("valid"):
        return False
    e1, e2 = frame(normal)
    other = (sid != segment_id) & (np.abs((xyz - center) @ normal) <= 0.6)
    polygon = result["polygon_xyz_mm"]
    for other_point in xyz[other]:
        p2 = np.column_stack(((polygon - other_point) @ e1, (polygon - other_point) @ e2))
        if contains_origin(p2):
            return True
    return False


def add_reasons(result: dict, reasons: list[str]) -> dict:
    if not reasons:
        return result
    return {**result, "valid": False, "reason": ";".join(sorted(set(reasons + ([] if result["reason"] == "ok" else [result["reason"]]))))}


def section_geometry_gate(result: dict, center: np.ndarray, normal: np.ndarray, seg: dict,
                          local_s: float, xyz: np.ndarray, sid: np.ndarray, cfg: dict, atlas_s: np.ndarray | None = None) -> dict:
    """Shared P4/P6 source-independent geometry gate, retaining raw validity."""
    reasons = []
    start, end = seg["start_s_mm"], seg["end_s_mm"]
    if not start <= local_s <= end:
        reasons.append("outside_segment_extent")
    if seg["parent_id"] != -1 and local_s - start < cfg["junction_exclusion_mm"]:
        reasons.append("junction_exclusion")
    if seg["children"] and end - local_s < cfg["junction_exclusion_mm"]:
        reasons.append("junction_exclusion")
    if ((seg["parent_id"] == -1 and local_s - start < cfg["endpoint_exclusion_mm"])
            or (not seg["children"] and end - local_s < cfg["endpoint_exclusion_mm"])):
        reasons.append("endpoint_exclusion")
    if other_branch_in_contour(result, center, normal, xyz, sid, seg["segment_id"]):
        reasons.append("section_contains_other_centerline_branch")
    crossings = {"inside_crossing_groups": [], "extra_nonlocal_groups": [], "local_group_index": None}
    if result.get("valid") and atlas_s is not None:
        rows = np.flatnonzero(sid == seg["segment_id"])
        rows = rows[np.argsort(atlas_s[rows])]
        crossings = same_segment_nonlocal_crossings(result["polygon_xyz_mm"], center, normal, xyz[rows], atlas_s[rows], local_s,
                                                    nonlocal_gap_mm=cfg["same_segment_nonlocal_s_gap_mm"], plane_tol_mm=cfg["plane_intersection_tolerance_mm"])
        if crossings["extra_nonlocal_groups"]:
            reasons.append("section_contains_same_segment_nonlocal_plane_crossing")
    gated = add_reasons(result, reasons)
    return {**gated, "raw_contour_valid": bool(result["valid"]), "raw_contour_reason": result["reason"],
            "pipeline_valid": bool(gated["valid"]), "pipeline_reason": gated["reason"],
            "same_segment_crossing_group_count": len(crossings["inside_crossing_groups"]),
            "same_segment_nonlocal_crossing_count": len(crossings["extra_nonlocal_groups"]), "same_segment_plane_intersections": crossings}


def opening_geometry_gate(result: dict, center: np.ndarray | None, normal: np.ndarray | None,
                          root_segment: int, xyz: np.ndarray, sid: np.ndarray, extra_reasons=(), atlas_s=None, cfg=None) -> dict:
    """Opening-specific gate: openings are not rejected for being endpoints."""
    reasons = list(extra_reasons)
    if center is not None and normal is not None and other_branch_in_contour(result, center, normal, xyz, sid, root_segment):
        reasons.append("opening_contour_contains_other_centerline_branch")
    crossings = {"inside_crossing_groups": [], "extra_nonlocal_groups": [], "local_group_index": None}
    if result.get("valid") and atlas_s is not None and cfg is not None:
        rows = np.flatnonzero(sid == root_segment)
        rows = rows[np.argsort(atlas_s[rows])]
        crossings = same_segment_nonlocal_crossings(result["polygon_xyz_mm"], center, normal, xyz[rows], atlas_s[rows], float(atlas_s[rows[0]]),
                                                    nonlocal_gap_mm=cfg["same_segment_nonlocal_s_gap_mm"], plane_tol_mm=cfg["plane_intersection_tolerance_mm"])
        if crossings["extra_nonlocal_groups"]:
            reasons.append("opening_contains_same_segment_nonlocal_plane_crossing")
    gated = add_reasons(result, reasons)
    return {**gated, "raw_contour_valid": bool(result["valid"]), "raw_contour_reason": result["reason"],
            "pipeline_valid": bool(gated["valid"]), "pipeline_reason": gated["reason"],
            "same_segment_crossing_group_count": len(crossings["inside_crossing_groups"]),
            "same_segment_nonlocal_crossing_count": len(crossings["extra_nonlocal_groups"]), "same_segment_plane_intersections": crossings}


def branch_context(s: np.ndarray, area: np.ndarray, valid: np.ndarray, window_mm: float) -> dict[str, np.ndarray]:
    """Small local upstream/downstream descriptors on contiguous valid runs.

    Windows stop at a missing section or a branch junction. Thus these do not
    masquerade as a whole-tree hemodynamic or outlet-resistance model.
    """
    out = {k: np.full(len(s), np.nan) for k in ("upstream_min_area_ratio", "downstream_min_area_ratio",
                                               "upstream_min_distance_mm", "downstream_min_distance_mm")}
    idx = np.flatnonzero(valid & np.isfinite(area) & (area > 0))
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if len(run) < 2:
            continue
        for i in run:
            for name, candidates in (("upstream", run[(s[run] < s[i]) & (s[run] >= s[i] - window_mm)]),
                                     ("downstream", run[(s[run] > s[i]) & (s[run] <= s[i] + window_mm)])):
                if len(candidates):
                    minimum = np.min(area[candidates])
                    ties = candidates[np.isclose(area[candidates], minimum, rtol=1e-8, atol=1e-10)]
                    k = ties[np.argmin(np.abs(s[ties] - s[i]))]
                    out[f"{name}_min_area_ratio"][i] = area[k] / area[i]
                    out[f"{name}_min_distance_mm"][i] = abs(s[k] - s[i])
    return out


def case_candidate(canonical_id: str, output: Path, cfg: dict,
                   source_root: Path | None = None) -> dict:
    start = time.time()
    code_hashes = code_fingerprints()
    source = C.case_dir(canonical_id, root=source_root or C.SNAPSHOT_ROOT) / "case.h5"
    with h5py.File(source, "r") as h5:
        wall = h5["wall_static/xyz_mm"][()].astype(float)
        wall_ids = h5["wall_static/node_id_cas"][()]
        triangles = h5["topology/wall_triangles"][()]
        table = h5["geometry/atlas_table"][()]
        columns = json.loads(h5["geometry"].attrs["atlas_columns"])
        segments_src = json.loads(h5["geometry"].attrs["atlas_segments"])
        provenance = json.loads(h5["geometry"].attrs["atlas_provenance"])
        spacing = float(h5["geometry"].attrs["wall_point_spacing_mm"])
    col = lambda k: table[:, columns.index(k)]
    xyz = np.column_stack([col(k + "_mm") for k in "xyz"])
    tangent = unit(np.column_stack([col("tangent_" + k) for k in "xyz"]))
    sid, s = col("segment_id").astype(np.int16), col("s_local_mm")
    radius = col("radius_mm")
    segments, topology = topology_semantics(segments_src, xyz, sid, s)
    segments, topology = correct_topology(canonical_id, segments, topology)
    slicer = SurfaceSlicer(wall, triangles)
    inside, enclosed_audit = slicer.enclosed(xyz)
    audit_core = (col("dist_to_endpoint_mm") > 2.5) & (col("is_child_duplicate") < 0.5)
    openings, opening_audit = extract_openings(slicer, segments)
    openings = correct_openings(canonical_id, openings, segments)
    mapped = project_wall(wall, xyz, sid, s)
    pc_tree = cKDTree(wall)
    reference, pc_sections = [], []
    section_locations = []
    for seg in segments:
        rows = np.flatnonzero(sid == seg["segment_id"])
        rows = rows[np.argsort(s[rows])]
        sample_s = np.linspace(s[rows[0]], s[rows[-1]], max(3, int(np.ceil((s[rows[-1]] - s[rows[0]]) / cfg["section_spacing_mm"])) + 1))
        cc = np.column_stack([np.interp(sample_s, s[rows], xyz[rows, j]) for j in range(3)])
        tt = unit(np.column_stack([np.interp(sample_s, s[rows], tangent[rows, j]) for j in range(3)]))
        rr = np.interp(sample_s, s[rows], radius[rows])
        for sj, center, normal, rmis in zip(sample_s, cc, tt, rr):
            result = slicer.slice(center, normal)
            result = section_geometry_gate(result, center, normal, seg, sj, xyz, sid, cfg, s)
            search_radius = max(4 * rmis, 12.0)
            nearby = np.asarray(pc_tree.query_ball_point(center, search_radius), dtype=int)
            eligible = (mapped["segment_id"][nearby] == seg["segment_id"])
            pc = pointcloud_section(wall[nearby[eligible]], center, normal, rmis, spacing)
            pc = section_geometry_gate(pc, center, normal, seg, sj, xyz, sid, cfg, s)
            reference.append(result)
            pc_sections.append(pc)
            section_locations.append({"segment_id": seg["segment_id"], "s_local_mm": float(sj), "center_mm": center,
                                      "normal": normal, "radius_mis_mm": float(rmis)})
    section_sid = field_array(section_locations, "segment_id", dtype=np.int16)
    section_s = field_array(section_locations, "s_local_mm")
    area, valid = field_array(reference, "area_mm2"), field_array(reference, "valid", False, bool)
    pc_area, pc_valid = field_array(pc_sections, "area_mm2"), field_array(pc_sections, "valid", False, bool)
    slopes, pc_slopes = np.full(len(area), np.nan), np.full(len(area), np.nan)
    wall_area, wall_pc_area = np.full(len(wall), np.nan), np.full(len(wall), np.nan)
    wall_valid, wall_pc_valid = np.zeros(len(wall), bool), np.zeros(len(wall), bool)
    for seg in segments:
        si = np.flatnonzero(section_sid == seg["segment_id"])
        wi = np.flatnonzero(mapped["segment_id"] == seg["segment_id"])
        slopes[si] = log_area_slopes(section_s[si], area[si], valid[si])
        pc_slopes[si] = log_area_slopes(section_s[si], pc_area[si], pc_valid[si])
        wall_area[wi], wall_valid[wi] = interpolate_valid_field(mapped["s_local_mm"][wi], section_s[si], area[si], valid[si])
        wall_pc_area[wi], wall_pc_valid[wi] = interpolate_valid_field(mapped["s_local_mm"][wi], section_s[si], pc_area[si], pc_valid[si])
    wall_valid &= ~mapped["ambiguous"]
    wall_pc_valid &= ~mapped["ambiguous"]
    raw_wall_area, raw_wall_pc_area = wall_area.copy(), wall_pc_area.copy()
    wall_area[~wall_valid] = np.nan
    wall_pc_area[~wall_pc_valid] = np.nan
    # P6: actual recalculation at perturbed planes and 50% cloud density.
    stability = []
    rng = np.random.default_rng(cfg["seed"])
    half_mask = rng.random(len(wall)) < 0.5
    for seg in segments:
        indices = np.flatnonzero((section_sid == seg["segment_id"]) & valid)
        if not len(indices):
            continue
        chosen = np.unique(indices[np.rint(np.linspace(0, len(indices) - 1, cfg["stability_sections_per_segment"])).astype(int)])
        for k in chosen:
            loc = section_locations[k]
            c, t, r = loc["center_mm"], loc["normal"], loc["radius_mis_mm"]
            e1, _ = frame(t)
            angle = np.radians(cfg["stability_tilt_deg"])
            recipes = [
                ("shift_minus_1mm", c - cfg["stability_shift_mm"] * t, t, loc["s_local_mm"] - cfg["stability_shift_mm"]),
                ("shift_plus_1mm", c + cfg["stability_shift_mm"] * t, t, loc["s_local_mm"] + cfg["stability_shift_mm"]),
                ("tilt_minus_5deg", c, np.cos(angle) * t - np.sin(angle) * e1, loc["s_local_mm"]),
                ("tilt_plus_5deg", c, np.cos(angle) * t + np.sin(angle) * e1, loc["s_local_mm"]),
            ]
            variants = []
            seg_rows = np.flatnonzero(sid == seg["segment_id"])
            seg_rows = seg_rows[np.argsort(s[seg_rows])]
            for name, vc, vt, vs in recipes:
                m = section_geometry_gate(slicer.slice(vc, vt), vc, vt, seg, vs, xyz, sid, cfg, s)
                vr = float(np.interp(vs, s[seg_rows], radius[seg_rows]))
                near = np.asarray(pc_tree.query_ball_point(vc, max(4 * vr, 12.0)), dtype=int)
                near = near[mapped["segment_id"][near] == seg["segment_id"]]
                pm = section_geometry_gate(pointcloud_section(wall[near], vc, vt, vr, spacing), vc, vt, seg, vs, xyz, sid, cfg, s)
                for source_name, variant_name, result in (("reference_surface", name, m), ("pointcloud", "pointcloud_" + name, pm)):
                    baseline_area = area[k] if source_name == "reference_surface" else pc_area[k]
                    baseline_valid = valid[k] if source_name == "reference_surface" else pc_valid[k]
                    comparison_valid = bool(result["pipeline_valid"] and baseline_valid and baseline_area > 0)
                    raw_change = result.get("area_mm2", np.nan) / area[k] - 1
                    raw_same_source_change = result.get("area_mm2", np.nan) / baseline_area - 1 if baseline_area > 0 else None
                    variants.append({"name": variant_name, "source": source_name, "area_mm2": result.get("area_mm2"),
                                     **{key: result[key] for key in ("valid", "reason", "raw_contour_valid", "raw_contour_reason", "pipeline_valid", "pipeline_reason", "same_segment_crossing_group_count", "same_segment_nonlocal_crossing_count")},
                                     "comparison_valid": comparison_valid, "relative_area_change": raw_change if comparison_valid else None,
                                     "raw_relative_area_change": raw_change,
                                     "relative_to_same_source_baseline": raw_same_source_change if comparison_valid else None,
                                     "raw_relative_to_same_source_baseline": raw_same_source_change})
            nearby = np.asarray(pc_tree.query_ball_point(c, max(4 * r, 12.0)), dtype=int)
            eligible = (mapped["segment_id"][nearby] == seg["segment_id"])
            half = section_geometry_gate(pointcloud_section(wall[nearby[eligible & half_mask[nearby]]], c, t, r, spacing), c, t, seg, loc["s_local_mm"], xyz, sid, cfg, s)
            for name, m in [("pointcloud_full", pc_sections[k]), ("pointcloud_half", half)]:
                comparison_valid = bool(m["pipeline_valid"] and valid[k])
                raw_change = m.get("area_mm2", np.nan) / area[k] - 1
                variants.append({"name": name, "source": "pointcloud", "area_mm2": m.get("area_mm2"),
                                 **{key: m[key] for key in ("valid", "reason", "raw_contour_valid", "raw_contour_reason", "pipeline_valid", "pipeline_reason", "same_segment_crossing_group_count", "same_segment_nonlocal_crossing_count")},
                                 "comparison_valid": comparison_valid, "relative_area_change": raw_change if comparison_valid else None,
                                 "raw_relative_area_change": raw_change})
            density_comparison_valid = bool(pc_sections[k]["pipeline_valid"] and half["pipeline_valid"] and pc_area[k] > 0)
            raw_density_change = half.get("area_mm2", np.nan) / pc_area[k] - 1 if pc_area[k] > 0 else None
            stability.append({"section_index": int(k), "segment_id": seg["segment_id"], "s_local_mm": loc["s_local_mm"],
                              "reference_area_mm2": area[k], "variants": variants,
                              "pc_density_comparison_valid": density_comparison_valid,
                              "pc_density_relative_change": raw_density_change if density_comparison_valid else None,
                              "pc_density_relative_change_raw": raw_density_change})
    inlet = next((a for a in openings if a["label_provisional"] == "inlet"), None)
    opening_stability = []
    inlet_pc = {"valid": False, "reason": "inlet_boundary_not_identified"}
    inlet_pc_opening = None
    if inlet:
        for delta in [-5., -3., 3., 5.]:
            # Outward normal is defined by geometry; outside anatomy can be invalid.
            shift_center = inlet["center_mm"] + delta * inlet["normal"]
            m = opening_geometry_gate(slicer.slice(shift_center, inlet["normal"]), shift_center, inlet["normal"], inlet["segment_id"], xyz, sid, atlas_s=s, cfg=cfg)
            comparison_valid = bool(m["pipeline_valid"] and inlet["valid"])
            raw_change = m.get("area_mm2", np.nan) / inlet["area_mm2"] - 1
            opening_stability.append({"shift_along_outward_normal_mm": delta, "area_mm2": m.get("area_mm2"),
                                      **{key: m[key] for key in ("valid", "reason", "raw_contour_valid", "raw_contour_reason", "pipeline_valid", "pipeline_reason")},
                                      "comparison_valid": comparison_valid, "relative_area_change": raw_change if comparison_valid else None,
                                      "raw_relative_area_change": raw_change})
    # Point-cloud inlet independently locates its opening from the atlas root
    # and cloud rim. It does not inherit a reference-mesh center/normal/radius.
    trunk = next((row for row in segments if row["role"] == "trunk"), None)
    if trunk:
        trunk_rows = np.flatnonzero(sid == trunk["segment_id"])
        root_row = trunk_rows[np.argmin(s[trunk_rows])]
        for name, cloud in (("pointcloud_full", wall), ("pointcloud_half", wall[half_mask])):
            fit = _rim_plane(xyz[root_row], -tangent[root_row], radius[root_row], cloud, cKDTree(cloud))
            m = {"valid": False, "reason": "pointcloud_rim_plane_unavailable"}
            if fit is not None:
                m = pointcloud_section(cloud, fit["center"], fit["normal"], fit["radius"], spacing)
            m = opening_geometry_gate(m, fit["center"] if fit else None, fit["normal"] if fit else None, trunk["segment_id"], xyz, sid,
                                      ["rim_plane_tilt_requires_review"] if fit and fit["tilt_deg"] > 40 else [], atlas_s=s, cfg=cfg)
            if name == "pointcloud_full":
                inlet_pc = m
                inlet_pc_opening = None if fit is None else {"center_mm": fit["center"], "normal": fit["normal"],
                                                            "radius_hint_mm": fit["radius"], "rim_samples": fit["rim_samples"],
                                                            "tilt_deg": fit["tilt_deg"], "polygon_xyz_mm": m.get("polygon_xyz_mm", []),
                                                            "source": "atlas_root_and_bare_pointcloud_rim_plane"}
            comparison_valid = bool(m["pipeline_valid"] and inlet and inlet["valid"])
            raw_change = m.get("area_mm2", np.nan) / inlet["area_mm2"] - 1 if inlet else None
            opening_stability.append({"name": name, "area_mm2": m.get("area_mm2"),
                                      **{key: m[key] for key in ("valid", "reason", "raw_contour_valid", "raw_contour_reason", "pipeline_valid", "pipeline_reason")},
                                      "comparison_valid": comparison_valid, "relative_area_change": raw_change if comparison_valid else None,
                                      "raw_relative_area_change": raw_change,
                                      "opening_fit_recomputed_for_density": True})
    arrays = {
        "wall_xyz_mm": wall.astype(np.float32), "wall_triangles": triangles.astype(np.int32), "wall_node_id": wall_ids,
        "centerline_xyz_mm": xyz.astype(np.float32), "centerline_segment_id": sid,
        "centerline_s_local_mm": s.astype(np.float32), "centerline_radius_mis_mm": radius.astype(np.float32),
        "centerline_tangent": tangent.astype(np.float32), "centerline_parent_id": col("parent_id").astype(np.int16),
        "centerline_inside_reference": inside, "centerline_audit_core_mask": audit_core,
        "section_center_mm": field_array(section_locations, "center_mm").astype(np.float32),
        "section_tangent": field_array(section_locations, "normal").astype(np.float32),
        "section_segment_id": section_sid, "section_s_local_mm": section_s.astype(np.float32),
        "section_radius_mis_mm": field_array(section_locations, "radius_mis_mm").astype(np.float32),
        "section_area_slope_per_mm": slopes.astype(np.float32), "pc_section_area_slope_per_mm": pc_slopes.astype(np.float32),
        "wall_map_segment_id": mapped["segment_id"], "wall_map_s_local_mm": mapped["s_local_mm"].astype(np.float32),
        "wall_map_distance_mm": mapped["distance_mm"].astype(np.float32), "wall_map_ambiguous": mapped["ambiguous"],
        "wall_map_area_mm2": wall_area.astype(np.float32), "wall_map_valid": wall_valid,
        "wall_map_pc_area_mm2": wall_pc_area.astype(np.float32), "wall_map_pc_valid": wall_pc_valid,
        "wall_map_raw_area_mm2": raw_wall_area.astype(np.float32), "wall_map_pc_raw_area_mm2": raw_wall_pc_area.astype(np.float32),
    }
    for prefix, records in (("section_", reference), ("pc_section_", pc_sections)):
        for key in ("area_mm2", "req_mm", "perimeter_mm", "roundness", "eccentricity"):
            arrays[prefix + key] = field_array(records, key).astype(np.float32)
        arrays[prefix + "centroid_mm"] = np.asarray([r.get("centroid_mm", [np.nan] * 3) for r in records], np.float32)
        arrays[prefix + "valid"] = field_array(records, "valid", False, bool)
        arrays[prefix + "reason"] = field_array(records, "reason", "missing", str)
        arrays[prefix + "raw_contour_valid"] = field_array(records, "raw_contour_valid", False, bool)
        arrays[prefix + "raw_contour_reason"] = field_array(records, "raw_contour_reason", "missing", str)
        arrays[prefix + "same_segment_crossing_group_count"] = field_array(records, "same_segment_crossing_group_count", 0, np.int16)
        arrays[prefix + "same_segment_nonlocal_crossing_count"] = field_array(records, "same_segment_nonlocal_crossing_count", 0, np.int16)
        arrays[prefix + "polygon_xyz_mm"], arrays[prefix + "polygon_offsets"] = packed_polygons(records, "polygon_xyz_mm")
    arrays["section_intersection_segments_xyz_mm"], arrays["section_intersection_offsets"] = packed_polygons(reference, "intersection_segments_xyz_mm", (2, 3))
    # Finite, explicitly unapproved geometry groups G1--G6. No training stats,
    # WSS targets, model feature registry or experiment configuration is touched.
    for prefix, wall_prefix, values, accepted in (("section_", "wall_map_", area, valid),
                                                 ("pc_section_", "wall_map_pc_", pc_area, pc_valid)):
        context = {k: np.full(len(values), np.nan) for k in ("upstream_min_area_ratio", "downstream_min_area_ratio", "upstream_min_distance_mm", "downstream_min_distance_mm")}
        for seg in segments:
            si = np.flatnonzero(section_sid == seg["segment_id"])
            pack = branch_context(section_s[si], values[si], accepted[si], cfg["context_window_mm"])
            for key, v in pack.items():
                context[key][si] = v
        for key, v in context.items():
            arrays[prefix + key] = v.astype(np.float32)
        for key in ("area_slope_per_mm", "roundness", "eccentricity", *context):
            v = arrays[prefix + key]
            mapped_v = np.full(len(wall), np.nan)
            mapped_good = np.zeros(len(wall), bool)
            for seg in segments:
                si = np.flatnonzero(section_sid == seg["segment_id"])
                wi = np.flatnonzero(mapped["segment_id"] == seg["segment_id"])
                mapped_v[wi], mapped_good[wi] = interpolate_valid_field(mapped["s_local_mm"][wi], section_s[si], v[si], accepted[si] & np.isfinite(v[si]))
            mapped_good &= ~mapped["ambiguous"]
            mapped_v[~mapped_good] = np.nan
            arrays[wall_prefix + key] = mapped_v.astype(np.float32)
            arrays[wall_prefix + key + "_valid"] = mapped_good
    cia_of = {}
    by_segment = {r["segment_id"]: r for r in segments}
    for row in segments:
        current, seen = row, set()
        while current["role"] != "cia" and current["parent_id"] in by_segment and current["segment_id"] not in seen:
            seen.add(current["segment_id"])
            current = by_segment[current["parent_id"]]
        cia_of[row["segment_id"]] = current["segment_id"] if current["role"] == "cia" else -1
    wall_cia = np.array([cia_of.get(int(i), -1) for i in mapped["segment_id"]], np.int16)
    arrays["wall_cia_segment_id"] = wall_cia
    arrays["wall_cia_geometry_valid"] = (wall_cia >= 0) & ~mapped["ambiguous"] & topology["valid"]
    arrays["wall_cia_semantic_valid"] = np.array([by_segment[int(i)]["semantic_valid"] if i >= 0 else False for i in wall_cia])
    for key in ("length_mm", "tortuosity", "parent_branch_angle_deg", "daughter_angle_deg"):
        arrays["wall_cia_" + key] = np.array([by_segment[int(i)].get(key, np.nan) if i >= 0 else np.nan for i in wall_cia], np.float32)
    paired = valid & pc_valid & (area > 0)
    pc_error = np.abs(pc_area[paired] / area[paired] - 1)
    outside_core = np.flatnonzero(audit_core & ~inside)
    source_hash = digest_arrays({"wall": wall, "triangles": triangles, "atlas": table})
    warnings = []
    if not topology["valid"]:
        warnings.append("topology_failed")
    if not enclosed_audit["closed_reference_valid"] or len(outside_core):
        warnings.append("centerline_inside_audit_needs_review")
    if topology["mixed_side_cia_segments"]:
        warnings.append("mixed_CFD_sides_anatomy_unknown")
    if not opening_audit["valid"]:
        warnings.append("opening_loop_count_needs_review")
    if float(valid.mean()) < cfg["reference_coverage_review_threshold"]:
        warnings.append("reference_section_coverage_low")
    if float(pc_valid.mean()) < cfg["pointcloud_coverage_review_threshold"]:
        warnings.append("pointcloud_section_coverage_low")
    if len(pc_error) and np.percentile(pc_error, 90) > cfg["pointcloud_area_p90_review_threshold"]:
        warnings.append("pointcloud_reference_disagreement")
    report = {
        "schema": SCHEMA, "canonical_id": canonical_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "awaiting_user_review": True, "training_allowed": False, "user_approved": False,
        "status": "candidate_needs_review" if warnings else "candidate_ready_for_visual_review", "warnings": warnings,
        "config": cfg, "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "algorithm_code": code_hashes,
        "source": {"case_h5": str(source), "geometry_arrays_sha256": source_hash, "atlas_provenance": provenance,
                   "surface_source": "cfd_anatomy_wall_geometry_reference", "pointcloud_source": "same_wall_xyz_without_connectivity",
                   "fields_read": ["wall_static/xyz_mm", "wall_static/node_id_cas", "topology/wall_triangles", "geometry/atlas_table", "geometry attributes"],
                   "label_fields_read": [], "pointcloud_portability_proven": False, "frozen_assets_modified": False},
        "algorithm": {"reference_section": "triangle_plane_intersection_closed_contour_containing_centerline",
                      "pointcloud_section": "branch_filtered_slab_radial_sector_medians_star_convex_candidate",
                      "pointcloud_limit": "nonstar_sections_may_not_be_identifiable; angular_gap_and_radial_multimodality_failures_retained",
                      "pointcloud_mask_source": "pointcloud_polygon_and_centerline_geometry_only; reference_polygon_not_consumed",
                      "same_segment_gate": "continuous_polyline_plane_edge_intersections; duplicate_vertex_and_contiguous_near_coplanar_hits_merged_by_s; extra_inside_group_more_than_5mm_from_local_s_rejected",
                      "wall_mapping": "continuous_projection_nearest_midpoint_8_edges_per_branch; cross_branch_distance_ratio_1.2",
                      "area_slope_per_mm": "d(log(area_mm2))/ds_mm; separate_contiguous_valid_runs",
                      "context_minimum_ties": "choose_nearest_in_each_direction_among_area_minima_isclose_rtol1e-8_atol1e-10_mm2",
                      "eccentricity": "distance_of_section_area_centroid_from_centerline / equivalent_area_radius",
                      "semantic_policy": "topology_entities_separate_from_CFD_zone_names; all_anatomy_names_await_user_review",
                      "radius_area_check": "A/(pi*smoothed_Rmis^2)_diagnostic_only_not_a_hard_gate"},
        "P1_centerline": {**enclosed_audit, "reused_without_coordinate_changes": True, "core_points": int(audit_core.sum()),
                          "outside_core_points": int(len(outside_core)), "outside_core_atlas_rows": outside_core.tolist(),
                          "outside_core_fraction": float(np.mean(~inside[audit_core])) if audit_core.any() else None},
        "P2_topology": topology, "P3_CIA": [r for r in segments if r["role"] == "cia"], "segments": segments,
        "openings": openings, "opening_audit": opening_audit, "inlet_reference_area_mm2": inlet["area_mm2"] if inlet else None,
        "inlet_pointcloud_area_mm2": inlet_pc.get("area_mm2"), "inlet_pointcloud_valid": inlet_pc["valid"],
        "inlet_pointcloud_reason": inlet_pc["reason"],
        "inlet_pointcloud_dependency": "atlas_root_and_bare_pointcloud_rim_plane; reference_center_normal_and_area_not_consumed",
        "inlet_pointcloud_opening": inlet_pc_opening,
        "P4_sections": {"count": len(area), "reference_valid": int(valid.sum()), "pointcloud_valid": int(pc_valid.sum()),
                        "reference_valid_fraction": float(valid.mean()), "pointcloud_valid_fraction": float(pc_valid.mean()),
                        "reference_reasons": dict(Counter(r["reason"] for r in reference)),
                        "pointcloud_reasons": dict(Counter(r["reason"] for r in pc_sections)), "paired_valid": int(paired.sum()),
                        "pointcloud_reference_relative_area_median_abs": float(np.median(pc_error)) if len(pc_error) else None,
                        "pointcloud_reference_relative_area_p90_abs": float(np.percentile(pc_error, 90)) if len(pc_error) else None},
        "same_segment_crossing_diagnostics": [{"source": source_name, "section_index": k, "segment_id": int(section_sid[k]),
                                               "s_local_mm": float(section_s[k]), **item["same_segment_plane_intersections"]}
                                              for source_name, records in (("reference_surface", reference), ("pointcloud", pc_sections))
                                              for k, item in enumerate(records) if item["same_segment_nonlocal_crossing_count"] > 0],
        "P5_wall_mapping": {"points": len(wall), "reference_valid_fraction": float(wall_valid.mean()),
                            "pointcloud_valid_fraction": float(wall_pc_valid.mean()), "ambiguous_fraction": float(mapped["ambiguous"].mean()),
                            "invalid_values_retained_as_nan": True},
        "feature_candidates": {"status": "computed_awaiting_geometry_user_review_not_registered_as_training_features",
                               "G1_inlet": "reference_and_pointcloud_area_separate; fixed_common_Q_is_only_a_scale_reencoding",
                               "G2_section": "area_and_equivalent_radius_for_two_sources",
                               "G3_slope": "dlogA/ds_by_contiguous_valid_branch_run",
                               "G4_shape": "roundness_and_area_centroid_offset_over_Req",
                               "G5_context": f"minimum_area_ratio_and_distance_upstream_downstream_within_{cfg['context_window_mm']}mm_same_branch_valid_run",
                               "G6_CIA": "topology_CIA_id_length_tortuosity_parent_and_daughter_angles_mapped_to_CIA_and_descendant_walls; anatomy_valid_separate",
                               "normalization": "raw_physical_quantities_only; no_train_or_test_statistics_fit"},
        "stability": stability, "opening_stability": opening_stability,
        "P6_stability": {"sampled_sections": len(stability), "cloud_density_fraction": 0.5, "random_seed": cfg["seed"],
                         "slab_thickness_fixed_across_density_variants": True, "shift_mm": cfg["stability_shift_mm"], "tilt_deg": cfg["stability_tilt_deg"],
                         "section_gate": "same_P4_endpoint_junction_other_branch_gate_recomputed_for_each_variant; raw_contour_and_pipeline_validity_separate",
                         "perturbation_sources": "reference_surface_and_pointcloud_both_shifted_and_tilted",
                         "scope": "section_reconstruction_and_geometry_gate_only; does_not_recompute_whole_branch_slopes_context_wall_mapping_or_A5_PCA_curvature",
                         "opening_gate": "opening_specific_contour_and_other_branch_gate; endpoint_exclusion_not_applicable_to_openings; rim_fit_recomputed_at_half_density",
                         "half_density_mapping": "retained_points_reuse_their_geometry_only_nearest_branch_projection; atlas_and_spacing_recipe_fixed",
                         "interpretation": "geometric_sensitivity_probe_not_training_accuracy_or_user_approval"},
        "seconds": round(time.time() - start, 2), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    dest = output / "cases" / canonical_id.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    temp = dest / "geometry.tmp.npz"
    np.savez_compressed(temp, **arrays)
    temp.replace(dest / "geometry.npz")
    report["geometry_npz"] = str(dest / "geometry.npz")
    report = correct_report(report)
    write_json(dest / "report.json", report)
    return report


def all_cases(split_path: Path | None = None) -> list[str]:
    # Frozen split identity is allowed; labels and outcomes are never read.
    split = json.loads((split_path or C.SPLIT_PATH).read_text())
    train = set(split.get("train_cases", []))
    if not train:
        for fold in split["cv"]["folds"]:
            train.update(fold["train_cases"])
            train.update(fold["validation_cases"])
    return sorted(train | set(split["test_cases"]))


def summarize(output: Path, expected: list[str]) -> dict:
    reports = []
    for cid in expected:
        path = output / "cases" / cid.replace("/", "__") / "report.json"
        if path.is_file():
            r = json.loads(path.read_text())
            reports.append({"canonical_id": cid, "status": r["status"], "warnings": r.get("warnings", []),
                            "reference_valid_fraction": r.get("P4_sections", {}).get("reference_valid_fraction"),
                            "pointcloud_valid_fraction": r.get("P4_sections", {}).get("pointcloud_valid_fraction"),
                            "pointcloud_reference_relative_area_p90_abs": r.get("P4_sections", {}).get("pointcloud_reference_relative_area_p90_abs"),
                            "outside_core_points": r.get("P1_centerline", {}).get("outside_core_points"),
                            "mixed_side_cia_segments": r.get("P2_topology", {}).get("mixed_side_cia_segments"),
                            "report_json": str(path), "geometry_npz": r.get("geometry_npz"),
                            "awaiting_user_review": True, "training_allowed": False})
        else:
            reports.append({"canonical_id": cid, "status": "not_processed", "awaiting_user_review": True, "training_allowed": False})
    manifest = {"schema": SCHEMA, "phase": "P_geometry_candidate_only", "created_at": datetime.now(timezone.utc).isoformat(),
                "awaiting_user_review": True, "training_allowed": False, "user_approved": False,
                "expected_cases": len(expected), "status_counts": dict(Counter(r["status"] for r in reports)), "cases": reports,
                "no_experiment_may_consume_this_output_until_user_confirmation": True,
                "pointcloud_portability_proven": False}
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--split", type=Path, default=C.SPLIT_PATH,
                    help="geometry cohort identity; leaves the frozen contract default unchanged")
    ap.add_argument("--source-root", type=Path, default=C.SNAPSHOT_ROOT,
                    help="read-only HDF5 snapshot root containing cases/<key>/case.h5")
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--case-index", type=int, help="zero-based index into sorted cases from --split, for Slurm arrays")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    output = args.output.resolve()
    source_root = args.source_root.resolve()
    for protected in (C.SNAPSHOT_ROOT, source_root, C.ATLAS_DIR, C.ROOT / "data_wss_v5/views"):
        if output == protected.resolve() or protected.resolve() in output.parents:
            ap.error("output must be a separate candidate directory, never a frozen asset/view")
    cfg = json.loads(args.config.read_text())
    expected = all_cases(args.split)
    if args.summarize:
        m = summarize(output, expected)
        print(json.dumps(m["status_counts"], ensure_ascii=False), flush=True)
        return
    if not os.environ.get("SLURM_JOB_ID"):
        ap.error("case geometry processing must run inside a Slurm allocation; use --summarize for metadata only")
    selected = args.cases or (cfg["pilot_cases"] if args.pilot else expected)
    if args.case_index is not None:
        if not 0 <= args.case_index < len(expected):
            ap.error(f"--case-index must be between 0 and {len(expected) - 1} for this split")
        selected = [expected[args.case_index]]
    unknown = set(selected) - set(expected)
    if unknown:
        ap.error(f"cases outside formal geometry cohort: {sorted(unknown)}")
    failed = 0
    for cid in selected:
        dest = output / "cases" / cid.replace("/", "__")
        existing = dest / "report.json"
        if args.resume and existing.is_file():
            old = json.loads(existing.read_text())
            expected_source = C.case_dir(cid, root=source_root) / "case.h5"
            same_source = Path(old.get("source", {}).get("case_h5", "")).resolve() == expected_source.resolve()
            if old.get("status") != "processing_failed" and (dest / "geometry.npz").is_file() and old.get("config") == cfg and old.get("algorithm_code") == code_fingerprints() and same_source:
                print(json.dumps({"case": cid, "status": "resumed_existing_candidate"}), flush=True)
                continue
        try:
            r = case_candidate(cid, output, cfg, source_root=source_root)
            print(json.dumps({"case": cid, "status": r["status"], "seconds": r["seconds"], "P4": r["P4_sections"]}, ensure_ascii=False), flush=True)
        except Exception as exc:
            failed += 1
            write_json(existing, {"schema": SCHEMA, "canonical_id": cid, "status": "processing_failed", "error": str(exc),
                                  "traceback": traceback.format_exc(), "awaiting_user_review": True, "training_allowed": False})
            print(json.dumps({"case": cid, "status": "processing_failed", "error": str(exc)}), flush=True)
    # Array workers do not race on a shared global manifest; run --summarize afterwards.
    if args.case_index is None:
        summarize(output, expected)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
