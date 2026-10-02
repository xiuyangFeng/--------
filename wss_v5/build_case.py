"""Build one V5 case bundle: sources -> mesh/identity -> frames -> features -> point-cloud gate -> HDF5 + manifest."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from wss_pinn.utils import git_state, sha256_file, utc_now

from . import contract as C
from .centerline_features import POINT_FEATURES, alignment_metrics, load_atlas, map_points, relabel_outlets
from .conditions import read_conditions, waveform_samples
from wss_pinn.v4.new_case_sources import udf_to_dataset_outlets
from .mesh_topology import MeshData, distance_to_wall, load_mesh, wall_triangles
from .pointcloud import build_oriented_cloud, generate_internal_queries, inside_hybrid, inside_score_vote, patch_atlas_end_radius, winding_number
from .raw_frames import VolumeFrames, WallFrames, read_volume_frames, read_wall_frames
from .sources import CaseSources, Registry, locate
from .store import CaseWriter, aggregate_digest

PEAK_STEP = 1162
MM = C.LENGTH_M_TO_MM


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _interface_tables(md: MeshData, vol: VolumeFrames, semantics: dict[int, str]) -> dict[str, dict[str, Any]]:
    row_of_cell = -np.ones(md.mesh.cell_count + 1, dtype=np.int64)
    row_of_cell[vol.cell_id_cas] = np.arange(len(vol.cell_id_cas))
    tables: dict[str, dict[str, Any]] = {}
    for interface in md.interfaces:
        label = "inlet" if interface.distal_bc_type in (10, 4, 20) else semantics.get(int(interface.distal_bc_zone_id), "")
        rows = row_of_cell[interface.anatomy_cells]
        if np.any(rows < 0):
            raise ValueError(f"interface {label}: adjacent anatomy cell missing from the export")
        area_vec = interface.area_vectors_m2
        area = np.linalg.norm(area_vec, axis=1)
        p_mean = np.einsum("tf,f->t", vol.pressure[:, rows].astype(np.float64), area) / area.sum()
        flux = np.einsum("tfk,fk->t", vol.velocity[:, rows, :].astype(np.float64), area_vec)  # outward positive
        rim_rows = md.node_row(interface.node_ids)
        tables[label] = {
            "extension_zone": interface.extension_zone_name,
            "center_mm": (np.average(interface.centres_m, axis=0, weights=area) * MM),
            "normal_out": interface.unit_normal,
            "area_m2": float(area.sum()),
            "planarity": interface.planarity,
            "face_center_mm": interface.centres_m * MM,
            "face_area_vec_m2": area_vec,
            "adjacent_volume_row": rows,
            "rim_wall_rows": rim_rows[rim_rows >= 0],
            "p_mean_pa": p_mean,
            "flux_outward_m3s": flux,
        }
    return tables


def _pointcloud_gate(md: MeshData, vol: VolumeFrames, atlas, wall_xyz_mm: np.ndarray, interfaces: dict[str, dict[str, Any]],
                     seed: int, n_sub: int, n_generate: int) -> dict[str, Any]:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    cloud, info = build_oriented_cloud(wall_xyz_mm, atlas)  # bare point cloud: PCA normals, nominal areas
    n_wall = len(wall_xyz_mm)
    pca = cloud.normals_out[:n_wall]
    cosang = np.clip(np.einsum("ij,ij->i", pca, md.wall_node_normals_out), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosang))
    normals = {"angle_median_deg": float(np.median(angle)), "angle_p95_deg": float(np.percentile(angle, 95)),
               "flipped_fraction": float(np.mean(angle > 90.0))}
    # truth sets (judging only)
    anatomy_mm = vol.xyz_m * MM
    all_c = md.centroids_m[1:]
    is_anat = md.cell_zone[1:] == md.anatomy_zone_id
    ext_mm = all_c[~is_anat] * MM
    centers = np.stack([v["center_mm"] for v in interfaces.values()])
    normals_if = np.stack([np.asarray(v["normal_out"]) for v in interfaces.values()])
    d_center, which = cKDTree(centers).query(ext_mm, k=1)
    plane_dist = np.einsum("ij,ij->i", ext_mm - centers[which], normals_if[which])  # >0 beyond the cut, in the extension
    near = (d_center <= 15.0) & (plane_dist >= 2.0)
    near_band = (d_center <= 15.0) & (plane_dist < 2.0)
    ext_band_mm = ext_mm[near_band]
    ext_mm = ext_mm[near]
    # cap centre offset from the true interface plane (>0 = beyond the cut / outside), judging only
    cap_offsets = {}
    for cap in info["caps"]:
        if cap["label"] in interfaces:
            c = np.asarray(cap["center_mm"]); v = interfaces[cap["label"]]
            cap_offsets[cap["label"]] = {"plane_offset_mm": float((c - v["center_mm"]) @ np.asarray(v["normal_out"])),
                                         "center_gap_mm": float(np.linalg.norm(c - v["center_mm"])),
                                         "radius_ratio_cap_over_sqrt_area": float(cap["radius_mm"] / np.sqrt(v["area_m2"] * 1e6 / np.pi))}
    shell_off = rng.uniform(1.0, 3.0, size=n_wall)
    shell_mm = wall_xyz_mm + md.wall_node_normals_out * shell_off[:, None]

    def sub(a):
        return a[rng.choice(len(a), size=min(n_sub, len(a)), replace=False)] if len(a) else a

    sets = {"anatomy_cells": sub(anatomy_mm), "extension_cells_near_interface": sub(ext_mm),
            "extension_cells_within_2mm_of_cut": sub(ext_band_mm), "shell_1_3mm_outside": sub(shell_mm)}  # rng draw order unchanged
    # 2026-09-22: a shell point 1-3 mm outside one wall can lie inside another segment of the same lumen (post-op
    # ILO limbs running side by side: 2-5 % of the shell). Those points are inside by ground truth (they fall in a
    # CFD cell), so they are not "false inside" samples; the gate judges the true-outside subset. Library cases
    # only move down (their max was 1.9 %), the original metric is kept for comparability.
    shell_sub = sets["shell_1_3mm_outside"]
    cell_size_mm = np.cbrt(md.volumes_m3[1:]) * MM
    d_cell, i_cell = cKDTree(all_c * MM).query(shell_sub, k=1) if len(shell_sub) else (np.zeros(0), np.zeros(0, dtype=int))
    shell_in_volume = d_cell <= 0.8 * cell_size_mm[i_cell] if len(shell_sub) else np.zeros(0, dtype=bool)
    sets["shell_1_3mm_outside_true_outside"] = shell_sub[~shell_in_volume]
    results: dict[str, Any] = {}
    for name, pts in sets.items():
        if not len(pts):
            results[name] = {"n": 0}
            continue
        score = inside_score_vote(pts, cloud)
        w = winding_number(pts, cloud)
        hybrid, hdiag = inside_hybrid(pts, cloud)
        results[name] = {
            "n": int(len(pts)),
            "inside_fraction_vote": float(np.mean(score > 0)),
            "inside_fraction_winding": float(np.mean(w > 0.5)),
            "inside_fraction_hybrid": float(np.mean(hybrid)),
            "winding_median": float(np.median(w)),
            **hdiag,
        }
    results["shell_in_cfd_volume_fraction"] = float(np.mean(shell_in_volume)) if len(shell_sub) else 0.0  # 2026-09-22 judging only
    results["cap_offsets"] = cap_offsets
    generated, gdiag = generate_internal_queries(atlas, cloud, n_target=n_generate, seed=seed)
    # leak check of generated points against the true cell zones (nearest cell, distance-limited)
    tree_all = cKDTree(all_c * MM)
    d, idx = tree_all.query(generated, k=1)
    size = np.cbrt(md.volumes_m3[1:][idx]) * MM
    far = d > 2.0 * size
    leak = ~is_anat[idx] | far
    gdiag.update({"generated_kept": int(len(generated)), "leak_fraction": float(np.mean(leak)) if len(generated) else None,
                  "leak_far_fraction": float(np.mean(far)) if len(generated) else None})
    return {"cloud": info, "normals_pca_vs_mesh": normals, "inside_test": results, "generated_queries": gdiag,
            "generated_sample_mm": generated[: min(len(generated), 50_000)], "pca_normals_out": pca.astype(np.float32),
            "seconds": time.time() - t0}


def _label_checks(md: MeshData, vol: VolumeFrames, wall: WallFrames) -> dict[str, Any]:
    t_peak = C.EXPECTED_STEPS.index(PEAK_STEP)
    scalar = wall.wss_scalar[:, wall.valid].astype(np.float64)
    vector = wall.wss_vector[:, wall.valid].astype(np.float64)
    norm = np.linalg.norm(vector, axis=2)
    gap = (scalar - norm) / np.maximum(scalar, 1.0e-12)
    tree = cKDTree(vol.xyz_m)
    _, nearest = tree.query(md.wall_node_coords_m[wall.valid], k=1)
    delta = wall.pressure[t_peak, wall.valid].astype(np.float64) - vol.pressure[t_peak, nearest].astype(np.float64)
    tang = np.abs(np.einsum("ij,ij->i", vector[t_peak], md.wall_node_normals_out[wall.valid])) / np.maximum(norm[t_peak], 1.0e-12)
    return {
        "all_finite": bool(np.isfinite(scalar).all() and np.isfinite(vector).all() and np.isfinite(vol.velocity).all() and np.isfinite(vol.pressure).all()),
        "wss_negative_rows": int(np.sum(scalar < 0)),
        "vector_norm_le_scalar_fraction": float(np.mean(norm <= scalar * (1.0 + 1.0e-3) + 1.0e-9)),
        "scalar_minus_vector_rel_median": float(np.median(gap)),
        "wall_pressure_minus_adjacent_cell_median_pa": float(np.median(delta)),
        "wall_pressure_minus_adjacent_cell_p95_abs_pa": float(np.percentile(np.abs(delta), 95)),
        "tangency_p95": float(np.percentile(tang[norm[t_peak] > 1.0e-12], 95)) if np.any(norm[t_peak] > 1.0e-12) else None,
        "wss_peak_p50_pa": float(np.percentile(scalar[t_peak], 50)),
        "wss_peak_p99_pa": float(np.percentile(scalar[t_peak], 99)),
        "wss_peak_max_pa": float(scalar[t_peak].max()),
        "speed_peak_max_m_s": float(np.linalg.norm(vol.velocity[t_peak], axis=1).max()),
        "pressure_peak_mean_pa": float(vol.pressure[t_peak].mean()),
    }


def _gates(report: dict[str, Any], src: CaseSources) -> dict[str, Any]:
    g = C.GATES
    m = report["metrics"]
    waived = set()
    if not src.wall_audit.get("gate_pass", True):
        waived.update(src.wall_audit.get("failed_gates", []))
    checks = {
        "source_cas_sha_matches_audit": report["sources"]["cas_sha256"] == src.cas_sha256_expected,
        "volume_frames_81": len(report["frames"]["volume"]["files"]) == len(C.EXPECTED_STEPS),
        "wall_frames_81": len(report["frames"]["wall"]["files"]) == len(C.EXPECTED_STEPS),
        "volume_identity_bijective": bool(m["identity"]["volume"].get("bijective", False)),
        "anatomy_rows_match_audit": m["counts"]["n_volume"] == int(src.topology_audit["anatomy_rows"]),
        "wall_missing_nodes_within_limit": m["counts"]["n_wall_missing"] <= g["wall_missing_nodes_max"],
        # whole-domain node exports (mixed-zone cases, 2026-09-22) legitimately carry interior/extension rows: they are
        # dropped by coordinate matching and the anatomy coverage is still enforced by wall_missing_nodes_within_limit
        "wall_rows_outside_anatomy_zero": report["frames"]["wall"]["rows_outside_anatomy"] == 0
                                          or report["frames"]["wall"].get("export_scope") == "whole_domain",
        "labels_finite": bool(m["labels"]["all_finite"]),
        "wss_non_negative": m["labels"]["wss_negative_rows"] == 0,
        "wall_same_solution_as_volume": abs(m["labels"]["wall_pressure_minus_adjacent_cell_median_pa"]) <= g["wall_pressure_delta_median_pa_max"],
        "volume_sum_matches_mesh": abs(m["integration"]["volume_sum_m3"] - m["integration"]["anatomy_volume_mesh_m3"]) <= g["volume_sum_rel_tol"] * m["integration"]["anatomy_volume_mesh_m3"],
        "area_sum_matches_mesh": abs(m["integration"]["node_area_sum_m2"] - m["integration"]["anatomy_wall_area_mesh_m2"]) <= g["area_sum_rel_tol"] * m["integration"]["anatomy_wall_area_mesh_m2"],
        "interfaces_five_labelled": sorted(m["interfaces"].keys()) == sorted(C.INTERFACE_ORDER),
        "atlas_aligned": m["atlas_alignment"]["median_abs_rel_gap"] <= g["atlas_alignment_median_rel_gap_max"],
        "atlas_outlet_labels_resolved": bool(m.get("atlas_relabel", {}).get("all_leaves_resolved", True)),
        "pca_normals_ok": (m["pointcloud"]["normals_pca_vs_mesh"]["angle_median_deg"] <= g["pca_normal_angle_median_deg_max"]
                           and m["pointcloud"]["normals_pca_vs_mesh"]["angle_p95_deg"] <= g["pca_normal_angle_p95_deg_max"]
                           and m["pointcloud"]["normals_pca_vs_mesh"]["flipped_fraction"] <= g["pca_normal_flipped_fraction_max"]),
        "domain_inside_recall": m["pointcloud"]["inside_test"]["anatomy_cells"].get("inside_fraction_hybrid", 0.0) >= g["inside_recall_min"],
        "domain_extension_rejected": m["pointcloud"]["inside_test"]["extension_cells_near_interface"].get("inside_fraction_hybrid", 1.0) <= g["extension_false_inside_max"],
        "domain_shell_rejected": m["pointcloud"]["inside_test"].get("shell_1_3mm_outside_true_outside", m["pointcloud"]["inside_test"]["shell_1_3mm_outside"]).get("inside_fraction_hybrid", 1.0) <= g["shell_false_inside_max"],
    }
    waiver_map = {"wall_same_solution_as_volume": "same_solution_as_volume"}
    applied = {}
    for name, audit_gate in waiver_map.items():
        if not checks[name] and audit_gate in waived:
            applied[name] = f"waived: signed-off wall audit already records '{audit_gate}' for this case (constant gauge offset)"
    for name, reason in getattr(C, "CASE_GATE_WAIVERS", {}).get(report["canonical_id"], {}).items():  # 2026-09-22 user sign-offs
        if name in checks and not checks[name]:
            applied[name] = f"waived (case sign-off): {reason}"
    required_pass = all(v or (k in applied) for k, v in checks.items())
    return {"checks": checks, "waivers": applied, "pass": required_pass}


def build_case(canonical_id: str, registry: Registry, out_root: Path, *, hash_files: bool = True, seed: int = 0,
               domain_subsample: int = C.WINDING_SUBSAMPLE, n_generate: int = 100_000, keep_h5: bool = True) -> dict[str, Any]:
    started = time.time()
    timings: dict[str, float] = {}
    case_out = C.case_dir(canonical_id, out_root)
    case_out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"schema_version": C.SCHEMA_VERSION, "geometry_program_version": C.GEOMETRY_PROGRAM_VERSION,
                              "canonical_id": canonical_id, "created_at": utc_now(), "status": "running"}
    writer = None
    try:
        src = locate(canonical_id, registry)
        t = time.time(); cas_sha = sha256_file(src.cas_path); timings["cas_sha"] = time.time() - t
        t = time.time(); md = load_mesh(src.cas_path); timings["mesh"] = time.time() - t
        t = time.time(); vol = read_volume_frames(src.raw_dir, md, hash_files=hash_files); timings["volume_frames"] = time.time() - t
        t = time.time(); wall = read_wall_frames(src.raw_dir, md, hash_files=hash_files); timings["wall_frames"] = time.time() - t
        t = time.time()
        interfaces = _interface_tables(md, vol, src.interface_semantics())
        timings["interfaces"] = time.time() - t
        t = time.time()
        atlas, relabel = relabel_outlets(load_atlas(src.atlas_npz, src.atlas_summary), {k: v["center_mm"] for k, v in interfaces.items()})
        wall_xyz_mm = md.wall_node_coords_m * MM
        atlas_end_patch = patch_atlas_end_radius(atlas, wall_xyz_mm)  # 2026-09-22: escaped-sphere endpoint radius -> held at the measured opening
        vol_xyz_mm = vol.xyz_m * MM
        wall_feats = map_points(wall_xyz_mm, atlas)
        vol_feats = map_points(vol_xyz_mm, atlas)
        align = alignment_metrics(wall_xyz_mm, atlas)
        dist_m, tri = distance_to_wall(vol.xyz_m, md)
        timings["features"] = time.time() - t
        volumes = md.volumes_m3[vol.cell_id_cas]
        p_vol_mean = np.einsum("tn,n->t", vol.pressure.astype(np.float64), volumes) / volumes.sum()
        wave = waveform_samples()
        p_cycle = float(np.trapz(p_vol_mean, wave["time_s"]) / (wave["time_s"][-1] - wave["time_s"][0]))
        cond = read_conditions(src.udf_path, md.inlet_bc_face_area_m2, {k: v["area_m2"] for k, v in interfaces.items()}, src.cohort,
                               outlet_relabel=udf_to_dataset_outlets(src.canonical_id, src.raw_dir))
        t = time.time()
        pc = _pointcloud_gate(md, vol, atlas, wall_xyz_mm, interfaces, seed, domain_subsample, n_generate)
        timings["pointcloud_gate"] = time.time() - t
        labels = _label_checks(md, vol, wall)
        n_missing = int(np.sum(~wall.valid))
        report.update({
            "cohort": src.cohort, "role": src.role, "validation_fold": src.validation_fold, "patient_group": src.patient_group,
            "sources": {
                "raw_dir": str(src.raw_dir), "cas_path": str(src.cas_path), "cas_sha256": cas_sha, "cas_sha256_expected": src.cas_sha256_expected,
                "atlas_npz": str(src.atlas_npz), "atlas_created_at": json.loads(src.atlas_summary.read_text())["created_at"],
                "udf": cond["udf"], "udf_alternates": src.udf_alternates, "monitor_files": src.monitor_files,
                "solver": src.solver, "wall_audit": src.wall_audit,
            },
            "frames": {
                "volume": {"files": vol.frame_files, "reordered_steps": vol.frames_reordered, "export_rows_total": vol.export_rows_total, "zone_counts": vol.zone_counts},
                "wall": {"files": wall.frame_files, "rematched_steps": wall.frames_rematched, "duplicate_rows": wall.duplicate_rows,
                         "rows_outside_anatomy": wall.rows_outside_anatomy, "export_kind": wall.export_kind, "delimiter": wall.delimiter,
                         "export_scope": wall.export_scope},
            },
            "metrics": {
                "counts": {"n_volume": int(len(vol.cell_id_cas)), "n_wall": int(len(md.wall_node_ids)), "n_wall_missing": n_missing, "n_frames": len(C.EXPECTED_STEPS),
                           "n_atlas_samples": int(len(atlas.table))},
                "identity": {"volume": vol.identity, "wall": wall.identity},
                "integration": {"volume_sum_m3": float(volumes.sum()), "anatomy_volume_mesh_m3": md.anatomy_volume_m3,
                                "node_area_sum_m2": float(md.wall_node_area_m2.sum()), "anatomy_wall_area_mesh_m2": md.anatomy_wall_area_m2,
                                "dist_to_wall_mm": {"min": float(dist_m.min() * MM), "median": float(np.median(dist_m) * MM), "max": float(dist_m.max() * MM),
                                                     "within_0_5mm_fraction": float(np.mean(dist_m * MM <= 0.5)), "within_1_5mm_fraction": float(np.mean(dist_m * MM <= 1.5))}},
                "interfaces": {k: {"area_m2": v["area_m2"], "planarity": v["planarity"], "faces": int(len(v["adjacent_volume_row"])), "rim_wall_nodes": int(len(v["rim_wall_rows"])),
                                   "flux_outward_peak_m3s": float(v["flux_outward_m3s"][C.EXPECTED_STEPS.index(PEAK_STEP)]),
                                   "p_mean_peak_pa": float(v["p_mean_pa"][C.EXPECTED_STEPS.index(PEAK_STEP)])} for k, v in interfaces.items()},
                "mass_balance_peak_rel_inlet": float(abs(sum(v["flux_outward_m3s"][C.EXPECTED_STEPS.index(PEAK_STEP)] for v in interfaces.values()))
                                                     / max(abs(interfaces["inlet"]["flux_outward_m3s"][C.EXPECTED_STEPS.index(PEAK_STEP)]), 1e-12)) if "inlet" in interfaces else None,
                "atlas_alignment": align,
                "atlas_relabel": relabel,
                "atlas_end_radius_patch": atlas_end_patch,
                "pointcloud": {k: v for k, v in pc.items() if k not in ("generated_sample_mm", "pca_normals_out")},
                "labels": labels,
                "pressure_reference": {"p_volume_mean_peak_pa": float(p_vol_mean[C.EXPECTED_STEPS.index(PEAK_STEP)]), "p_case_cycle_mean_pa": p_cycle,
                                       "p_inlet_mean_peak_pa": float(interfaces["inlet"]["p_mean_pa"][C.EXPECTED_STEPS.index(PEAK_STEP)]) if "inlet" in interfaces else None},
                "conditions": {"protocol_id": cond["protocol"]["protocol_id"], "r_total_parallel_dc": cond["protocol"]["r_total_parallel_dc"],
                               "inlet_area_ratio": cond["inlet_area_ratio"], "left_conductance_share": cond["protocol"]["left_conductance_share"],
                               "protocol_exception": cond["protocol"]["protocol_exception"], "protocol_exception_reasons": cond["protocol"]["protocol_exception_reasons"]},
            },
        })
        report["gates"] = _gates(report, src)

        # ------------------------------------------------------------ write HDF5
        t = time.time()
        h5_path = case_out / "case.h5"
        writer = CaseWriter(h5_path)
        writer.attrs("/", {"schema_version": C.SCHEMA_VERSION, "canonical_id": canonical_id, "cohort": src.cohort, "role": src.role,
                           "patient_group": src.patient_group, "validation_fold": src.validation_fold, "created_at": report["created_at"],
                           "length_units": "mm (Fluent m x 1000)", "time_units": "s", "coordinate_frame": "Fluent case frame (no rotation/translation applied)",
                           "dependency_tags": [C.TAG_MODEL_FEATURE, C.TAG_QUADRATURE, C.TAG_AUDIT, C.TAG_OFFLINE_REF, C.TAG_LABEL]})
        # geometry / atlas
        writer.write("geometry/atlas_table", atlas.table, units="mixed (see columns)", tag=C.TAG_MODEL_FEATURE, description="signed-off 7-segment centerline atlas, true mm")
        writer.attrs("geometry", {"atlas_columns": atlas.columns, "atlas_segments": atlas.segments, "semantic_labels": C.SEMANTIC_LABELS,
                                  "semantic_of_segment": atlas.semantic_of_segment, "atlas_provenance": atlas.provenance,
                                  "outlet_relabel": relabel})
        writer.write("geometry/atlas_frame_n", atlas.frame_n.astype(np.float32), units="unit", tag=C.TAG_MODEL_FEATURE, description="rotation-minimising normal per atlas sample")
        writer.write("geometry/atlas_frame_b", atlas.frame_b.astype(np.float32), units="unit", tag=C.TAG_MODEL_FEATURE)
        caps = pc["cloud"]["caps"]
        writer.write("geometry/cap_center_mm", np.array([c["center_mm"] for c in caps]), units="mm", tag=C.TAG_MODEL_FEATURE)
        writer.write("geometry/cap_radius_mm", np.array([c["radius_mm"] for c in caps]), units="mm", tag=C.TAG_MODEL_FEATURE)
        writer.attrs("geometry", {"cap_labels": [c["label"] for c in caps], "wall_point_spacing_mm": pc["cloud"]["spacing_mm"],
                                  "atlas_end_radius_patch": atlas_end_patch, "geometry_program_patches": ["end-radius-hold-2026-09-22"] if atlas_end_patch else []})
        # wall static
        writer.write("wall_static/xyz_mm", wall_xyz_mm, units="mm", tag=C.TAG_MODEL_FEATURE, description="anatomy wall nodes (all, incl. non-exported)")
        writer.write("wall_static/node_id_cas", md.wall_node_ids, units="1-based Fluent node id", tag=C.TAG_AUDIT)
        writer.write("wall_static/source_row", wall.source_row, units="row in reference wall ASCII frame (-1 missing)", tag=C.TAG_AUDIT)
        writer.write("wall_static/valid", wall.valid, units="bool", tag=C.TAG_LABEL, description="label available on all 81 frames")
        writer.write("wall_static/normal_out_mesh", md.wall_node_normals_out.astype(np.float32), units="unit", tag=C.TAG_AUDIT, description="area-weighted outward normal from the CFD mesh (judging only)")
        writer.write("wall_static/normal_out_pca", pc["pca_normals_out"], units="unit", tag=C.TAG_MODEL_FEATURE, description="PCA normal from the bare point cloud, oriented away from the nearest centerline sample")
        writer.write("wall_static/area_m2", md.wall_node_area_m2, units="m^2", tag=C.TAG_QUADRATURE, description="equal share of incident anatomy wall face areas")
        writer.write("wall_static/rim_mask", md.wall_node_rim, units="bool", tag=C.TAG_QUADRATURE, description="node shared with an interface cut")
        for name in POINT_FEATURES:
            writer.write(f"wall_static/{name}", wall_feats[name], units="mm|rad|id|bool", tag=C.TAG_MODEL_FEATURE if name != "atlas_row" else C.TAG_AUDIT)
        # wall temporal
        writer.write("wall_temporal/step", np.asarray(C.EXPECTED_STEPS, dtype=np.int32), units="solver step", tag=C.TAG_AUDIT)
        writer.write("wall_temporal/time_s", wave["time_s"], units="s", tag=C.TAG_MODEL_FEATURE)
        writer.write("wall_temporal/phase_s", wave["phase_s"], units="s within cycle 8", tag=C.TAG_MODEL_FEATURE)
        writer.write("wall_temporal/wss_scalar_pa", wall.wss_scalar, units="Pa", tag=C.TAG_LABEL, description="Fluent node 'wall-shear' scalar (NaN where invalid)")
        writer.write("wall_temporal/wss_vector_pa", wall.wss_vector, units="Pa", tag=C.TAG_LABEL, description="Fluent node x/y/z-wall-shear")
        writer.write("wall_temporal/pressure_pa", wall.pressure, units="Pa", tag=C.TAG_AUDIT, description="wall export pressure (use volume pressure for labels)")
        # volume static
        writer.write("volume_static/xyz_mm", vol_xyz_mm, units="mm", tag=C.TAG_MODEL_FEATURE, description="anatomy cell centres in export order")
        writer.write("volume_static/cell_id_cas", vol.cell_id_cas, units="1-based Fluent cell id", tag=C.TAG_AUDIT)
        writer.write("volume_static/source_row", vol.source_row, units="row in reference volume ASCII frame", tag=C.TAG_AUDIT)
        writer.write("volume_static/volume_m3", volumes, units="m^3", tag=C.TAG_QUADRATURE)
        writer.write("volume_static/dist_to_wall_mm", (dist_m * MM).astype(np.float32), units="mm", tag=C.TAG_MODEL_FEATURE, description="exact distance to anatomy wall triangles")
        writer.write("volume_static/nearest_wall_triangle", tri.astype(np.int32), units="index into topology/wall_triangles", tag=C.TAG_AUDIT)
        for name in POINT_FEATURES:
            writer.write(f"volume_static/{name}", vol_feats[name], units="mm|rad|id|bool", tag=C.TAG_MODEL_FEATURE if name != "atlas_row" else C.TAG_AUDIT)
        # volume temporal
        writer.write("volume_temporal/velocity_m_s", vol.velocity, units="m/s", tag=C.TAG_LABEL)
        writer.write("volume_temporal/pressure_pa", vol.pressure, units="Pa", tag=C.TAG_LABEL, description="raw static gauge pressure; references in pressure_reference/")
        # interfaces
        for label, tab in interfaces.items():
            g = f"interfaces/{label}"
            writer.attrs(g, {"extension_zone": tab["extension_zone"], "area_m2": tab["area_m2"], "planarity": tab["planarity"],
                             "center_mm": tab["center_mm"].tolist(), "normal_out": np.asarray(tab["normal_out"]).tolist()})
            writer.write(f"{g}/face_center_mm", tab["face_center_mm"], units="mm", tag=C.TAG_QUADRATURE)
            writer.write(f"{g}/face_area_vec_m2", tab["face_area_vec_m2"], units="m^2 (outward)", tag=C.TAG_QUADRATURE)
            writer.write(f"{g}/adjacent_volume_row", tab["adjacent_volume_row"], units="row", tag=C.TAG_QUADRATURE)
            writer.write(f"{g}/rim_wall_rows", tab["rim_wall_rows"], units="row", tag=C.TAG_QUADRATURE)
            writer.write(f"{g}/p_mean_pa", tab["p_mean_pa"], units="Pa", tag=C.TAG_QUADRATURE, description="area-weighted adjacent-cell pressure per frame")
            writer.write(f"{g}/flux_outward_m3s", tab["flux_outward_m3s"], units="m^3/s", tag=C.TAG_OFFLINE_REF, description="first-order flux from adjacent cell-centre velocity")
        # pressure references
        writer.write("pressure_reference/p_volume_mean_pa", p_vol_mean, units="Pa", tag=C.TAG_QUADRATURE, description="volume-weighted anatomy mean per frame")
        writer.write("pressure_reference/p_case_cycle_mean_pa", np.array(p_cycle), units="Pa", tag=C.TAG_QUADRATURE, description="trapezoid time mean of p_volume_mean over the 81 frames")
        if "inlet" in interfaces:
            writer.write("pressure_reference/p_inlet_mean_pa", interfaces["inlet"]["p_mean_pa"], units="Pa", tag=C.TAG_QUADRATURE)
        # conditions (never model input)
        writer.attrs("conditions", {k: v for k, v in cond.items()})
        writer.attrs("conditions", {"solver": src.solver, "wall_audit": src.wall_audit, "split": {"role": src.role, "validation_fold": src.validation_fold, "patient_group": src.patient_group}})
        writer.write("conditions/q_nom_m3s", wave["q_nom_m3s"], units="m^3/s", tag=C.TAG_OFFLINE_REF, description="public inlet waveform at the 81 steps")
        writer.write("conditions/dq_nom_dt_m3s2", wave["dq_nom_dt_m3s2"], units="m^3/s^2", tag=C.TAG_OFFLINE_REF)
        # topology sidecar
        tris, face_of_tri = wall_triangles(md)
        writer.write("topology/wall_face_offsets", md.wall["face_offsets"], units="index", tag=C.TAG_QUADRATURE)
        writer.write("topology/wall_face_nodes", md.node_row(md.wall["face_nodes"]), units="wall row", tag=C.TAG_QUADRATURE)
        writer.write("topology/wall_face_area_vec_m2", md.wall["area_vectors_m2"], units="m^2 (outward)", tag=C.TAG_QUADRATURE)
        row_of_cell = -np.ones(md.mesh.cell_count + 1, dtype=np.int64); row_of_cell[vol.cell_id_cas] = np.arange(len(vol.cell_id_cas))
        writer.write("topology/wall_face_adjacent_volume_row", row_of_cell[md.wall["adjacent_cells"]], units="row", tag=C.TAG_QUADRATURE)
        writer.write("topology/wall_triangles", tris.astype(np.int32), units="wall row", tag=C.TAG_QUADRATURE)
        writer.write("topology/wall_triangle_face", face_of_tri.astype(np.int32), units="face index", tag=C.TAG_AUDIT)
        # deployment-side sample outputs
        writer.write("query_geometry/generated_internal_sample_mm", pc["generated_sample_mm"], units="mm", tag=C.TAG_MODEL_FEATURE,
                     description="internal query points generated from the bare wall point cloud + atlas (hybrid inside test)")
        writer.attrs("query_geometry", {"recipe": {"pca_k": C.PCA_NORMAL_K, "vote_k": C.INSIDE_VOTE_K, "tube_margin": C.TUBE_MARGIN, "seed": seed,
                                                   "geometry_program_version": C.GEOMETRY_PROGRAM_VERSION},
                                        "generated_queries": pc["generated_queries"], "inside_test": pc["inside_test"]})
        h5_final = writer.close()
        writer = None
        timings["write_h5"] = time.time() - t
        datasets = {}
        import h5py
        with h5py.File(h5_final, "r") as h5:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    datasets[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype), "units": obj.attrs.get("units", ""), "dependency": obj.attrs.get("dependency", "")}
            h5.visititems(visit)
        report["bundle"] = {"path": str(h5_final), "bytes": h5_final.stat().st_size, "datasets": len(datasets),
                            "dataset_digest_sha256": aggregate_digest({k: {"sha256": v} for k, v in _digests_from(h5_final).items()})}
        report["datasets"] = datasets
        report["status"] = "built"
    except Exception as exc:  # noqa: BLE001
        if writer is not None:
            writer.abort()
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report.setdefault("gates", {"pass": False, "checks": {}, "waivers": {}})
    report["timings_s"] = {**timings, "total": time.time() - started}
    report["code"] = git_state()
    payload = _json_ready(report)
    (case_out / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    if report["status"] == "built":
        manifest = {k: payload[k] for k in ("schema_version", "geometry_program_version", "canonical_id", "cohort", "role", "validation_fold", "patient_group", "created_at", "sources", "bundle", "datasets", "gates", "code")}
        manifest["counts"] = payload["metrics"]["counts"]
        manifest["conditions_summary"] = payload["metrics"]["conditions"]
        (case_out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def _digests_from(h5_path: Path) -> dict[str, str]:
    """Dataset digests recomputed from the written file (binds the manifest to the bytes on disk)."""
    import h5py
    from .store import array_digest
    out = {}
    with h5py.File(h5_path, "r") as h5:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = array_digest(obj[()])
        h5.visititems(visit)
    return out
