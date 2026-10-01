"""Truth-free PF6/VF6 deployment, portable volume exports and report."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import analysis as A, centerline as CL, geometry as G, metrics as M, morphology as MORPH, narrative as NARR, report as R
from .io_utils import atomic_json, file_sha256, resolve_job_path
from .paths import BRANCH_CN
from .schema import (build_results, field_descriptor, model_release_metadata,
                     single_frame_time_axis, stable_run_identity, summary_provenance, write_run_manifest)


def _statistics(values):
    values = np.asarray(values)
    return {"count": int(values.size), "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "weighting": "sample_points"}


def _export(job_dir, points, pressure, velocity, wall, wall_pressure, vertices, faces, vertex_pressure, geom, lines):
    import pyvista as pv
    from .pipeline import atomic_output
    # v0.14: each file is written to a hidden temporary name and renamed into place (atomic).
    speed = np.linalg.norm(velocity, axis=1)
    interior = pv.PolyData(np.asarray(points, np.float32))
    interior.point_data.update({"pressure_pa": pressure, "velocity_m_s": velocity,
                                "speed_m_s": speed, "segment_id": geom["segment_id"][len(wall):]})
    with atomic_output(job_dir / "volume_fields.vtp") as tmp:
        interior.save(str(tmp))
    with atomic_output(job_dir / "wall_pressure.vtp") as tmp:
        R.write_vtp(tmp, vertices, faces,
                    {"pressure_pa": vertex_pressure, "pressure_covered": np.isfinite(vertex_pressure)})
    with atomic_output(job_dir / "points_volume.csv") as tmp:
        np.savetxt(tmp, np.column_stack((points, pressure, velocity, speed)),
                   delimiter=",", fmt="%.7g", comments="",
                   header="x_mm,y_mm,z_mm,pressure_pa,vx_m_s,vy_m_s,vz_m_s,speed_m_s")
    if lines:
        xyz = np.concatenate([line["points"] for line in lines])
        cells = []; offset = 0
        for line in lines:
            size = len(line["points"])
            cells.extend([size, *range(offset, offset+size)]); offset += size
        traces = pv.PolyData(xyz, lines=np.asarray(cells))
        traces.point_data["speed_m_s"] = np.concatenate([line["speed_m_s"] for line in lines])
        with atomic_output(job_dir / "streamlines.vtp") as tmp:
            traces.save(str(tmp))


def stage_b_volume(job_dir, mapping, release, *, smooth_mm=1., spacing_mm=.5,
                   confirmed=False, case_id=None, device="auto", seed_count=None, threads=None,
                   progress=None, cancelled=None):
    from .pipeline import (_archive_previous_run, _clean_mesh, _now, _resampled, VertexInterpolation, cache_record,
                           inference_threads, prune_geometry_cache, run_inference, save_npz_atomic)
    from . import geometry_cache as GC
    from .volume_geometry import make_inside_test
    from .volume_cache import build_volume_case_cached
    from .volume_report import build_html
    from .streamlines import ball_certified_inside, centerline_seeds, integrate_streamlines, thin_lines, volume_seeds
    from wss_features import contract as feature_contract
    job_dir = Path(job_dir).resolve()
    a = json.loads((job_dir / "stage_a.json").read_text())
    if a.get("stage") != "A" or not a.get("input_check", {}).get("ok"):
        raise ValueError("输入检查尚未通过，不能开始预测。")
    if not confirmed: raise ValueError("必须确认出口命名后才能开始预测。")
    clean_stl = resolve_job_path(job_dir, a["input_check"]["clean_stl"])
    progress = progress or (lambda *_: None)
    cancelled = cancelled or (lambda: False)

    def check():
        if cancelled(): raise InterruptedError("任务已取消")

    check()
    previous_archive = _archive_previous_run(job_dir)
    timing = dict(a["timing_s"])
    atlas = CL.apply_mapping(CL.load_vessel_geom_atlas(job_dir / "centerline"), mapping)
    # v0.14: clean mesh, resampled wall and morphology sections come from <job_dir>/geometry_cache when
    # their exact-input keys match (pipeline.precompute_geometry_cache); otherwise computed here.
    cache, cache_lock = GC.GeometryCache(job_dir), GC.job_lock(job_dir)
    with cache_lock:
        vertices, faces = _clean_mesh(job_dir, clean_stl, cache)
    vertices = np.asarray(vertices, np.float64); faces = np.asarray(faces, np.int64)
    sampling_seed = G.stable_sampling_seed(a["input_sha256"])
    progress("geometry", "正在构建壁面支持点和封闭管腔")
    start = time.perf_counter()
    with cache_lock:
        smoothed, wall = _resampled(vertices, faces, smooth_mm=smooth_mm, spacing_mm=spacing_mm,
                                    seed=sampling_seed, cache=cache)
    timing["smooth_resample"] = time.perf_counter()-start
    check()
    progress("features", "正在生成管腔内部采样点和 PF6 / VF6 特征")
    start = time.perf_counter()
    with cache_lock:
        case, aux = build_volume_case_cached(wall, smoothed, faces, atlas, release.input_features,
                                             cache=cache, mapping=mapping,
                                             case_name="input-" + a["input_sha256"][:24],
                                             n_internal=20000, seed=sampling_seed)
    timing["volume_features"] = time.perf_counter()-start
    check()
    progress("inference", "正在分别预测相对压力与体内速度")
    # v0.14: CPU inference without an explicit thread count uses min(32, cores) (WSS_DEPLOY_CPU_THREADS).
    cpu_threads = inference_threads(getattr(release, "device", None), threads)
    start = time.perf_counter(); prediction = run_inference(release, case, threads=cpu_threads)
    timing["inference_volume"] = time.perf_counter()-start
    check()
    n_wall = int(case["n_wall"])
    all_points = case["wall_coords_raw"]
    points = all_points[n_wall:]; wall = all_points[:n_wall]
    pressure = np.asarray(prediction["pressure_pa"], np.float32)
    velocity = np.asarray(prediction["velocity_m_s"], np.float32)
    if pressure.shape != (len(all_points),) or velocity.shape != points.shape:
        raise ValueError("体场输出与壁面／内部查询点不匹配。")
    if not np.isfinite(pressure).all() or not np.isfinite(velocity).all():
        raise ValueError("体场预测包含非有限值，已停止导出。")
    speed = np.linalg.norm(velocity, axis=1)
    geom = aux["geom"]
    stats = {"pressure_interior_pa": _statistics(pressure[n_wall:]),
             "pressure_wall_pa": _statistics(pressure[:n_wall]), "speed_m_s": _statistics(speed)}
    progress("metrics", "正在积分体内稳态流线并汇总压力与速度")
    start = time.perf_counter()
    closed = aux["closed_surface"]
    inside = ball_certified_inside(make_inside_test(closed["vertices"], closed["faces"]), points, closed["vertices"], closed["faces"])
    seeds = np.concatenate([centerline_seeds(atlas), volume_seeds(points, n=120, seed=sampling_seed)])
    lines, line_info = integrate_streamlines(points, velocity, seeds, inside)
    lines, line_info["vertex_stride"] = thin_lines(lines)
    interp = VertexInterpolation(wall, vertices)   # one KD-tree for the pressure and the segment labels
    vertex_pressure = interp.values(pressure[:n_wall])
    timing["streamlines_and_interpolation"] = time.perf_counter()-start
    check()
    semantic_labels = ("root", "left_cia", "right_cia", "out-le", "out-li", "out-re", "out-ri")
    branch_names = {str(sid): BRANCH_CN[semantic_labels[int(label)]] for sid, label in atlas.semantic_of_segment.items()}
    # Analysis layer (contract §1-§3) on the interior predictions; nothing here changes a prediction.
    interior_feats = {key: np.asarray(geom[key])[n_wall:] for key in ("segment_id", "s_local_mm", "s_from_root_mm", "radius_mm")}
    branch_geometry = M.geometry_table(atlas, {int(k): v for k, v in branch_names.items()})
    profiles_block = A.profiles(atlas, interior_feats, {"speed": speed, "pressure": pressure[n_wall:]}, branch_names=branch_names)
    findings_block = A.findings_volume(points, speed, pressure[n_wall:], interior_feats, atlas, branch_geometry, branch_names=branch_names)
    # Lumen morphology (contract §17.1) on the same clean wall mesh the report embeds.
    progress("metrics", "正在测量沿程管腔截面与瘤体形态")
    start = time.perf_counter()
    with cache_lock:
        morphology_block = MORPH.compute(vertices, faces, atlas, branch_names=branch_names, findings=findings_block,
                                         interior={"segment_id": interior_feats["segment_id"], "speed": speed},
                                         section_cache=cache)
    timing["morphology"] = time.perf_counter()-start
    A.apply_morphology(findings_block, morphology_block, branch_names)
    vertex_segment = interp.labels(np.asarray(geom["segment_id"])[:n_wall])
    caps = list((aux["diag"].get("closure") or {}).get("caps") or [])
    dist_to_wall = np.asarray(case["dist_to_wall_mm"], np.float32)[n_wall:]
    segments = atlas.col("segment_id").astype(int); index = atlas.col("sample_index")
    edges = []
    for sid in np.unique(segments):
        rows = np.flatnonzero(segments == sid); rows = rows[np.argsort(index[rows])]
        edges.extend(zip(rows[:-1], rows[1:]))
    fields = {
        "pressure": {**field_descriptor("pressure", label="相对压力", units="Pa", location="wall_and_interior",
                                        array_key="pressure_pa", statistics_key="pressure_interior_pa"),
                     "reference": "volume_mean_relative", "coordinates": "pts", "point_kind_key": "point_kind"},
        "velocity": {**field_descriptor("velocity", label="体内速度", units="m/s", location="interior",
                                        kind="vector", components=3, array_key="velocity_m_s",
                                        axis_order=("point", "component"), statistics_key="speed_m_s"),
                     "frame": "world", "coordinates": "internal_pts", "component_labels": ["vx", "vy", "vz"]},
    }
    model_frame = {"target": "peak_systole", "step": 1162, "time_s": .21, "label": "peak_systole"}
    axis = single_frame_time_axis(model_frame)
    record = {}
    if (job_dir / "job.json").is_file(): record = json.loads((job_dir / "job.json").read_text())
    mapping_history = [{k: row[k] for k in ("at", "source", "confidence", "inlet", "suggested", "confirmed", "acknowledged", "confidence_gate") if k in row}
                       for row in record.get("mapping_history", []) if isinstance(row, dict)]
    case_metadata = {key: record.get(key, [] if key == "tags" else "")
                     for key in ("patient_id", "scan_label", "scan_date", "tags", "notes")}
    counts = {"pressure": len(prediction["seed_pressure_pa"]), "velocity": len(prediction["seed_velocity_m_s"])}
    parameters = {"smooth_mm": smooth_mm, "spacing_mm": spacing_mm, "internal_points": len(points),
                  "device": prediction["device"], "seed_count": counts["pressure"], "seeds_per_field": counts,
                  "threads": threads, "internal_sampling": aux["diag"]["geometry_version"],
                  "resolved_units": a["input_check"].get("resolved_units") or a["input_check"].get("units"),
                  "inlet_override": a.get("inlet_override"),
                  "clean_stl_sha256": file_sha256(clean_stl)}
    release_meta = model_release_metadata(release)
    meta = {"schema_version": "wss-deploy.summary/v1", "case_id": case_id or job_dir.name,
            "created_at": _now(), "release": release.name, "model_release": release_meta,
            "device": prediction["device"], "gpu": prediction.get("gpu"), "case_metadata": case_metadata,
            "input_sha256": a["input_sha256"], "input_check": a["input_check"], "centerline": a["centerline"],
            "outlets_confirmed": True, "mapping": mapping, "flags": a["proposal"].get("flags", []),
            "branch_names": branch_names,
            # Per-branch centreline geometry; kept under its own key because the summary key
            # ``geometry`` is reserved for the wall family in the golden-regression comparison.
            "branch_geometry": branch_geometry, "profiles": profiles_block, "findings": findings_block,
            "morphology": morphology_block,
            "proposal_confidence": a["proposal"].get("confidence"),
            "proposal_side_confidence": a["proposal"].get("side_confidence", {}),
            "proposal_confirmation_required": a["proposal"].get("confirmation_required"),
            "proposal_confidence_gate": a["proposal"].get("confidence_gate", {}),
            "cloud": {"n_wall": n_wall, "n_internal": len(points), "spacing_mm": spacing_mm},
            "volume_geometry": aux["diag"], "volume_statistics": stats, "streamlines": line_info,
            "pressure_reference": "相对于该病例当前帧的体积平均压力；不可解释为绝对血压。",
            "fields": fields, "time_axis": axis, "model_frame": model_frame,
            "run_parameters": parameters, "sampling_seed": sampling_seed,
            "frame_transform": A.frame_transform(case["frame_rotation"], geom["frame_origin_mm"],
                                                 source="vessel_geom atlas + anatomical_frame",
                                                 direction_source=a["input_check"].get("orientation_source"),
                                                 vector_to_world="v_aligned @ rotation"),
            "release_hash": file_sha256(Path(release.dir) / "MANIFEST.sha256"),
            "feature_contract": feature_contract(),
            "seconds_per_model": prediction["seconds_per_model"],
            **({"inference_threads": cpu_threads} if cpu_threads is not None else {}),
            "interpolation": {"field": "wall_pressure", "source": "wall_query_predictions", "method": "Gaussian",
                              "sigma_mm": .5, "max_dist_mm": 1.5, "covered_vertices": int(np.isfinite(vertex_pressure).sum())},
            "audit": {"mapping_history": mapping_history, "previous_run_archive": previous_archive},
            "notes": ["PF6 压力相对于当前帧体积平均压力；不能恢复绝对血压。",
                      "VF6 仅查询体内速度，导出向量位于 STL 世界坐标。",
                      "流线为固定收缩期向量场的稳态积分；截面为有限厚度内部点云切片。",
                      "统计按查询点等权；不代表体积积分或流量。"]}
    meta["run_identity"] = stable_run_identity(input_sha256=a["input_sha256"], release=release_meta,
                                               mapping=mapping, parameters=parameters)
    from .reference import evaluate as evaluate_reference
    # The geometry reference check reads ``geometry``; evaluate on a shallow view so the volume
    # summary keeps its own ``branch_geometry`` key.
    meta["reference_assessment"] = evaluate_reference({**meta, "geometry": branch_geometry}, getattr(release, "info", {}))
    mesh_trust, cloud_trust, meta["trust"] = A.trust_volume(vertices, vertex_pressure, vertex_segment, points,
                                                            interior_feats["segment_id"], caps,
                                                            meta["reference_assessment"], branch_names)
    from .schema import summary_display_name
    meta["display_name"] = summary_display_name(meta)          # C7 (2026-09-30)
    meta["narrative"] = NARR.build_narrative(meta)
    meta["results"] = build_results(time_axis=axis, fields=fields, statistics=stats, compatibility={})
    meta["geometry_cache"] = cache_record(cache)
    meta["exports"] = {"vtp": True, "csv": True, "html": True, "internal_field": True,
                       "wall_pressure": True, "streamlines": bool(lines), "run_manifest": True}
    meta["run_manifest"] = {"path": "run_manifest.json", "schema_version": "wss-deploy.run-manifest/v1"}
    progress("export", "正在导出体场、壁面压力、流线和交互截面报告")
    start = time.perf_counter()
    save_npz_atomic(job_dir / "field.npz", pts=all_points.astype(np.float32), internal_pts=points.astype(np.float32),
                    pressure_pa=pressure, velocity_m_s=velocity, speed_m_s=speed,
                    point_kind=np.r_[np.zeros(n_wall, np.uint8), np.ones(len(points), np.uint8)],
                    segment_id=geom["segment_id"], vertices=vertices.astype(np.float32), faces=faces.astype(np.int32),
                    vertex_pressure_pa=vertex_pressure, seed_pressure_pa=prediction["seed_pressure_pa"],
                    seed_velocity_m_s=prediction["seed_velocity_m_s"])
    _export(job_dir, points, pressure[n_wall:], velocity, wall, pressure[:n_wall], vertices, faces,
            vertex_pressure, geom, lines)
    from .pipeline import atomic_output, report_meta
    with atomic_output(job_dir / "report.html") as report_tmp:
        build_html(report_tmp, report_meta(meta),
                   mesh={"vertices": vertices, "faces": faces, "pressure_pa": vertex_pressure, "trust": mesh_trust},
                   cloud={"pts": points, "pressure_pa": pressure[n_wall:], "velocity_m_s": velocity,
                          "segment": geom["segment_id"][n_wall:], "trust": cloud_trust,
                          "s_from_root_mm": interior_feats["s_from_root_mm"].astype(np.float32),
                          "radius_mm": interior_feats["radius_mm"].astype(np.float32), "dist_to_wall_mm": dist_to_wall},
                   centerline={"xyz": atlas.xyz, "tangent": atlas.tangent, "radius_mm": atlas.col("radius_mm"),
                               "edges": np.asarray(edges, np.uint32), "segment": segments}, streamlines=lines)
        timing["export"] = time.perf_counter()-start
        timing["total"] = sum(value for key, value in timing.items() if key != "total")
        meta["timing_s"] = {k: round(v, 2) for k, v in timing.items()}
        # v0.14 provenance (analysis_version, deploy_version, git_describe, git_dirty, code_source_hash).
        meta.update(summary_provenance())
        R.update_html_meta(report_tmp, report_meta(meta))
    atomic_json(job_dir / "summary.json", meta)
    names = ["summary.json", "report.html", "field.npz", "stage_a.json", "volume_fields.vtp", "wall_pressure.vtp", "points_volume.csv"]
    if lines: names.append("streamlines.vtp")
    write_run_manifest(job_dir, meta, outputs=names)
    prune_geometry_cache(cache)
    return meta


def precompute_volume_case(job_dir, job, *, mapping, cancel_event=None) -> dict:
    """Fill ``<job_dir>/geometry_cache`` with the PF6/VF6 geometry of the *proposed* outlet mapping (v0.16.1).

    Runs in the background precompute thread after ``pipeline.precompute_geometry_cache`` while the outlets of a
    job with a volume companion are being confirmed.  The companion's stage B builds the same case from the same
    clean mesh, resampled wall, atlas and sampling seed; when the confirmed mapping equals the proposal (whatever
    its order) the exact-input key matches and stage B reuses the entry, otherwise it misses and computes as
    before — the result is the same either way.  The input features are the PF6/VF6 contract
    (``families.VOLUME_FEATURES``, checked for every volume release), so no release is loaded.  The job lock is
    held only for cache I/O, never during the computation.  Never raises: returns ``{"ok", ...}``.
    """
    from .pipeline import DEFAULT_SMOOTH_MM, DEFAULT_SPACING_MM, _clean_mesh, _resampled, _stage_a_record
    from . import geometry_cache as GC, volume_cache as VC
    from .families import VOLUME_FEATURES
    from .volume_geometry import build_volume_case
    started = time.perf_counter()
    cancelled = lambda: bool(cancel_event is not None and cancel_event.is_set())
    out = {"ok": False, "seconds": 0.0}
    if not GC.enabled():
        return {**out, "skipped": "disabled"}
    job_dir = Path(job_dir).resolve()
    cache, lock = GC.GeometryCache(job_dir), GC.job_lock(job_dir)
    try:
        a = _stage_a_record(job_dir, job)
        if a.get("stage") != "A" or not (a.get("input_check") or {}).get("ok"):
            return {**out, "skipped": "stage A not passed"}
        params = (job or {}).get("params") or {}
        smooth_mm = float(params.get("smooth_mm", DEFAULT_SMOOTH_MM))
        spacing_mm = float(params.get("spacing_mm", DEFAULT_SPACING_MM))
        seed = G.stable_sampling_seed(a["input_sha256"])
        clean_stl = resolve_job_path(job_dir, a["input_check"]["clean_stl"])
        with lock:
            vertices, faces = _clean_mesh(job_dir, clean_stl, cache)
        vertices, faces = np.asarray(vertices, np.float64), np.asarray(faces, np.int64)
        with lock:
            smoothed, wall = _resampled(vertices, faces, smooth_mm=smooth_mm, spacing_mm=spacing_mm, seed=seed, cache=cache)
        if cancelled():
            return {**out, "cancelled": True}
        atlas = CL.apply_mapping(CL.load_vessel_geom_atlas(job_dir / "centerline"), mapping)
        kwargs = dict(target="pressure_velocity", case_name="input-" + a["input_sha256"][:24], n_internal=20000, seed=seed)
        key = VC.volume_case_key(wall, smoothed, faces, atlas, VOLUME_FEATURES, mapping=mapping, **kwargs)
        with lock:
            hit = VC.load_volume_case(cache, key)
        if hit is None:
            if cancelled():
                return {**out, "cancelled": True}
            result = build_volume_case(wall, smoothed, faces, atlas, VOLUME_FEATURES, **kwargs)
            with lock:
                VC.store_volume_case(cache, key, result)
        return {**out, "ok": True, "cancelled": False, "reused": hit is not None,
                "seconds": round(time.perf_counter() - started, 2), "cache": cache.summary()}
    except Exception as exc:  # noqa: BLE001 — a failed precompute only means the companion computes it itself
        return {**out, "error": f"{type(exc).__name__}: {exc}"[:500], "seconds": round(time.perf_counter() - started, 2)}
