"""Orchestration: stage A (ingest + centreline + naming proposal, CPU) -> [human confirms outlets] -> stage B (geometry + inference + report)."""
from __future__ import annotations
import hashlib, json, time, datetime, shutil
from pathlib import Path
import numpy as np
from wss_features import contract as feature_contract
from wss_features.atlas import Atlas
from wss_features.stl import load_stl
from . import centerline as CL, geometry as G, metrics as M, report as R
from .ingest import ingest
from .infer import Release
from .paths import OUTLET_CN
from .io_utils import file_sha256, portable_job_path, resolve_job_path
from .schema import (build_results, field_descriptor, model_release_metadata,
                     single_frame_time_axis, write_run_manifest, wss_compatibility)
from .schema import stable_run_identity
from .quality import ensemble_quality
from . import analysis as A, morphology as MORPH, narrative as NARR


def _preview_surface(path: Path, max_faces: int = 18000) -> dict:
    """Small display mesh; model inference always uses the full cleaned STL."""
    vertices, faces = load_stl(path)
    vertices, faces = np.asarray(vertices, np.float32), np.asarray(faces, np.int64)
    if len(faces) > max_faces:
        faces = faces[np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)]
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return {"vertices": vertices[used].round(3).tolist(), "faces": inverse.reshape(-1, 3).tolist(),
            "display_faces": int(len(faces)), "display_only": True}


def report_meta(meta: dict) -> dict:
    """Summary trimmed for embedding in the report page.

    ``summary.json`` keeps the full morphology stations (positions, closure flags, areas); the page
    only draws the diameter curves and the largest-station ring, so the station tables are reduced
    to the three series contract §17.3 asks for.  Nothing else is changed.
    """
    morphology = meta.get("morphology")
    if not isinstance(morphology, dict):
        return meta
    return {**meta, "morphology": MORPH.trim_for_report(morphology)}


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _archive_previous_run(job_dir: Path) -> str | None:
    """Preserve a completed result before a retry/manual override rewrites it."""
    summary = Path(job_dir) / "summary.json"
    if not summary.is_file():
        return None
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + f"_{time.time_ns() % 1000000:06d}"
    digest = hashlib.sha256(summary.read_bytes()).hexdigest()[:10]
    archive = Path(job_dir) / "history" / f"{stamp}_{digest}"
    archive.mkdir(parents=True, exist_ok=False)
    for name in ("summary.json", "run_manifest.json", "report.html", "wall_wss.vtp",
                 "points_wss.csv", "field.npz", "quality_audit.json", "volume_fields.vtp",
                 "points_volume.csv", "wall_pressure.vtp", "streamlines.vtp"):
        source = Path(job_dir) / name
        if source.is_file():
            shutil.copy2(source, archive / name)
    return archive.relative_to(job_dir).as_posix()


def stage_a(stl_path: Path, job_dir: Path, *, inlet: int | None = None, smooth_iterations: int = 0,
            units: str = "mm", remove_fragments: bool = False, progress=None, cancelled=None) -> dict:
    job_dir = Path(job_dir).resolve(); job_dir.mkdir(parents=True, exist_ok=True); T = {}
    progress = progress or (lambda phase, detail: None)
    cancelled = cancelled or (lambda: False)
    if cancelled(): raise InterruptedError("任务已取消")
    progress("ingest", "正在读取并检查 STL")
    t = time.perf_counter(); ic = ingest(Path(stl_path), job_dir, units=units, remove_fragments=remove_fragments); T["ingest"] = time.perf_counter() - t
    if ic["status"] != "pass":
        out = {"stage": "input", "created_at": _now(), "stl": portable_job_path(job_dir, stl_path), "input_sha256": file_sha256(Path(stl_path)), "input_check": ic,
               "centerline": None, "proposal": None, "inlet_override": inlet, "timing_s": {k: round(v, 2) for k, v in T.items()}}
        (job_dir / "stage_a.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        return out
    if cancelled(): raise InterruptedError("任务已取消")
    progress("centerline", "正在提取中心线和开口")
    cl_dir = job_dir / "centerline"
    clean_stl = resolve_job_path(job_dir, ic["clean_stl"])
    t = time.perf_counter(); cl = CL.run_vessel_geom(clean_stl, cl_dir, inlet=inlet, smooth_iterations=smooth_iterations, cancelled=cancelled); T["centerline"] = time.perf_counter() - t
    if cancelled(): raise InterruptedError("任务已取消")
    atlas = CL.load_vessel_geom_atlas(cl_dir)
    proposal = CL.propose_outlets(atlas, orientation_source=ic.get("orientation_source", "unknown_stl"))
    proposal["direction_note"] = ic.get("direction_note", "")
    out = {"stage": "A", "created_at": _now(), "stl": portable_job_path(job_dir, stl_path), "input_sha256": file_sha256(Path(stl_path)), "input_check": ic,
           "centerline": cl, "proposal": proposal, "preview": _preview_surface(clean_stl),
           "inlet_override": inlet, "timing_s": {k: round(v, 2) for k, v in T.items()}}
    (job_dir / "stage_a.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def stage_b(job_dir: Path, mapping: dict[str, str], release: Release, *, smooth_mm: float = 1.0, spacing_mm: float = 0.5,
            confirmed: bool = False, case_id: str | None = None, device: str = "auto",
            seed_count: int | None = None, threads: int | None = None, progress=None, cancelled=None) -> dict:
    """Stage B for any release: the model family decides how inputs, prediction and exports are built."""
    from .families import family_for_release
    family = family_for_release(release)
    return family.stage_b(job_dir, mapping, release, smooth_mm=smooth_mm, spacing_mm=spacing_mm,
                          confirmed=confirmed, case_id=case_id, device=device, seed_count=seed_count,
                          threads=threads, progress=progress, cancelled=cancelled)


def stage_b_wall(job_dir: Path, mapping: dict[str, str], release: Release, *, smooth_mm: float = 1.0, spacing_mm: float = 0.5,
                 confirmed: bool = False, case_id: str | None = None, device: str = "auto",
                 seed_count: int | None = None, threads: int | None = None, progress=None, cancelled=None) -> dict:
    """Wall WSS family: smooth + resample -> 27-D features -> ensemble -> metrics -> exports."""
    job_dir = Path(job_dir).resolve(); previous_archive = _archive_previous_run(job_dir); a = json.loads((job_dir / "stage_a.json").read_text(encoding="utf-8")); T = dict(a["timing_s"])
    progress = progress or (lambda phase, detail: None)
    cancelled = cancelled or (lambda: False)
    if a.get("stage") != "A" or not a.get("input_check", {}).get("ok"):
        raise ValueError("输入检查尚未通过，不能开始预测。")
    if not confirmed:
        raise ValueError("必须确认出口命名后才能开始预测。")
    if cancelled(): raise InterruptedError("任务已取消")
    progress("geometry", "正在平滑表面并重采样")
    atlas = CL.apply_mapping(CL.load_vessel_geom_atlas(job_dir / "centerline"), mapping)
    vertices, faces = load_stl(resolve_job_path(job_dir, a["input_check"]["clean_stl"])); vertices = np.asarray(vertices, np.float64); faces = np.asarray(faces, np.int64)
    t = time.perf_counter(); v_s, pts = G.smooth_and_resample(vertices, faces, smooth_mm=smooth_mm, spacing_mm=spacing_mm, seed=G.stable_sampling_seed(a["input_sha256"])); T["smooth_resample"] = time.perf_counter() - t
    if cancelled(): raise InterruptedError("任务已取消")
    progress("features", "正在构建 27 维几何特征")
    capfit_rule = Path(release.dir) / "rules" / "flow_split_rule_train136.json"
    # Model sampling uses the case unit id.  A task directory is ephemeral and
    # would make identical inputs produce different predictions across reruns.
    stable_case_name = "input-" + str(a["input_sha256"])[:24]
    t = time.perf_counter(); case, aux = G.build_case(pts, atlas, release.input_features,
                                                       case_name=stable_case_name,
                                                       capfit_rule_path=capfit_rule); T["features"] = time.perf_counter() - t
    if cancelled(): raise InterruptedError("任务已取消")
    progress("inference", "正在运行五模型集成")
    old_threads = None
    if threads is not None:
        try:
            import torch
            old_threads = torch.get_num_threads()
            torch.set_num_threads(int(threads))
        except (ImportError, RuntimeError, ValueError):
            old_threads = None
    try:
        t = time.perf_counter(); pred = release.predict(case); T["inference_5_models"] = time.perf_counter() - t
    finally:
        if old_threads is not None:
            try:
                import torch
                torch.set_num_threads(old_threads)
            except (ImportError, RuntimeError, ValueError):
                pass
    wss = pred["wss_pa"]; geom = aux["geom"]; diag = aux["diag"]
    # Extra wall scalar fields (e.g. TAWSS / OSI from a three-head release) ride along with the prediction;
    # everything below treats them generically: descriptor, statistics, npz/csv/vtp arrays and report colouring.
    extra_fields = {str(k): dict(v) for k, v in (pred.get("extra_fields") or {}).items() if isinstance(v, dict) and "values" in v}
    for key, item in extra_fields.items():
        values = np.asarray(item["values"], dtype=np.float64)
        if values.shape != np.shape(wss) or not np.isfinite(values).all():
            raise ValueError(f"额外壁面字段 {key} 必须与预测点一一对应且全部有限。")
        item["values"] = values
    quality = ensemble_quality(wss, pred["seed_sd_pa"], seed_count=len(pred["seed_pred_pa"]))
    progress("metrics", "正在汇总预测点云统计")
    t = time.perf_counter()
    met = M.compute(pts, wss, geom, atlas, diag, a["input_check"]["area_mm2"])
    vw = R.interpolate_to_vertices(pts, wss.astype(np.float32), vertices); vseg = R.nearest_label(pts, geom["segment_id"], vertices)
    # Surface statistics are face based; derive a deterministic majority label
    # from the three interpolated vertex labels for the branch breakdown.
    face_segments = np.asarray(vseg, dtype=np.int64)[faces]
    face_labels = np.where(face_segments[:, 0] == face_segments[:, 1], face_segments[:, 0],
                           np.where(face_segments[:, 0] == face_segments[:, 2], face_segments[:, 0], face_segments[:, 1]))
    met["surface_statistics"] = M.surface_metrics(vertices, faces, vw,
                                                   covered=np.isfinite(vw), face_labels=face_labels)
    # centreline arrays for the viewer
    seg = atlas.col("segment_id").astype(int); idx = atlas.col("sample_index").astype(int); edges = []
    for s in atlas.segments:
        rows = np.flatnonzero(seg == int(s["segment_id"])); rows = rows[np.argsort(idx[rows])]; edges += [(rows[i], rows[i + 1]) for i in range(len(rows) - 1)]
    endpoints = []
    shares = {}
    for c in atlas.endpoints():
        lab = c["label"]; endpoints.append({"kind": "inlet" if lab == "inlet" else "outlet", "name": lab, "name_cn": OUTLET_CN.get(lab, lab), "radius_mm": float(diag["caps"].get(lab, {}).get("radius_mm", c["radius_mm"])), "center_mm": np.asarray(c["center_mm"]).round(2).tolist(), "segment_id": int(c["segment_id"])})
    # Murray shares are keyed by segment id in diag; attach to leaf endpoints
    for e in endpoints:
        if e["kind"] == "outlet":
            sh = diag["murray_shares"].get(str(e["segment_id"]));  e["share"] = sh
    per_branch_s = {}
    for sid in np.unique(geom["segment_id"]):
        per_branch_s[str(int(sid))] = float(np.median(geom["s_from_root_mm"][geom["segment_id"] == sid]))
    # Analysis layer (contract §1-§2): along-branch profiles and the findings list.  Derived from
    # the prediction cloud and the atlas only; nothing here changes a prediction or field.npz.
    thresholds = met["wss_field_pa"]["thresholds_pa"]
    profiles_block = A.profiles(atlas, geom, {"wss": wss}, branch_names=met["branch_names"])
    T["metrics_and_interpolation"] = time.perf_counter() - t
    # Lumen morphology (contract §17.1): cross-sections of the same clean mesh the report embeds.
    progress("metrics", "正在测量沿程管腔截面与瘤体形态")
    t = time.perf_counter()
    morphology_block = MORPH.compute(vertices, faces, atlas, branch_names=met["branch_names"],
                                     per_branch=met["per_branch"], thresholds=thresholds,
                                     cloud={"segment_id": geom["segment_id"], "wss": wss})
    T["morphology"] = time.perf_counter() - t
    t = time.perf_counter()
    findings_block = A.findings_wall(pts, wss, geom, atlas, met["geometry"], thresholds=thresholds,
                                     total_area_mm2=a["input_check"]["area_mm2"], spacing_mm=diag["spacing_mm"],
                                     branch_names=met["branch_names"], morphology=morphology_block)
    T["metrics_and_interpolation"] += time.perf_counter() - t
    # Mapping provenance lives in the service job record.  Copy only the
    # review fields into the portable run record; never copy owner/session
    # credentials or the rest of job.json.
    mapping_history = []
    job_record_path = job_dir / "job.json"
    if job_record_path.is_file():
        try:
            job_record = json.loads(job_record_path.read_text(encoding="utf-8"))
            for item in job_record.get("mapping_history", []):
                if isinstance(item, dict):
                    mapping_history.append({key: item[key] for key in (
                        "at", "source", "confidence", "inlet", "suggested", "confirmed", "acknowledged",
                        "confidence_gate") if key in item})
        except (OSError, ValueError, TypeError):
            mapping_history = []
    model_frame = {"target": "peak_systole", "step": 1162, "time_s": 0.21, "label": "peak_systole"}
    time_axis = single_frame_time_axis(model_frame)
    fields = {"wss": field_descriptor("wss", label="壁面切应力", units="Pa", location="wall",
                                       kind="scalar", array_key="wss_pa", time_indices=(0,),
                                       axis_order=("point",), statistics_key="wss")}
    if extra_fields:
        from .cycle_fields import descriptor as cycle_descriptor
        for key, item in extra_fields.items():
            fields[key] = cycle_descriptor(key, item["values"])
    case_metadata = {}
    job_record_path = job_dir / "job.json"
    if job_record_path.is_file():
        try:
            jr = json.loads(job_record_path.read_text(encoding="utf-8"))
            case_metadata = {key: jr.get(key, [] if key == "tags" else "")
                             for key in ("patient_id", "scan_label", "scan_date", "tags", "notes")}
        except (OSError, ValueError, TypeError):
            case_metadata = {}
    meta = {"schema_version": "wss-deploy.summary/v1", "case_id": case_id or job_dir.name, "release": release.name, "device": pred["device"], "gpu": pred["gpu"], "created_at": _now(), "input_sha256": a["input_sha256"],
            "case_metadata": case_metadata,
            "input_check": a["input_check"], "centerline": a["centerline"], "outlets_confirmed": confirmed, "mapping": mapping,
            "proposal_confidence": a["proposal"].get("confidence"),
            "proposal_side_confidence": a["proposal"].get("side_confidence", {}),
            "proposal_confirmation_required": a["proposal"].get("confirmation_required"),
            "proposal_confidence_gate": a["proposal"].get("confidence_gate", {}),
            "flags": a["proposal"].get("flags", []),
            "cloud": {"n_points": int(len(pts)), "spacing_mm": diag["spacing_mm"], "smooth_mm": smooth_mm, "surface_variation_median": diag["surface_variation_median"], "junction_ambiguous_fraction": diag["junction_ambiguous_fraction"]},
            "caps": diag["caps"], "murray_shares": diag["murray_shares"], "endpoints": endpoints, "per_branch_s": per_branch_s, "timing_s": {k: round(v, 2) for k, v in T.items()},
            "seconds_per_model": [round(x, 2) for x in pred["seconds_per_model"]],
            "sampling_seed": G.stable_sampling_seed(a["input_sha256"]),
            "model_frame": model_frame,
            # Generic result metadata.  The historical WSS keys below remain
            # in the summary for old reports and scripts.
            "model_release": model_release_metadata(release),
            "time_axis": time_axis,
            "fields": fields,
            "run_parameters": {"smooth_mm": smooth_mm, "spacing_mm": spacing_mm, "device": pred["device"], "seed_count": len(pred["seed_pred_pa"]), "threads": threads},
            "frame_transform": A.frame_transform(geom["frame_rotation"], geom["frame_origin_mm"],
                                                 source="vessel_geom atlas + anatomical_frame",
                                                 direction_source=a["input_check"].get("orientation_source")),
            "release_hash": file_sha256(Path(release.dir) / "MANIFEST.sha256") if (Path(release.dir) / "MANIFEST.sha256").is_file() else release.name,
            # Identity of the frozen STL -> feature program; a release may pin it (see infer.Release).
            "feature_contract": feature_contract(),
            "quality": quality["quality"],
            "profiles": profiles_block, "findings": findings_block, "morphology": morphology_block,
            "audit": {"mapping_history": mapping_history, "mapping_history_count": len(mapping_history),
                      "previous_run_archive": previous_archive,
                      "quality_audit": quality["audit"], "quality_audit_path": "quality_audit.json"},
            "interpolation": {"method": "Gaussian", "sigma_mm": 0.5, "max_dist_mm": 1.5, "covered_vertices": int(np.isfinite(vw).sum()), "total_vertices": int(len(vw))}, **met}
    if extra_fields:
        from .cycle_fields import cycle_block
        meta["cycle"] = cycle_block(extra_fields, geom["segment_id"], met["branch_names"], a["input_check"]["area_mm2"])
        if isinstance(pred.get("cycle"), dict):
            meta["cycle"]["definition"] = {**meta["cycle"]["definition"], **pred["cycle"]}
    from .reference import evaluate as evaluate_reference
    release_info = getattr(release, "info", {})
    meta["reference_assessment"] = evaluate_reference(meta, release_info)
    # Trust mask (contract §3) needs the reference assessment for the geometry-out-of-range bit.
    trust_vertices, meta["trust"] = A.trust_wall(vertices, vw, pts, geom["surface_variation"], vseg,
                                                 meta["reference_assessment"], met["branch_names"])
    meta["run_identity"] = stable_run_identity(input_sha256=meta["input_sha256"],
                                                release=meta["model_release"],
                                                mapping=meta["mapping"],
                                                parameters=meta["run_parameters"])
    meta["narrative"] = NARR.build_narrative(meta)
    statistics = {"wss": met["wss_field_pa"], "surface": met["surface_statistics"], "quality": quality["quality"]}
    if extra_fields:
        statistics.update({key: meta["cycle"]["fields"][key] for key in extra_fields if key in meta["cycle"]["fields"]})
        if "stagnation" in meta["cycle"]:
            statistics["stagnation"] = meta["cycle"]["stagnation"]
    meta["results"] = build_results(time_axis=time_axis, fields=fields, statistics=statistics,
                                     compatibility=wss_compatibility(met))
    progress("export", "正在写入 CSV、VTP 和 HTML 报告")
    export_started = time.perf_counter()
    extra_npz, extra_vertex = {}, {}
    for key, item in extra_fields.items():
        ak = fields[key]["array_key"]
        extra_vertex[ak] = R.interpolate_to_vertices(pts, item["values"].astype(np.float32), vertices)
        extra_npz[ak] = item["values"].astype(np.float32); extra_npz[f"vertex_{ak}"] = extra_vertex[ak]
        if item.get("seed_pred") is not None: extra_npz[f"seed_{ak}"] = np.asarray(item["seed_pred"], np.float32)
        if item.get("seed_sd") is not None: extra_npz[f"seed_sd_{ak}"] = np.asarray(item["seed_sd"], np.float32)
    np.savez_compressed(job_dir / "field.npz", pts=pts.astype(np.float32), wss_pa=wss.astype(np.float32), seed_pred_pa=pred["seed_pred_pa"].astype(np.float32), seed_sd_pa=pred["seed_sd_pa"].astype(np.float32), segment_id=geom["segment_id"],
                        s_from_root_mm=geom["s_from_root_mm"], theta_rad=geom["theta_rad"], radius_mm=geom["radius_mm"], vertices=vertices.astype(np.float32), faces=faces.astype(np.int32), vertex_wss_pa=vw, **extra_npz)
    with open(job_dir / "points_wss.csv", "w") as f:
        extra_keys = [fields[key]["array_key"] for key in extra_fields]
        f.write("x_mm,y_mm,z_mm,wss_pa,segment_id,branch,s_from_inlet_mm,theta_rad" + "".join("," + k for k in extra_keys) + "\n")
        bn = met["branch_names"]
        # Same row format as before; plain Python floats format ~10x faster than per-element numpy indexing.
        segment_list = geom["segment_id"].astype(int).tolist()
        rows = zip(pts[:, 0].tolist(), pts[:, 1].tolist(), pts[:, 2].tolist(), wss.tolist(), segment_list,
                   [bn.get(str(s), "") for s in segment_list], geom["s_from_root_mm"].tolist(), geom["theta_rad"].tolist(),
                   *[extra_npz[k].astype(np.float64).tolist() for k in extra_keys])
        f.write("\n".join(f"{x:.3f},{y:.3f},{z:.3f},{w:.4f},{s},{b},{sr:.2f},{th:.4f}" + "".join(f",{e:.4f}" for e in extras)
                          for x, y, z, w, s, b, sr, th, *extras in rows) + "\n")
    vtp_ok = R.write_vtp(job_dir / "wall_wss.vtp", vertices, faces, {"wss_pa": vw, "wss_covered": np.isfinite(vw), "segment_id": vseg.astype(np.int32), **extra_vertex})
    R.build_html(job_dir / "report.html", report_meta(meta),
                 mesh={"vertices": vertices, "faces": faces, "wss": vw, "segment": vseg, "trust": trust_vertices, "extra": extra_vertex},
                 cloud={"pts": pts, "wss": wss, "segment": geom["segment_id"], "s_from_root_mm": geom["s_from_root_mm"], "theta_rad": geom["theta_rad"], "radius_mm": geom["radius_mm"], "dist_to_junction_mm": geom["dist_to_junction_mm"],
                        "extra": {ak: extra_npz[ak] for ak in extra_vertex}},
                 centerline={"xyz": atlas.xyz, "radius_mm": atlas.col("radius_mm"), "edges": np.asarray(edges, dtype=np.uint32), "segment": seg.astype(np.int16)})
    T["export"] = time.perf_counter() - export_started
    T["total"] = float(sum(v for k, v in T.items() if k not in {"total"}))
    meta["timing_s"] = {k: round(v, 2) for k, v in T.items()}
    meta["run_manifest"] = {"path": "run_manifest.json", "schema_version": "wss-deploy.run-manifest/v1"}
    meta["exports"] = {"vtp": bool(vtp_ok), "csv": True, "html": True, "internal_field": True, "run_manifest": True}
    R.update_html_meta(job_dir / "report.html", report_meta(meta))
    (job_dir / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    # The manifest is intentionally a separate, portable record.  It is
    # written after summary/report so their hashes are stable.  Changing model
    # weights creates a new, auditable release rather than overwriting a run.
    quality_audit_path = job_dir / "quality_audit.json"
    # Maintenance-only record: keeps numeric ensemble diagnostics and the
    # provenance needed to identify a result, while the public report receives
    # only the compact quality level/reasons.
    quality_audit_record = {
        **quality["audit"], "created_at": meta["created_at"],
        "input_sha256": meta["input_sha256"], "release_hash": meta["release_hash"],
        "model_release": meta["model_release"], "mapping_history": mapping_history,
    }
    quality_audit_path.write_text(json.dumps(quality_audit_record, ensure_ascii=False, indent=1), encoding="utf-8")
    output_names = ("report.html", "summary.json", "wall_wss.vtp", "points_wss.csv", "field.npz", "quality_audit.json", "stage_a.json")
    write_run_manifest(job_dir, meta, outputs=output_names)
    return meta


def run_all(stl_path: Path, out_dir: Path, release: Release, *, mapping: dict[str, str] | None = None,
            inlet: int | None = None, smooth_mm: float = 1.0, spacing_mm: float = 0.5,
            case_id: str | None = None, units: str = "mm",
            allow_unvalidated_auto: bool = False) -> dict:
    a = stage_a(stl_path, out_dir, inlet=inlet, units=units)
    if a.get("stage") != "A":
        raise RuntimeError("输入检查未通过：" + "; ".join(a["input_check"].get("errors", []) + a["input_check"].get("flags", [])))
    gate = CL.evaluate_confidence_gate(
        a["proposal"],
        orientation_source=a["input_check"].get("orientation_source", "unknown_stl"),
        release=release,
    )
    a["proposal"]["confidence_gate"] = gate
    a["proposal"]["confidence_profile"] = gate.get("profile_id")
    a["proposal"]["confidence_is_calibrated"] = gate.get("calibration_status") == "validated"
    a["proposal"]["confirmation_required"] = not bool(gate.get("passed"))
    (Path(out_dir) / "stage_a.json").write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    if mapping is None:
        if not a["proposal"]["auto_ok"]:
            raise RuntimeError("automatic outlet naming failed: " + "; ".join(a["proposal"]["flags"]))
        if not gate["passed"] and not allow_unvalidated_auto:
            reasons = "; ".join(gate.get("reasons") or [])
            raise RuntimeError("automatic outlet naming is not independently validated; "
                               "provide an explicit mapping after review. " + reasons)
        mapping = a["proposal"]["mapping"]
    return stage_b(out_dir, mapping, release, smooth_mm=smooth_mm, spacing_mm=spacing_mm, confirmed=True, case_id=case_id)
