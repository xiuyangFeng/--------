"""Orchestration: stage A (ingest + centreline + naming proposal, CPU) -> [human confirms outlets] -> stage B (geometry + inference + report)."""
from __future__ import annotations
import hashlib, json, time, datetime
from pathlib import Path
import numpy as np
from wss_v5.centerline_features import Atlas
from . import centerline as CL, geometry as G, metrics as M, report as R
from .ingest import ingest
from .infer import Release
from .paths import OUTLET_CN
from .io_utils import file_sha256


def _preview_surface(path: Path, max_faces: int = 18000) -> dict:
    """Small display mesh; model inference always uses the full cleaned STL."""
    from training_wss_min.surface import load_stl
    vertices, faces = load_stl(path)
    vertices, faces = np.asarray(vertices, np.float32), np.asarray(faces, np.int64)
    if len(faces) > max_faces:
        faces = faces[np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)]
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return {"vertices": vertices[used].round(3).tolist(), "faces": inverse.reshape(-1, 3).tolist(),
            "display_faces": int(len(faces)), "display_only": True}


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stage_a(stl_path: Path, job_dir: Path, *, inlet: int | None = None, smooth_iterations: int = 0,
            units: str = "mm", remove_fragments: bool = False, progress=None, cancelled=None) -> dict:
    job_dir = Path(job_dir).resolve(); job_dir.mkdir(parents=True, exist_ok=True); T = {}
    progress = progress or (lambda phase, detail: None)
    cancelled = cancelled or (lambda: False)
    if cancelled(): raise InterruptedError("任务已取消")
    progress("ingest", "正在读取并检查 STL")
    t = time.perf_counter(); ic = ingest(Path(stl_path), job_dir, units=units, remove_fragments=remove_fragments); T["ingest"] = time.perf_counter() - t
    if ic["status"] != "pass":
        out = {"stage": "input", "created_at": _now(), "stl": str(stl_path), "input_sha256": file_sha256(Path(stl_path)), "input_check": ic,
               "centerline": None, "proposal": None, "inlet_override": inlet, "timing_s": {k: round(v, 2) for k, v in T.items()}}
        (job_dir / "stage_a.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        return out
    if cancelled(): raise InterruptedError("任务已取消")
    progress("centerline", "正在提取中心线和开口")
    cl_dir = job_dir / "centerline"
    t = time.perf_counter(); cl = CL.run_vessel_geom(Path(ic["clean_stl"]), cl_dir, inlet=inlet, smooth_iterations=smooth_iterations); T["centerline"] = time.perf_counter() - t
    if cancelled(): raise InterruptedError("任务已取消")
    atlas = CL.load_vessel_geom_atlas(cl_dir)
    proposal = CL.propose_outlets(atlas)
    proposal["direction_note"] = ic.get("direction_note", "")
    out = {"stage": "A", "created_at": _now(), "stl": str(stl_path), "input_sha256": file_sha256(Path(stl_path)), "input_check": ic,
           "centerline": cl, "proposal": proposal, "preview": _preview_surface(Path(ic["clean_stl"])),
           "inlet_override": inlet, "timing_s": {k: round(v, 2) for k, v in T.items()}}
    (job_dir / "stage_a.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def stage_b(job_dir: Path, mapping: dict[str, str], release: Release, *, smooth_mm: float = 1.0, spacing_mm: float = 0.5, confirmed: bool = False, case_id: str | None = None, progress=None, cancelled=None) -> dict:
    job_dir = Path(job_dir).resolve(); a = json.loads((job_dir / "stage_a.json").read_text(encoding="utf-8")); T = dict(a["timing_s"])
    progress = progress or (lambda phase, detail: None)
    cancelled = cancelled or (lambda: False)
    if a.get("stage") != "A" or not a.get("input_check", {}).get("ok"):
        raise ValueError("输入检查尚未通过，不能开始预测。")
    if not confirmed:
        raise ValueError("必须确认出口命名后才能开始预测。")
    if cancelled(): raise InterruptedError("任务已取消")
    progress("geometry", "正在平滑表面并重采样")
    atlas = CL.apply_mapping(CL.load_vessel_geom_atlas(job_dir / "centerline"), mapping)
    from training_wss_min.surface import load_stl
    vertices, faces = load_stl(Path(a["input_check"]["clean_stl"])); vertices = np.asarray(vertices, np.float64); faces = np.asarray(faces, np.int64)
    t = time.perf_counter(); v_s, pts = G.smooth_and_resample(vertices, faces, smooth_mm=smooth_mm, spacing_mm=spacing_mm, seed=G.stable_sampling_seed(a["input_sha256"])); T["smooth_resample"] = time.perf_counter() - t
    if cancelled(): raise InterruptedError("任务已取消")
    progress("features", "正在构建 27 维几何特征")
    t = time.perf_counter(); case, aux = G.build_case(pts, atlas, release.input_features, case_name=job_dir.name); T["features"] = time.perf_counter() - t
    if cancelled(): raise InterruptedError("任务已取消")
    progress("inference", "正在运行五模型集成")
    t = time.perf_counter(); pred = release.predict(case); T["inference_5_models"] = time.perf_counter() - t
    wss = pred["wss_pa"]; geom = aux["geom"]; diag = aux["diag"]
    progress("metrics", "正在汇总预测点云统计")
    t = time.perf_counter()
    met = M.compute(pts, wss, geom, atlas, diag, a["input_check"]["area_mm2"])
    vw = R.interpolate_to_vertices(pts, wss.astype(np.float32), vertices); vseg = R.nearest_label(pts, geom["segment_id"], vertices)
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
    T["metrics_and_interpolation"] = time.perf_counter() - t
    meta = {"case_id": case_id or job_dir.name, "release": release.name, "device": pred["device"], "gpu": pred["gpu"], "created_at": _now(), "input_sha256": a["input_sha256"],
            "input_check": a["input_check"], "centerline": a["centerline"], "outlets_confirmed": confirmed, "mapping": mapping, "flags": a["proposal"].get("flags", []),
            "cloud": {"n_points": int(len(pts)), "spacing_mm": diag["spacing_mm"], "smooth_mm": smooth_mm, "surface_variation_median": diag["surface_variation_median"], "junction_ambiguous_fraction": diag["junction_ambiguous_fraction"]},
            "caps": diag["caps"], "murray_shares": diag["murray_shares"], "endpoints": endpoints, "per_branch_s": per_branch_s, "timing_s": {k: round(v, 2) for k, v in T.items()},
            "seconds_per_model": [round(x, 2) for x in pred["seconds_per_model"]],
            "sampling_seed": G.stable_sampling_seed(a["input_sha256"]),
            "model_frame": {"target": "peak_systole", "step": 1162, "time_s": 0.21},
            "run_parameters": {"smooth_mm": smooth_mm, "spacing_mm": spacing_mm, "device": pred["device"], "seed_count": len(pred["seed_pred_pa"])},
            "frame_transform": {"source": "vessel_geom atlas + anatomical_frame", "direction_source": a["input_check"].get("orientation_source")},
            "release_hash": file_sha256(Path(release.dir) / "MANIFEST.sha256") if (Path(release.dir) / "MANIFEST.sha256").is_file() else release.name,
            "interpolation": {"method": "Gaussian", "sigma_mm": 0.5, "max_dist_mm": 1.5, "covered_vertices": int(np.isfinite(vw).sum()), "total_vertices": int(len(vw))}, **met}
    progress("export", "正在写入 CSV、VTP 和 HTML 报告")
    export_started = time.perf_counter()
    np.savez_compressed(job_dir / "field.npz", pts=pts.astype(np.float32), wss_pa=wss.astype(np.float32), seed_pred_pa=pred["seed_pred_pa"].astype(np.float32), segment_id=geom["segment_id"],
                        s_from_root_mm=geom["s_from_root_mm"], theta_rad=geom["theta_rad"], radius_mm=geom["radius_mm"], vertices=vertices.astype(np.float32), faces=faces.astype(np.int32), vertex_wss_pa=vw)
    with open(job_dir / "points_wss.csv", "w") as f:
        f.write("x_mm,y_mm,z_mm,wss_pa,segment_id,branch,s_from_inlet_mm,theta_rad\n")
        bn = met["branch_names"]
        for i in range(len(pts)):
            f.write(f"{pts[i,0]:.3f},{pts[i,1]:.3f},{pts[i,2]:.3f},{wss[i]:.4f},{int(geom['segment_id'][i])},{bn.get(str(int(geom['segment_id'][i])),'')},{geom['s_from_root_mm'][i]:.2f},{geom['theta_rad'][i]:.4f}\n")
    vtp_ok = R.write_vtp(job_dir / "wall_wss.vtp", vertices, faces, {"wss_pa": vw, "wss_covered": np.isfinite(vw), "segment_id": vseg.astype(np.int32)})
    R.build_html(job_dir / "report.html", meta,
                 mesh={"vertices": vertices, "faces": faces, "wss": vw, "segment": vseg},
                 cloud={"pts": pts, "wss": wss, "segment": geom["segment_id"], "s_from_root_mm": geom["s_from_root_mm"], "theta_rad": geom["theta_rad"], "radius_mm": geom["radius_mm"], "dist_to_junction_mm": geom["dist_to_junction_mm"]},
                 centerline={"xyz": atlas.xyz, "radius_mm": atlas.col("radius_mm"), "edges": np.asarray(edges, dtype=np.uint32), "segment": seg.astype(np.int16)})
    T["export"] = time.perf_counter() - export_started
    T["total"] = float(sum(v for k, v in T.items() if k not in {"total"}))
    meta["timing_s"] = {k: round(v, 2) for k, v in T.items()}
    meta["exports"] = {"vtp": bool(vtp_ok), "csv": True, "html": True, "internal_field": True}
    R.update_html_meta(job_dir / "report.html", meta)
    (job_dir / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def run_all(stl_path: Path, out_dir: Path, release: Release, *, mapping: dict[str, str] | None = None, inlet: int | None = None, smooth_mm: float = 1.0, spacing_mm: float = 0.5, case_id: str | None = None) -> dict:
    a = stage_a(stl_path, out_dir, inlet=inlet, units="mm")
    if a.get("stage") != "A":
        raise RuntimeError("输入检查未通过：" + "; ".join(a["input_check"].get("errors", []) + a["input_check"].get("flags", [])))
    if mapping is None:
        if not a["proposal"]["auto_ok"]:
            raise RuntimeError("automatic outlet naming failed: " + "; ".join(a["proposal"]["flags"]))
        mapping = a["proposal"]["mapping"]
    return stage_b(out_dir, mapping, release, smooth_mm=smooth_mm, spacing_mm=spacing_mm, confirmed=True, case_id=case_id)
