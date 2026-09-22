"""Rebuild report.html (and streamlines) for finished jobs without re-running the model.

    PYTHONPATH=. python -m wss_deploy.rebuild_report [--jobs-root outputs/wss_deploy_jobs] [--job ID ...]

Predictions are read back from ``field.npz``; only the presentation layer is regenerated:
the embedded viewer, and for volume jobs the streamlines (re-integrated from the stored
velocity samples with the current seeding/coverage rules).  ``summary.json``,
``run_manifest.json`` and the job record are updated so the output hashes stay truthful.
Stop the service (or restart it afterwards) before touching jobs it holds in memory.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import analysis as A, centerline as CL, metrics as M, morphology as MORPH, narrative as NARR, report as R
from .io_utils import atomic_json, resolve_job_path
from .pipeline import report_meta
from .schema import write_run_manifest


def _centerline_arrays(atlas) -> dict:
    seg = atlas.col("segment_id").astype(int); idx = atlas.col("sample_index")
    edges = []
    for sid in np.unique(seg):
        rows = np.flatnonzero(seg == sid); rows = rows[np.argsort(idx[rows])]
        edges.extend(zip(rows[:-1], rows[1:]))
    return {"xyz": atlas.xyz, "tangent": atlas.tangent, "radius_mm": atlas.col("radius_mm"),
            "edges": np.asarray(edges, np.uint32), "segment": seg}


def _record_timing(meta: dict, key: str, seconds: float) -> None:
    """Keep ``timing_s`` truthful for a step this rebuild actually ran (the rest stays the original run's)."""
    timing = meta.get("timing_s")
    if isinstance(timing, dict):
        timing[key] = round(float(seconds), 2)


def _load_job(job_dir: Path):
    record = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    meta = json.loads((job_dir / "summary.json").read_text(encoding="utf-8"))
    if record.get("status") != "done" or not record.get("mapping"):
        raise ValueError("只能重建已完成且已确认出口的任务")
    atlas = CL.apply_mapping(CL.load_vessel_geom_atlas(job_dir / "centerline"), record["mapping"])
    return record, meta, atlas


def _refresh_reference(meta: dict) -> dict:
    """Re-evaluate the release's reference profiles (geometry ranges, OOF population) for a stored summary.

    Reads only ``release.json`` and the optional ``reference.json`` sidecar of the release the run
    was bound to; no weights are loaded.  Unknown release → assessment stays ``unknown``.
    """
    from .infer import load_reference_sidecar
    from .paths import RELEASE_DIR
    from .reference import evaluate
    release = meta.get("model_release") or {}
    release_id = release.get("registry_id") or release.get("release") or release.get("name") or meta.get("release")
    info = {}
    if isinstance(release_id, str) and release_id:
        release_dir = RELEASE_DIR.parent / release_id
        if (release_dir / "release.json").is_file():
            info = json.loads((release_dir / "release.json").read_text(encoding="utf-8"))
            try:
                load_reference_sidecar(release_dir, str(info.get("release", release_id)), info)
            except ValueError:
                info = {}
    view = dict(meta)
    if "geometry" not in view and isinstance(view.get("branch_geometry"), dict):
        view["geometry"] = view["branch_geometry"]   # volume family stores the branch table under this key
    return evaluate(view, info)


def rebuild_volume(job_dir: Path, *, streamlines: bool = True) -> dict:
    from .streamlines import centerline_seeds, integrate_streamlines, thin_lines, volume_seeds
    from .volume_geometry import close_lumen, make_inside_test
    from .volume_report import build_html
    record, meta, atlas = _load_job(job_dir)
    z = np.load(job_dir / "field.npz")
    vertices = np.asarray(z["vertices"], np.float64); faces = np.asarray(z["faces"], np.int64)
    pts = np.asarray(z["pts"], np.float32); internal = np.asarray(z["internal_pts"], np.float32)
    n_wall = len(pts) - len(internal)
    pressure = np.asarray(z["pressure_pa"], np.float32); velocity = np.asarray(z["velocity_m_s"], np.float32)
    segment_id = np.asarray(z["segment_id"]); vertex_pressure = np.asarray(z["vertex_pressure_pa"], np.float32)
    # Analysis layer recomputed from the stored predictions (deterministic; field.npz untouched).
    from wss_features.atlas import map_points
    from .volume_geometry import _surface, _wall_distances
    branch_names = dict(meta.get("branch_names") or {})
    feats = map_points(internal.astype(np.float64), atlas)
    interior_feats = {key: np.asarray(feats[key]) for key in ("segment_id", "s_local_mm", "s_from_root_mm", "radius_mm")}
    speed = np.linalg.norm(velocity, axis=1)
    branch_geometry = M.geometry_table(atlas, {int(k): v for k, v in branch_names.items()})
    meta["branch_geometry"] = branch_geometry
    meta["profiles"] = A.profiles(atlas, interior_feats, {"speed": speed, "pressure": pressure[n_wall:]}, branch_names=branch_names)
    meta["findings"] = A.findings_volume(internal, speed, pressure[n_wall:], interior_feats, atlas, branch_geometry, branch_names=branch_names)
    # Morphology (contract §17.1) is re-measured from the stored clean mesh; the ΔP items above fill
    # ``branches[].delta_p_pa``, so the max-diameter finding is patched afterwards instead of
    # recomputing the whole findings list.
    started = time.perf_counter()
    meta["morphology"] = MORPH.compute(vertices, faces, atlas, branch_names=branch_names, findings=meta["findings"],
                                       interior={"segment_id": interior_feats["segment_id"], "speed": speed})
    _record_timing(meta, "morphology", time.perf_counter() - started)
    A.apply_morphology(meta["findings"], meta["morphology"], branch_names)
    vertex_segment = R.nearest_label(pts[:n_wall], segment_id[:n_wall], vertices)
    caps = list(((meta.get("volume_geometry") or {}).get("closure") or {}).get("caps") or [])
    if not caps:
        caps = [{"center_mm": np.asarray(e["center_mm"]).tolist(), "radius_mm": float(e["radius_mm"])} for e in atlas.endpoints()]
    meta["reference_assessment"] = _refresh_reference(meta)
    mesh_trust, cloud_trust, meta["trust"] = A.trust_volume(vertices, vertex_pressure, vertex_segment, internal,
                                                            interior_feats["segment_id"], caps,
                                                            meta.get("reference_assessment"), branch_names)
    dist_to_wall = _wall_distances(internal.astype(np.float64), _surface(vertices, faces)).astype(np.float32)
    frame = meta.get("frame_transform") if isinstance(meta.get("frame_transform"), dict) else {}
    if frame.get("rotation") is not None and frame.get("origin_mm") is not None:
        meta["frame_transform"] = A.frame_transform(frame["rotation"], frame["origin_mm"],
                                                    source=frame.get("source", "vessel_geom atlas + anatomical_frame"),
                                                    direction_source=frame.get("direction_source") or (meta.get("input_check") or {}).get("orientation_source"),
                                                    **({"vector_to_world": frame["vector_to_world"]} if frame.get("vector_to_world") else {}))
    lines = []
    if not streamlines:
        lines = _load_streamlines(job_dir / "streamlines.vtp")   # page-only rebuild keeps the stored lines
    if streamlines:
        closed, _ = close_lumen(vertices, faces, atlas)
        inside = make_inside_test(closed.points, np.asarray(closed.faces).reshape(-1, 4)[:, 1:])
        seed = int(meta.get("sampling_seed") or 0)
        seeds = np.concatenate([centerline_seeds(atlas), volume_seeds(internal, n=120, seed=seed)])
        lines, info = integrate_streamlines(internal, velocity, seeds, inside)
        lines, info["vertex_stride"] = thin_lines(lines)
        meta["streamlines"] = info
        vtp = job_dir / "streamlines.vtp"
        if lines:
            import pyvista as pv
            xyz = np.concatenate([line["points"] for line in lines]); cells = []; offset = 0
            for line in lines:
                size = len(line["points"]); cells.extend([size, *range(offset, offset + size)]); offset += size
            traces = pv.PolyData(xyz, lines=np.asarray(cells))
            traces.point_data["speed_m_s"] = np.concatenate([line["speed_m_s"] for line in lines])
            traces.save(str(vtp))
        elif vtp.is_file():
            vtp.unlink()
        meta.setdefault("exports", {})["streamlines"] = bool(lines)
    build_html(job_dir / "report.html", report_meta(meta),
               mesh={"vertices": vertices, "faces": faces, "pressure_pa": vertex_pressure, "trust": mesh_trust},
               cloud={"pts": internal, "pressure_pa": pressure[n_wall:], "velocity_m_s": velocity, "segment": segment_id[n_wall:],
                      "trust": cloud_trust, "s_from_root_mm": interior_feats["s_from_root_mm"].astype(np.float32),
                      "radius_mm": interior_feats["radius_mm"].astype(np.float32), "dist_to_wall_mm": dist_to_wall},
               centerline=_centerline_arrays(atlas), streamlines=lines)
    return _finish(job_dir, record, meta, ["summary.json", "report.html", "field.npz", "stage_a.json", "volume_fields.vtp",
                                            "wall_pressure.vtp", "points_volume.csv"] + (["streamlines.vtp"] if lines else []),
                   {"streamlines": len(lines), "regenerated": bool(streamlines)})


def _load_streamlines(vtp: Path) -> list:
    """Read back the exported polylines (points + speed) so a page-only rebuild keeps them."""
    if not vtp.is_file():
        return []
    import pyvista as pv
    poly = pv.read(str(vtp))
    cells = np.asarray(poly.lines); speed = np.asarray(poly.point_data["speed_m_s"], np.float32)
    points = np.asarray(poly.points, np.float32)
    lines, at = [], 0
    while at < len(cells):
        size = int(cells[at]); ids = cells[at + 1:at + 1 + size]; at += 1 + size
        lines.append({"points": points[ids], "speed_m_s": speed[ids]})
    return lines


def rebuild_wall(job_dir: Path) -> dict:
    from wss_features.atlas import map_points
    from wss_features.cloud import pca_normals
    from wss_features.frame import anatomical_frame
    record, meta, atlas = _load_job(job_dir)
    z = np.load(job_dir / "field.npz")
    pts = np.asarray(z["pts"], np.float64); wss = np.asarray(z["wss_pa"], np.float32)
    vertices = np.asarray(z["vertices"], np.float64); faces = np.asarray(z["faces"], np.int64)
    feats = map_points(pts, atlas)
    seg = atlas.col("segment_id").astype(int)
    vseg = R.nearest_label(pts, z["segment_id"], vertices)
    vw = np.asarray(z["vertex_wss_pa"], np.float32)
    # Analysis layer recomputed from the stored predictions (deterministic; field.npz untouched).
    branch_names = dict(meta.get("branch_names") or {})
    geometry_table = meta.get("geometry") or M.geometry_table(atlas, {int(k): v for k, v in branch_names.items()})
    thresholds = ((meta.get("wss_field_pa") or {}).get("thresholds_pa")) or [M.LOW_PA, M.HIGH_PA, M.VERY_HIGH_PA]
    spacing = float((meta.get("cloud") or {}).get("spacing_mm") or A._median_spacing(pts))
    area = float((meta.get("input_check") or {}).get("area_mm2") or 0.0)
    if area <= 0:
        tri = vertices[faces]; area = float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() * 0.5)
    meta["profiles"] = A.profiles(atlas, feats, {"wss": wss}, branch_names=branch_names)
    started = time.perf_counter()
    meta["morphology"] = MORPH.compute(vertices, faces, atlas, branch_names=branch_names, thresholds=thresholds,
                                       per_branch=meta.get("per_branch"),
                                       cloud={"segment_id": np.asarray(z["segment_id"]), "wss": wss})
    _record_timing(meta, "morphology", time.perf_counter() - started)
    meta["findings"] = A.findings_wall(pts, wss, feats, atlas, geometry_table, thresholds=thresholds,
                                       total_area_mm2=area, spacing_mm=spacing, branch_names=branch_names,
                                       morphology=meta["morphology"])
    _, variation = pca_normals(pts, atlas)
    meta["reference_assessment"] = _refresh_reference(meta)
    trust_vertices, meta["trust"] = A.trust_wall(vertices, vw, pts, variation, vseg, meta.get("reference_assessment"), branch_names)
    frame = anatomical_frame(atlas.table, list(atlas.columns), {str(k): v for k, v in atlas.semantic_of_segment.items()})
    previous = meta.get("frame_transform") if isinstance(meta.get("frame_transform"), dict) else {}
    meta["frame_transform"] = A.frame_transform(frame["rotation"], frame["origin_mm"],
                                                source=previous.get("source", "vessel_geom atlas + anatomical_frame"),
                                                direction_source=previous.get("direction_source") or (meta.get("input_check") or {}).get("orientation_source"))
    # Extra wall scalar fields (three-head release: TAWSS / OSI) are stored in field.npz under the descriptor's
    # array_key (+ vertex_<key>); carry them into the rebuilt page so the field switch keeps working.
    extra_c, extra_m = {}, {}
    for field_id, desc in (meta.get("fields") or {}).items():
        key = desc.get("array_key") if isinstance(desc, dict) else None
        if field_id != "wss" and key and key in z.files and f"vertex_{key}" in z.files:
            extra_c[key] = np.asarray(z[key], np.float32); extra_m[key] = np.asarray(z[f"vertex_{key}"], np.float32)
    R.build_html(job_dir / "report.html", report_meta(meta),
                 mesh={"vertices": vertices, "faces": faces, "wss": vw, "segment": vseg, "trust": trust_vertices, "extra": extra_m},
                 cloud={"pts": pts, "wss": wss, "segment": z["segment_id"], "s_from_root_mm": z["s_from_root_mm"], "theta_rad": z["theta_rad"],
                        "radius_mm": z["radius_mm"], "dist_to_junction_mm": feats["dist_to_junction_mm"], "extra": extra_c},
                 centerline={"xyz": atlas.xyz, "radius_mm": atlas.col("radius_mm"), "edges": _centerline_arrays(atlas)["edges"], "segment": seg.astype(np.int16)})
    return _finish(job_dir, record, meta, ["report.html", "summary.json", "wall_wss.vtp", "points_wss.csv", "field.npz", "quality_audit.json", "stage_a.json"], {})


def embed_sidecars(job_dir: Path, meta: dict) -> dict:
    """Carry the reviewer's sidecars into the rebuilt page: ``annotations.json`` → ``meta.annotations``,
    ``findings_review.json`` → ``meta.findings.review``, ``narrative.json`` → the edited conclusion
    (contract §11.7 / §11.10 / §17.2).  Missing files leave meta untouched."""
    def _narrative(doc: dict) -> None:
        block = meta.get("narrative") if isinstance(meta.get("narrative"), dict) else {}
        meta["narrative"] = NARR.merge_edit(block or doc.get("auto"), doc.get("edited"),
                                            edited_by=doc.get("edited_by"), edited_at=doc.get("edited_at"))

    for name, apply in (("annotations.json", lambda doc: meta.__setitem__("annotations", doc)),
                        ("findings_review.json", lambda doc: meta.setdefault("findings", {}).__setitem__("review", doc)),
                        ("narrative.json", _narrative)):
        path = Path(job_dir) / name
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print(f"[rebuild] {job_dir.name}: cannot read {name}; skipped", file=sys.stderr)
            continue
        if isinstance(doc, dict):
            if not isinstance(meta.get("findings"), dict) and name == "findings_review.json":
                meta["findings"] = {}
            apply(doc)
    return meta


def _finish(job_dir: Path, record: dict, meta: dict, outputs: list, extra: dict) -> dict:
    stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    meta["narrative"] = NARR.build_narrative(meta)
    embed_sidecars(job_dir, meta)
    meta.setdefault("audit", {})["report_rebuilt_at"] = stamp
    R.update_html_meta(job_dir / "report.html", report_meta(meta))
    atomic_json(job_dir / "summary.json", meta)
    write_run_manifest(job_dir, meta, outputs=outputs)
    summary = record.get("summary")
    # The job record is owned by the running service; rewrite it only when the
    # export set changed (new streamlines), which needs a restart anyway.
    if isinstance(summary, dict) and extra.get("regenerated"):
        if "exports" in meta: summary["exports"] = meta["exports"]
        summary.setdefault("audit", {})
        if isinstance(summary["audit"], dict): summary["audit"]["report_rebuilt_at"] = stamp
        atomic_json(job_dir / "job.json", record)
    return {"job": job_dir.name, "family": "volume" if "velocity_m_s" in np.load(job_dir / "field.npz").files else "wall",
            "report_bytes": (job_dir / "report.html").stat().st_size, **extra}


def rebuild(job_dir: Path, *, streamlines: bool = True) -> dict:
    job_dir = Path(job_dir).resolve()
    if "velocity_m_s" in np.load(job_dir / "field.npz").files:
        return rebuild_volume(job_dir, streamlines=streamlines)
    return rebuild_wall(job_dir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs-root", default="outputs/wss_deploy_jobs")
    parser.add_argument("--job", action="append", default=None, help="job id (repeatable); default: every finished job")
    parser.add_argument("--no-streamlines", action="store_true", help="keep the stored streamlines; only rebuild the page")
    args = parser.parse_args(argv)
    root = Path(args.jobs_root)
    ids = args.job or [p.parent.name for p in sorted(root.glob("*/job.json"))
                       if json.loads(p.read_text(encoding="utf-8")).get("status") == "done" and (p.parent / "field.npz").is_file()
                       and (p.parent / "centerline" / "atlas.npz").is_file()]
    results, failed = [], []
    for job_id in ids:
        try:
            result = rebuild(root / job_id, streamlines=not args.no_streamlines)
            results.append(result); print(f"[rebuild] {job_id}: {result}", file=sys.stderr, flush=True)
        except Exception as exc:
            failed.append(job_id); print(f"[rebuild] {job_id}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    print(json.dumps({"rebuilt": [r["job"] for r in results], "failed": failed}, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
