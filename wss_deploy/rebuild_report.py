"""Rebuild report.html (and streamlines) for finished jobs without re-running the model.

    PYTHONPATH=. python -m wss_deploy.rebuild_report [--jobs-root outputs/wss_deploy_jobs] [--job ID ...] [--force]

Predictions are read back from ``field.npz``; only the presentation layer is regenerated:
the embedded viewer, and for volume jobs the streamlines (re-integrated from the stored
velocity samples with the current seeding/coverage rules).  ``summary.json``,
``run_manifest.json`` and the job record are updated so the output hashes stay truthful.
The command refuses while a service holds the jobs-root writer lock (v0.14); use
``cli service upgrade --rebuild ID`` instead, or ``--force`` when you know it is safe.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from . import analysis as A, centerline as CL, metrics as M, morphology as MORPH, narrative as NARR, report as R
from . import geometry_cache as GC
from .io_utils import atomic_json, file_sha256, resolve_job_path
from .pipeline import atomic_output, report_meta
from .schema import summary_provenance, write_run_manifest


class _NpzArrays:
    """``field.npz`` with each array decompressed at most once (v0.14: ``segment_id`` was read up to five
    times, every read inflating the member again).  Read-only use; ``files`` as on ``NpzFile``."""

    def __init__(self, path: Path):
        self._npz = np.load(path)
        self.files = list(self._npz.files)
        self._arrays: dict[str, np.ndarray] = {}

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._arrays:
            self._arrays[key] = self._npz[key]
        return self._arrays[key]


def _surface_variation(job_dir: Path, pts: np.ndarray, atlas) -> np.ndarray:
    """PCA surface variation of the stored cloud (the report's rough-surface trust bit).

    v0.14: cached in ``geometry_cache/`` (never in field.npz, which the golden regression compares) under
    a key over the stored points, the atlas geometry and the feature program; recomputed on any change.
    """
    from wss_features.cloud import pca_normals
    cache = GC.GeometryCache(job_dir)
    key = GC.key_of("rebuild-surface-variation", GC.feature_program_hash(), np.asarray(pts),
                    np.asarray(atlas.table), np.asarray(atlas.tree_rows), list(atlas.columns))
    hit = cache.load("rebuildvar", key)
    if hit is not None and "variation" in hit:
        return hit["variation"]
    _, variation = pca_normals(pts, atlas)
    cache.save("rebuildvar", key, {"variation": variation})
    return variation


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
    if "model_release" in meta:    # v0.15.1: strip server paths from provenance written by older code
        from .schema import redact_paths
        meta["model_release"] = redact_paths(meta["model_release"])
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
    from .paths import RELEASE_DIR, RETIRED_RELEASE_ROOT
    from .reference import evaluate
    release = meta.get("model_release") or {}
    release_id = release.get("registry_id") or release.get("release") or release.get("name") or meta.get("release")
    info = {}
    if isinstance(release_id, str) and release_id:
        release_dir = RELEASE_DIR.parent / release_id
        if not (release_dir / "release.json").is_file() and (RETIRED_RELEASE_ROOT / release_id / "release.json").is_file():
            release_dir = RETIRED_RELEASE_ROOT / release_id      # 2026-10-02: a retired package keeps its reference sidecar
        if (release_dir / "release.json").is_file():
            info = json.loads((release_dir / "release.json").read_text(encoding="utf-8"))
            try:
                load_reference_sidecar(release_dir, str(info.get("release", release_id)), info)
            except ValueError:
                info = {}
            if info:
                from .reference import merge_sidecar_v2   # E4: v2 population_references (listing only)
                info = merge_sidecar_v2(info, release_dir, str(info.get("release", release_id)))
    view = dict(meta)
    if "geometry" not in view and isinstance(view.get("branch_geometry"), dict):
        view["geometry"] = view["branch_geometry"]   # volume family stores the branch table under this key
    return evaluate(view, info)


def rebuild_volume(job_dir: Path, *, streamlines: bool = True) -> dict:
    from .streamlines import ball_certified_inside, centerline_seeds, integrate_streamlines, thin_lines, volume_seeds
    from .volume_geometry import close_lumen, make_inside_test
    from .volume_report import build_html
    record, meta, atlas = _load_job(job_dir)
    previous_findings = _previous_findings(meta)
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
                                       interior={"segment_id": interior_feats["segment_id"], "speed": speed},
                                       section_cache=GC.GeometryCache(job_dir))
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
        closed_faces = np.asarray(closed.faces).reshape(-1, 4)[:, 1:]
        inside = ball_certified_inside(make_inside_test(closed.points, closed_faces), internal, closed.points, closed_faces)
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
            with atomic_output(vtp) as tmp:
                traces.save(str(tmp))
        elif vtp.is_file():
            vtp.unlink()
        meta.setdefault("exports", {})["streamlines"] = bool(lines)
    with atomic_output(job_dir / "report.html") as report_tmp:
        build_html(report_tmp, report_meta(meta),
                   mesh={"vertices": vertices, "faces": faces, "pressure_pa": vertex_pressure, "trust": mesh_trust},
                   cloud={"pts": internal, "pressure_pa": pressure[n_wall:], "velocity_m_s": velocity, "segment": segment_id[n_wall:],
                          "trust": cloud_trust, "s_from_root_mm": interior_feats["s_from_root_mm"].astype(np.float32),
                          "radius_mm": interior_feats["radius_mm"].astype(np.float32), "dist_to_wall_mm": dist_to_wall},
                   centerline=_centerline_arrays(atlas), streamlines=lines)
    return _finish(job_dir, record, meta, ["summary.json", "report.html", "field.npz", "stage_a.json", "volume_fields.vtp",
                                            "wall_pressure.vtp", "points_volume.csv"] + (["streamlines.vtp"] if lines else []),
                   {"streamlines": len(lines), "regenerated": bool(streamlines)}, previous=previous_findings)


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


def _cycle_arrays(meta: dict, z) -> dict:
    """TAWSS / OSI point arrays of a three-head job from ``field.npz`` (descriptor ``array_key``); {} otherwise."""
    out = {}
    for field_id, default in (("tawss", "tawss_pa"), ("osi", "osi")):
        desc = (meta.get("fields") or {}).get(field_id)
        if not isinstance(desc, dict):
            continue
        key = desc.get("array_key") or default
        if key in z.files:
            out[field_id] = np.asarray(z[key], np.float64)
    return out


def _add_derived_indices(meta: dict, cycle: dict, segment_id: np.ndarray, branch_names: dict, area_mm2: float) -> dict:
    """RRT / ECAP point arrays from the stored TAWSS / OSI; descriptors, ``cycle.fields`` and ``results`` follow.

    Returns ``{}`` for a job without both cycle fields (peak-only releases stay byte-for-byte as before).
    """
    from . import cycle_fields as CF
    if not all(k in cycle for k in CF.DERIVED_FROM):
        return {}
    derived = CF.derive_indices(cycle["tawss"], cycle["osi"])
    fields = meta.setdefault("fields", {})
    block = meta.setdefault("cycle", {"definition": dict(CF.CYCLE_DEFINITION), "fields": {}})
    block.setdefault("definition", {}).update({k: CF.CYCLE_DEFINITION[k] for k in derived})
    results = meta.get("results") if isinstance(meta.get("results"), dict) else None
    for name, values in derived.items():
        fields[name] = CF.descriptor(name, values)
        block.setdefault("fields", {})[name] = CF.scalar_field_summary(name, values, segment_id, branch_names, area_mm2)
        if results is not None:
            results.setdefault("fields", {})[name] = fields[name]
            results.setdefault("statistics", {})[name] = block["fields"][name]
    return derived


def _append_csv_columns(path: Path, columns: dict) -> None:
    """Append (or replace) derived columns of ``points_wss.csv``; the existing columns keep their exact text."""
    if not path.is_file() or not columns:
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    keep = [i for i, name in enumerate(head) if name not in columns]
    rows = lines[1:]
    n = len(next(iter(columns.values())))
    if len(rows) != n:
        raise ValueError(f"points_wss.csv 行数 {len(rows)} 与预测点数 {n} 不一致")
    lists = [np.asarray(v, np.float64).tolist() for v in columns.values()]
    out = [",".join([head[i] for i in keep] + list(columns))]
    for k, row in enumerate(rows):
        cells = row.split(",")
        out.append(",".join([cells[i] for i in keep] + [f"{col[k]:.4f}" for col in lists]))
    with atomic_output(path) as tmp:
        tmp.write_text("\n".join(out) + "\n", encoding="utf-8")


def rebuild_wall(job_dir: Path) -> dict:
    from wss_features.atlas import map_points
    from wss_features.frame import anatomical_frame
    record, meta, atlas = _load_job(job_dir)
    previous_findings = _previous_findings(meta)
    z = _NpzArrays(job_dir / "field.npz")
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
    cycle = _cycle_arrays(meta, z)
    # v0.13: RRT / ECAP are derived from the stored TAWSS / OSI (field.npz stays untouched) and join the summary,
    # the page and the CSV / VTP exports, so an older three-head job gains them without re-running the model.
    derived = _add_derived_indices(meta, cycle, np.asarray(z["segment_id"]), branch_names, area)
    meta["profiles"] = A.profiles(atlas, feats, {"wss": wss, **cycle, **derived}, branch_names=branch_names)
    started = time.perf_counter()
    # v0.14: station tables and the lumen volume come from geometry_cache/ when mesh, atlas and code are unchanged.
    meta["morphology"] = MORPH.compute(vertices, faces, atlas, branch_names=branch_names, thresholds=thresholds,
                                       per_branch=meta.get("per_branch"),
                                       cloud={"segment_id": np.asarray(z["segment_id"]), "wss": wss},
                                       section_cache=GC.GeometryCache(job_dir))
    _record_timing(meta, "morphology", time.perf_counter() - started)
    meta["findings"] = A.findings_wall(pts, wss, feats, atlas, geometry_table, thresholds=thresholds,
                                       total_area_mm2=area, spacing_mm=spacing, branch_names=branch_names,
                                       morphology=meta["morphology"], cycle=cycle or None)
    # U10 (2026-09-30): anatomical zones from the stored cloud labels (the arrays stage B used).
    meta["zones"] = A.zones_wall(np.asarray(z["segment_id"]), np.asarray(z["s_from_root_mm"]), {"wss": wss, **cycle},
                                 morphology=meta["morphology"], branch_names=branch_names, total_area_mm2=area,
                                 thresholds={"wss": tuple(thresholds[:2])})
    variation = _surface_variation(job_dir, pts, atlas)
    meta["reference_assessment"] = _refresh_reference(meta)
    A.grade_high_findings(meta["findings"], meta["reference_assessment"])    # U11
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
    outputs = ["report.html", "summary.json", "wall_wss.vtp", "points_wss.csv", "field.npz", "quality_audit.json", "stage_a.json"]
    if derived:
        from .cycle_fields import derived_display_arrays
        extra_d = {name: {"values": values, "derived": True} for name, values in derived.items()}
        derived_points, derived_vertex = derived_display_arrays(meta["fields"], extra_d, extra_m)
        extra_c.update(derived_points); extra_m.update(derived_vertex)
        _append_csv_columns(job_dir / "points_wss.csv", derived_points)
        with atomic_output(job_dir / "wall_wss.vtp") as vtp_tmp:
            R.write_vtp(vtp_tmp, vertices, faces, {"wss_pa": vw, "wss_covered": np.isfinite(vw),
                                                   "segment_id": vseg.astype(np.int32), **extra_m})
    with atomic_output(job_dir / "report.html") as report_tmp:
        R.build_html(report_tmp, report_meta(meta),
                     mesh={"vertices": vertices, "faces": faces, "wss": vw, "segment": vseg, "trust": trust_vertices, "extra": extra_m},
                     cloud={"pts": pts, "wss": wss, "segment": z["segment_id"], "s_from_root_mm": z["s_from_root_mm"], "theta_rad": z["theta_rad"],
                            "radius_mm": z["radius_mm"], "dist_to_junction_mm": feats["dist_to_junction_mm"], "extra": extra_c},
                     centerline={"xyz": atlas.xyz, "radius_mm": atlas.col("radius_mm"), "edges": _centerline_arrays(atlas)["edges"], "segment": seg.astype(np.int16)})
    return _finish(job_dir, record, meta, outputs, {"derived_fields": sorted(derived)} if derived else {},
                   previous=previous_findings)


def _embedded_json(html: str, script_id: str) -> tuple[str, dict]:
    match = re.search(r'<script id="' + re.escape(script_id) + r'"[^>]*>(.*?)</script>', html, re.S)
    if not match:
        raise ValueError(f"报告缺少嵌入数据：{script_id}")
    raw = match.group(1)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"报告嵌入数据不是 JSON 对象：{script_id}")
    return raw, value


def _replace_embedded_json(html: str, script_id: str, raw: str) -> str:
    pattern = r'(<script id="' + re.escape(script_id) + r'"[^>]*>).*?(</script>)'
    replaced, count = re.subn(pattern, lambda m: m.group(1) + raw + m.group(2), html, count=1, flags=re.S)
    if count != 1:
        raise ValueError(f"无法保留报告嵌入数据：{script_id}")
    return replaced


def _decode_array(arrays: dict, key: str, dtype, *, width: int = 1, required: bool = True) -> np.ndarray | None:
    raw = arrays.get(key)
    if raw is None:
        if required:
            raise ValueError(f"报告缺少数组：{key}")
        return None
    try:
        values = np.frombuffer(base64.b64decode(raw, validate=True), dtype=dtype).copy()
    except (ValueError, TypeError) as exc:
        raise ValueError(f"报告数组无法解码：{key}") from exc
    if width > 1:
        if values.size % width:
            raise ValueError(f"报告数组形状无效：{key}")
        values = values.reshape(-1, width)
    return values


def _refresh_manifest_report(job_dir: Path) -> None:
    """Update only the report output record; all other provenance stays byte-for-byte untouched."""
    path = Path(job_dir) / "run_manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs")
    record = outputs.get("report.html") if isinstance(outputs, dict) else None
    if isinstance(record, dict):
        record["path"] = "report.html"
        record["present"] = True
        record["size_bytes"] = int((Path(job_dir) / "report.html").stat().st_size)
        record["sha256"] = file_sha256(Path(job_dir) / "report.html")
        atomic_json(path, manifest)


def refresh_ui_only(job_dir: Path) -> dict:
    """Rebuild a finished report with the current templates without touching predictions or summary.

    The self-contained HTML already carries every numeric array needed by the viewer.  Decode those
    arrays, render the current template, then restore the original embedded JSON text and update only
    ``run_manifest.outputs['report.html']``.  This makes a presentation refresh safe for historical
    jobs and avoids the scientific recomputation performed by :func:`rebuild`.
    """
    job_dir = Path(job_dir).resolve()
    report_path = job_dir / "report.html"
    if not report_path.is_file():
        raise ValueError("任务没有 report.html")
    source = report_path.read_text(encoding="utf-8")
    meta_raw, meta = _embedded_json(source, "wss-report-meta")
    array_id = "volume-arrays" if 'id="volume-arrays"' in source else "wss-report-arrays"
    arrays_raw, arrays = _embedded_json(source, array_id)
    temp_path = report_path.with_name(report_path.name + ".ui.tmp")
    if array_id == "wss-report-arrays":
        from .report import build_html
        # v0.14 pages store branch ids as uint8 when they fit and record it in ``arrays["dt"]``; older pages are int32.
        dtypes = arrays.get("dt") if isinstance(arrays.get("dt"), dict) else {}
        seg_type = lambda key: np.uint8 if dtypes.get(key) == "u8" else np.int32
        vertices = _decode_array(arrays, "mv", np.float32, width=3)
        faces = _decode_array(arrays, "mf", np.uint32, width=3)
        center = _decode_array(arrays, "cv", np.float32, width=3)
        cloud_pts = _decode_array(arrays, "pv", np.float32, width=3)
        dist_to_junction = _decode_array(arrays, "p_dj", np.float32, required=False)
        if dist_to_junction is None:
            dist_to_junction = np.zeros(len(cloud_pts), np.float32)
        build_html(temp_path, meta,
                   mesh={"vertices": vertices, "faces": faces, "wss": _decode_array(arrays, "mw", np.float32),
                         "segment": _decode_array(arrays, "ms", seg_type("ms")),
                         "trust": _decode_array(arrays, "mt", np.uint8, required=False),
                         "extra": {key: _decode_array(item, "m", np.float32) for key, item in (arrays.get("xf") or {}).items()}},
                   cloud={"pts": cloud_pts, "wss": _decode_array(arrays, "pw", np.float32),
                          "segment": _decode_array(arrays, "ps", seg_type("ps")),
                          "s_from_root_mm": _decode_array(arrays, "p_s", np.float32),
                          "theta_rad": _decode_array(arrays, "p_th", np.float32),
                          "radius_mm": _decode_array(arrays, "p_r", np.float32),
                          "dist_to_junction_mm": dist_to_junction,
                          "extra": {key: _decode_array(item, "p", np.float32) for key, item in (arrays.get("xf") or {}).items()}},
                   centerline={"xyz": center, "radius_mm": _decode_array(arrays, "cr", np.float32),
                               "edges": _decode_array(arrays, "ce", np.uint32, width=2),
                               "segment": _decode_array(arrays, "cs", seg_type("cs"))})
    else:
        from .volume_report import build_html
        pts = _decode_array(arrays, "pts", np.float32, width=3)
        vertices = _decode_array(arrays, "vertices", np.float32, width=3)
        faces = _decode_array(arrays, "faces", np.uint32, width=3)
        center = _decode_array(arrays, "center", np.float32, width=3, required=False)
        center = center if center is not None else np.empty((0, 3), np.float32)
        streamlines = []
        for line in arrays.get("streamlines") or []:
            streamlines.append({"points": _decode_array(line, "points", np.float32, width=3),
                                "speed_m_s": _decode_array(line, "speed_m_s", np.float32)})
        build_html(temp_path, meta,
                   mesh={"vertices": vertices, "faces": faces,
                         "pressure_pa": _decode_array(arrays, "wall_pressure_pa", np.float32, required=False),
                         "trust": _decode_array(arrays, "wall_trust", np.uint8, required=False)},
                   cloud={"pts": pts, "is_wall": _decode_array(arrays, "is_wall", np.uint8),
                          "segment": _decode_array(arrays, "segment", np.int32),
                          "pressure_pa": _decode_array(arrays, "pressure_pa", np.float32, required=False),
                          "velocity_m_s": _decode_array(arrays, "velocity_m_s", np.float32, width=3, required=False),
                          "trust": _decode_array(arrays, "trust", np.uint8, required=False),
                          "s_from_root_mm": _decode_array(arrays, "s_from_root_mm", np.float32, required=False),
                          "radius_mm": _decode_array(arrays, "radius_mm", np.float32, required=False),
                          "dist_to_wall_mm": _decode_array(arrays, "dist_to_wall_mm", np.float32, required=False)},
                   centerline={"xyz": center, "segment": _decode_array(arrays, "center_segment", np.int32, required=False),
                               "tangent": _decode_array(arrays, "tangent", np.float32, width=3, required=False),
                               "radius_mm": _decode_array(arrays, "center_radius_mm", np.float32, required=False),
                               "edges": _decode_array(arrays, "edges", np.uint32, width=2, required=False)},
                   streamlines=streamlines)
    refreshed = temp_path.read_text(encoding="utf-8")
    # Keep existing metadata/arrays exactly as embedded.  The only intended change is the surrounding
    # presentation template and inlined viewer code; this also makes the operation auditable.
    refreshed = _replace_embedded_json(refreshed, "wss-report-meta", meta_raw)
    refreshed = _replace_embedded_json(refreshed, array_id, arrays_raw)
    temp_path.write_text(refreshed, encoding="utf-8")
    if _embedded_json(refreshed, "wss-report-meta")[1] != meta or _embedded_json(refreshed, array_id)[1] != arrays:
        temp_path.unlink(missing_ok=True)
        raise ValueError("UI 刷新改变了报告中的科学数据")
    temp_path.replace(report_path)
    _refresh_manifest_report(job_dir)
    return {"job": job_dir.name, "report_bytes": report_path.stat().st_size, "ui_only": True}


def _previous_findings(meta: dict) -> dict:
    """The stored findings list and analysis version, kept before a rebuild re-derives them (review remap)."""
    findings = meta.get("findings") if isinstance(meta.get("findings"), dict) else {}
    items = [dict(item) for item in (findings.get("items") or []) if isinstance(item, dict)]
    return {"items": items, "analysis_version": meta.get("analysis_version")}


def remap_findings_review(job_dir: Path, meta: dict, previous: dict | None, record: dict | None = None) -> dict | None:
    """Move reviewer decisions of ``findings_review.json`` onto the re-derived findings ids (2026-09-30).

    Decisions are keyed by finding id; new listing rules renumber the list.  ``analysis.remap_review``
    matches every decided old finding by kind + position; unmatched decisions are kept under ``legacy``
    (「规则更新前的判定」) and never applied to another finding.  The sidecar (and the job record's copy) is
    rewritten only when something moved.  Returns the report of ``remap_review`` or None without a review.
    """
    path = Path(job_dir) / "findings_review.json"
    if previous is None or not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    from .schema import ANALYSIS_VERSION
    new_items = ((meta.get("findings") or {}).get("items") or []) if isinstance(meta.get("findings"), dict) else []
    remapped, report = A.remap_review(doc, previous.get("items"), new_items,
                                      from_version=previous.get("analysis_version"), to_version=ANALYSIS_VERSION)
    if remapped is not doc and isinstance(remapped, dict):
        atomic_json(path, remapped)
        if isinstance(record, dict):
            record["findings_review"] = remapped
    return report


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


def _finish(job_dir: Path, record: dict, meta: dict, outputs: list, extra: dict, *, previous: dict | None = None) -> dict:
    from .clock import now_iso
    from .schema import redact_paths, summary_display_name
    stamp = now_iso()
    meta = redact_paths(meta)     # v0.15.2: no server paths anywhere in summary / manifest / report META (strings only)
    meta["display_name"] = summary_display_name(meta)      # C7 (2026-09-30)
    meta["narrative"] = NARR.build_narrative(meta)
    # 2026-09-30: reviewer decisions follow their findings across the new listing rules (kind + position).
    review_report = remap_findings_review(job_dir, meta, previous, record)
    embed_sidecars(job_dir, meta)
    meta.setdefault("audit", {})["report_rebuilt_at"] = stamp
    # v0.14: a rebuild re-derives the analysis layer, so it records the analysis / code version it ran with.
    meta.update(summary_provenance())
    R.update_html_meta(job_dir / "report.html", report_meta(meta))
    atomic_json(job_dir / "summary.json", meta)
    audit_path = job_dir / "quality_audit.json"      # v0.15.1: the audit copy of model_release is redacted too
    if audit_path.is_file():
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if isinstance(audit, dict) and redact_paths(audit) != audit:
                atomic_json(audit_path, redact_paths(audit))
        except (OSError, ValueError):
            pass
    write_run_manifest(job_dir, meta, outputs=outputs)
    summary = record.get("summary")
    # The job record is owned by the running service (stop it, or restart afterwards).  Keep the few
    # summary mirrors it shows in step with the rebuilt summary: the conclusion text always, the export
    # set when streamlines were regenerated; ``findings_top`` is dropped so the service re-derives it.
    review_moved = bool(review_report and (review_report.get("moved") or review_report.get("legacy")))
    if review_moved:
        # remap_findings_review updated record["findings_review"]; the record must reach job.json either way.
        extra = {**extra, "findings_review": {"moved": review_report["moved"], "legacy": review_report["legacy"]}}
        if not isinstance(summary, dict):
            atomic_json(job_dir / "job.json", record)
    if isinstance(summary, dict):
        changed = summary.get("narrative") != meta.get("narrative") or "findings_top" in summary or review_moved
        if "narrative" in meta:
            summary["narrative"] = meta["narrative"]
        summary.pop("findings_top", None)
        if "model_release" in meta and summary.get("model_release") != meta["model_release"]:
            summary["model_release"] = meta["model_release"]     # v0.15.1: the mirror carries the redacted provenance
            changed = True
        if extra.get("regenerated"):
            if "exports" in meta: summary["exports"] = meta["exports"]
            changed = True
        if changed:
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
    parser.add_argument("--ui-only", action="store_true", help="只替换报告模板和内嵌查看器；不重算 summary、分析或流线")
    parser.add_argument("--force", action="store_true",
                        help="服务持有任务目录写锁时仍然执行（运行中的服务可能覆盖本命令写入的内容）")
    args = parser.parse_args(argv)
    root = Path(args.jobs_root)
    from .clock import configure as configure_clock
    configure_clock(root.resolve())      # O1: timestamps in the zone of this jobs root (service.json env.TZ)
    # v0.14: this command rewrites job.json / summary.json / report.html.  A running service keeps its jobs in
    # memory and holds <root>/.service.lock; refuse unless --force (the service itself rebuilds during
    # ``service upgrade --rebuild ID``).
    from .service import ServiceLockError, offline_lock
    try:
        lock = offline_lock(root, force=args.force, purpose="rebuild_report",
                            out=lambda text: print(text, file=sys.stderr))
    except ServiceLockError as exc:
        print(f"错误：{exc} 服务运行时请改用 `python -m wss_deploy.cli service upgrade --rebuild <任务ID>`，"
              "由服务在停机窗口内重建。", file=sys.stderr)
        return 2
    try:
        ids = args.job or [p.parent.name for p in sorted(root.glob("*/job.json"))
                           if json.loads(p.read_text(encoding="utf-8")).get("status") == "done" and (p.parent / "field.npz").is_file()
                           and (p.parent / "centerline" / "atlas.npz").is_file()]
        results, failed = [], []
        for job_id in ids:
            try:
                result = refresh_ui_only(root / job_id) if args.ui_only else rebuild(root / job_id, streamlines=not args.no_streamlines)
                results.append(result); print(f"[rebuild] {job_id}: {result}", file=sys.stderr, flush=True)
            except Exception as exc:
                failed.append(job_id); print(f"[rebuild] {job_id}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        if lock is not None:
            lock.release()
    print(json.dumps({"rebuilt": [r["job"] for r in results], "failed": failed}, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
