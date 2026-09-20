"""Offline Plotly review pages for the independent V6 geometry candidate.

Reads geometry only. It neither promotes candidate features nor modifies training.
The browser needs no server/network: data and a pinned local Plotly bundle are used.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import html
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree
from .anatomy_overrides import correct_report

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs/03-汇报材料/WSS_V6_新几何验收_20260909"
DEFAULT_CANDIDATE = ROOT / "outputs/wss_v6_geometry_candidate_20260909"
DEFAULT_SNAPSHOT = ROOT / "data_wss_v5/anatomy_pointcloud_v5_20260906"
COLORS = ["#2563eb", "#d97706", "#16a34a", "#9333ea", "#dc2626", "#0891b2", "#db2777"]
PLOTLY_ASSET = "plotly-2.35.2.min.js"


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dumps(value):
    return json.dumps(clean(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def array_attr(group, key, default):
    value = group.attrs.get(key, default)
    return json.loads(value) if isinstance(value, str) else value


def h5_geometry(path):
    """Pre-candidate geometry preview. No fabricated sections or approval status."""
    with h5py.File(path) as h:
        geom = h["geometry"]
        table = geom["atlas_table"][()]
        columns = array_attr(geom, "atlas_columns", [])
        col = lambda key: table[:, columns.index(key)]
        xyz = np.column_stack([col(f"{axis}_mm") for axis in "xyz"])
        old_segments = array_attr(geom, "atlas_segments", [])
        semantics = array_attr(geom, "semantic_of_segment", {})
        labels = array_attr(geom, "semantic_labels", {})
        segments = []
        for segment in old_segments:
            sid = int(segment["segment_id"])
            rows = np.flatnonzero(col("segment_id") == sid)
            rows = rows[np.argsort(col("s_local_mm")[rows])]
            semantic = semantics.get(str(sid), -1)
            role = "trunk" if segment.get("starts_at_root") else "terminal" if segment.get("ends_at_leaf") else "cia"
            segments.append(dict(sid=sid, parent_id=int(segment["parent_id"]), role=role,
                                 anatomy_label=labels.get(str(semantic), "unresolved"),
                                 cfd_zone_label=segment.get("outlet_name", ""),
                                 semantic_valid=False, semantic_reason="既有 atlas 标签，仅供候选人工核对",
                                 start_mm=xyz[rows[0]], end_mm=xyz[rows[-1]], length_mm=segment["length_mm"]))
        openings = []
        for label in h["interfaces"]:
            group = h[f"interfaces/{label}"]
            points = h["wall_static/xyz_mm"][()][group["rim_wall_rows"][()]]
            center = np.asarray(array_attr(group, "center_mm", [0, 0, 0]))
            normal = np.asarray(array_attr(group, "normal_out", [0, 0, 1]))
            e1, e2 = plane_frame(normal)
            if len(points):
                delta = points - center
                points = points[np.argsort(np.arctan2(delta @ e2, delta @ e1))]
            openings.append(dict(center_mm=center, normal=normal, area_mm2=float(group.attrs["area_m2"]) * 1e6,
                                 polygon_xyz_mm=points, label_provisional=label))
        z = dict(wall_xyz_mm=h["wall_static/xyz_mm"][()], wall_triangles=h["topology/wall_triangles"][()],
                 centerline_xyz_mm=xyz, centerline_segment_id=col("segment_id").astype(int),
                 centerline_s_local_mm=col("s_local_mm"), centerline_radius_mis_mm=col("radius_mm"),
                 centerline_tangent=np.column_stack([col(f"tangent_{axis}") for axis in "xyz"]),
                 wall_map_segment_id=h["wall_static/segment_id"][()], wall_map_s_local_mm=h["wall_static/s_local_mm"][()])
        cid = str(h.attrs["canonical_id"])
    report = dict(canonical_id=cid, segments=segments, openings=openings, stability=[],
                  awaiting_user_review=True, training_allowed=False,
                  surface_source="cfd_anatomy_wall_geometry", pointcloud_portability_proven=False,
                  preview_only=True)
    return z, report


def plane_frame(normal):
    normal = np.asarray(normal, dtype=float)
    normal /= max(np.linalg.norm(normal), 1e-12)
    hint = np.eye(3)[np.argmin(np.abs(normal))]
    e1 = np.cross(normal, hint)
    e1 /= max(np.linalg.norm(e1), 1e-12)
    return e1, np.cross(normal, e1)


def scatter3(points, name, color="#475569", mode="lines", **kwargs):
    points = np.asarray(points).reshape(-1, 3)
    trace = dict(type="scatter3d", x=points[:, 0], y=points[:, 1], z=points[:, 2], name=name, mode=mode)
    trace["line" if mode == "lines" else "marker"] = dict(color=color, **({"width": 5} if mode == "lines" else {"size": 4}))
    trace.update(kwargs)
    return trace


def scene_layout(title):
    return dict(title=title, height=680, margin=dict(l=0, r=0, b=0, t=45),
                scene=dict(aspectmode="data", camera=dict(eye=dict(x=2.0, y=2.0, z=1.5)), xaxis=dict(title="Fluent x / mm"),
                           yaxis=dict(title="Fluent y / mm"), zaxis=dict(title="Fluent z / mm")),
                legend=dict(x=0, y=1), uirevision="geometry")


def mesh_trace(z, max_faces=60000):
    vertices = z["wall_xyz_mm"]
    triangles = z["wall_triangles"].astype(int)
    if len(triangles) > max_faces:
        triangles = triangles[np.linspace(0, len(triangles) - 1, max_faces).astype(int)]
    used, inverse = np.unique(triangles, return_inverse=True)
    p = vertices[used]
    t = inverse.reshape(-1, 3)
    return dict(type="mesh3d", x=p[:, 0], y=p[:, 1], z=p[:, 2], i=t[:, 0], j=t[:, 1], k=t[:, 2],
                color="#94a3b8", opacity=0.12, name="解剖壁面（仅显示可抽样）", hoverinfo="skip", showscale=False)


def segment_id(segment):
    return int(segment.get("sid", segment.get("segment_id", -1)))


def overview(z, report):
    traces = [mesh_trace(z)]
    cxyz = z["centerline_xyz_mm"]
    csid = z["centerline_segment_id"]
    for segment in report.get("segments", []):
        sid = segment_id(segment)
        rows = np.flatnonzero(csid == sid)
        rows = rows[np.argsort(z["centerline_s_local_mm"][rows])]
        status = '标签已确认' if segment.get('anatomy_label_user_confirmed') else '候选/推导'
        name = f"S{sid} · {segment.get('role', '')} · {status} {segment.get('anatomy_label', '?')}"
        traces.append(scatter3(cxyz[rows], name, COLORS[sid % len(COLORS)]))
        if segment.get("role") == "cia":
            for endpoint, label in [("start_mm", "CIA起点"), ("end_mm", "CIA终点")]:
                if endpoint in segment:
                    traces.append(scatter3([segment[endpoint]], f"S{sid} {label}", COLORS[sid % 7], "markers+text",
                                           text=[f"S{sid} {label}"], textposition="top center", showlegend=False))
        if any(int(child.get("parent_id", -2)) == sid for child in report.get("segments", [])):
            traces.append(scatter3([cxyz[rows[-1]]], f"原图节点（S{sid}末端）", "#111827", "markers+text",
                                   text=[f"图节点 S{sid}"], textposition="bottom center", showlegend=False))
    for opening in report.get("openings", []):
        polygon = np.asarray(opening.get("polygon_xyz_mm", []))
        label = opening.get("anatomy_outlet_label", opening.get("label_provisional", opening.get("label", "未定开口")))
        if len(polygon):
            traces.append(scatter3(np.vstack([polygon, polygon[0]]), f"开口 {label}", "#111827"))
        traces.append(scatter3([opening["center_mm"]], str(label), "#111827", "markers+text", text=[str(label)], textposition="top center", showlegend=False))
    pc_inlet = report.get("inlet_pointcloud_opening", report.get("pc_inlet_geometry"))
    if pc_inlet and len(pc_inlet.get("polygon_xyz_mm", [])):
        polygon = np.asarray(pc_inlet["polygon_xyz_mm"])
        pc_inlet_valid = bool(report.get("inlet_pointcloud_valid", False))
        pc_color = "#9333ea" if pc_inlet_valid else "#64748b"
        traces.append(scatter3(np.vstack([polygon, polygon[0]]), "独立点云入口轮廓" if pc_inlet_valid else "独立点云入口（无效候选）", pc_color, line=dict(color=pc_color, width=6, dash="dash")))
        traces.append(scatter3([pc_inlet["center_mm"]], "点云入口中心", "#9333ea", "markers+text", text=["PC入口"], textposition="top center", showlegend=False))
    if "centerline_inside_reference" in z:
        bad = ~z["centerline_inside_reference"].astype(bool)
        bad &= z.get("centerline_audit_core_mask", np.ones(len(bad), bool)).astype(bool)
        if np.any(bad):
            traces.append(scatter3(cxyz[bad], "中心线核心区出界候选", "#ef4444", "markers", marker=dict(color="#ef4444", size=6, symbol="x")))
    return dict(data=traces, layout=scene_layout("透明壁面 · 七段中心线 · 五开口 · 分叉与 CIA 起止"))


def tree_plot(report):
    segments = report.get("segments", [])
    by_id = {segment_id(s): s for s in segments}
    children = {sid: sorted(segment_id(s) for s in segments if int(s.get("parent_id", -2)) == sid) for sid in by_id}
    positions = {}
    leaf_slot = [0]

    def place(sid, depth=0):
        if sid in positions:
            return positions[sid][0]
        if children[sid]:
            x = np.mean([place(child, depth + 1) for child in children[sid]])
        else:
            x = leaf_slot[0]
            leaf_slot[0] += 1
        positions[sid] = (x, -depth)
        return x

    for sid, segment in by_id.items():
        if int(segment.get("parent_id", -1)) not in by_id:
            place(sid)
    traces = []
    for sid, s in by_id.items():
        parent = int(s.get("parent_id", -1))
        if parent in positions and sid in positions:
            x0, y0 = positions[parent]
            x1, y1 = positions[sid]
            traces.append(dict(type="scatter", mode="lines", x=[x0, x1], y=[y0, y1], line=dict(color="#94a3b8"), hoverinfo="skip", showlegend=False))
    for sid, (x, y) in positions.items():
        s = by_id[sid]
        status = '已确认标签' if s.get('anatomy_label_user_confirmed') else '候选/推导'
        outlet = s.get('anatomy_outlet_label', s.get('cfd_zone_label', ''))
        label = f"S{sid} {s.get('role', '')}<br>{s.get('anatomy_label', '?')}（{status}）<br>{outlet}"
        traces.append(dict(type="scatter", mode="markers+text", x=[x], y=[y], text=[label], textposition="bottom center",
                           marker=dict(color=COLORS[sid % 7], size=16), showlegend=False,
                           hovertext=[f"长度 {s.get('length_mm', '?')} mm · {s.get('semantic_reason', '')}"], hoverinfo="text"))
    return dict(data=traces, layout=dict(title="拓扑树：图中左右排版不代表解剖左右", height=420,
                                        xaxis=dict(visible=False), yaxis=dict(visible=False, range=[-3, .4]), margin=dict(t=45, b=30)))


def polygon_at(z, prefix, k):
    offsets = z.get(prefix + "polygon_offsets")
    points = z.get(prefix + "polygon_xyz_mm")
    return np.empty((0, 3)) if offsets is None or points is None else points[int(offsets[k]):int(offsets[k + 1])]


def sections_payload(z):
    out = []
    n = len(z.get("section_center_mm", []))
    for k in range(n):
        center = z["section_center_mm"][k]
        normal = z["section_tangent"][k]
        e1, e2 = plane_frame(normal)
        selected = polygon_at(z, "section_", k)
        pc = polygon_at(z, "pc_section_", k)
        offsets = z.get("section_intersection_offsets")
        edges = z.get("section_intersection_segments_xyz_mm")
        raw = np.empty((0, 3))
        if offsets is not None and edges is not None:
            es = edges[int(offsets[k]):int(offsets[k + 1])]
            raw = np.full((len(es), 3, 3), np.nan)
            raw[:, :2, :] = es
            raw = raw.reshape(-1, 3)
        transform = lambda p: np.column_stack([(p - center) @ e1, (p - center) @ e2])
        item = dict(index=k, center=center, normal=normal, segment=int(z["section_segment_id"][k]), s=float(z["section_s_local_mm"][k]),
                    valid=bool(z["section_valid"][k]), reason=str(z["section_reason"][k]), raw=raw,
                    selected=selected, pc=pc, raw2=transform(raw), selected2=transform(selected), pc2=transform(pc))
        for key in ["area_mm2", "req_mm", "roundness", "eccentricity", "area_slope_per_mm"]:
            item[key] = float(z["section_" + key][k]) if "section_" + key in z else None
            item["pc_" + key] = float(z["pc_section_" + key][k]) if "pc_section_" + key in z else None
        item["pc_valid"] = bool(z["pc_section_valid"][k]) if "pc_section_valid" in z else None
        item["pc_reason"] = str(z["pc_section_reason"][k]) if "pc_section_reason" in z else "尚未提供点云截面"
        item["raw_contour_valid"] = bool(z["section_raw_contour_valid"][k]) if "section_raw_contour_valid" in z else None
        item["pc_raw_contour_valid"] = bool(z["pc_section_raw_contour_valid"][k]) if "pc_section_raw_contour_valid" in z else None
        out.append(item)
    return out


def profiles(z):
    plots = []
    specs = [("area_mm2", "截面积 A / mm²"), ("req_mm", "等面积半径 Req 与 MIS 半径 / mm"),
             ("area_slope_per_mm", "dlog(A)/ds / mm⁻¹"), ("roundness", "圆度 4πA/P²"), ("eccentricity", "中心线偏心度 |截面质心−中心线| / Req")]
    for key, title in [("upstream_min_area_ratio", "同分支上游窗口最小面积比"), ("downstream_min_area_ratio", "同分支下游窗口最小面积比"),
                       ("upstream_min_distance_mm", "到上游窗口最小面积站点距离 / mm"), ("downstream_min_distance_mm", "到下游窗口最小面积站点距离 / mm")]:
        if "section_" + key in z:
            specs.append((key, title))
    for key, title in specs:
        traces = []
        for sid in np.unique(z.get("section_segment_id", [])):
            sid = int(sid)
            rows = np.flatnonzero(z["section_segment_id"] == sid)
            rows = rows[np.argsort(z["section_s_local_mm"][rows])]
            for prefix, suffix, dash in [("section_", "面几何", "solid"), ("pc_section_", "点云候选", "dot")]:
                if prefix + key not in z:
                    continue
                values = z[prefix + key][rows].astype(float).copy()
                valid = z.get(prefix + "valid", np.ones(len(z["section_segment_id"]), bool))[rows]
                values[~valid.astype(bool)] = np.nan
                traces.append(dict(type="scatter", x=z["section_s_local_mm"][rows], y=values, mode="lines+markers",
                                   marker=dict(size=3), name=f"S{sid} {suffix}", line=dict(color=COLORS[sid % 7], dash=dash)))
            if key == "req_mm":
                cr = np.flatnonzero(z["centerline_segment_id"] == sid)
                traces.append(dict(type="scatter", x=z["centerline_s_local_mm"][cr], y=z["centerline_radius_mis_mm"][cr],
                                   mode="lines", name=f"S{sid} Rmis", line=dict(color=COLORS[sid % 7], dash="dash")))
        plots.append(dict(data=traces, layout=dict(title=title, height=340, xaxis=dict(title="段内弧长 s / mm"),
                                                  yaxis=dict(title=title), margin=dict(l=65, r=20, t=45, b=45))))
    return plots


def wall_maps(z):
    xyz = z["wall_xyz_mm"]
    rows = np.linspace(0, len(xyz) - 1, min(len(xyz), 15000)).astype(int)
    p = xyz[rows]
    maps = {}
    specs = [("wall_map_segment_id", "壁面分支映射", "Turbo"), ("wall_map_area_mm2", "面几何参考：壁面映射截面积 / mm²", "Viridis"),
             ("wall_map_valid", "面几何映射 valid：1有效 / 0无效", "RdYlGn"),
             ("wall_map_pc_area_mm2", "点云候选：壁面映射截面积 / mm²", "Viridis"),
             ("wall_map_pc_valid", "点云映射 valid：1有效 / 0无效", "RdYlGn"),
             ("wall_map_ambiguous", "分支归属 ambiguous：1歧义", "YlOrRd")]
    for prefix, source in [("wall_map_", "面几何"), ("wall_map_pc_", "点云")]:
        for key, title in [("area_slope_per_mm", "dlogA/ds"), ("roundness", "圆度"), ("eccentricity", "中心线偏心度"),
                           ("upstream_min_area_ratio", "上游最小面积比"), ("downstream_min_area_ratio", "下游最小面积比")]:
            if prefix + key in z:
                specs.append((prefix + key, f"{source}：{title}", "Viridis"))
    for key in sorted(z):
        if key.startswith("wall_cia_") and np.asarray(z[key]).shape == (len(xyz),):
            specs.append((key, "CIA候选：" + key.removeprefix("wall_cia_"), "Viridis"))
    for key, title, colorscale in specs:
        if key not in z:
            continue
        value = np.asarray(z[key])[rows].astype(float)
        if key == "wall_map_segment_id":
            traces = [scatter3(p[value == sid], f"S{int(sid)}", COLORS[int(sid) % 7], "markers", marker=dict(size=2.5, color=COLORS[int(sid) % 7])) for sid in np.unique(value)]
            maps[key] = dict(data=traces, layout=scene_layout(title + "（与中心线相同颜色）"))
            continue
        if key == "wall_map_area_mm2" and "wall_map_valid" in z:
            value[~np.asarray(z["wall_map_valid"])[rows].astype(bool)] = np.nan
        if key == "wall_map_pc_area_mm2" and "wall_map_pc_valid" in z:
            value[~np.asarray(z["wall_map_pc_valid"])[rows].astype(bool)] = np.nan
        if key + "_valid" in z:
            value[~np.asarray(z[key + "_valid"])[rows].astype(bool)] = np.nan
        if key.startswith("wall_cia_") and not key.endswith("_valid") and "wall_cia_geometry_valid" in z:
            value[~np.asarray(z["wall_cia_geometry_valid"])[rows].astype(bool)] = np.nan
        finite = np.isfinite(value)
        trace = scatter3(p[finite], title, mode="markers", marker=dict(size=2.5, color=value[finite], colorscale=colorscale, showscale=True, colorbar=dict(title=title)))
        traces = [trace]
        if np.any(~finite):
            traces.append(scatter3(p[~finite], "无效 / 未赋面积", "#94a3b8", "markers", marker=dict(size=2, color="#94a3b8", opacity=.4)))
        maps[key] = dict(data=traces, layout=scene_layout(title + "（仅显示抽样，mask不强填）"))
    return maps


def stability_plots(report):
    items = report.get("stability", [])
    if isinstance(items, dict):
        items = items.get("sections", items.get("probes", []))
    groups = {}
    same_source_groups = {}
    density_x, density_y = [], []
    rows = []
    for item in items:
        label = f"S{item.get('segment_id', '?')} @ {float(item.get('s_local_mm', 0)):.1f}mm"
        for variant in item.get("variants", []):
            name = variant.get("name", "unknown")
            delta = variant.get("relative_area_change")
            accepted = bool(variant.get("comparison_valid", variant.get("pipeline_valid", variant.get("valid", False))))
            groups.setdefault(name, []).append((label, 100 * delta if delta is not None and accepted else None))
            same_source_delta = variant.get("relative_to_same_source_baseline")
            if same_source_delta is not None:
                same_source_groups.setdefault(name, []).append((label, 100 * same_source_delta if accepted else None))
            rows.append(dict(section=label, reference_area_mm2=item.get("reference_area_mm2"),
                             pc_density_relative_change=item.get("pc_density_relative_change"), **variant))
        density_x.append(label)
        density_change = item.get("pc_density_relative_change")
        density_y.append(100 * density_change if density_change is not None and item.get("pc_density_comparison_valid", True) else None)
    traces = [dict(type="scatter", mode="markers+lines", name=name, x=[i[0] for i in vals], y=[i[1] for i in vals]) for name, vals in groups.items()]
    plot = dict(data=traces, layout=dict(title="稳定性：相对面几何基准截面积变化（%）", height=480,
                                       yaxis=dict(title="ΔA / Aref (%)"), margin=dict(t=50, b=150)))
    same_plot = dict(data=[dict(type="scatter", mode="markers+lines", name=name, x=[i[0] for i in vals], y=[i[1] for i in vals]) for name, vals in same_source_groups.items()],
                     layout=dict(title="位移/倾角：相对各自来源的未扰动面积（%）", height=430, yaxis=dict(title="ΔA / 同源基准 A (%)"), margin=dict(t=50, b=150)))
    density_plot = dict(data=[dict(type="scatter", mode="markers", name="点云 half vs full", x=density_x, y=density_y)],
                        layout=dict(title="点云密度：half 相对 full（仅双方pipeline有效）", height=360, yaxis=dict(title="(Ahalf / Afull − 1) × 100 (%)"), margin=dict(t=50, b=150)))
    return plot, rows, same_plot, density_plot


def coverage(z, cid, fixed_radii, support_n=5000):
    xyz = z["wall_xyz_mm"]
    rng = np.random.default_rng(1234)
    rows = np.sort(rng.choice(len(xyz), min(support_n, len(xyz)), replace=False))
    p = xyz[rows]
    bundle = ROOT / "data_wss_v5/views/wss_min_view_v1" / cid / "bundle.npz"
    if not bundle.is_file():
        return dict(available=False, reason="缺少原训练视图 coord_scale；不估造归一化半径", plots=[])
    with np.load(bundle, allow_pickle=False) as b:
        scale = float(b["coord_scale"])
    norm_radii = np.asarray([.015, .03, .06]) * scale
    tree = cKDTree(p)
    records = []
    traces = []
    maps = []
    for kind, radii, color in [("当前归一半径", norm_radii, "#2563eb"), ("固定mm候选", fixed_radii, "#d97706")]:
        for j, radius in enumerate(radii):
            counts = tree.query_ball_point(p, float(radius), return_length=True, workers=1)
            records.append(dict(mode=kind, scale=j + 1, radius_mm=float(radius), points=len(p),
                                count_p10=float(np.percentile(counts, 10)), count_median=float(np.median(counts)),
                                below16_fraction=float(np.mean(counts < 16)), self_only_fraction=float(np.mean(counts <= 1))))
            traces.append(dict(type="box", y=np.minimum(counts, 64), name=f"{kind} {j + 1}<br>{radius:.2f} mm", boxpoints=False,
                               marker=dict(color=color), hovertemplate="邻居数（显示上限64）=%{y}<extra>%{fullData.name}</extra>"))
            if j == 0:
                maps.append(dict(data=[scatter3(p, kind, mode="markers", marker=dict(size=2.5, color=np.minimum(counts, 16),
                                                cmin=1, cmax=16, colorscale="Viridis", showscale=True))],
                                 layout=scene_layout(f"{kind} 第一尺度 {radius:.2f} mm：有效邻居≤16")))
    plots = [dict(data=traces, layout=dict(title="同一随机 support：邻居覆盖比较（含自身；网络邻居上限16）", height=420,
                                         yaxis=dict(title="候选邻居数，显示裁剪到64"), margin=dict(t=50, b=90)))] + maps
    return dict(available=True, coord_scale_mm=scale, seed=1234, support_points=len(p), fixed_radii_mm=fixed_radii,
                normalized_radii_mm=norm_radii, note="诊断随机点集，两方式完全同点；非正式训练采样重放。固定mm仅候选，未训练。", records=records, plots=plots)


def bifurcation_payload(diagnostic):
    """Compact, precomputed local slices; no geometric computation in browser."""
    result = []
    if not diagnostic:
        return result
    for j in diagnostic.get('junctions', []):
        scan = next((s for s in j.get('scans', []) if s['mode'] == 'daughter'), None)
        if scan is None:
            continue
        records = []
        for r in scan['records']:
            if 'selected_contours_xyz_mm' not in r:
                continue
            c = np.asarray(r['center_mm'])
            e1, e2 = plane_frame(r['normal'])
            def project(p):
                p = np.asarray(p).reshape(-1, 3) - c
                return np.column_stack((p @ e1, p @ e2))
            loops = r['selected_contours_xyz_mm']
            records.append({**r, 'loops2': [project(p) for p in loops],
                            'daughter_points2': project(r.get('daughter_points_mm', []))})
        result.append(dict(parent_segment=j['parent_segment'], daughter_segments=j['daughter_segments'],
                           graph_node_mm=j['graph_node_mm'], candidate=scan.get('candidate'),
                           status=scan.get('status', 'candidate' if scan.get('candidate') else 'no_candidate'),
                           records=records, scan_states=[dict(s_mm=r['s_mm'], state=r['state']) for r in scan['records']]))
    return result


CSS = """
body{margin:0;background:#f1f5f9;color:#172033;font:15px/1.65 system-ui,-apple-system,'Noto Sans CJK SC',sans-serif}
header,main{max-width:1500px;margin:auto;padding:22px 28px}header{padding-bottom:8px}h1{font-size:27px;margin:0}h2{font-size:20px}
a{color:#1d4ed8}nav{display:flex;gap:8px;flex-wrap:wrap;position:sticky;top:0;background:#f1f5f9;padding:10px 0;z-index:4}
button,select,input{font:inherit;padding:7px 12px;border:1px solid #cbd5e1;border-radius:7px;background:white}button{cursor:pointer}select{max-width:100%}
button.active{background:#1e40af;color:white}.card{background:white;padding:18px 22px;border-radius:12px;margin:14px 0;border:1px solid #e2e8f0}
.notice{background:#fffbeb;border-left:4px solid #d97706;padding:12px 18px}.muted{color:#64748b}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}
.plot{width:100%;min-height:340px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #e2e8f0;vertical-align:top}th{background:#f8fafc;position:sticky;top:0}
.table-wrap{overflow:auto;max-height:620px}pre{white-space:pre-wrap;overflow-wrap:anywhere}code{overflow-wrap:anywhere}.pill{display:inline-block;background:#e2e8f0;border-radius:20px;padding:3px 10px;margin:3px}.bad{color:#b91c1c}.good{color:#166534}.pane[hidden]{display:none}
.section-controls{position:sticky;top:var(--review-nav-height,65px);z-index:3;background:#fff;border:1px solid #cbd5e1;border-radius:8px;padding:10px 12px;box-shadow:0 3px 8px #17203312}
.section-controls .control-row{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.section-controls select{flex:1;min-width:180px}.section-controls input[type=range]{flex:1;min-width:140px;padding:0;accent-color:#1e40af}.section-controls input[type=number]{width:75px}.section-controls p{margin:4px 0;font-size:13px}.section-controls button:disabled{opacity:.4;cursor:default}#section-status{font-weight:600}.section-help{margin:10px 0}.section-controls label{white-space:nowrap}
@media(max-width:650px){header,main{padding-left:12px;padding-right:12px}.card{padding:12px}.grid{grid-template-columns:minmax(0,1fr)}nav button{padding:6px 8px;font-size:13px}}
"""


JS = """
const config={responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}};
function plot(id,p){Plotly.newPlot(id,p.data,p.layout,config)}
function table(items){if(!items.length)return '<p class="muted">未提供此项数据。</p>';let keys=[...new Set(items.flatMap(Object.keys))];return '<div class="table-wrap"><table><thead><tr>'+keys.map(k=>'<th>'+esc(k)+'</th>').join('')+'</tr></thead><tbody>'+items.map(r=>'<tr>'+keys.map(k=>'<td>'+esc(typeof r[k]==='object'?JSON.stringify(r[k]):r[k]??'—')+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>'}
function esc(x){return String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function line3(p,name,color){let q=p.length?[...p,p[0]]:[];return {type:'scatter3d',mode:'lines',x:q.map(x=>x[0]),y:q.map(x=>x[1]),z:q.map(x=>x[2]),name,line:{color,width:6}}}
function line2(p,name,color,close=true){let q=close&&p.length?[...p,p[0]]:p;return {type:'scatter',mode:'lines',x:q.map(x=>x[0]),y:q.map(x=>x[1]),name,line:{color,width:2}}}
let sectionPending=null,sectionWork=null,currentSection=0;
function sectionControls(k){for(const id of ['section-select','section-range','section-number'])document.getElementById(id).value=String(k);document.getElementById('section-prev').disabled=k===0;document.getElementById('section-next').disabled=k===D.sections.length-1}
function showSection(k){if(!D.sections.length||!Number.isFinite(Number(k)))return Promise.resolve();sectionPending=Math.max(0,Math.min(D.sections.length-1,Math.trunc(Number(k))));sectionControls(sectionPending);
 if(!sectionWork)sectionWork=(async()=>{try{while(sectionPending!==null){let target=sectionPending;sectionPending=null;await renderSection(target)}}catch(error){document.getElementById('section-status').textContent='切换失败：'+error.message;console.error(error)}finally{sectionWork=null;document.getElementById('section-status').setAttribute('aria-busy','false')}})();return sectionWork}
async function renderSection(k){let s=D.sections[k];if(!s)return;document.getElementById('section-status').setAttribute('aria-busy','true');document.getElementById('section-status').textContent='正在切换到 #'+k+'…';let raw={type:'scatter3d',mode:'lines',x:s.raw.map(p=>p[0]),y:s.raw.map(p=>p[1]),z:s.raw.map(p=>p[2]),name:'全部交线',line:{color:'#f59e0b',width:3}};
 let refName=s.valid?'面几何有效轮廓':'面几何无效候选',pcName=s.pc_valid?'点云有效候选':'点云无效候选',refColor=s.valid?'#16a34a':'#94a3b8',pcColor=s.pc_valid?'#9333ea':'#64748b';
 let data=[D.overview.data[0],raw,line3(s.selected,refName,refColor),line3(s.pc,pcName,pcColor),{type:'scatter3d',mode:'markers',x:[s.center[0]],y:[s.center[1]],z:[s.center[2]],name:'中心线站点',marker:{color:'#dc2626',size:5}}];
 await Promise.all([Plotly.react('section3',data,{...structuredClone(D.overview.layout),uirevision:'section-camera',title:'截面 #'+s.index+' · S'+s.segment+' · s='+s.s.toFixed(1)+' mm'},config),Plotly.react('section2',[line2(s.raw2,'全部交线','#f59e0b',false),line2(s.selected2,refName,refColor),line2(s.pc2,pcName,pcColor)],{title:'截面 #'+s.index+' · 局部平面（mm）',height:600,xaxis:{title:'局部 e1 / mm'},yaxis:{title:'局部 e2 / mm',scaleanchor:'x',scaleratio:1},margin:{t:45}},config)]);
 currentSection=k;if(sectionPending===null)sectionControls(k);document.getElementById('section-meta').innerHTML=table([{index:s.index,segment:s.segment,s_mm:s.s,valid:s.valid,reason:s.reason,A_valid_mm2:s.valid?s.area_mm2:null,raw_candidate_A_mm2:s.area_mm2,Req_valid_mm:s.valid?s.req_mm:null,pc_valid:s.pc_valid,pc_reason:s.pc_reason,pc_A_valid_mm2:s.pc_valid?s.pc_area_mm2:null}]);document.getElementById('section-status').textContent='已显示 #'+s.index+' / '+(D.sections.length-1)+' · 分支 S'+s.segment+' · 沿分支 '+s.s.toFixed(1)+' mm · 面几何'+(s.valid?'有效':'无效')+' / 点云'+(s.pc_valid?'有效':'无效');}
const bifNames={one:'同一管腔',two:'两支独立管腔',ambiguous:'无法唯一判定'};
let bifPending=null,bifWork=null,currentBif=null;
function bifControls(j,k){let r=D.bifurcations[j].records;document.getElementById('bif-junction').value=j;for(const id of ['bif-range','bif-number']){let e=document.getElementById(id);e.max=Math.max(0,r.length-1);e.value=k;e.disabled=!r.length}document.getElementById('bif-prev').disabled=k===0;document.getElementById('bif-next').disabled=k>=r.length-1}
function chooseBif(j){let a=D.bifurcations[j];let k=a.candidate?a.records.findIndex(r=>Math.abs(r.s_mm-a.candidate.s_mm)<1e-6):0;return showBif(j,Math.max(0,k))}
function showBif(j,k){let a=D.bifurcations[j];if(!a)return Promise.resolve();if(!a.records.length){bifControls(j,0);document.getElementById('bif-status').textContent='当前分叉未得到可用切片：'+a.status;for(const id of ['bif3','bif2'])Plotly.purge(id);return Promise.resolve()}k=Math.max(0,Math.min(a.records.length-1,Math.trunc(k)));bifControls(j,k);bifPending=[j,k];if(!bifWork)bifWork=(async()=>{try{while(bifPending){const v=bifPending;bifPending=null;await renderBif(...v)}}catch(e){document.getElementById('bif-status').textContent='切换失败：'+e.message;console.error(e)}finally{bifWork=null}})();return bifWork}
async function renderBif(j,k){const a=D.bifurcations[j],r=a.records[k],pp=r.daughter_points_mm,qq=r.daughter_points2;const marker=(p,name,color)=>({type:'scatter3d',mode:'markers',x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2]),name,marker:{color,size:5}});const traces=[...D.overview.data.filter(t=>t.type==='mesh3d'||(t.name||'').startsWith('S')),marker([a.graph_node_mm],'原图节点','#111827'),marker([r.center_mm],'当前切面中心','#c026d3'),marker(pp,'两子支真实穿面点','#dc2626')];r.selected_contours_xyz_mm.forEach((p,i)=>traces.push(line3(p,'包含子支的轮廓 '+(i+1),['#16a34a','#2563eb'][i%2])));const flat=r.loops2.map((p,i)=>line2(p,'包含子支的轮廓 '+(i+1),['#16a34a','#2563eb'][i%2]));flat.push({type:'scatter',mode:'markers+text',x:qq.map(p=>p[0]),y:qq.map(p=>p[1]),text:a.daughter_segments.map(v=>'S'+v),textposition:'top center',name:'子支穿面点',marker:{color:'#dc2626',size:9}});const title='J'+a.parent_segment+' · 切片 '+k+' · s='+r.s_mm.toFixed(2)+' mm · '+bifNames[r.state];await Promise.all([Plotly.react('bif3',traces,{...structuredClone(D.overview.layout),title,uirevision:'bif-'+j},config),Plotly.react('bif2',flat,{title,height:600,xaxis:{title:'局部 e1 / mm'},yaxis:{title:'局部 e2 / mm',scaleanchor:'x',scaleratio:1},margin:{t:55}},config)]);currentBif=[j,k];if(!bifPending)bifControls(j,k);document.getElementById('bif-status').textContent=title+(a.candidate?' · 候选区间 '+a.candidate.bracket_mm.map(v=>v.toFixed(2)).join('–')+' mm':' · 本处未确定过渡区间');}
function initBif(){let a=D.bifurcations||[];if(!a.length){document.getElementById('bif-status').textContent='尚未计算此病例的快速定位';return}const sel=document.getElementById('bif-junction');sel.innerHTML=a.map((j,i)=>'<option value="'+i+'">J'+j.parent_segment+'：S'+j.parent_segment+' → S'+j.daughter_segments.join(' / S')+(j.candidate?' · 候选 '+j.candidate.s_mm.toFixed(2)+' mm':' · 未确定')+'</option>').join('');sel.onchange=()=>chooseBif(+sel.value);document.getElementById('bif-range').oninput=e=>showBif(+sel.value,+e.target.value);const num=document.getElementById('bif-number');num.onchange=()=>{if(num.value!=='')showBif(+sel.value,+num.value)};num.onkeydown=e=>{if(e.key==='Enter'&&num.value!=='')showBif(+sel.value,+num.value)};document.getElementById('bif-prev').onclick=()=>showBif(+sel.value,+num.value-1);document.getElementById('bif-next').onclick=()=>showBif(+sel.value,+num.value+1);document.getElementById('bif-candidate').onclick=()=>chooseBif(+sel.value);document.getElementById('bif-table').innerHTML=table(a.map(j=>({分叉:'J'+j.parent_segment,子支:j.daughter_segments.join('/'),区间_mm:j.candidate?.bracket_mm??'未确定',状态:j.status})));chooseBif(0)}
let done={};function showTab(name){document.querySelectorAll('.pane').forEach(p=>p.hidden=p.id!==name);document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('active',b.dataset.tab===name));if(!done[name]){done[name]=true;
 if(name==='overview'){plot('overview-plot',D.overview);plot('tree-plot',D.tree)}
 if(name==='bifurcation')initBif();
 if(name==='sections'&&D.sections.length){let sel=document.getElementById('section-select');sel.innerHTML=D.sections.map((s,i)=>'<option value="'+i+'">#'+i+' S'+s.segment+' s='+s.s.toFixed(1)+'mm · REF '+(s.valid?'valid':'INVALID '+esc(s.reason))+' · PC '+(s.pc_valid?'valid':'INVALID '+esc(s.pc_reason))+'</option>').join('');sel.onchange=()=>showSection(+sel.value);let slider=document.getElementById('section-range'),number=document.getElementById('section-number');slider.max=number.max=String(D.sections.length-1);slider.oninput=()=>showSection(+slider.value);number.onchange=()=>{if(number.value!=='')showSection(+number.value)};number.onkeydown=e=>{if(e.key==='Enter'&&number.value!=='')showSection(+number.value)};document.getElementById('section-prev').onclick=()=>showSection(Number(slider.value)-1);document.getElementById('section-next').onclick=()=>showSection(Number(slider.value)+1);showSection(0)}
 if(name==='profiles')D.profiles.forEach((p,i)=>plot('profile-'+i,p));
 if(name==='mapping'){let sel=document.getElementById('map-select');sel.innerHTML=Object.entries(D.maps).map(([k,p])=>'<option value="'+k+'">'+esc(p.layout.title)+'</option>').join('');sel.onchange=()=>plot('map-plot',D.maps[sel.value]);if(sel.value)sel.onchange()}
 if(name==='stability'){plot('stability-plot',D.stability);plot('stability-same-source',D.stability_same_source);plot('stability-density',D.stability_density);document.getElementById('stability-table').innerHTML=table(D.stability_rows);document.getElementById('opening-stability').innerHTML=table(D.report.opening_stability||[])}
 if(name==='coverage'){D.coverage.plots.forEach((p,i)=>plot('coverage-'+i,p));document.getElementById('coverage-table').innerHTML=table(D.coverage.records||[])}
 }setTimeout(()=>window.dispatchEvent(new Event('resize')),50)}
function initialize(){document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>showTab(b.dataset.tab));document.getElementById('segments-table').innerHTML=table(D.report.segments||[]);document.getElementById('openings-table').innerHTML=table((D.report.openings||[]).map(({polygon_xyz_mm,...r})=>r));document.getElementById('load-status').textContent='离线数据已加载';const nav=document.querySelector('nav');new ResizeObserver(()=>document.documentElement.style.setProperty('--review-nav-height',nav.getBoundingClientRect().height+'px')).observe(nav);showTab('overview')}
"""


def case_html(payload, asset_prefix="../assets/"):
    cid = html.escape(payload["canonical_id"])
    preview = payload["report"].get("preview_only", False)
    tabs = [("overview", "整体与拓扑"), ("bifurcation", "分叉快速定位"), ("sections", "截面逐站检查"), ("profiles", "沿程几何曲线"),
            ("mapping", "壁面映射与mask"), ("stability", "扰动与密度稳定性"), ("coverage", "M6邻域覆盖候选")]
    nav = "".join(f'<button data-tab="{key}">{name}</button>' for key, name in tabs)
    profile_divs = "".join(f'<div class="card"><div class="plot" id="profile-{i}"></div></div>' for i in range(len(payload["profiles"])))
    coverage_divs = "".join(f'<div class="card"><div class="plot" id="coverage-{i}"></div></div>' for i in range(len(payload["coverage"]["plots"])))
    missing = '<p class="notice">当前仅有既有几何预览；截面、点云可移植性和稳定性尚未计算，不表示通过验收。</p>' if preview else ''
    compressed = base64.b64encode(gzip.compress(dumps(payload).encode("utf-8"), compresslevel=6, mtime=0)).decode("ascii")
    n_sections = len(payload["sections"])
    valid_sections = sum(s["valid"] for s in payload["sections"])
    pc_valid_sections = sum(bool(s["pc_valid"]) for s in payload["sections"])
    unresolved = [f"S{segment_id(s)}: {s.get('semantic_reason', 'unknown')}" for s in payload["report"].get("segments", []) if not s.get("semantic_valid", False)]
    stats = f'面几何截面 {valid_sections}/{n_sections} 有效 · 点云截面 {pc_valid_sections}/{n_sections} 有效' if n_sections else '截面候选尚未生成'
    if unresolved:
        stats += '<br>语义待核查：' + html.escape('；'.join(unresolved))
    inlet_info = ''
    if not preview:
        ref_area = payload['report'].get('inlet_reference_area_mm2')
        pc_area = payload['report'].get('inlet_pointcloud_area_mm2')
        format_area = lambda area: '缺失' if area is None or not np.isfinite(area) else f'{area:.2f} mm²'
        pc_display = format_area(pc_area) if payload['report'].get('inlet_pointcloud_valid') else f'无效（原候选 {format_area(pc_area)}）'
        inlet_info = f"<p>入口面积：面几何参考 {format_area(ref_area)} · 独立点云 {pc_display} · PC valid={payload['report'].get('inlet_pointcloud_valid')} ({html.escape(str(payload['report'].get('inlet_pointcloud_reason', '')))})</p>"
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{cid} · V6 几何候选验收</title><style>{CSS}</style><script src="{asset_prefix}{PLOTLY_ASSET}"></script>
<header><a href="../index.html">← 全队列索引</a><h1>{cid}</h1><p>{stats}</p><p class="muted">待人工确认 · 未接入训练 · Fluent 坐标 / mm · 颜色编码分支 ID，画面左右不代表解剖左右</p><p class="muted" id="load-status">正在加载离线几何数据…</p>{missing}</header>
<main><nav>{nav}</nav>
<section class="pane" id="overview"><div class="card"><p>自动诊断：{html.escape(', '.join(payload['report'].get('warnings', [])) or '未报告警告；仍待人工确认')}</p>{inlet_info}<div class="plot" id="overview-plot"></div></div><div class="card"><div id="tree-plot"></div></div><div class="card"><h2>七段定义与候选语义</h2><div id="segments-table"></div><h2>五开口</h2><div id="openings-table"></div></div></section>
<section class="pane" id="bifurcation" hidden><div class="card"><div class="section-controls"><div class="control-row"><label for="bif-junction">选择分叉</label><select id="bif-junction"></select><button id="bif-candidate">跳到候选位置</button></div><div class="control-row"><button id="bif-prev">← 前一片</button><label for="bif-number">局部切片编号</label><input id="bif-number" type="number" min="0" step="1" value="0"><button id="bif-next">后一片 →</button><input id="bif-range" type="range" min="0" step="1" value="0" aria-label="分叉局部切片"></div><p id="bif-status" role="status">加载后选择分叉</p></div><p>先选 J0（主动脉）或 J1/J2（髂总动脉末端），点击“跳到候选位置”，再用滑块检查前后切片。红点为两条子支与同一切面的真实交点；绿色和蓝色为包含它们的闭合轮廓。重点看同一个轮廓是否持续分成两个、且每个各包住一条子支。</p><p class="muted">s 是两子支原局部弧长的配对扫描参数；局部切片编号与“截面逐站检查”的编号不同。黑点为原中心线图节点，紫点为当前切面中心。候选用于快速找位置，中心线分段和 CIA 长度尚未按它重建；切面中心也不是壁面分流嵴。</p><div class="grid"><div class="plot" id="bif3"></div><div class="plot" id="bif2"></div></div><div id="bif-table"></div></div></section>
<section class="pane" id="sections" hidden><div class="card"><div class="section-controls" data-ui-version="20260910"><div class="control-row"><button id="section-prev" aria-label="前一站">← 前一站</button><label for="section-number">站点编号</label><input id="section-number" type="number" min="0" step="1" value="0"><button id="section-next" aria-label="后一站">后一站 →</button><input id="section-range" type="range" min="0" step="1" value="0" aria-label="切换截面站点"></div><div class="control-row"><label for="section-select">按分支选站</label><select id="section-select"></select></div><p id="section-status" role="status" aria-live="polite">选择站点后同步更新左右两图</p></div><p class="section-help">拖动上方站点滑块，或输入编号后按回车；左右两图会同步切换。编号从0开始。拖动右图只平移视图，不切站；滚动表格只查看字段。</p><p class="muted">左图定位切面在血管中的位置，右图查看该切面的全部交线与轮廓。橙色：全部交线；绿色：面几何有效轮廓；紫色：点云有效轮廓；灰色：无效候选。第0站常位于入口排除区，灰色不代表零面积。</p><div id="section-meta"></div><div class="grid"><div class="plot" id="section3"></div><div class="plot" id="section2"></div></div></div></section>
<section class="pane" id="profiles" hidden><p>实线为面几何参考，点线为点云候选；无效站点留断点。Rmis 是原中心线半径，Req = √(A/π)；二者分别展示。坡度定义为 dlog(A)/ds。</p>{profile_divs}</section>
<section class="pane" id="mapping" hidden><div class="card"><p>显示抽样不参与计算。无效面积不作插值强填；分支归属与面积有效性分别检查。</p><select id="map-select"></select><div class="plot" id="map-plot"></div></div></section>
<section class="pane" id="stability" hidden><div class="card"><p>位移 ±1 mm、倾角 ±5° 按各自来源对比；点云 full/half 单独检查密度敏感性。曲线只纳入 comparison_valid；原始闭合与完整pipeline门控在下表分列。</p><div class="plot" id="stability-same-source"></div><div class="plot" id="stability-density"></div><div class="plot" id="stability-plot"></div><div id="stability-table"></div><h2>开口稳定性</h2><div id="opening-stability"></div></div></section>
<section class="pane" id="coverage" hidden><div class="card"><p class="notice">M6 仅邻域覆盖可视化，未训练。固定 mm 三尺度为待确认候选；同一 seed 的随机 support 只用于两方式同点比较，不宣称重放正式训练采样。</p><p>当前病例 coord_scale = {payload['coverage'].get('coord_scale_mm', '缺失')} mm。两组半径和覆盖率见下表。</p><div id="coverage-table"></div></div>{coverage_divs}</section>
<div class="card muted"><p>surface_source: {html.escape(str(payload['report'].get('surface_source', '未注明')))} · pointcloud_portability_proven: {html.escape(str(payload['report'].get('pointcloud_portability_proven', False)))}</p><p>完整候选与审计字段以源 NPZ/JSON 为准。图形仅按需显示，所有数据已嵌入本页；Plotly 从相邻 assets 离线加载。</p></div></main>
<script>let D;{JS}
(async()=>{{const bytes=Uint8Array.from(atob('{compressed}'),c=>c.charCodeAt(0));const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));D=JSON.parse(await new Response(stream).text());initialize()}})().catch(error=>{{document.getElementById('load-status').textContent='数据加载失败：'+error.message;console.error(error)}});
</script></html>'''


def make_case(candidate_dir, snapshot, out, fixed_radii, preview=False):
    key = candidate_dir.name
    npz = candidate_dir / "geometry.npz"
    source_report = candidate_dir / "report.json"
    if npz.is_file() and source_report.is_file():
        with np.load(npz, allow_pickle=False) as bundle:
            z = {k: bundle[k] for k in bundle.files}
        report = read_json(source_report)
        report["preview_only"] = False
        for source_field in ["surface_source", "pointcloud_portability_proven"]:
            if source_field not in report:
                report[source_field] = report.get("source", {}).get(source_field)
    elif preview:
        z, report = h5_geometry(snapshot / "cases" / key / "case.h5")
    else:
        raise FileNotFoundError(f"candidate NPZ/report missing: {candidate_dir}")
    report = correct_report(report)
    cid = report.get("canonical_id", report.get("case_id", key.replace("__", "/")))
    if cid.replace("/", "__") != key:
        raise ValueError(f"canonical ID {cid!r} does not match candidate directory {key!r}")
    stability, stability_rows, stability_same, stability_density = stability_plots(report)
    payload = dict(canonical_id=cid, report=report, overview=overview(z, report), tree=tree_plot(report),
                   sections=sections_payload(z), profiles=profiles(z), maps=wall_maps(z), stability=stability,
                   stability_rows=stability_rows, stability_same_source=stability_same, stability_density=stability_density,
                   coverage=coverage(z, cid, fixed_radii))
    diagnostic_path = candidate_dir / "bifurcation_locator.json"
    diagnostic = read_json(diagnostic_path)
    if diagnostic:
        if diagnostic['canonical_id'] != cid or diagnostic['source_geometry_sha256'] != hashlib.sha256(npz.read_bytes()).hexdigest() or diagnostic['source_report_sha256'] != hashlib.sha256(source_report.read_bytes()).hexdigest():
            raise ValueError(f'Stale or mismatched bifurcation diagnostic: {diagnostic_path}')
    payload['bifurcations'] = bifurcation_payload(diagnostic)
    for junction in payload['bifurcations']:
        if junction['candidate']:
            payload['overview']['data'].append(scatter3([junction['candidate']['center_mm']], f"J{junction['parent_segment']} 过渡切面候选", '#c026d3', 'markers+text', text=[f"J{junction['parent_segment']} 候选切面"], textposition='top center', marker=dict(color='#c026d3',size=6,symbol='diamond')))
    dest = out / "cases" / (key + ".html")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".tmp.html")
    temp.write_text(case_html(payload), encoding="utf-8")
    temp.replace(dest)
    valid = np.asarray(z.get("section_valid", []), dtype=bool)
    pc_valid = np.asarray(z.get("pc_section_valid", []), dtype=bool)
    map_valid = np.asarray(z.get("wall_map_valid", []), dtype=bool)
    summary = dict(canonical_id=cid, key=key, page="cases/" + dest.name, preview_only=bool(report.get("preview_only")),
                   sections=len(valid), section_valid_fraction=float(valid.mean()) if len(valid) else None,
                   pc_section_valid_fraction=float(pc_valid.mean()) if len(pc_valid) else None,
                   wall_map_valid_fraction=float(map_valid.mean()) if len(map_valid) else None,
                   coord_scale_mm=payload["coverage"].get("coord_scale_mm"),
                   semantic_unconfirmed=sum(not bool(s.get("anatomy_label_user_confirmed", False)) for s in report.get("segments", [])),
                   semantic_unresolved=sum(not bool(s.get("semantic_valid", False)) for s in report.get("segments", [])),
                   warnings=report.get("warnings", []), outside_core_points=report.get("P1_centerline", {}).get("outside_core_points", 0),
                   source_report=str(source_report), awaiting_user_review=True, training_allowed=False,
                   source_schema=report.get("schema"), source_config_sha256=report.get("config_sha256"),
                   html_bytes=dest.stat().st_size, review_update='20260911',
                   bifurcation_candidates=sum(bool(j['candidate']) for j in payload['bifurcations']),
                   bifurcation_junctions=len(payload['bifurcations']), anatomy_correction=report.get('anatomy_correction'))
    (out / "cases" / (key + ".summary.json")).write_text(json.dumps(clean(summary), ensure_ascii=False, indent=2))
    return summary


def write_index(out, snapshot, candidates):
    summaries = {r["key"]: r for r in [read_json(p) for p in sorted((out / "cases").glob("*.summary.json"))]}
    keys = sorted(p.name for p in (snapshot / "cases").glob("*") if (p / "case.h5").is_file())
    unexpected = set(summaries) - set(keys)
    if unexpected:
        raise ValueError(f"review summaries contain non-snapshot IDs: {sorted(unexpected)}")
    for key, record in summaries.items():
        if record["canonical_id"].replace("/", "__") != key:
            raise ValueError(f"summary identity mismatch: {key}")
    rows = []
    for key in keys:
        r = summaries.get(key)
        cid = r["canonical_id"] if r else key.replace("__", "/")
        ready = (candidates / "cases" / key / "geometry.npz").is_file()
        status = "候选详页·待确认" if r and not r["preview_only"] else "既有几何预览·待计算" if r else "候选已生成·待渲染" if ready else "待候选计算"
        def pct(k):
            value = r.get(k) if r else None
            return "—" if value is None else f"{100 * value:.1f}%"
        name = f'<a href="{html.escape(r["page"])}">{html.escape(cid)}</a>' if r else html.escape(cid)
        rows.append(f'<tr data-search="{html.escape(cid.lower())}"><td>{name}</td><td>{status}</td><td>{r["sections"] if r else "—"}</td><td>{pct("section_valid_fraction")}</td><td>{pct("pc_section_valid_fraction")}</td><td>{pct("wall_map_valid_fraction")}</td><td>{r.get("coord_scale_mm", "—") if r else "—"}</td></tr>')
    completed = [r for r in summaries.values() if not r["preview_only"]]
    representatives = []
    for cohort in ["AG", "AAA", "ILO"]:
        pool = [r for r in completed if r["canonical_id"].startswith(cohort + "/")]
        if not pool:
            pool = [r for r in summaries.values() if r["canonical_id"].startswith(cohort + "/")]
        if pool:
            pool.sort(key=lambda r: (r.get("section_valid_fraction") or 0, r["canonical_id"]))
            representatives.append(pool[len(pool) // 2])
    representative_links = "".join(f'<li><a href="{html.escape(r["page"])}">{html.escape(r["canonical_id"])}</a> · {"既有几何预览" if r["preview_only"] else "本队列面截面有效率居中的代表"}</li>' for r in representatives)
    ranked = sorted(completed, key=lambda r: (-r.get("semantic_unresolved", 0), -r.get("outside_core_points", 0), min(r.get("section_valid_fraction") or 0, r.get("pc_section_valid_fraction") or 0, r.get("wall_map_valid_fraction") or 0)))[:12]
    exceptions = "".join(f'<li><a href="{html.escape(r["page"])}">{html.escape(r["canonical_id"])}</a> · {html.escape("；".join(r.get("warnings", [])) or "截面/点云/映射覆盖率排序")}</li>' for r in ranked)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WSS V6 新几何候选验收</title><style>{CSS}</style>
<header><h1>WSS V6 新几何候选验收</h1><p>全队列 {len(keys)} 例 · 已有候选详页 {len(completed)} · 既有几何预览 {len(summaries)-len(completed)}</p><p class="notice">所有结果待人工确认，尚未接入训练。当前页面不自动批准解剖标签、截面算法、几何输入或 M6 固定毫米邻域。面几何参考与点云候选分别展示。</p></header>
<main><div class="card"><h2>如何检查</h2><p>先检查透明壁面、七段中心线、五开口、三个分叉及 CIA 起止；再沿段查看截面交线与选中轮廓，对照 A、Req 与 Rmis；最后看无效 mask、壁面映射和位移/倾角/密度扰动。所有病例保留现有候选命名，画面左右不代表解剖左右。</p><p>M6 页只比较同一随机 support 的邻域覆盖，包含当前归一化半径与固定 mm 候选；未提交 M6 训练。</p></div>
<div class="card"><h2>从各队列代表例开始</h2><ul>{representative_links}</ul><p class="muted">代表例按几何有效率排序选择，不读取 WSS 或速度标签。</p></div>
<div class="card"><h2>优先审阅：候选异常</h2>{'<ul>'+exceptions+'</ul>' if exceptions else '<p class="muted">候选尚未齐备，异常排序在完成计算和渲染后生成。</p>'}</div>
<div class="card"><input id="search" placeholder="搜索病例 / 队列" style="width:min(600px,90%)"><p class="muted">有效率只是诊断读数，不是自动通过标准。缺失显示 —，不写成零误差。</p><div class="table-wrap"><table><thead><tr><th>病例</th><th>状态</th><th>截面数</th><th>面截面有效率</th><th>点云截面有效率</th><th>壁面映射有效率</th><th>coord_scale/mm</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></div>
<div class="card muted"><p>离线打开 index.html，保留 cases/ 与 assets/ 相对目录。候选源：{html.escape(str(candidates))}</p><p>生成器 wss_v5/render_geometry_candidate.py；详情见同目录 README.md 与 render_manifest.json。</p></div></main>
<script>document.getElementById('search').oninput=e=>{{let q=e.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.dataset.search.includes(q))}}</script></html>'''
    (out / "index.html").write_text(page, encoding="utf-8")
    asset = out / "assets" / PLOTLY_ASSET
    manifest = dict(total_cases=len(keys), candidate_detail_pages=len(completed), preview_pages=len(summaries)-len(completed),
                    candidate_root=str(candidates), snapshot_root=str(snapshot), awaiting_user_review=True, training_allowed=False,
                    plotly_asset=PLOTLY_ASSET, plotly_sha256=hashlib.sha256(asset.read_bytes()).hexdigest() if asset.exists() else None,
                    cases=list(summaries.values()))
    (out / "render_manifest.json").write_text(json.dumps(clean(manifest), ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cases", nargs="*", help="canonical IDs or flattened case directory names; default all available candidates")
    parser.add_argument("--preview-existing", action="store_true", help="Allow explicitly labelled H5 geometry-only previews")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--fixed-radii-mm", type=float, nargs=3, default=[2.5, 5, 10], help="Untrained M6 review candidates")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cases").mkdir(exist_ok=True)
    if not (args.out / "assets" / PLOTLY_ASSET).is_file():
        raise FileNotFoundError(f"Copy the pinned offline asset to {args.out / 'assets' / PLOTLY_ASSET}")
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        parser.error("require 0 <= shard-index < shards")
    if not args.index_only:
        if args.cases:
            keys = [cid.replace("/", "__") for cid in args.cases]
        else:
            source = args.snapshot_root if args.preview_existing else args.candidate_root
            keys = sorted(p.name for p in (source / "cases").glob("*") if (p / ("case.h5" if args.preview_existing else "geometry.npz")).is_file())
        for i, key in enumerate(keys):
            if i % args.shards != args.shard_index:
                continue
            summary = make_case(args.candidate_root / "cases" / key, args.snapshot_root, args.out, args.fixed_radii_mm, args.preview_existing)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    # Shards write separate pages only; a sequential --index-only finalizes once all finish.
    if args.shards == 1 or args.index_only:
        manifest = write_index(args.out, args.snapshot_root, args.candidate_root)
        print(json.dumps({k: manifest[k] for k in ["total_cases", "candidate_detail_pages", "preview_pages"]}), flush=True)


if __name__ == "__main__":
    main()
