"""Offline pressure/velocity viewer using actual volume predictions in physical units.

Slice panels show a finite-thickness selection of prediction points, not an
interpolated CFD plane. Streamlines, when supplied, must be integrated from the
predicted vector field by the caller; a centerline is never substituted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .glossary import glossary_document
from .paths import STATIC_DIR
from .report import _b64, _script_json

def _common_js() -> str:
    """Shared report library (contract §12), embedded verbatim.

    Mandatory since phase 2: measurements, presets, colour bars and the message
    protocol all read from it, so a report built without it would be silently
    degraded.  Fail loudly instead."""
    path = STATIC_DIR / "report_common.js"
    if not path.is_file():
        raise FileNotFoundError(
            f"共享报告库缺失：{path}（合同 §12；体场报告必须内嵌 report_common.js）")
    return path.read_text(encoding="utf-8")


def _points(value, name, *, empty=False):
    out = np.asarray(value, dtype=np.float32)
    if empty and out.size == 0:
        out = out.reshape(0, 3)
    if out.ndim != 2 or out.shape[1] != 3 or (not empty and not len(out)) or not np.isfinite(out).all():
        raise ValueError(f"{name} must contain finite (N, 3) coordinates")
    return out


def _scalar(value, count, name, *, allow_nan=False):
    out = np.asarray(value, dtype=np.float32)
    if out.shape != (count,) or np.isinf(out).any() or (not allow_nan and not np.isfinite(out).all()):
        raise ValueError(f"{name} must contain {count} finite scalar values")
    return out


def _indices(value, count, width, name):
    out = np.asarray(value)
    if out.size == 0:
        out = out.reshape(0, width)
    if (out.ndim != 2 or out.shape[1] != width or not np.isfinite(out).all()
            or np.any(out != np.floor(out)) or np.any(out < 0) or np.any(out >= count)):
        raise ValueError(f"{name} must contain valid integer point indices")
    return out.astype(np.uint32)


def _segments(value, count, name):
    out = _scalar(value, count, name)
    if np.any(out != np.floor(out)) or np.any(np.abs(out) > np.iinfo(np.int32).max):
        raise ValueError(f"{name} must contain integer segment identifiers")
    return out.astype(np.int32)


def _trust_bits(value, count, name):
    out = np.asarray(value)
    if out.shape != (count,) or out.dtype.kind not in "iub" or np.any(out < 0) or np.any(out > 255):
        raise ValueError(f"{name} must contain {count} uint8 trust bit masks")
    return out.astype(np.uint8)


def _per_interior(value, wall, count, name, kind):
    """Accept an array aligned to ``cloud.pts`` or, when the cloud carries wall rows,
    one aligned to its interior rows only (the analysis layer works per interior point)."""
    out = np.asarray(value)
    interior = int(np.sum(wall == 0))
    if out.ndim == 1 and out.shape[0] == interior and interior != count:
        full = np.zeros(count, dtype=np.uint8 if kind == "bits" else np.float32)
        if kind != "bits":
            full[:] = np.nan
        full[wall == 0] = out
        out = full
    return _trust_bits(out, count, name) if kind == "bits" else _scalar(out, count, name, allow_nan=True)


def build_html(out_path: Path, meta: dict, mesh: dict, cloud: dict, centerline: dict,
               streamlines=None) -> None:
    """Export a self-contained viewer; coordinates mm, pressure Pa, velocity m/s.

    ``mesh``: vertices, triangle faces, optional pressure_pa (NaN = unsupported),
              optional ``trust`` (uint8 bit mask per vertex; contract §3).
    ``cloud``: pts, optional pressure_pa / velocity_m_s, optional is_wall / segment,
               optional ``trust`` (uint8 per interior point) and the probe helpers
               ``s_from_root_mm`` / ``radius_mm`` / ``dist_to_wall_mm`` (float32 per interior point).
    ``centerline``: xyz, optional tangent / segment / radius_mm / edges.
    ``streamlines``: optional list of {points, speed_m_s}, already integrated.
    Every optional key may be absent: the viewer hides the matching panel.
    """
    pts = _points(cloud["pts"], "cloud.pts")
    vertices = _points(mesh["vertices"], "mesh.vertices")
    faces = _indices(mesh["faces"], len(vertices), 3, "mesh.faces")
    center = _points(centerline.get("xyz", []), "centerline.xyz", empty=True)
    wall = np.asarray(cloud.get("is_wall", np.zeros(len(pts), dtype=np.uint8)))
    if wall.shape != (len(pts),) or not np.isin(wall, [0, 1]).all():
        raise ValueError("cloud.is_wall must contain one Boolean per prediction point")
    arrays = {
        "pts": _b64(pts, np.float32), "vertices": _b64(vertices, np.float32),
        "faces": _b64(faces, np.uint32), "is_wall": _b64(wall, np.uint8),
        "segment": _b64(_segments(cloud.get("segment", np.zeros(len(pts))), len(pts), "cloud.segment"), np.int32),
        "center": _b64(center, np.float32),
        "center_segment": _b64(_segments(centerline.get("segment", np.zeros(len(center))), len(center), "centerline.segment"), np.int32),
        "has_segments": cloud.get("segment") is not None,
        "streamlines": [],
    }
    if cloud.get("segment") is not None:
        # Keep a small human-readable index beside the packed segment array so
        # the offline viewer can offer module names and counts without
        # decoding the complete point cloud a second time.
        segment_ids = np.asarray(cloud["segment"], dtype=np.int32)
        wall_values = wall.astype(bool)
        arrays["modules"] = [
            {"segment": int(segment_id),
             "count": int(np.sum(segment_ids == segment_id)),
             "interior": int(np.sum((segment_ids == segment_id) & ~wall_values)),
             "wall": int(np.sum((segment_ids == segment_id) & wall_values))}
            for segment_id in sorted(set(int(value) for value in segment_ids))
        ]
    if cloud.get("pressure_pa") is not None:
        arrays["pressure_pa"] = _b64(_scalar(cloud["pressure_pa"], len(pts), "cloud.pressure_pa"), np.float32)
    if cloud.get("velocity_m_s") is not None:
        velocity = _points(cloud["velocity_m_s"], "cloud.velocity_m_s")
        if velocity.shape != pts.shape:
            raise ValueError("cloud.velocity_m_s must match cloud.pts")
        arrays["velocity_m_s"] = _b64(velocity, np.float32)
    if not ({"pressure_pa", "velocity_m_s"} & arrays.keys()):
        raise ValueError("A volume report requires pressure or velocity predictions")
    if mesh.get("pressure_pa") is not None:
        if "pressure_pa" not in arrays:
            raise ValueError("Wall pressure requires the corresponding volume pressure field")
        arrays["wall_pressure_pa"] = _b64(_scalar(mesh["pressure_pa"], len(vertices), "mesh.pressure_pa", allow_nan=True), np.float32)
    if cloud.get("trust") is not None:
        arrays["trust"] = _b64(_per_interior(cloud["trust"], wall, len(pts), "cloud.trust", "bits"), np.uint8)
    for key in ("s_from_root_mm", "radius_mm", "dist_to_wall_mm"):
        if cloud.get(key) is not None:
            arrays[key] = _b64(_per_interior(cloud[key], wall, len(pts), f"cloud.{key}", "scalar"), np.float32)
    if mesh.get("trust") is not None:
        arrays["wall_trust"] = _b64(_trust_bits(mesh["trust"], len(vertices), "mesh.trust"), np.uint8)
    if centerline.get("tangent") is not None:
        tangent = _points(centerline["tangent"], "centerline.tangent", empty=True)
        if tangent.shape != center.shape:
            raise ValueError("centerline.tangent must match centerline.xyz")
        arrays["tangent"] = _b64(tangent, np.float32)
    if centerline.get("radius_mm") is not None:
        # Separate key: ``radius_mm`` belongs to the prediction points (probe), this one to the
        # centreline samples (C7 measurements need the inscribed radius along the tree).
        arrays["center_radius_mm"] = _b64(_scalar(centerline["radius_mm"], len(center), "centerline.radius_mm"), np.float32)
    if centerline.get("edges") is not None:
        arrays["edges"] = _b64(_indices(centerline["edges"], len(center), 2, "centerline.edges"), np.uint32)
    for line in ([] if streamlines is None else streamlines):
        if "velocity_m_s" not in arrays:
            raise ValueError("Streamlines require a predicted velocity vector field")
        coords = _points(line["points"], "streamline.points")
        speeds = _scalar(line["speed_m_s"], len(coords), "streamline.speed_m_s")
        if len(coords) < 2 or np.any(speeds < 0):
            raise ValueError("Streamlines require at least two points and nonnegative speed")
        arrays["streamlines"].append({"points": _b64(coords, np.float32), "speed_m_s": _b64(speeds, np.float32)})
    viewer = (STATIC_DIR / "volume_viewer.js").read_text(encoding="utf-8")
    # Glossary (C17) and the shared library (§12) are inlined so the file stays self-contained offline.
    html = (TEMPLATE.replace("__THREE__", (STATIC_DIR / "three.min.js").read_text(encoding="utf-8"))
            .replace("__ORBIT__", (STATIC_DIR / "OrbitControls.js").read_text(encoding="utf-8"))
            .replace("__COMMON__", _common_js()).replace("__VIEWER__", viewer)
            .replace("__GLOSSARY__", _script_json(glossary_document()))
            .replace("__ARRAYS__", _script_json(arrays))
            .replace("__META__", _script_json(meta)))
    Path(out_path).write_text(html, encoding="utf-8")


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>体场预测报告</title><style>
:root{--ink:#203049;--muted:#5a6d80;--line:#dce4ed;--accent:#176caa;--soft:#f4f7fa;--success:#2f855a;--warning:#b7791f;--danger:#c0392b}*{box-sizing:border-box}html,body{height:100%}body{height:100dvh;margin:0;background:#eef3f7;color:var(--ink);font:14px/1.55 system-ui,-apple-system,"Microsoft YaHei",sans-serif;display:flex;flex-direction:column;overflow:hidden}
header{flex:0 0 auto;background:white;border-bottom:1px solid var(--line);padding:10px 18px;display:flex;align-items:center;gap:18px;flex-wrap:wrap}h1{margin:0;font-size:19px}h2{font-size:15px;margin:0 0 10px}h3{font-size:13px;margin:12px 0 6px;color:#3f5a72}small,.note{color:var(--muted);font-size:12px}.note{margin:6px 0;overflow-wrap:anywhere}
header .actions{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap}button,select,input{font:inherit;max-width:100%}button,select{padding:6px 10px;border:1px solid var(--line);border-radius:7px;background:white;color:var(--ink)}button{cursor:pointer}button:disabled{opacity:.45;cursor:default}button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.mode-tabs{display:flex;gap:4px;background:var(--soft);padding:4px;border-radius:9px}.mode-tabs .tab{border:0;background:transparent;padding:6px 14px;border-radius:6px;font-weight:600;color:#4d6478}.mode-tabs .tab.on{background:var(--accent);color:white}.mode-tabs .tab:disabled{color:#a9b7c3}
button.pick{border-color:#e0b4bd;color:#a3324d}button.pick.on{background:#a3324d;color:white;border-color:#a3324d}
main{flex:1 1 auto;min-height:0;display:grid;grid-template-columns:300px minmax(0,1fr)}
aside.menu{padding:12px;background:#f9fbfd;border-right:1px solid var(--line);overflow-y:auto;overflow-x:hidden;height:100%;min-height:0;overscroll-behavior:contain}
details.menu-card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:0 13px;margin-bottom:10px}details.menu-card>summary{cursor:pointer;font-weight:650;padding:11px 0;list-style:none;display:flex;align-items:center}details.menu-card>summary::before{content:"▸";display:inline-block;width:16px;color:var(--accent);transition:transform .15s}details.menu-card[open]>summary::before{transform:rotate(90deg)}details.menu-card>summary::-webkit-details-marker{display:none}details.menu-card>.body{padding:0 0 12px}
details.sub-details{border-top:1px solid var(--line);margin-top:8px;padding-top:4px}details.sub-details>summary{cursor:pointer;font-size:13px;color:#3f5a72;padding:6px 0}
label{display:block;margin:8px 0}label.inline{display:flex;gap:8px;align-items:center;font-weight:400}label select{width:100%;margin-top:4px}.range-row{display:flex;gap:8px;align-items:center}.range-row input[type=range]{flex:1;min-width:0}.range-row output{min-width:52px;text-align:right;font-variant-numeric:tabular-nums;font-size:12px}
.row{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}.row button{font-size:12px;padding:5px 9px}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:8px}.two-col label{margin:4px 0}
.view-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:8px 0}.view-grid button{padding:6px 4px;font-size:13px}
#findings-list{list-style:none;padding:0;margin:6px 0}#findings-list li{display:flex;gap:8px;align-items:flex-start;padding:7px 8px;border:1px solid var(--line);border-radius:7px;margin-bottom:5px;cursor:pointer;font-size:12px;background:#fff}#findings-list li:hover{background:#f0f6fb}#findings-list li.on{border-color:#ff2fa8;background:#fff0f7}.sev{flex:0 0 auto;border-radius:4px;padding:1px 6px;font-size:12px;font-weight:600;color:#fff;background:#8093a2}.sev.attention{background:#c0392b}.sev.note{background:#176caa}.sev.info{background:#5f7c8a}
#profile-canvas{width:100%;height:230px;display:block;border:1px solid var(--line);background:#fff;cursor:crosshair}
#trust-legend ul{margin:4px 0 0;padding-left:16px;font-size:12px;color:var(--muted)}
#volume-view{position:relative;height:100%;min-height:0;overflow:hidden}#volume-view>canvas{display:block}
#right-dock{position:absolute;right:14px;top:14px;bottom:14px;width:330px;display:flex;flex-direction:column;gap:10px;z-index:2;pointer-events:none}#right-dock>*{pointer-events:auto}#legend{background:#fffffff0;padding:9px 11px;border-radius:8px;font-size:12px;flex:0 0 auto;border:1px solid var(--line)}#legend-bar{height:12px;margin:6px 0;border-radius:2px}.ends{display:flex;justify-content:space-between}
#trust-legend{background:#fffffff0;padding:8px 11px;border-radius:8px;font-size:12px;flex:0 0 auto;border:1px solid var(--line)}
.probe{background:#fffffff5;border:1px solid var(--line);border-radius:10px;padding:9px 12px;font-size:12px;flex:0 0 auto}.probe .metrics{font-size:12px}
.slice-zoom{position:fixed;inset:0;background:#0b1a2ab8;z-index:40;display:flex;align-items:center;justify-content:center;padding:18px}.slice-zoom-box{background:#fff;border-radius:12px;padding:12px 16px 10px;width:min(96vw,1180px);max-height:96vh;display:flex;flex-direction:column;gap:6px;box-shadow:0 12px 40px #0006}.slice-zoom-head{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap}.slice-zoom-head button{font-size:12px;padding:4px 10px}#slice-zoom-canvas{width:100%;height:auto;max-height:calc(96vh - 130px);object-fit:contain;border:1px solid var(--line);background:#fff;display:block}#slice-zoom-readout{min-height:1.2em;font-variant-numeric:tabular-nums}.slice-tools{position:absolute;top:12px;left:50%;transform:translateX(-50%);display:flex;gap:6px;align-items:center;flex-wrap:wrap;justify-content:center;max-width:94%;background:#fffffff0;border:1px solid var(--line);border-radius:10px;padding:6px 10px;font-size:12px;z-index:3;box-shadow:0 4px 18px #16334b14}.slice-tools button{font-size:12px;padding:3px 9px}.slice-tools button.on{background:var(--accent);color:#fff;border-color:var(--accent)}.slice-tools-label{color:var(--muted)}.slice-tools-hint{flex-basis:100%;text-align:center;color:var(--muted);font-size:12px}.pick-hint{position:absolute;top:70px;left:50%;transform:translateX(-50%);background:#a3324d;color:#fff;padding:8px 14px;border-radius:8px;font-size:13px;z-index:3;box-shadow:0 4px 18px #16334b2a;max-width:80%;text-align:center}#view-note{position:absolute;bottom:14px;left:14px;max-width:46%;background:#fffffff0;padding:9px 11px;border-radius:8px;font-size:12px;pointer-events:none}#volume-error{position:absolute;top:40%;left:20px;right:20px;background:#fff0db;padding:15px;border-radius:8px;z-index:3}
.slice-panel{flex:0 1 auto;min-height:0;overflow:auto;overscroll-behavior:contain;background:#fffffff5;border:1px solid var(--line);border-radius:10px;padding:10px 12px;box-shadow:0 4px 18px #16334b14}.inline-check{font-size:12px;color:var(--muted);margin-right:4px}.inline-check input{vertical-align:-2px}.slice-panel-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}.slice-panel-head button{font-size:12px;padding:3px 8px}#slice-canvas{width:100%;height:260px;border:1px solid var(--line);background:#f6f8fb;display:block}
.metrics{display:grid;grid-template-columns:1fr auto;gap:5px;font-size:13px}.metrics b{font-variant-numeric:tabular-nums;text-align:right}.hidden,[hidden]{display:none!important}#volume-snapshot{display:none;width:100%}
.warning{background:#fff5e1;padding:9px;border-radius:6px;color:#6d501f;font-size:12px}.coords{font-size:12px;overflow-wrap:anywhere}
footer{flex:0 0 auto;display:flex;gap:12px;flex-wrap:wrap;align-items:center;padding:6px 18px;background:#fff;border-top:1px solid var(--line);font-size:12px;color:var(--muted)}
.gloss{display:inline-block;border:1px solid #cfdbe6;border-radius:999px;font-size:12px;line-height:15px;padding:0 6px;margin-left:4px;background:#f4f7fa;color:#3f5a72;cursor:help;vertical-align:middle;font-weight:400}
.gloss-pop{position:fixed;z-index:20;max-width:300px;background:#fff;border:1px solid var(--line);border-radius:8px;padding:8px 10px;box-shadow:0 6px 20px #16334b22;font-size:12px}.gloss-pop .gloss-head{display:flex;justify-content:space-between;align-items:center;gap:8px}.gloss-pop button{padding:0 6px;font-size:12px}.gloss-pop p{margin:6px 0 0;color:var(--muted)}
.branch-list{display:grid;grid-template-columns:1fr 1fr;gap:2px 8px}.branch-list label{margin:2px 0;font-size:12px}
.table-wrap{overflow:auto;max-height:180px;border:1px solid var(--line);border-radius:6px}#probe-log{border-collapse:collapse;font-size:12px;width:100%}#probe-log th,#probe-log td{padding:3px 5px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}#probe-log button{padding:0 5px;font-size:12px}
#findings-list li{flex-wrap:wrap}#findings-list li.rejected{opacity:.55;border-style:dashed}#findings-list li.confirmed{border-color:#3f8f6b}.sev.manual{background:#7d5ba6}.finding-review{flex:0 0 100%;display:flex;gap:4px;align-items:center;margin-top:4px}.finding-review button{font-size:12px;padding:2px 6px}.finding-review button.on{background:var(--accent);color:#fff;border-color:var(--accent)}.finding-review input{flex:1;min-width:0;font-size:12px;padding:2px 5px;border:1px solid var(--line);border-radius:5px}
.count{display:inline-block;min-width:16px;padding:0 5px;border-radius:999px;background:#e7eef4;font-size:12px;text-align:center}
.item-list{list-style:none;margin:6px 0;padding:0;display:grid;gap:4px}.item{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px;align-items:center;padding:4px 7px;border:1px solid var(--line);border-radius:6px;background:#fff;font-size:12px}.item>span{overflow-wrap:anywhere}.item .tools{display:flex;gap:3px}.item .tools button{font-size:12px;padding:2px 6px}.item input{width:100%;font-size:12px;padding:2px 5px;border:1px solid var(--line);border-radius:5px}
.mode-row button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
#labels{position:absolute;inset:0;pointer-events:none;z-index:1;overflow:hidden}#labels div{position:absolute;transform:translate(-50%,-50%);background:#fffffff0;border:1px solid var(--line);border-radius:5px;padding:1px 6px;font-size:12px;white-space:nowrap}#labels div.meas{background:#e8f6f8;border-color:#5bbcc9}#labels div.annot{background:#fff7e6;border-color:#e0a84a;max-width:240px;white-space:normal}
/* §17.3 automatic labels: findings (colour by severity, clickable), branch names, the max-diameter ring */
#labels div.flabel{pointer-events:auto;cursor:pointer;font-weight:600;background:#eef4f9f2;border-color:#8093a2}#labels div.flabel.attention{background:#fdeceaf2;border-color:#c0392b;color:#8d2114}#labels div.flabel.note{background:#eaf3fbf2;border-color:#176caa;color:#12547a}#labels div.flabel.info{background:#eef1f3f2;border-color:#5f7c8a;color:#3f5a72}#labels div.flabel.manual{background:#f3ecfaf2;border-color:#7d5ba6;color:#5b3f7d}
#labels div.blabel{background:#f7fbf9f2;border-color:#3f8f6b;color:#2c6b50;font-size:12px}#labels div.dlabel{background:#f6eefaf2;border-color:#8e44ad;color:#6d2f88;font-weight:600}
/* §21.4 leader from a moved chip back to its anchor: a 1 px line rotated about its start (left / top = anchor) */
#labels div.leader{height:1px;padding:0;border:0;border-radius:0;transform-origin:0 50%;opacity:.6;pointer-events:none;background:#33475b}
#narrative-card{margin:0 0 8px;padding:7px 9px;border:1px solid #cfe0ec;border-radius:7px;background:#f4f9fd}#narrative-card h3{margin:0 0 4px}#narrative-body p{margin:3px 0;font-size:12px;overflow-wrap:anywhere}#narrative-edited{background:#fff1d6;color:#8a5a12}
#profile-morphology{margin:6px 0;display:flex;gap:6px;align-items:center;flex-wrap:wrap;font-size:12px}#profile-morphology button{font-size:12px;padding:2px 7px}
.gloss-pop b{display:block;margin-bottom:3px}.gloss-pop span{color:var(--muted)}
.metrics .kv{display:contents}
/* compact layout (narrow window / compare-page iframe): adaptive menu columns, slim header, small legend, probe strip */
@media screen{
.compact-only{display:none}.compact .compact-only{display:inline-flex}
/* Title, display modes and actions occupy separate rows in half-width reports. */
.compact header{display:grid;grid-template-columns:auto minmax(0,1fr);gap:5px 8px;padding:6px 10px;overflow:visible}
.compact header>div:nth-of-type(1){min-width:0;overflow:hidden}
.compact header h1{font-size:14px}
.compact #volume-subtitle{font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin:0}
.compact header .gloss{display:none}
.compact header .mode-tabs{grid-column:1/-1;min-width:0;display:flex;flex-wrap:wrap;padding:3px}
.compact .mode-tabs button{flex:1 1 auto;padding:4px 9px;font-size:12px;white-space:nowrap}
.compact header .actions{grid-column:1/-1;display:flex;flex-wrap:wrap;justify-content:flex-end;gap:4px;margin:0;padding-top:5px;border-top:1px solid var(--line);min-width:0}
.compact header .actions button,.compact #menu-toggle{padding:4px 8px;font-size:12px;white-space:nowrap}
header .mode-tabs{flex-wrap:wrap}header>div:nth-of-type(1){min-width:0;overflow-wrap:anywhere}
.compact main{display:block;position:relative}.compact aside.menu{position:absolute;left:0;top:0;bottom:0;width:min(320px,88%);z-index:6;transform:translateX(-102%);transition:transform .15s;box-shadow:4px 0 18px #16334b26}.compact aside.menu.open{transform:none}.compact #volume-view{height:100%;min-width:0}.compact main.menu-open{display:grid;grid-template-columns:minmax(190px,42%) minmax(0,1fr)}.compact main.menu-open aside.menu{position:relative;left:auto;top:auto;bottom:auto;width:auto;transform:none;transition:none;box-shadow:4px 0 18px #16334b26;z-index:2}.compact main.menu-open #menu-tab{display:none}
@media(max-width:560px){.compact main.menu-open{display:block}.compact main.menu-open aside.menu{position:absolute;left:0;top:0;bottom:0;width:min(320px,88%);transform:none}}
.menu-head{display:none}.compact .menu-head{display:flex;justify-content:space-between;align-items:center;margin:0 0 8px;font-weight:650}.menu-head button{font-size:12px;padding:3px 8px}
#menu-tab{display:none}.compact #menu-tab{display:block;position:absolute;left:0;top:50%;transform:translateY(-50%);z-index:3;writing-mode:vertical-rl;padding:10px 4px;border-radius:0 8px 8px 0;border:1px solid var(--line);border-left:0;background:#fffffff0;font-size:12px;letter-spacing:2px;color:#3f5a72}
.compact footer{display:none}
.compact #right-dock{inset:8px;width:auto;display:block;pointer-events:none}
.compact #legend{position:absolute;top:0;right:0;width:200px;padding:6px 8px;font-size:12px}.compact #legend-note{display:none}.compact #legend-bar{height:9px;margin:4px 0}
.compact #trust-legend{position:absolute;top:80px;right:0;width:200px;font-size:12px;padding:6px 8px}
.compact .probe{position:absolute;left:0;right:0;bottom:0;padding:4px 8px;font-size:12px}.compact .probe .slice-panel-head{margin-bottom:2px}.compact .probe .slice-panel-head button{font-size:12px;padding:2px 7px}.compact #probe-note{display:none}
.compact .probe .metrics{display:flex;flex-wrap:wrap;gap:1px 12px;font-size:12px}.compact .probe .metrics .kv{display:inline-flex;gap:4px;white-space:nowrap}.compact .probe .metrics .kv span{color:var(--muted)}
.compact .slice-panel{position:absolute;top:80px;right:0;width:230px;max-height:calc(100% - 150px);padding:6px 8px}.compact #trust-legend:not([hidden])~.slice-panel{top:172px}.compact #slice-canvas{height:170px}.compact .slice-panel .inline-check,.compact #slice-details,.compact #slice-statistics{display:none}.compact .slice-panel-head strong{font-size:12px}
.compact #view-note{bottom:auto;top:8px;left:8px;max-width:38%;font-size:12px;padding:5px 8px}
.compact .slice-tools{top:8px;left:auto;right:220px;transform:none;max-width:calc(100% - 240px);padding:4px 8px}.compact .slice-tools-hint,.compact .slice-tools-label{display:none}.compact .slice-tools button{font-size:12px;padding:2px 7px}.compact .pick-hint{top:44px;font-size:12px}
.mode-tabs .tab.on:hover,.mode-row button.on:hover{background:#125889;color:#fff}
}
@media print{body{overflow:visible;display:block}header .actions,header .mode-tabs,aside.menu,#volume-view>canvas,#legend,#view-note,.slice-tools,.pick-hint,.slice-zoom,.probe,#labels,.gloss,.gloss-pop,#menu-tab,#menu-toggle{display:none!important}main{display:block}#volume-view{height:auto;min-height:0}#volume-snapshot{display:block;max-height:120mm;object-fit:contain}#right-dock{position:static;width:100%}.slice-panel{box-shadow:none}body,aside{background:white}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}

/* Shared workbench scale: readable controls and a restrained report palette. */
@media screen{
header{padding:12px 18px;gap:16px}header h1{font-size:18px}button,select{min-height:34px;border-radius:8px}button:hover:not(:disabled){border-color:var(--line);background:#edf4fa}button.on:hover:not(:disabled){background:#125889;color:#fff}.mode-tabs button.on:hover{background:#125889}
main{grid-template-columns:340px minmax(0,1fr)}aside.menu{padding:14px;background:#f7f9fc}details.menu-card{border-radius:12px;margin-bottom:10px}details.menu-card>summary{padding:12px 0;font-size:14px}details.menu-card>.body{font-size:13px;line-height:1.6;gap:9px}.row{gap:8px}.row button{font-size:13px;padding:6px 10px}.card h3{font-size:14px}.kv,table,.metric-bar,.finding,.item{font-size:12px}
details.sub-details{border-top:1px solid var(--line);margin-top:8px;padding-top:4px}details.sub-details>summary{font-size:13px;color:var(--muted);padding:8px 0;cursor:pointer}details.sub-details>.body{display:grid;gap:9px;padding-bottom:4px}.body input[type=number],.body input[type=text],.body select{border:1px solid var(--line);border-radius:7px;padding:5px 7px;background:#fff;color:var(--ink)}input[type=checkbox],input[type=range]{accent-color:#176caa}.badge,.sev{font-size:12px}.badge.attention,.sev.attention{background:#fde7e3;color:var(--danger)}.badge.note,.sev.note{background:#fff2d5;color:#8b6519}.badge.info,.sev.info{background:#e5f0fa;color:#176caa}.warning{background:#fff2d5;color:#74551b}#labels div.flabel.attention{background:#fde7e3f0;border-color:#e2a294;color:#8d3223}#labels div.flabel.note{background:#fff2d5f0;border-color:#e6c47a;color:#7a5814}#labels div.flabel.info{background:#e5f0faf0;border-color:#9dc3e6;color:#175b8c}.flag{background:#fff2d5;border-color:#e8c889;color:#74551b}.flag.ok{background:#e9f6ee;border-color:#a9d8bb;color:var(--success)}
.compact header h1{font-size:15px}.compact .actions button,.compact .mode-tabs button{min-height:32px}.compact aside.menu{padding:10px}.compact details.menu-card>.body{font-size:13px}.compact .legend-note,.compact #trust-legend,.compact #tip,.compact #view-note{font-size:12px}
}
/* v0.12 (§19.9): the "?" glossary buttons are 16 px circles everywhere (the shared button rules above used to
   stretch them into tall ovals in menu titles and the footer); text chips keep a pill. */
button.gloss{display:inline-flex;align-items:center;justify-content:center;flex:0 0 auto;box-sizing:border-box;margin:0 0 0 4px;border:1px solid #cfdbe6;background:#f4f7fa;color:#3f5a72;font-weight:700;line-height:1;vertical-align:middle;cursor:help}
button.gloss:not(.chip){width:16px!important;height:16px!important;min-width:16px;min-height:0!important;padding:0!important;border-radius:50%!important;font-size:12px}
button.gloss.chip{min-height:0;height:auto;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:400;line-height:18px;margin:2px 4px 2px 0}
button.gloss:hover:not(:disabled){background:#e3eef8;border-color:#9dc3e6;color:#176caa}
.gloss-pop{z-index:35}
/* header: subtitle and its "?" share one line */
.subline{display:flex;align-items:center;min-width:0;margin-top:2px}.subline #volume-subtitle{margin:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
/* footer: review · release · time, technical details in a popover */
footer{gap:14px}footer b{color:var(--ink);font-weight:600}
footer .tech-toggle{margin-left:auto;min-height:0;padding:2px 10px;font-size:12px;border-radius:6px;color:#3f5a72}
.tech-pop{position:fixed;right:16px;bottom:44px;z-index:30;width:min(480px,calc(100vw - 32px));max-height:70vh;overflow:auto;background:#fff;border:1px solid var(--line);border-radius:10px;box-shadow:0 12px 36px #16334b2e;padding:10px 14px;font-size:12px}
.tech-head{display:flex;justify-content:space-between;align-items:center;gap:8px}.tech-head button{min-height:0;padding:2px 9px;font-size:12px;margin-left:4px}
.tech-pop dl{display:grid;grid-template-columns:auto minmax(0,1fr);gap:5px 12px;margin:8px 0}.tech-pop dt{color:var(--muted);white-space:nowrap;display:flex;align-items:center}.tech-pop dd{margin:0;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}
/* B4: thin closable warning banner on top of the viewport */
.warn-banner{position:absolute;top:0;left:0;right:0;z-index:5;display:flex;align-items:center;gap:8px;padding:5px 10px 5px 14px;background:#fff4d6;border-bottom:1px solid #e8c889;color:#6b4e12;font-size:12px;line-height:1.45}
.warn-banner .warn-icon{flex:0 0 auto;font-weight:700}#warn-banner-text{flex:1 1 auto;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.warn-banner button{flex:0 0 auto;min-height:0!important;padding:1px 8px;font-size:12px;background:#fffaf0;border:1px solid #e8c889;color:#6b4e12;border-radius:6px}
/* the slice toolbar centres in the space left of the 330 px right dock instead of running under the legend */
@media screen{body:not(.compact) .slice-tools{left:calc(50% - 172px);max-width:calc(100% - 380px)}}
#volume-view.has-banner #right-dock{top:44px}#volume-view.has-banner .slice-tools{top:42px}#volume-view.has-banner .pick-hint{top:100px}
.compact #volume-view.has-banner #right-dock{inset:38px 8px 8px 8px}.compact #volume-view.has-banner #view-note,.compact #volume-view.has-banner .slice-tools{top:38px}
.ref-list{margin:4px 0 8px;padding-left:18px;font-size:12px;color:#74551b}
.slice-controls{display:flex;flex-wrap:wrap;gap:4px 10px;align-items:center;margin:0 0 6px;font-size:12px}.slice-controls select,.slice-zoom-tools select{min-height:0;padding:1px 4px;font-size:12px;border-radius:6px}
.slice-controls label,.slice-zoom-tools label{display:inline-flex;align-items:center;gap:4px;margin:0;white-space:nowrap}.slice-controls label select,.slice-zoom-tools label select{width:auto;margin:0}
.slice-manual input{width:64px;min-height:0;padding:1px 4px;font-size:12px;border:1px solid var(--line);border-radius:5px}.slice-zoom-tools{display:flex;flex-wrap:wrap;gap:4px 10px;align-items:center;justify-content:flex-end}
@media print{.warn-banner{position:static}.warn-banner button,footer .tech-toggle,.tech-pop{display:none!important}}
/* v0.15: 「← 工作台」 like the wall report (served pages only), and 44 px controls on touch screens */
.back-link{flex:0 0 auto;font-size:13px;font-weight:600;color:var(--acc,#176caa);text-decoration:none;padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:#fff;white-space:nowrap}.back-link:hover{background:#edf4fa}.back-link:focus-visible{outline:2px solid #176caa;outline-offset:2px}
.compact header.has-back{grid-template-columns:auto auto minmax(0,1fr)}.compact .back-link{padding:4px 8px;font-size:12px}
@media print{.back-link{display:none!important}}
@media (pointer:coarse){button:not(.gloss),select,summary,.back-link{min-height:44px!important}button:not(.gloss){min-width:40px}input[type=range]{min-height:40px}.gloss{position:relative}.gloss:after{content:"";position:absolute;inset:-12px}input[type=checkbox]{width:22px;height:22px}}
</style></head><body>
<header><a id="back-to-workbench" class="back-link" href="/" title="回到工作台（本病例）" hidden>← 工作台</a><button type="button" id="menu-toggle" class="compact-only" aria-expanded="false" title="打开菜单">☰ 菜单</button><div><h1>体场预测报告</h1><div class="subline"><span id="volume-subtitle" class="note"></span><button type="button" class="gloss" data-gloss="fixed_frame" aria-label="固定时相说明">?</button></div></div>
<div class="mode-tabs" role="group" aria-label="显示方式"><button type="button" aria-pressed="false" aria-controls="volume-view" class="tab" id="mode-cloud">体内点云</button><button type="button" aria-pressed="false" aria-controls="volume-view" class="tab" id="mode-slice">截面</button><button type="button" aria-pressed="false" aria-controls="volume-view" class="tab" id="mode-wall">壁面压力</button><button type="button" aria-pressed="false" aria-controls="volume-view" class="tab" id="mode-streamlines">流线</button></div>
<div class="actions"><button type="button" class="pick" id="pick-toggle" aria-pressed="false">点选定位截面</button><button type="button" id="volume-fit">复位视角</button><button type="button" id="volume-save" title="把当前三维视图保存为 PNG 截图">保存截图</button><button type="button" id="volume-print">打印 / 保存 PDF</button><button type="button" id="layout-toggle" title="在紧凑布局与完整布局之间切换（窄窗口与并排比较时自动紧凑）">紧凑布局</button></div></header>
<main><aside class="menu"><div class="menu-head"><span>菜单</span><button type="button" id="menu-close">◂ 收起</button></div>
<details id="menu-display" class="menu-card" open><summary>显示</summary><div class="body">
<label>物理量<select id="volume-field"></select></label>
<div hidden><select id="volume-mode"><option value="cloud">体内点云</option><option value="slice">横截面</option><option value="wall">壁面压力</option><option value="streamlines">速度流线</option></select></div>
<div class="two-col"><label>色标<select id="colormap"></select></label><label>色带分段<select id="color-bands"></select></label></div>
<label>血管模块（切割后）<select id="volume-module"></select></label>
<label>血管轮廓透明度<div class="range-row"><input id="volume-opacity" type="range" min="0" max="0.5" step="0.01" value="0.1"><output id="volume-opacity-value">0.10</output></div></label>
<details class="sub-details"><summary>高级显示设置</summary>
<div class="two-col"><label>压力单位 <button type="button" class="gloss" data-gloss="pressure_units" aria-label="术语解释">?</button><select id="pressure-unit"></select></label><label>速度单位 <button type="button" class="gloss" data-gloss="speed" aria-label="术语解释">?</button><select id="velocity-unit"></select></label></div>
<label class="inline" id="velocity-log-row"><input type="checkbox" id="velocity-log">对数色标（仅速度；下限 = max(范围下限, 上限 / 200, 0.001 m/s)）</label>
<div class="row"><button type="button" id="defaults-save">设为默认口径</button></div><p id="defaults-status" class="note"></p>
<div id="branch-visibility-row" hidden><h3>分支显隐</h3><div id="branch-visibility" class="branch-list"></div><p class="note">统计不受显隐影响，只改变显示。</p></div>
<h3>自动标注</h3><div class="two-col"><label>自动标注发现<select id="labels-findings"><option value="0">关</option><option value="3">前 3</option><option value="5">前 5</option><option value="10">前 10</option></select></label><label class="inline"><input type="checkbox" id="labels-branches">分支名</label></div>
<p class="note">按发现列表顺序（驳回项不标）在三维上钉出标签；点标签即飞到该处。隐藏的分支不标。标签随导出 PNG、拼图与一页纸配图一起合成。</p>
<div id="vectors-row"><label class="inline"><input type="checkbox" id="volume-vectors">显示速度方向箭头</label></div>
<div id="trust-row" hidden><label class="inline"><input type="checkbox" id="trust-overlay">可信区域叠加（灰化不可信点）</label></div>
<div id="streamline-settings" hidden><h3>流线 <button type="button" class="gloss" data-gloss="streamlines" aria-label="术语解释">?</button></h3><label>流线粗细<div class="range-row"><input id="streamline-width" type="range" min="0.4" max="3" step="0.1" value="1"><output id="streamline-width-value">1.0×</output></div></label>
<label>流线密度<select id="streamline-density"><option value="1">全部</option><option value="2">一半</option><option value="3">三分之一</option><option value="5">五分之一</option></select></label>
<label class="inline"><input type="checkbox" id="streamline-thin">细线模式（显卡较弱时）</label></div>
</details>
<p id="volume-field-note" class="note"></p><p id="streamline-note" class="note"></p></div></details>
<details id="menu-findings" class="menu-card" hidden><summary>发现 <button type="button" class="gloss" data-gloss="finding_decision" aria-label="术语解释">?</button></summary><div class="body"><p id="findings-note" class="note"></p><ul id="findings-list"></ul><p id="findings-hint" class="note"></p><div class="row"><button type="button" id="findings-clear">取消高亮</button></div>
<div class="row"><input id="finding-add-text" type="text" maxlength="200" placeholder="新增发现的文字（人工）"><button type="button" id="finding-add">新增发现 · 点选位置</button></div><p id="findings-review-status" class="note"></p></div></details>
<details id="menu-measure" class="menu-card"><summary>测量 <button type="button" class="gloss" data-gloss="local_diameter" aria-label="术语解释">?</button></summary><div class="body">
<div id="measure-tools">
<div class="row mode-row" id="measure-modes"><button type="button" id="measure-distance">距离</button><button type="button" id="measure-arc">弧长</button><button type="button" id="measure-diameter">管径</button><button type="button" id="measure-segment">分段</button></div>
<p id="measure-hint" class="note">选择模式后在壁面或体内点上点击取点（距离 / 弧长 / 分段各两点，管径一点）；再次点击该模式按钮退出。管径 = 中心线内切半径 × 2 <button type="button" class="gloss" data-gloss="local_diameter" aria-label="术语解释">?</button>，不是该处截面的最大直径；弧长沿中心线树计算 <button type="button" class="gloss" data-gloss="arc_distance" aria-label="术语解释">?</button>。</p>
<ul id="measure-list" class="item-list"></ul>
<div class="row"><button type="button" id="measure-copy">复制列表</button><button type="button" id="measure-clear">清空</button></div>
<p id="measure-note" class="note"></p></div>
<h3>探针记录 <span id="probe-log-count" class="count">0</span></h3>
<div class="table-wrap"><table id="probe-log"><thead><tr><th>编号</th><th>xyz (mm)</th><th>分支</th><th>s (mm)</th><th>半径 (mm)</th><th>数值</th><th></th></tr></thead><tbody id="probe-log-body"></tbody></table></div>
<div class="row"><button type="button" id="probe-log-copy">复制 TSV</button><button type="button" id="probe-log-export">导出 CSV</button><button type="button" id="probe-log-clear">清空</button></div>
<p id="probe-log-note" class="note"></p></div></details>
<details id="menu-view" class="menu-card"><summary>导出与视图</summary><div class="body">
<h3>标准视角 <button type="button" class="gloss" data-gloss="standard_views" aria-label="术语解释">?</button></h3><div class="view-grid"><button type="button" id="view-front">前</button><button type="button" id="view-back">后</button><button type="button" id="view-left">左</button><button type="button" id="view-right">右</button><button type="button" id="view-top">上</button><button type="button" id="view-bottom">下</button></div>
<p id="view-direction-note" class="note"></p>
<div id="shortcuts-row" class="row" hidden><button type="button" id="shortcuts-help">快捷键说明（?）</button></div>
<h3>视图状态</h3><div class="row"><button type="button" id="view-save">保存到浏览器</button><button type="button" id="view-restore">恢复</button><button type="button" id="view-export">导出 JSON</button><label class="inline" style="margin:0"><input id="view-import" type="file" accept=".json" style="font-size:12px;max-width:150px"></label></div>
<div class="row"><button type="button" id="view-link">复制复现链接</button><button type="button" id="view-replay">复现此图（按链接）</button></div>
<p id="view-status" class="note"></p><p class="note">视图状态包含相机、物理量、显示方式、色标、单位、截面参数、点选点、分支显隐、探针记录与导图设置；导出的链接在同一报告文件中打开即可复现。</p>
<div class="row"><button type="button" id="export-png">导出 PNG</button><button type="button" id="view-six">导出六视角</button></div>
<p id="export-status" class="note"></p>
<details class="sub-details"><summary>导出设置与多视角配图</summary><h3>出版级导图</h3>
<div class="two-col"><label>分辨率<select id="export-scale"><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select></label><label>背景<select id="export-background"><option value="white">白色</option><option value="transparent">透明</option><option value="current">当前</option></select></label></div>
<div class="two-col"><label>色标<select id="export-colorbar"><option value="overlay">叠加在图上</option><option value="none">不含</option><option value="svg">另存 SVG 色标</option></select></label><label>标签语言<select id="export-lang"><option value="zh">中文</option><option value="en">English</option></select></label></div>
<label class="inline"><input type="checkbox" id="export-hide-ui">隐藏界面元素（图例与说明）</label>

<p class="note">离屏渲染当前视角；显卡不支持高分辨率时自动降到 1× 并提示。文件名 病例_视角_物理量_倍数x.png。</p><h3>多视角拼图 <button type="button" class="gloss" data-gloss="standard_views" aria-label="术语解释">?</button></h3><div id="montage-views" class="view-grid"><label class="inline-check"><input type="checkbox" id="mv-front" checked>前</label><label class="inline-check"><input type="checkbox" id="mv-back">后</label><label class="inline-check"><input type="checkbox" id="mv-left" checked>左</label><label class="inline-check"><input type="checkbox" id="mv-right">右</label><label class="inline-check"><input type="checkbox" id="mv-top">上</label><label class="inline-check"><input type="checkbox" id="mv-bottom">下</label><label class="inline-check"><input type="checkbox" id="mv-current" checked>当前</label><label class="inline-check"><input type="checkbox" id="mv-slice" checked>截面</label></div><div class="two-col"><label>列数<select id="montage-columns"><option value="2">2 列</option><option value="3">3 列</option><option value="4">4 列</option></select></label><label class="inline" style="align-self:end"><button type="button" id="montage-export">导出多视角拼图</button></label></div><p id="montage-status" class="note"></p><p class="note">每幅按上面的分辨率 / 背景 / 语言离屏渲染，图上不叠色标，拼图右侧共用一条色标；截面用当前平面视图。文件名 病例_montage_物理量_倍数x.png。</p><h3>一页纸配图</h3><div class="row"><button type="button" id="onepage-shots">生成一页纸配图</button></div><p id="onepage-status" class="note"></p><p class="note">仅在线可用：把前视、左视、上视、当前视角（有截面时再加一张截面图）按 2×、白底、含色标上传到本任务，一页纸报告的「配图」区即用这几张。</p></details></div></details>
<details id="menu-slice" class="menu-card"><summary>截面 <button type="button" class="gloss" data-gloss="slice" aria-label="术语解释">?</button></summary><div class="body">
<h3>放置截面</h3><p class="note">三种方式：① 直接拖动三维视图里的蓝色截面——默认沿法向上下移动，Shift 拖动旋转，Alt 拖动面内平移，滚轮沿法向移动，↑↓ ←→ PgUp/PgDn 微调（视图顶部工具条可切换拖动含义）；② 点选定位：点「点选定位截面」后在血管壁上点一下，截面垂直于该处中心线，再点一下可让截面通过两点；③ 下方滑杆精确调节。空白处拖动仍旋转视图。</p><p id="pick-status" class="note"></p><div class="row"><button type="button" class="pick" id="pick-toggle-menu" aria-pressed="false">点选定位截面</button><button type="button" id="pick-clear">清除选点</button></div>
<details class="sub-details"><summary>滑杆精确调节</summary>
<label>平面定位<select id="slice-basis"><option value="centerline">沿中心线移动（垂直局部切线）</option><option value="pick">点选平面</option><option value="x">垂直 X 轴</option><option value="y">垂直 Y 轴</option><option value="z">垂直 Z 轴</option></select></label>
<label id="slice-branch-label">分支<select id="slice-branch"></select></label>
<label><span id="slice-position-label">位置</span><div class="range-row"><input id="slice-position" type="range" min="0" max="100" step="0.2" value="50"><output id="slice-position-value">50%</output></div></label>
<label>绕平面横轴旋转<div class="range-row"><input id="slice-pitch" type="range" min="-85" max="85" value="0"><output id="slice-pitch-value">0°</output></div></label>
<label>绕平面纵轴旋转<div class="range-row"><input id="slice-yaw" type="range" min="-85" max="85" value="0"><output id="slice-yaw-value">0°</output></div></label>
<label>截面厚度（mm）<div class="range-row"><input id="slice-thickness" type="range" min="0.2" max="12" step="0.2" value="2"><output id="slice-thickness-value">2.0</output></div></label>
<label>截面左右偏移（mm）<div class="range-row"><input id="slice-offset-u" type="range" min="-30" max="30" step="0.2" value="0"><output id="slice-offset-u-value">0.0</output></div></label>
<label>截面上下偏移（mm）<div class="range-row"><input id="slice-offset-v" type="range" min="-30" max="30" step="0.2" value="0"><output id="slice-offset-v-value">0.0</output></div></label>
<div class="row"><button type="button" id="slice-show">选择并显示截面</button><button type="button" id="slice-cut">按当前截面切割</button><button type="button" id="slice-clear-cut">取消切割</button></div><p id="slice-cut-status" class="note"></p>
<p class="note">截面也可在三维视图里直接拖动（见上）；切割后可在“血管模块”中选择截面两侧或全部。平面视图默认「补全截面」：用截面与血管壁的交线做边界，壁面按无滑移（速度 0）或壁面压力作为边界条件，连同厚度内的预测点插值填满整个管腔；离预测点较远、主要靠壁面边界补出来的格子画成淡色，底部标注直接支撑的比例。取消勾选则回到只画有预测点支撑区域、其余留灰的严格模式。</p></details>
<details class="sub-details"><summary>自动截面</summary><label>分支预设<select id="auto-presets"></select></label><div class="row"><button type="button" id="slice-apply-auto">应用截面</button></div><p class="note">按每条中心线分支弧长自动选择 5%、20%、40%、60%、80%、95%；平面垂直局部切线。</p><h3>截面系列拼图</h3><div class="two-col"><label>分支<select id="slice-series-branch"></select></label><label>站位数<input id="slice-series-count" type="number" min="2" max="12" step="1" value="6"></label></div><div class="row"><button type="button" id="slice-series-export">导出系列拼图</button></div><p id="slice-series-note" class="note">逐站沿该分支画平面视图，拼成一张 PNG（3 列、共用当前色标与厚度 / 补全设置）；6 站 = 5%、20%、40%、60%、80%、95%，其余站位数在 5%–95% 之间均分。</p></details>
</div></details>
<details id="menu-stats" class="menu-card"><summary>统计与口径</summary><div class="body">
<div id="narrative-card" hidden><h3>结论（参考） <button type="button" class="gloss" data-gloss="narrative" aria-label="术语解释">?</button> <span id="narrative-edited" class="count" hidden>审阅人已修改</span></h3><div id="narrative-body"></div><p class="note">自动生成的参考描述，只读；编辑在工作台详情页。非诊断结论。</p></div>
<div id="reference-card" hidden><h3>参考范围</h3><p id="reference-status" class="note"></p><ul id="reference-list" class="ref-list"></ul></div>
<div id="volume-statistics" class="metrics"></div>
<p class="note">术语：<button type="button" class="gloss chip" data-gloss="relative_pressure">相对压力</button><button type="button" class="gloss chip" data-gloss="delta_p">ΔP 压差</button><button type="button" class="gloss chip" data-gloss="speed">速度大小</button><button type="button" class="gloss chip" data-gloss="trust_low_sample_support">采样支撑</button><button type="button" class="gloss chip" data-gloss="trust_near_opening">近切口</button><button type="button" class="gloss chip" data-gloss="trust_geometry_out_of_range">几何越界</button><button type="button" class="gloss chip" data-gloss="trust_interpolation_uncovered">插值无支撑</button></p><p id="pressure-reference" class="note"></p><p id="volume-protocol" class="note"></p><p class="warning">速度主要查看体内点、向量和流线；壁面无滑移速度不作为主要展示。压力可以同时查看体内与壁面。单个预测时相的流线不是粒子随时间运动的轨迹。</p>
<div class="row"><button type="button" id="tech-info-menu" aria-expanded="false" aria-controls="tech-info">技术信息（发布包、特征合同、run_identity、哈希）</button></div></div></details>
<details id="menu-profiles" class="menu-card" hidden><summary>沿程</summary><div class="body">
<div id="profile-morphology" hidden><span id="max-diameter-text"></span><button type="button" id="max-diameter-fly">飞到</button><label class="inline" style="margin:0"><input type="checkbox" id="labels-max-diameter" checked>显示最大直径环</label><button type="button" class="gloss" data-gloss="max_diameter" aria-label="术语解释">?</button></div>
<div class="two-col"><label>分支<select id="profile-branch"></select></label><label>物理量<select id="profile-quantity"></select></label></div>
<canvas id="profile-canvas" width="560" height="460" aria-label="沿分支弧长的截面平均值曲线"></canvas>
<p class="note">实线为截面平均，虚线为最大 / 最低；下图为局部半径，有形态数据时叠加截面最大直径（实线）与等效直径（虚线）。点击曲线任一处，三维截面跳到该弧长。压力为相对量，只用于差值。</p><div class="row"><button type="button" id="profile-svg">导出曲线 SVG</button></div><p id="profile-svg-note" class="note">导出当前分支的两张矢量图：所选物理量（按当前显示单位）一张、半径一张（有形态数据时同图含最大 / 等效直径）；横轴为距入口弧长。</p>
<details class="sub-details"><summary>两截面之间的区域统计</summary>
<label>分支<select id="region-branch"></select></label>
<label>近端位置<div class="range-row"><input id="region-smin" type="range" min="0" max="100" step="1" value="10"><output id="region-smin-value">10%</output></div></label>
<label>远端位置<div class="range-row"><input id="region-smax" type="range" min="0" max="100" step="1" value="90"><output id="region-smax-value">90%</output></div></label>
<label class="inline"><input type="checkbox" id="region-highlight">在三维视图中高亮该区域</label>
<div id="region-stats" class="metrics"></div><p id="region-note" class="note"></p></details>
</div></details>
<details id="menu-annot" class="menu-card"><summary>标注</summary><div class="body">
<div class="row"><input id="annot-text" type="text" maxlength="200" placeholder="标注文字（留空则弹窗输入）"><button type="button" id="annot-add">钉标注 · 点选位置</button></div>
<div class="row"><button type="button" id="annot-save">保存到服务</button><button type="button" id="annot-clear">清空</button></div>
<p id="annot-status" class="note"></p><ul id="annot-list" class="item-list"></ul>
<p class="note">标注钉在预测点或壁面上，随视图状态、导图与六视角输出；在线时保存到任务目录，审阅锁定后只读。</p></div></details>
<details id="menu-presets" class="menu-card"><summary>预设</summary><div class="body">
<div id="preset-builtin" class="row"></div><p class="note">内置预设按本例的分支名与发现列表计算，合并到当前视图后应用。</p>
<label>名称<input id="preset-name" type="text" maxlength="40" placeholder="我的视图"></label>
<div class="row"><button type="button" id="preset-save">存为预设</button><button type="button" id="preset-export">导出 JSON</button></div>
<ul id="preset-user" class="item-list"></ul><p id="preset-note" class="note"></p></div></details>
</aside>
<div id="volume-view" role="region" aria-label="三维体场视图"><div id="warn-banner" class="warn-banner" role="status" hidden><span class="warn-icon" aria-hidden="true">⚠</span><span id="warn-banner-text"></span><button type="button" id="warn-banner-more">详情</button><button type="button" id="warn-banner-close" aria-label="关闭提示">×</button></div><img id="volume-snapshot" alt="体场视图"><div id="labels" aria-hidden="true"></div><button type="button" id="menu-tab" title="打开菜单">菜单</button><div id="view-note"></div><div id="slice-tools" class="slice-tools" hidden><span class="slice-tools-label">拖动蓝色截面 =</span><button type="button" id="drag-move" class="on" aria-pressed="true">沿法向移动</button><button type="button" id="drag-rotate" aria-pressed="false">旋转</button><button type="button" id="drag-offset" aria-pressed="false">面内平移</button><button type="button" id="slice-reset-angle">回正</button><button type="button" class="pick" id="pick-toggle-2" aria-pressed="false">点选定位截面</button><span class="slice-tools-hint">Shift 拖动 = 旋转 · Alt 拖动 = 平移 · 滚轮 = 沿法向移动（Shift 滚轮改厚度）· 方向键 / PgUp PgDn 微调 · 空白处拖动旋转视图</span></div><div id="pick-hint" class="pick-hint" hidden></div><div id="slice-zoom" class="slice-zoom" hidden role="dialog" aria-label="截面平面视图（放大）"><div class="slice-zoom-box"><div class="slice-zoom-head"><strong id="slice-zoom-title">截面平面视图</strong><span class="slice-zoom-tools"><label class="inline-check">色标 <select id="slice-zoom-range" aria-label="截面色标范围"><option value="section">本截面</option><option value="global">全局</option><option value="manual">手动</option></select></label><span id="slice-zoom-manual" class="slice-manual" hidden><input id="slice-zoom-min" type="number" step="any" aria-label="色标下限（显示单位）"> – <input id="slice-zoom-max" type="number" step="any" aria-label="色标上限（显示单位）"></span><label class="inline-check" id="slice-zoom-quantity-row">着色 <select id="slice-zoom-quantity" aria-label="截面着色量"><option value="speed">速度大小</option><option value="normal">穿面速度（沿法向为正）</option></select></label><label class="inline-check" id="slice-zoom-arrows-row"><input type="checkbox" id="slice-zoom-arrows" checked> 面内流向箭头</label> <label class="inline-check"><input type="checkbox" id="slice-zoom-points" checked> 显示预测点</label> <label class="inline-check"><input type="checkbox" id="slice-zoom-fill" checked> 补全截面</label> <button type="button" id="slice-zoom-png">导出 PNG</button><button type="button" id="slice-zoom-csv">导出 CSV</button><button type="button" id="slice-zoom-close">关闭</button></span></div><canvas id="slice-zoom-canvas" width="1400" height="1200" aria-label="截面平面视图（放大）"></canvas><p id="slice-zoom-readout" class="note"></p><p id="slice-zoom-caption" class="note"></p><p id="slice-zoom-status" class="note"></p></div></div><div id="volume-error" hidden></div>
<div id="right-dock"><div id="legend"><strong id="legend-title"></strong><div id="legend-bar"></div><div class="ends"><span id="legend-min"></span><span id="legend-max"></span></div><small id="legend-note">当前物理量统一色标</small></div>
<div id="trust-legend" hidden><strong>可信区域</strong><ul id="trust-legend-list"></ul><small>灰化点仅提示证据不足，不改变预测数值。</small></div>
<div id="probe-card" class="probe" hidden><div class="slice-panel-head"><strong>探针</strong><span><button type="button" id="probe-record">记录</button> <button type="button" id="probe-clear">清除</button></span></div><div id="probe-body" class="metrics"></div><p id="probe-note" class="note"></p></div>
<div id="slice-panel" class="slice-panel" hidden><div class="slice-panel-head"><strong>截面平面视图</strong><span><label class="inline-check"><input type="checkbox" id="slice-fill" checked> 补全截面</label> <button type="button" id="slice-zoom-open">放大</button><button type="button" id="slice-panel-collapse">收起</button></span></div><div id="slice-panel-body"><div id="slice-controls" class="slice-controls"><label class="inline-check">色标 <select id="slice-range-mode" aria-label="截面色标范围"><option value="section">本截面</option><option value="global">全局</option><option value="manual">手动</option></select></label><span id="slice-manual" class="slice-manual" hidden><input id="slice-min" type="number" step="any" aria-label="色标下限（显示单位）"> – <input id="slice-max" type="number" step="any" aria-label="色标上限（显示单位）"></span><label class="inline-check" id="slice-quantity-row">着色 <select id="slice-quantity" aria-label="截面着色量"><option value="speed">速度大小</option><option value="normal">穿面速度（沿法向为正）</option></select></label><label class="inline-check" id="slice-arrows-row"><input type="checkbox" id="slice-arrows" checked> 面内流向箭头</label></div><canvas id="slice-canvas" width="560" height="520" aria-label="预测点在有限厚度截面内的投影"></canvas><p id="slice-fill-note" class="note"></p><p id="slice-details" class="note coords"></p><div id="slice-statistics" class="metrics"></div></div></div></div>
</div></main>
<footer><span>审阅<button type="button" class="gloss" data-gloss="review_status" aria-label="术语解释">?</button>：<b id="footer-review"></b></span><span>发布包<button type="button" class="gloss" data-gloss="release" aria-label="术语解释">?</button> <b id="footer-release"></b></span><span id="footer-time"></span><button type="button" id="tech-info-toggle" class="tech-toggle" aria-expanded="false" aria-controls="tech-info" title="发布包、特征合同、run_identity、哈希与重建时间">技术信息</button></footer>
<div id="tech-info" class="tech-pop" role="dialog" aria-label="技术信息" hidden><div class="tech-head"><strong>技术信息</strong><span><button type="button" id="tech-info-copy">复制</button><button type="button" id="tech-info-close" aria-label="关闭">×</button></span></div><dl id="tech-info-rows"><dt>特征合同</dt><dd id="footer-feature"></dd><dt>run_identity</dt><dd id="footer-identity"></dd></dl><p id="tech-info-status" class="note"></p><p class="note">用于复现与核对；不影响任何预测数值。</p></div>
<div id="gloss-pop" class="gloss-pop" role="dialog" aria-live="polite" hidden><div class="gloss-head"><strong id="gloss-title"></strong><button type="button" id="gloss-close" aria-label="关闭">×</button></div><p id="gloss-desc"></p></div>
<!--WSS_META_START--><script id="wss-report-meta" type="application/json">__META__</script><!--WSS_META_END-->
<script id="volume-arrays" type="application/json">__ARRAYS__</script><script id="wss-glossary" type="application/json">__GLOSSARY__</script><script>__THREE__</script><script>__ORBIT__</script><script>__COMMON__</script><script>__VIEWER__</script></body></html>"""
