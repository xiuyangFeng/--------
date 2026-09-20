#!/usr/bin/env python3
"""Render the delivered triangle-surface VTPs without an X server.

Run with /public/newhome/cy/.conda/envs/GNN/bin/python (VTK, Matplotlib).
Uses the actual VTP polygons, an identical orthographic camera for the three
panels, and back-to-front triangle sorting.  No point splatting, mesh fitting,
interpolation, new inference, or metric recomputation is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties, fontManager, findfont
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import vtk
from vtk.util.numpy_support import vtk_to_numpy


SPECS = {
    "X5X11": {
        "filename": "surface_gaussian.vtp",
        "fields": ["wss_cfd_pa", "wss_pred_pa", "wss_error_pred_minus_cfd_pa"],
        "selfmax_fields": ["wss_cfd_selfmax", "wss_pred_selfmax", "wss_selfmax_error_pred_minus_cfd"],
        "target": "WSS", "unit": "Pa", "kind": "壁面 WSS · STL Gaussian 回插",
        "selfmax_unit_interval": True, "metric_scope": "原始壁面同点",
    },
    "PF6": {
        "filename": "surface_gaussian.vtp",
        "fields": ["pressure_cfd_pa", "pressure_pred_pa", "pressure_error_pred_minus_cfd_pa"],
        "selfmax_fields": ["pressure_cfd_selfmax", "pressure_pred_selfmax", "pressure_selfmax_error_pred_minus_cfd"],
        "target": "Relative pressure p − p_ref", "unit": "Pa", "kind": "壁面相对压力 p − p_ref · STL Gaussian 回插",
        "selfmax_unit_interval": False, "metric_scope": "原始壁面∪体内同点",
    },
    "VF6_wss": {
        "filename": "surface_gaussian.vtp",
        "fields": ["wss_cfd_pa", "wss_pred_pa", "wss_error_pred_minus_cfd_pa"],
        "selfmax_fields": ["wss_cfd_selfmax", "wss_pred_selfmax", "wss_selfmax_error_pred_minus_cfd"],
        "target": "WSS", "unit": "Pa",
        "kind": "VF6→冻结 V3 派生壁面 WSS · STL Gaussian 回插",
        "selfmax_unit_interval": True, "metric_scope": "原始壁面同点",
    },
    "X5D_v51": {
        "filename": "surface_gaussian.vtp",
        "fields": ["wss_cfd_pa", "wss_pred_pa", "wss_error_pred_minus_cfd_pa"],
        "selfmax_fields": ["wss_cfd_selfmax", "wss_pred_selfmax", "wss_selfmax_error_pred_minus_cfd"],
        "target": "WSS", "unit": "Pa",
        "kind": "壁面 WSS · STL Gaussian 回插 · v5.1 X5D_v51",
        "selfmax_unit_interval": True, "metric_scope": "原始壁面同点",
    },
    "V4_EMA": {
        "filename": "surface_gaussian.vtp",
        "fields": ["wss_cfd_pa", "wss_pred_pa", "wss_error_pred_minus_cfd_pa"],
        "selfmax_fields": ["wss_cfd_selfmax", "wss_pred_selfmax", "wss_selfmax_error_pred_minus_cfd"],
        "target": "WSS", "unit": "Pa",
        "kind": "历史 V4 EMA 派生壁面 WSS · STL Gaussian 回插",
        "selfmax_unit_interval": True, "metric_scope": "原始壁面同点",
        "caption_checkpoint": "s1234 / last_converged（epoch9999，未收敛）",
        "caption_selection": "病例角色按 test35 派生 WSS R² 选取；Median 是距分布中位最近的真实病例。历史 V4 稳态 peak，不是 V5 1162。",
    },
    "V4_EMA_pressure": {
        "folder": "V4_EMA",
        "filename": "pressure_surface_gaussian.vtp",
        "fields": ["pressure_cfd_pa", "pressure_pred_pa", "pressure_error_pred_minus_cfd_pa"],
        "selfmax_fields": ["pressure_cfd_selfmax", "pressure_pred_selfmax", "pressure_selfmax_error_pred_minus_cfd"],
        "target": "Relative pressure p − p_ref", "unit": "Pa",
        "kind": "历史 V4 EMA 壁面相对压力 · STL Gaussian 回插",
        "selfmax_unit_interval": False, "metric_scope": "正式体内同点；面片仅壁面",
        "metrics_key": "visualization_metrics_pressure",
        "normalization_key": "display_normalization_pressure",
        "caption_checkpoint": "s1234 / last_converged（epoch9999，未收敛）",
        "caption_selection": "病例角色按派生 WSS R² 选取，不是压力独立排名；本图为同例相对压力。历史 V4 稳态 peak。",
    },
}
DEFAULT_MODELS = ("X5X11", "PF6")
ROLES = ("best", "median", "worst")
ROLE_TEXT = {"best": "Best / 最好", "median": "Median / 中位代表", "worst": "Worst / 最差"}


def setup_font():
    for filename in ("NotoSansCJK-Regular.ttc", "NotoSansCJK-Bold.ttc"):
        font_path = Path("/usr/share/fonts/opentype/noto") / filename
        if font_path.exists():
            fontManager.addfont(str(font_path))
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 12,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_surface(path, field_names):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly.GetNumberOfPolys() == 0:
        raise ValueError(f"No polygons in {path}; a point cloud is not a surface")
    # The delivered STL contains triangles; reject unexpected topology instead
    # of fitting a different geometry for a preview.
    cells = poly.GetPolys()
    offsets = vtk_to_numpy(cells.GetOffsetsArray()).astype(np.int64)
    indices = vtk_to_numpy(cells.GetConnectivityArray()).astype(np.int64)
    if not np.all(np.diff(offsets) == 3):
        raise ValueError(f"Non-triangle polygons in {path}")
    triangles = indices.reshape(-1, 3)
    xyz = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    fields = {}
    for name in field_names:
        array = poly.GetPointData().GetArray(name)
        if array is None:
            raise ValueError(f"Missing field {name} in {path}")
        values = vtk_to_numpy(array).astype(float)
        if values.shape != (len(xyz),):
            raise ValueError(f"Expected scalar point field {name}: {values.shape}")
        fields[name] = values
    cfd, pred, err = (fields[name] for name in field_names)
    valid = np.isfinite(cfd) & np.isfinite(pred) & np.isfinite(err)
    if not valid.any():
        raise ValueError(f"No common finite field values in {path}")
    if not np.allclose(err[valid], pred[valid] - cfd[valid], rtol=1e-4, atol=1e-5):
        raise ValueError(f"Signed error does not match Pred - CFD in {path}")
    return xyz, triangles, fields


def camera_projection(xyz):
    # One fixed camera in the registered atlas frame, for all cases and models.
    # Camera x is 20 degrees from atlas x; camera up is 8 degrees from atlas z.
    theta, phi = np.deg2rad([20, 8])
    right = np.array([np.cos(theta), np.sin(theta), 0.0])
    up = np.array([np.sin(phi) * np.sin(theta), -np.sin(phi) * np.cos(theta), np.cos(phi)])
    depth = np.cross(right, up)
    basis = np.stack([right, up, depth], axis=1)
    origin = (xyz.max(axis=0) + xyz.min(axis=0)) / 2
    projected = (xyz - origin) @ basis
    return projected, basis, origin


def render_one(root, model, role, outdir, dpi, mode="physical"):
    spec = SPECS[model]
    case_dir = root / spec.get("folder", model) / role
    mesh_path = case_dir / spec["filename"]
    source_manifest = case_dir / "manifest.json"
    manifest = json.loads(source_manifest.read_text())
    is_selfmax = mode == "selfmax"
    field_names = spec["selfmax_fields"] if is_selfmax else spec["fields"]
    xyz, triangles, fields = load_surface(mesh_path, field_names)
    normalization = None
    if is_selfmax:
        normalization = manifest.get(spec.get("normalization_key", "display_normalization"))
        if not isinstance(normalization, dict):
            raise ValueError(f"Missing display_normalization in {source_manifest}")
        required = ("cfd_max", "pred_max", "denominator_scope")
        if any(key not in normalization for key in required):
            raise ValueError(f"Missing normalization denominator/scope in {source_manifest}")
        denominators = [float(normalization[key]) for key in required[:2]]
        if any(not math.isfinite(value) or value == 0 for value in denominators):
            raise ValueError(f"Invalid selfmax denominators in {source_manifest}: {denominators}")
        if spec.get("selfmax_unit_interval") and any(value < 0 for value in denominators):
            raise ValueError(f"WSS magnitude requires positive maxima: {source_manifest}")
        # Verify against the recorded *original full evaluation-domain* maxima;
        # never replace them with maxima of the smoothed surface.
        _, _, physical_fields = load_surface(mesh_path, spec["fields"])
        for normalized_name, physical_name, denominator in zip(field_names[:2], spec["fields"][:2], denominators):
            actual, expected = fields[normalized_name], physical_fields[physical_name] / denominator
            if not np.allclose(actual, expected, rtol=2e-5, atol=2e-6, equal_nan=True):
                raise ValueError(f"Selfmax field does not use recorded denominator: {mesh_path}/{normalized_name}")
    projected, basis, origin = camera_projection(xyz)
    # Observer is on positive depth, looking toward the origin: negative depth
    # is farther and must be painted first.
    face_order = np.argsort(projected[triangles, 2].mean(axis=1), kind="stable")
    sorted_triangles = triangles[face_order]
    polygons = projected[sorted_triangles, :2]
    cfd, pred, err = (fields[name] for name in field_names)
    joint = np.concatenate([cfd[np.isfinite(cfd)], pred[np.isfinite(pred)]])
    vmin, vmax = float(joint.min()), float(joint.max())
    if is_selfmax and spec.get("selfmax_unit_interval"):
        if vmin < -1e-6 or vmax > 1 + 1e-6:
            raise ValueError(f"WSS selfmax out of 0–1 range: {mesh_path}: {vmin}, {vmax}")
        vmin, vmax = 0.0, 1.0
    elif spec.get("selfmax_unit_interval"):
        vmin = min(0.0, vmin)
    if vmax <= vmin:
        vmax = vmin + 1e-12
    emax = max(float(np.nanmax(np.abs(err))), 1e-12)
    norms = [Normalize(vmin, vmax), Normalize(vmin, vmax), Normalize(-emax, emax)]
    cmaps = ["viridis", "viridis", "RdBu_r"]
    extent = np.ptp(projected[:, :2], axis=0)
    # Keep all three panels on exactly the same camera and spatial bounds.
    lo, hi = projected[:, :2].min(axis=0), projected[:, :2].max(axis=0)
    padding = max(float(extent.max()) * 0.035, 1e-6)
    limits = [(lo[i] - padding, hi[i] + padding) for i in (0, 1)]
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.035, 0.958, f"{model}  |  {ROLE_TEXT[role]}  |  {manifest['case_id']}",
             fontsize=19, weight="bold", color="#142c43", va="top")
    subtitle = spec["kind"]
    if is_selfmax:
        subtitle += "  |  Selfmax 各自除最大值：分布比较，幅值差已移除"
    fig.text(0.035, 0.905, subtitle, fontsize=13 if is_selfmax else 14, color="#526474")
    titles = (["CFD / CFDmax", "Pred / Predmax", "Signed error = Prednorm − CFDnorm"]
              if is_selfmax else ["CFD 真值", "Pred 预测", "Signed error = Pred − CFD"])
    for i, (values, cmap_name, norm, title) in enumerate(zip([cfd, pred, err], cmaps, norms, titles)):
        # A pale panel background keeps white (near-zero) signed-error regions
        # visible without altering the scalar colors with artificial lighting.
        fig.add_artist(FancyBboxPatch((0.035 + i * 0.329, 0.264), 0.30, 0.577,
                                      boxstyle="round,pad=0.004,rounding_size=0.008",
                                      transform=fig.transFigure, facecolor="#edf1f5",
                                      edgecolor="none", zorder=-1))
        ax = fig.add_axes([0.025 + i * 0.329, 0.275, 0.32, 0.55])
        cmap = matplotlib.colormaps[cmap_name]
        face_values = values[sorted_triangles]
        valid_faces = np.isfinite(face_values).all(axis=1)
        rgba = np.full((len(sorted_triangles), 4), [0.80, 0.81, 0.83, 1.0])
        rgba[valid_faces] = cmap(norm(face_values[valid_faces].mean(axis=1)))
        ax.add_collection(PolyCollection(polygons, facecolors=rgba, edgecolors="none",
                                         linewidths=0, antialiaseds=False, rasterized=True))
        ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
        ax.axis("off")
        # Draw titles in figure coordinates so differing aspect ratios do not
        # change the alignment of the three panels.
        fig.text(0.185 + i * 0.329, 0.853, title, ha="center", fontsize=15, weight="bold", color="#193047")
    field_cax = fig.add_axes([0.08, 0.217, 0.535, 0.018])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norms[0], cmap=cmaps[0]), cax=field_cax, orientation="horizontal")
    if is_selfmax:
        field_label = ("WSS / own max（无量纲）— CFD / Pred 共用 0–1"
                       if spec.get("selfmax_unit_interval") else "p_rel / own max（无量纲）— CFD / Pred 共用完整范围，保留负值")
    else:
        field_label = f"{spec['target']} ({spec['unit']}) — CFD / Pred 共用完整范围"
    cb.set_label(field_label, fontsize=11)
    err_cax = fig.add_axes([0.72, 0.217, 0.235, 0.018])
    cb_err = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norms[2], cmap=cmaps[2]), cax=err_cax, orientation="horizontal")
    error_label = "Prednorm − CFDnorm（无量纲）— 对称完整范围" if is_selfmax else f"Pred − CFD ({spec['unit']}) — 对称完整范围"
    cb_err.set_label(error_label, fontsize=10 if is_selfmax else 11)
    for colorbar in (cb, cb_err):
        colorbar.ax.tick_params(labelsize=10)
        colorbar.outline.set_linewidth(0.5)
    metrics = manifest.get(spec.get("metrics_key", "visualization_metrics"), {})
    metric = metrics.get("r2")
    if metric is None or not math.isfinite(float(metric)):
        raise ValueError(f"Missing {spec.get('metrics_key', 'visualization_metrics')}.r2; refusing to substitute the selection mean: {source_manifest}")
    peak = manifest.get("peak_step", "see manifest")
    metric_scope = spec.get("metric_scope", "原始同点")
    ckpt_label = spec.get("caption_checkpoint", "s1234 / ckpt_best")
    sel_label = spec.get(
        "caption_selection",
        "病例角色按三 seed 病例 R² 均值选取；图中场与 R² 均来自 s1234。插值云图不重新定义正式指标。",
    )
    if is_selfmax:
        fig.text(0.035, 0.124,
                 f"{ckpt_label} {metric_scope}物理量 R² = {float(metric):.4f}（仅作参考，不是 selfmax R²）  |  peak step {peak}",
                 fontsize=11.5, color="#193047")
        fig.text(0.035, 0.089,
                 f"原始完整评估域最大值：CFDmax = {denominators[0]:.6g} Pa；Predmax = {denominators[1]:.6g} Pa。两者分别作分母。",
                 fontsize=11, color="#193047")
        scope_label = "完整壁面" if spec.get("selfmax_unit_interval") else "完整壁面∪体内"
        gaussian_note = ("Gaussian 回插后最大值可小于 1，未再次缩放。" if all(value > 0 for value in denominators)
                         else "存在负最大值分母，除法会反转符号；保留完整范围，未再次缩放。")
        fig.text(0.035, 0.056,
                 f"分母范围：原始{scope_label}同点评估域；{gaussian_note}",
                 fontsize=10.5, color="#556674")
        fig.text(0.035, 0.024, sel_label + " 各自幅值差已移除，不能据此判断物理量幅值准确性。",
                 fontsize=10.5, color="#556674")
    else:
        fig.text(0.035, 0.118,
                 f"{ckpt_label} {metric_scope} R² = {float(metric):.4f}    |    peak step {peak}    |    "
                 f"STL: {len(xyz):,} vertices / {len(triangles):,} triangles",
                 fontsize=12, color="#193047")
        fig.text(0.035, 0.079, sel_label, fontsize=10.5, color="#556674")
        projection_note = "STL surface · Gaussian interpolation · full, unclipped color range."
        fig.text(0.035, 0.043, projection_note + "  Orthographic triangle view; gray = unmapped.",
                 fontsize=10.5, color="#556674")
    outfile = outdir / f"{model}_{role}_surface{'_selfmax' if is_selfmax else ''}_triptych.png"
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)
    return {
        "model": model, "role": role, "case_id": manifest["case_id"],
        "source_mesh": str(mesh_path.relative_to(root)), "source_mesh_sha256": sha256(mesh_path),
        "source_manifest": str(source_manifest.relative_to(root)), "source_manifest_sha256": sha256(source_manifest),
        "preview": str(outfile.relative_to(root)), "field_keys": field_names,
        "mode": mode, "display_normalization": normalization,
        "normalized_metrics_recomputed": False,
        "color_range": [vmin, vmax], "signed_error_range": [-emax, emax],
        "metric_value": float(metric),
        "metric_source": f"manifest.{spec.get('metrics_key', 'visualization_metrics')}.r2",
        "metric_note": "original colocated s1234 predictions; not recomputed on interpolated mesh",
        "n_vertices": len(xyz), "n_triangles": len(triangles),
        "camera": {"type": "orthographic", "basis_columns_right_up_depth": basis.tolist(), "origin_mm": origin.tolist()},
        "rendering": "complete VTP triangle topology; depth-sorted polygon projection; mean vertex scalar per triangle; no lighting",
    }


def contact_sheet(root, entries, outdir, mode="physical", models=None):
    models = [m for m in (models or list(DEFAULT_MODELS)) if any(e["model"] == m for e in entries)]
    width, header_height, cell_width, cell_height = 3000, 145, 1000, 562
    sheet = Image.new("RGB", (width, header_height + max(len(models), 1) * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    fontfile = findfont(FontProperties(family="Noto Sans CJK JP"))
    font = ImageFont.truetype(fontfile, 40)
    smallfont = ImageFont.truetype(fontfile, 27)
    if models == ["VF6_wss"]:
        title = ("VELWSS2 派生 WSS · 各自 max 归一化" if mode == "selfmax"
                 else "VELWSS2 派生 WSS · Gaussian 面片云图")
        subtitle = "列：Best / Median / Worst    VF6 预测速度 → 冻结 Profile-Secant V3"
    elif models == ["X5D_v51"]:
        title = ("X5D_v51 代表病例 · 各自 max 归一化 · 分布比较（幅值差已移除）" if mode == "selfmax"
                 else "X5D_v51 代表病例 · Gaussian 面片云图总览")
        subtitle = "列：Best / Median / Worst    v5.1 密度增广直接 WSS · s1234"
    elif models == ["V4_EMA"]:
        title = ("V4 EMA 派生 WSS · 各自 max 归一化" if mode == "selfmax"
                 else "V4 EMA 派生 WSS · Gaussian 面片云图")
        subtitle = "列：Best / Median / Worst    历史 PointNet BC+PDE-EMA · 冻结 V3 派生 WSS"
    elif models == ["V4_EMA_pressure"]:
        title = ("V4 EMA 相对压力 · 各自 max 归一化" if mode == "selfmax"
                 else "V4 EMA 相对压力 · Gaussian 面片云图")
        subtitle = "列：Best / Median / Worst    病例按派生 WSS 选取；本图为同例壁面相对压力"
    elif mode == "selfmax":
        title = "壁面代表病例 · 各自 max 归一化 · 分布比较（幅值差已移除）"
        subtitle = "列：Best / Median / Worst    行：X5X11 WSS / PF6 相对压力    分母取原始完整评估域；压力保留负值"
    else:
        title = "壁面代表病例 · Gaussian 面片云图总览"
        subtitle = "列：Best / Median / Worst      行：X5X11 WSS / PF6 压力      VF6 保留体内速度点云，另在 ParaView 查看"
    draw.text((35, 16), title, font=font, fill="#142c43")
    draw.text((35, 81), subtitle, font=smallfont, fill="#526474")
    lookup = {(entry["model"], entry["role"]): entry for entry in entries}
    for row, model in enumerate(models):
        for col, role in enumerate(ROLES):
            entry = lookup.get((model, role))
            if entry is None:
                continue
            with Image.open(root / entry["preview"]) as img:
                img = img.convert("RGB")
                img.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
                sheet.paste(img, (col * cell_width, header_height + row * cell_height))
    if models == list(DEFAULT_MODELS):
        outfile = outdir / f"surface{'_selfmax' if mode == 'selfmax' else ''}_triptychs_contact_sheet.png"
    else:
        stem = "_".join(models)
        outfile = outdir / f"{stem}_surface{'_selfmax' if mode == 'selfmax' else ''}_triptychs_contact_sheet.png"
    sheet.save(outfile)
    return outfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--models", nargs="+", choices=list(SPECS), default=list(DEFAULT_MODELS))
    parser.add_argument("--roles", nargs="+", choices=ROLES, default=list(ROLES))
    parser.add_argument("--modes", nargs="+", choices=("physical", "selfmax"), default=["physical", "selfmax"])
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    setup_font()
    root = args.root.resolve()
    outdir = root / "plots"
    outdir.mkdir(exist_ok=True)
    entries = []
    contacts = {}
    for mode in args.modes:
        mode_entries = []
        for model in args.models:
            for role in args.roles:
                entry = render_one(root, model, role, outdir, args.dpi, mode)
                mode_entries.append(entry)
                entries.append(entry)
                print(f"Rendered {entry['preview']} | raw s1234 R2={entry['metric_value']:.6f}", flush=True)
        contacts[mode] = str(contact_sheet(root, mode_entries, outdir, mode, args.models).relative_to(root))
    result = {"generator": Path(__file__).name, "case_count": len({(e['model'], e['role']) for e in entries}),
              "preview_count": len(entries), "contact_sheets": contacts, "previews": entries}
    manifest_name = "preview_manifest.json" if list(args.models) == list(DEFAULT_MODELS) else f"{'_'.join(args.models)}_preview_manifest.json"
    (outdir / manifest_name).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {len(entries)} previews and {len(contacts)} contact sheets", flush=True)


if __name__ == "__main__":
    main()
