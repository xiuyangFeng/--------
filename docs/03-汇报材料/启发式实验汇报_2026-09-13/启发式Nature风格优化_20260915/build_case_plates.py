#!/usr/bin/env python3
"""Evidence-led case plates from verified VTPs; never modifies source data.

Contract: spatial diagnosis, not a new model ranking or mechanism claim.
Rows show reference, prediction and signed discrepancy; columns establish a
best-case example, a median representative and the known failure boundary.
Structural adaptation of the previously verified Python field renderer uses
the same field mapping/camera and full geometry.  VF6 now draws every interior
point; no plotting subsample survives in this independent figure workflow.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from plot_style import OUT, ROOT, INK, MUTED, TEAL, LINE, page, save, sha


SOURCE = OUT.parent / "启发式v2_代表病例后处理_20260914"
ROLES = ("best", "median", "worst")
ROLE_LABELS = ("Best · 最好", "Median · 中位代表", "Worst · 最差")
FIELD_CMAP = LinearSegmentedColormap.from_list("case_field", ["#9EBDD0", "#7FB7BC", "#428D95", "#174659"])
ERROR_CMAP = LinearSegmentedColormap.from_list("case_error", ["#326A91", "#F8F8F8", "#B14C46"])
SPECS = {
    "X5X11": {
        "stem": "10_x5x11_cases", "prefix": "wss", "target": "WSS", "unit": "Pa",
        "physical_fields": ["wss_cfd_pa", "wss_pred_pa", "wss_error_pred_minus_cfd_pa"],
        "scope": "完整壁面", "mesh": "surface_gaussian.vtp",
        "title": "X5X11：高 R² 病例仍需检查局部 WSS 误差",
        "selfmax_title": "X5X11：去掉各自幅值后，空间差异还剩多少？",
        "subtitle": "从最好到最差病例，查看误差位置是否改变；图像观察仍需区域指标验证。",
        "questions": ["观察哪里？", "分叉、扩张段", "与出口附近", "", "下一步验证", "相同解剖区域的", "误差与热点重合率"],
    },
    "VF6": {
        "stem": "11_vf6_cases", "prefix": "speed", "target": "Speed", "unit": "m/s",
        "physical_fields": ["speed_cfd", "speed_pred", "speed_error_pred_minus_cfd"],
        "scope": "完整体内", "mesh": None,
        "title": "VF6：速度整体得分能否反映局部失真？",
        "selfmax_title": "VF6：幅值差移除后，速度分布仍有何差异？",
        "subtitle": "绘制全部体内速度点；正交点云投影用于观察，不能替代连续截面。",
        "questions": ["观察哪里？", "弯曲、分叉位置", "与低速区域", "", "下一步验证", "同位置截面", "和近壁速度剖面"],
    },
    "PF6": {
        "stem": "12_pf6_cases", "prefix": "pressure", "target": "p − p_ref", "unit": "Pa",
        "physical_fields": ["pressure_cfd_pa", "pressure_pred_pa", "pressure_error_pred_minus_cfd_pa"],
        "scope": "完整壁面∪体内", "mesh": "surface_gaussian.vtp",
        "title": "PF6：压力大趋势与局部差异需要分开判断",
        "selfmax_title": "PF6：各自除最大值后，压力分布如何变化？",
        "subtitle": "壁面展示相对压力；原始 R² 来自壁面∪体内同点评估，保留压力参考零点。",
        "questions": ["差异来自哪里？", "沿程压降", "还是局部区域？", "", "下一步验证", "同路径压力曲线", "与区域偏差"],
    },
}


def camera(xyz):
    theta, phi = np.deg2rad([20, 8])
    right = np.array([np.cos(theta), np.sin(theta), 0.0])
    up = np.array([np.sin(phi) * np.sin(theta), -np.sin(phi) * np.cos(theta), np.cos(phi)])
    depth = np.cross(right, up)
    # Roll the displayed camera by 90 degrees. Coordinates and vectors in the
    # source VTP stay unchanged; horizontal arteries fit the comparison grid.
    basis = np.stack([-up, right, depth], axis=1)
    origin = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    return (xyz - origin) @ basis, basis, origin


def read_case(model, role):
    spec = SPECS[model]
    case_dir = SOURCE / model / role
    mp = case_dir / "manifest.json"
    manifest = json.loads(mp.read_text())
    if manifest["schema_version"] != 3 or manifest["model"] != model:
        raise ValueError(f"Unexpected verified source contract: {model}/{role}")
    source = case_dir / (spec["mesh"] or manifest["files"]["pointcloud"])
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(source))
    reader.Update()
    poly = reader.GetOutput()
    xyz = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    if not np.isfinite(xyz).all():
        raise ValueError("Nonfinite coordinates in source")
    triangles = None
    if model == "VF6":
        if poly.GetNumberOfPolys() or manifest["files"].get("gaussian_surface"):
            raise ValueError("VF6 must remain a volume point cloud")
        if not np.all(vtk_to_numpy(poly.GetPointData().GetArray("point_kind")) == 1):
            raise ValueError("Expected exclusively interior velocity points")
    else:
        if not poly.GetNumberOfPolys():
            raise ValueError("Wall display must have real triangle faces")
        offsets = vtk_to_numpy(poly.GetPolys().GetOffsetsArray())
        if not np.all(np.diff(offsets) == 3):
            raise ValueError("Expected the delivered triangle topology")
        triangles = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).astype(np.int64).reshape(-1, 3)
    prefix = spec["prefix"]
    normalized_fields = [f"{prefix}_cfd_selfmax", f"{prefix}_pred_selfmax", f"{prefix}_selfmax_error_pred_minus_cfd"]
    fields = {}
    for name in spec["physical_fields"] + normalized_fields:
        arr = poly.GetPointData().GetArray(name)
        if arr is None:
            raise ValueError(f"Missing source field {name}")
        fields[name] = vtk_to_numpy(arr).astype(float)
        if fields[name].shape != (len(xyz),) or not np.isfinite(fields[name]).all():
            raise ValueError(f"Invalid source field {name}; no silent exclusion is allowed")
    for names in (spec["physical_fields"], normalized_fields):
        truth, pred, error = [fields[name] for name in names]
        if not np.allclose(error, pred - truth, rtol=2e-5, atol=2e-6):
            raise ValueError("Signed error must equal prediction minus reference")
    norm = manifest["display_normalization"]
    for name in ("cfd", "pred"):
        denominator = float(norm[f"{name}_max"])
        if not math.isfinite(denominator) or denominator == 0 or not norm[f"{name}_denominator"]["valid"]:
            raise ValueError("Recorded selfmax denominator is invalid")
        physical_name = spec["physical_fields"][0 if name == "cfd" else 1]
        if not np.allclose(fields[f"{prefix}_{name}_selfmax"], fields[physical_name] / denominator, rtol=2e-5, atol=2e-6):
            raise ValueError("Selfmax must preserve the original full-domain denominator")
    projected, basis, origin = camera(xyz)
    if triangles is None:
        order = np.argsort(projected[:, 2], kind="stable")
    else:
        order = np.argsort(projected[triangles, 2].mean(axis=1), kind="stable")
    return {"role": role, "manifest": manifest, "manifest_path": mp, "source": source,
            "xyz": xyz, "triangles": triangles, "fields": fields,
            "normalized_fields": normalized_fields, "projected": projected,
            "order": order, "basis": basis, "origin": origin}


def data_limits(projected, width, height):
    lower, upper = projected[:, :2].min(axis=0), projected[:, :2].max(axis=0)
    center = (lower + upper) / 2
    extent = (upper - lower) * 1.12
    box_ratio = width * 16 / (height * 9)
    extent[0] = max(extent[0], extent[1] * box_ratio)
    extent[1] = extent[0] / box_ratio
    return [(float(center[i] - extent[i] / 2), float(center[i] + extent[i] / 2)) for i in (0, 1)]


def add_field(ax, case, values, norm, cmap, limits):
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc="#F3F5F6", ec="none", zorder=-1))
    projected = case["projected"]
    if case["triangles"] is None:
        index = case["order"]
        # Every source point is drawn. Tiny opaque rasterized marks preserve
        # the field colour; no sample, clip, field threshold or alpha averaging.
        ax.scatter(projected[index, 0], projected[index, 1], c=values[index],
                   s=0.035, cmap=cmap, norm=norm, linewidths=0, edgecolors="none", rasterized=True)
    else:
        tri = case["triangles"][case["order"]]
        face_values = values[tri].mean(axis=1)
        ax.add_collection(PolyCollection(projected[tri, :2], facecolors=cmap(norm(face_values)),
                                         linewidths=0, edgecolors="none", antialiaseds=False, rasterized=True))
    ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
    ax.axis("off")


def colorbar(fig, rect, norm, cmap, label):
    ax = fig.add_axes(rect)
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation="horizontal")
    bar.ax.tick_params(labelsize=9, length=2, pad=5)
    if norm.vmin == 0 and norm.vmax == 1:
        bar.set_ticks([0, .5, 1])
    else:
        bar.locator = MaxNLocator(nbins=3)
        bar.update_ticks()
    bar.outline.set_linewidth(.45)
    bar.set_label(label, fontsize=10, labelpad=3)
    return ax


def draw_figure(model, cases, selfmax=False):
    spec = SPECS[model]
    stem = spec["stem"] + ("_selfmax" if selfmax else "")
    title = spec["selfmax_title"] if selfmax else spec["title"]
    if selfmax:
        subtitle = "CFD 与 Pred 分别除以原始完整评估域最大值；幅值差已移除，图中 R² 仅为原物理量参考。"
    else:
        subtitle = spec["subtitle"]
    footer = "三 seed 均值选例；展示 s1234 / ckpt_best，peak 1162。每列独立色标，同列 CFD/Pred 共用；误差 = Pred − CFD。"
    if selfmax and model in ("X5X11", "VF6"):
        footer = "三 seed 均值选例；展示 s1234 / ckpt_best，peak 1162。CFD/Pred 统一 0–1；误差为归一化 Pred − CFD，逐列对称色标。"
    elif selfmax:
        footer = "三 seed 均值选例；展示 s1234 / ckpt_best，peak 1162。同列 CFD/Pred 共用完整色标；归一化误差逐列对称，保留负值。"
    fig = page(title, subtitle, kicker=f"{'附录 · SELFMAX' if selfmax else '空间诊断'}  /  {model}", footer=footer)
    # Equal final plot rectangles (including physical axis ratio) are explicit.
    lefts, width, height = [0.145, 0.370, 0.595], 0.185, 0.135
    bottoms = [0.555, 0.395, 0.235]
    row_text = ["CFD", "Pred", "Error"]
    axes, sources, rows, panel_notes = [], [], [], []
    for row, label in enumerate(row_text):
        fig.text(.052, bottoms[row] + height / 2, label, fontsize=17, weight="bold", va="center")
    fig.text(.815, .750, "讨论焦点", fontsize=15, weight="bold", color=TEAL)
    if selfmax:
        questions = ["仅比较分布", "幅值差已移除", "", "对照物理量页", "差图改变多少？", "", "压力保留负值" if model == "PF6" else "全病例统一 0–1"]
    else:
        questions = spec["questions"]
    for i, line in enumerate(questions):
        fig.text(.815, .694 - i * .043, line, fontsize=14 if i not in (0, 4) else 15,
                 weight="bold" if i in (0, 4) else "normal", color=INK if i in (0, 4) else MUTED)
    if model == "VF6":
        fig.text(.815, .295, "全部体内点", fontsize=13, weight="bold", color=INK)
        fig.text(.815, .262, "无抽样 · 无面片", fontsize=12, color=MUTED)
        fig.text(.815, .229, "投影不是截面", fontsize=12, color=MUTED)
    elif selfmax:
        fig.text(.815, .295, "原始分母固定", fontsize=13, weight="bold", color=INK)
        fig.text(.815, .262, "回插后不再取 max", fontsize=12, color=MUTED)
    for col, case in enumerate(cases):
        mp = case["manifest"]
        case_id = mp["case_id"]
        short_id = case_id.split("/")[-1]
        r2 = float(mp["visualization_metrics"]["r2"])
        xc = lefts[col] + width / 2
        fig.text(xc, .783, ROLE_LABELS[col], fontsize=16, ha="center", weight="bold")
        fig.text(xc, .752, short_id, fontsize=12.5, ha="center")
        fig.text(xc, .724, f"s1234  R² = {r2:.3f}", fontsize=12, ha="center", color=MUTED)
        names = case["normalized_fields"] if selfmax else spec["physical_fields"]
        values = [case["fields"][name] for name in names]
        vmin, vmax = min(float(v.min()) for v in values[:2]), max(float(v.max()) for v in values[:2])
        if selfmax and model in ("X5X11", "VF6"):
            if vmin < -1e-6 or vmax > 1 + 1e-6:
                raise ValueError("Nonnegative scalar selfmax exceeds declared 0–1 scale")
            vmin, vmax = 0.0, 1.0
        elif model in ("X5X11", "VF6"):
            vmin = min(0.0, vmin)
        if vmax <= vmin:
            raise ValueError("Degenerate source field color range")
        emax = max(float(np.max(np.abs(values[2]))), 1e-12)
        field_norm, error_norm = Normalize(vmin, vmax), Normalize(-emax, emax)
        limits = data_limits(case["projected"], width, height)
        for row, (value, name) in enumerate(zip(values, names)):
            ax = fig.add_axes([lefts[col], bottoms[row], width, height])
            panel = chr(ord("a") + row * 3 + col)
            ax.set_gid(panel)
            add_field(ax, case, value, field_norm if row < 2 else error_norm,
                      FIELD_CMAP if row < 2 else ERROR_CMAP, limits)
            ax.text(.018, .920, panel, transform=ax.transAxes, fontsize=11, weight="bold", va="top")
            axes.append(ax)
            panel_notes.append({"panel": panel, "model": model, "role": case["role"], "case_id": case_id,
                "field": name, "evidence_role": ["reference anatomy/field", "same-point prediction", "local discrepancy/failure boundary"][row],
                "n_source_points": len(case["xyz"]), "n_points_excluded": 0,
                "uncertainty": "none: one fixed checkpoint field, not a seed aggregate", "limits_mm": limits})
        common_label = f"{spec['target']} ({spec['unit']})" if not selfmax else f"{spec['target']} / own max"
        if selfmax and model == "PF6":
            common_label = "(p − p_ref) / own max"
        if selfmax and model in ("X5X11", "VF6"):
            if col == 0:
                colorbar(fig, [lefts[0], .188, lefts[-1] + width - lefts[0], .009], field_norm, FIELD_CMAP,
                         common_label + " · CFD/Pred 共用")
        else:
            colorbar(fig, [lefts[col], .188, width, .009], field_norm, FIELD_CMAP, common_label)
        colorbar(fig, [lefts[col], .122, width, .009], error_norm, ERROR_CMAP, "Δ normalized" if selfmax else f"Δ ({spec['unit']})")
        if selfmax:
            normalization = mp["display_normalization"]
            if col == 0:
                fig.text(.815, .192, f"原始 max C/P ({spec['unit']})", fontsize=10, color=MUTED)
            fig.text(.815, .162 - col * .031,
                     f"{case['role'].capitalize()}: {normalization['cfd_max']:.3g} / {normalization['pred_max']:.3g}",
                     fontsize=10, color=MUTED)
        ntri = 0 if case["triangles"] is None else len(case["triangles"])
        row = {"model": model, "role": case["role"], "case_id": case_id,
               "physical_r2_s1234": r2, "selection_three_seed_r2_mean": mp["selection"]["r2_case_mean"],
               "physical_metric_scope": spec["scope"], "source_point_count": len(case["xyz"]),
               "drawn_point_count": len(case["xyz"]), "triangle_count": ntri, "excluded_point_count": 0,
               "field_min": vmin, "field_max": vmax, "signed_error_min": -emax, "signed_error_max": emax,
               "cfd_max_denominator": mp["display_normalization"]["cfd_max"],
               "pred_max_denominator": mp["display_normalization"]["pred_max"],
               "denominator_scope": mp["display_normalization"]["denominator_scope"],
               "coordinate_frame": mp["frame"], "unit": "dimensionless" if selfmax else spec["unit"]}
        rows.append(row)
        sources.extend([case["source"], case["manifest_path"]])
    # The custom axes were inserted column by column. Explicit comparable rows
    # and columns exclude colorbars and side notes from image alignment.
    ordered = sorted(axes, key=lambda ax: ax.get_gid())
    source_dir = OUT / "source_data"
    with (source_dir / f"{stem}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    source_meta = {"model": model, "mode": "selfmax" if selfmax else "physical",
        "field_mapping": {"rows": names, "columns": list(ROLES)}, "case_summary": rows,
        "source_vtps": [{"case_id": c["manifest"]["case_id"], "vtp": str(c["source"].resolve()),
                         "vtp_sha256": sha(c["source"]), "manifest": str(c["manifest_path"].resolve()),
                         "manifest_sha256": sha(c["manifest_path"]),
                         "display_normalization": c["manifest"]["display_normalization"],
                         "visualization_metrics": c["manifest"]["visualization_metrics"],
                         "selection": c["manifest"]["selection"],
                         "camera_basis_columns": c["basis"].tolist(), "camera_origin_mm": c["origin"].tolist()}
                        for c in cases],
        "archetype": "image plate + fixed-checkpoint quantitative reference",
        "transforms": "same-source fields; fixed orthographic camera with 90-degree display roll; depth sorting; no source modification",
        "surface_rendering": "complete delivered Gaussian triangle topology; face colour is mean of its three vertex values",
        "volume_rendering": "complete original interior point cloud; opaque rasterized marks; no downsampling or interpolation",
        "exclusions": [], "normalized_r2_recomputed": False, "panels": panel_notes,
        "caveats": ["projection occludes deeper anatomy/points", "per-case color scales differ except nonnegative selfmax 0–1",
                    "spatial observations require separate region/slice quantification", "selfmax removes separate amplitude scales"]}
    (source_dir / f"{stem}.json").write_text(json.dumps(source_meta, ensure_ascii=False, indent=2) + "\n")
    claim = title + "；这是空间诊断问题，尚非机制或区域误差量化结论。"
    result = save(fig, stem, claim=claim, sources=sources, axes=ordered,
                  row_groups=[["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]],
                  column_groups=[["a", "d", "g"], ["b", "e", "h"], ["c", "f", "i"]],
                  note="All delivered source vertices/interior points retained; colourbars excluded through explicit image axes.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(SPECS), default=list(SPECS))
    parser.add_argument("--modes", nargs="+", choices=["physical", "selfmax"], default=["physical", "selfmax"])
    args = parser.parse_args()
    for directory in ("figures", "qa", "source_data"):
        (OUT / directory).mkdir(exist_ok=True)
    for model in args.models:
        cases = [read_case(model, role) for role in ROLES]
        for mode in args.modes:
            draw_figure(model, cases, selfmax=mode == "selfmax")


if __name__ == "__main__":
    main()
