#!/usr/bin/env python3
"""Render VF6 interior point-cloud previews, without surface interpolation.

Use the GNN Python environment.  All three panels and both display modes use
identical deterministic sample indices, camera, and depth order.  Color limits
are obtained from the full source point cloud, not the plotting subsample.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from render_surface_previews import ROLES, ROLE_TEXT, camera_projection, setup_font, sha256


PHYSICAL_FIELDS = ["speed_cfd", "speed_pred", "speed_error_pred_minus_cfd"]
SELFMAX_FIELDS = ["speed_cfd_selfmax", "speed_pred_selfmax", "speed_selfmax_error_pred_minus_cfd"]
VELOCITY_SPECS = {
    "VF6": {
        "folder": "VF6",
        "pointcloud_key": "pointcloud",
        "metrics_key": "visualization_metrics",
        "normalization_key": "display_normalization",
        "require_no_gaussian": True,
        "title": "VF6",
        "caption_checkpoint": "s1234 / ckpt_best",
        "caption_selection": "病例角色按三 seed R² 均值选取；场来自 s1234。正交投影会遮挡深处点；可在 ParaView 中旋转全量点云查看。",
        "contact_title": "VF6 体内速度点云",
        "preview_prefix": "VF6",
        "manifest_name": "velocity_preview_manifest.json",
    },
    "V4_EMA": {
        "folder": "V4_EMA",
        "pointcloud_key": "interior_velocity",
        "metrics_key": "visualization_metrics_speed",
        "normalization_key": "display_normalization_speed",
        "require_no_gaussian": False,
        "title": "V4 EMA",
        "caption_checkpoint": "s1234 / last_converged（未收敛）",
        "caption_selection": "病例角色按派生 WSS R² 选取，不是速度独立排名；本图为同例体内速度。历史 V4 稳态 peak。",
        "contact_title": "V4 EMA 体内速度点云",
        "preview_prefix": "V4_EMA",
        "manifest_name": "V4_EMA_velocity_preview_manifest.json",
    },
}


def load_case(root, role, outdir, max_points, sample_seed, model="VF6"):
    spec = VELOCITY_SPECS[model]
    case_dir = root / spec["folder"] / role
    manifest_path = case_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if spec["require_no_gaussian"] and (
        manifest["model"] != model or manifest["files"].get("gaussian_surface")
    ):
        raise ValueError(f"Expected {model} interior-only point cloud: {manifest_path}")
    mesh_path = case_dir / manifest["files"][spec["pointcloud_key"]]
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(mesh_path))
    reader.Update()
    cloud = reader.GetOutput()
    if cloud.GetNumberOfPolys() != 0:
        raise ValueError(f"{model} preview requires point cloud, found surface faces: {mesh_path}")
    xyz = vtk_to_numpy(cloud.GetPoints().GetData()).astype(float)
    if not np.isfinite(xyz).all():
        raise ValueError(f"Nonfinite coordinates: {mesh_path}")
    kind_array = cloud.GetPointData().GetArray("point_kind")
    if kind_array is None or not np.all(vtk_to_numpy(kind_array) == 1):
        raise ValueError(f"{model} preview requires exclusively interior points (point_kind=1): {mesh_path}")
    fields = {}
    for name in PHYSICAL_FIELDS + SELFMAX_FIELDS:
        array = cloud.GetPointData().GetArray(name)
        if array is None:
            raise ValueError(f"Missing {name}: {mesh_path}")
        fields[name] = vtk_to_numpy(array).astype(float)
        if fields[name].shape != (len(xyz),) or not np.isfinite(fields[name]).all():
            raise ValueError(f"Invalid scalar field {name}: {mesh_path}")
    normalization = manifest[spec["normalization_key"]]
    if normalization["denominator_scope"] != "interior" or normalization["source_point_count"] != len(xyz):
        raise ValueError(f"Selfmax must use the complete original interior domain: {manifest_path}")
    denominators = [float(normalization["cfd_max"]), float(normalization["pred_max"])]
    if any(not math.isfinite(value) or value <= 0 for value in denominators):
        raise ValueError(f"Nonpositive or nonfinite speed selfmax denominator: {manifest_path}")
    for names in (PHYSICAL_FIELDS, SELFMAX_FIELDS):
        cfd, pred, error = [fields[name] for name in names]
        if not np.allclose(error, pred - cfd, rtol=2e-5, atol=2e-6):
            raise ValueError(f"Signed error mismatch: {mesh_path}/{names[2]}")
    for physical, normalized, maximum in zip(PHYSICAL_FIELDS[:2], SELFMAX_FIELDS[:2], denominators):
        if not np.allclose(fields[normalized], fields[physical] / maximum, rtol=2e-5, atol=2e-6):
            raise ValueError(f"Normalization does not use original maximum: {mesh_path}/{normalized}")
        if fields[normalized].min() < -1e-6 or fields[normalized].max() > 1 + 1e-6:
            raise ValueError(f"Speed selfmax outside 0–1: {mesh_path}/{normalized}")
    count = min(max_points, len(xyz))
    rng = np.random.default_rng(sample_seed)
    sample = np.sort(rng.choice(len(xyz), size=count, replace=False))
    projected, basis, origin = camera_projection(xyz)
    depth_order = np.argsort(projected[sample, 2], kind="stable")
    drawing_indices = sample[depth_order]
    indices_path = outdir / f"{spec['preview_prefix']}_{role}_plot_sample_indices.npy"
    np.save(indices_path, sample)
    return {
        "model": model, "spec": spec, "manifest": manifest, "manifest_path": manifest_path,
        "source": mesh_path, "fields": fields, "projected": projected, "sample": sample,
        "drawing_indices": drawing_indices, "sample_indices_path": indices_path,
        "sample_seed": sample_seed, "basis": basis, "origin": origin,
        "count": len(xyz), "normalization": normalization,
    }


def render_case(root, case, role, mode, outdir, dpi):
    manifest = case["manifest"]
    spec = case["spec"]
    model = case["model"]
    selfmax = mode == "selfmax"
    names = SELFMAX_FIELDS if selfmax else PHYSICAL_FIELDS
    cfd, pred, error = [case["fields"][name] for name in names]
    if selfmax:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = min(0.0, float(cfd.min()), float(pred.min())), max(float(cfd.max()), float(pred.max()))
    emax = max(float(np.abs(error).max()), 1e-12)
    norms = [Normalize(vmin, vmax), Normalize(vmin, vmax), Normalize(-emax, emax)]
    cmaps = ["viridis", "viridis", "RdBu_r"]
    sample = case["drawing_indices"]
    projected = case["projected"]
    lo, hi = projected[:, :2].min(axis=0), projected[:, :2].max(axis=0)
    padding = float((hi - lo).max()) * 0.035
    limits = [(lo[i] - padding, hi[i] + padding) for i in (0, 1)]
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.035, 0.958, f"{spec['title']}  |  {ROLE_TEXT[role]}  |  {manifest['case_id']}",
             fontsize=19, weight="bold", color="#142c43", va="top")
    subtitle = "体内速度点云正交投影 · 绘图抽样 · 无壁面插值 / 无连续截面"
    if selfmax:
        subtitle += "  |  各自 max：分布比较，幅值差已移除"
    fig.text(0.035, 0.905, subtitle, fontsize=12.5 if selfmax else 14, color="#526474")
    titles = (["CFD / CFDmax", "Pred / Predmax", "Signed error = Prednorm − CFDnorm"]
              if selfmax else ["CFD 体内速度", "Pred 体内速度", "Signed error = Pred − CFD"])
    for i, (values, cmap, norm, title) in enumerate(zip([cfd, pred, error], cmaps, norms, titles)):
        fig.add_artist(FancyBboxPatch((0.035 + i * 0.329, 0.264), 0.30, 0.577,
                                      boxstyle="round,pad=0.004,rounding_size=0.008",
                                      transform=fig.transFigure, facecolor="#edf1f5",
                                      edgecolor="none", zorder=-1))
        ax = fig.add_axes([0.025 + i * 0.329, 0.275, 0.32, 0.55])
        ax.scatter(projected[sample, 0], projected[sample, 1], c=values[sample],
                   s=0.45, marker="o", cmap=cmap, norm=norm, alpha=1.0,
                   linewidths=0, edgecolors="none", rasterized=True)
        ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
        ax.axis("off")
        fig.text(0.185 + i * 0.329, 0.853, title, ha="center", fontsize=15,
                 weight="bold", color="#193047")
    field_cax = fig.add_axes([0.08, 0.217, 0.535, 0.018])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norms[0], cmap=cmaps[0]), cax=field_cax, orientation="horizontal")
    cb.set_label("Speed / own max（无量纲）— CFD / Pred 共用 0–1" if selfmax
                 else "Speed (m/s) — CFD / Pred 共用完整源域范围", fontsize=11)
    err_cax = fig.add_axes([0.72, 0.217, 0.235, 0.018])
    cb_err = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norms[2], cmap=cmaps[2]), cax=err_cax, orientation="horizontal")
    cb_err.set_label("Prednorm − CFDnorm（无量纲）— 对称完整范围" if selfmax
                     else "Pred − CFD (m/s) — 对称完整源域范围", fontsize=10)
    for colorbar in (cb, cb_err):
        colorbar.ax.tick_params(labelsize=10)
        colorbar.outline.set_linewidth(0.5)
    r2 = float(manifest[spec["metrics_key"]]["r2"])
    if not math.isfinite(r2):
        raise ValueError(f"Invalid original physical R2: {case['manifest_path']}")
    reference = "（仅作参考，不是 selfmax R²）" if selfmax else ""
    fig.text(0.035, 0.124,
             f"{spec['caption_checkpoint']} 原始完整体内同点物理量 R² = {r2:.4f}{reference}  |  peak step {manifest['peak_step']}",
             fontsize=11.5, color="#193047")
    fig.text(0.035, 0.089,
             f"仅绘图抽样：{len(sample):,} / {case['count']:,} 点；CFD / Pred / Error 与两种显示模式共用同一组索引和视角。全量 VTP 保留。",
             fontsize=10.5, color="#193047")
    if selfmax:
        n = case["normalization"]
        note = (f"原始完整体内评估域分母：CFDmax = {n['cfd_max']:.6g} m/s；Predmax = {n['pred_max']:.6g} m/s。"
                "不按绘图样本重新取 max。")
    else:
        note = "色标来自全量体点云；图中只有独立采样点，没有三角面、壁面插值或连续 Slice。"
    fig.text(0.035, 0.056, note, fontsize=10.5, color="#556674")
    fig.text(0.035, 0.024, spec["caption_selection"], fontsize=10.5, color="#556674")
    output = outdir / f"{spec['preview_prefix']}_{role}_volume_pointcloud{'_selfmax' if selfmax else ''}_triptych.png"
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return {
        "model": model, "role": role, "case_id": manifest["case_id"], "mode": mode,
        "preview": str(output.relative_to(root)), "source": str(case["source"].relative_to(root)),
        "source_sha256": sha256(case["source"]),
        "source_manifest": str(case["manifest_path"].relative_to(root)),
        "source_manifest_sha256": sha256(case["manifest_path"]),
        "field_keys": names, "color_range": [vmin, vmax], "signed_error_range": [-emax, emax],
        "color_range_scope": "full original interior point cloud, not plotting sample",
        "full_point_count": case["count"], "plot_point_count": len(sample),
        "sample_indices": str(case["sample_indices_path"].relative_to(root)),
        "sample_indices_sha256": sha256(case["sample_indices_path"]), "sample_seed": case["sample_seed"],
        "sampling": "uniform without replacement; independent of truth/prediction/error; identical indices for all panels and modes",
        "camera": {"basis_columns_right_up_depth": case["basis"].tolist(), "origin_mm": case["origin"].tolist()},
        "rendering": "opaque independent scatter points in back-to-front depth order; no surface cells or interpolation; not a continuous Slice",
        "physical_metric_r2": r2, "metric_source": f"manifest.{spec['metrics_key']}.r2",
        "normalized_metrics_recomputed": False, "display_normalization": case["normalization"] if selfmax else None,
    }


def contact_sheet(root, entries, mode, outdir, spec):
    width, header, cell_width, cell_height = 3000, 145, 1000, 562
    sheet = Image.new("RGB", (width, header + cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    fontfile = findfont(FontProperties(family="Noto Sans CJK JP"))
    font, smallfont = ImageFont.truetype(fontfile, 40), ImageFont.truetype(fontfile, 27)
    suffix = "各自 max 归一化（分布比较，幅值差已移除）" if mode == "selfmax" else "物理量 m/s"
    draw.text((35, 16), f"{spec['contact_title']} · {suffix}", font=font, fill="#142c43")
    draw.text((35, 81), "列：Best / Median / Worst    确定性绘图抽样最多 30,000 点；全量 VTP 保留    无壁面插值 / 无连续截面", font=smallfont, fill="#526474")
    lookup = {entry["role"]: entry for entry in entries if entry["mode"] == mode}
    for col, role in enumerate(ROLES):
        entry = lookup.get(role)
        if entry:
            with Image.open(root / entry["preview"]) as img:
                img = img.convert("RGB")
                img.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
                sheet.paste(img, (col * cell_width, header))
    output = outdir / f"{spec['preview_prefix']}_volume_pointcloud{'_selfmax' if mode == 'selfmax' else ''}_contact_sheet.png"
    sheet.save(output)
    return str(output.relative_to(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--model", choices=list(VELOCITY_SPECS), default="VF6")
    parser.add_argument("--roles", nargs="+", choices=ROLES, default=list(ROLES))
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--sample-seed", type=int, default=20260914)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.max_points <= 30000:
        parser.error("--max-points must be between 1 and 30000")
    setup_font()
    root = args.root.resolve()
    outdir = root / "plots"
    outdir.mkdir(exist_ok=True)
    spec = VELOCITY_SPECS[args.model]
    entries = []
    for role in args.roles:
        case = load_case(root, role, outdir, args.max_points, args.sample_seed, args.model)
        for mode in ("physical", "selfmax"):
            entry = render_case(root, case, role, mode, outdir, args.dpi)
            entries.append(entry)
            print(f"Rendered {entry['preview']} | {entry['plot_point_count']}/{entry['full_point_count']} points", flush=True)
    contacts = {mode: contact_sheet(root, entries, mode, outdir, spec) for mode in ("physical", "selfmax")}
    result = {"generator": Path(__file__).name, "model": args.model, "case_count": len(args.roles),
              "preview_count": len(entries), "contact_sheets": contacts, "previews": entries}
    (outdir / spec["manifest_name"]).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {len(entries)} interior point-cloud previews and 2 contact sheets", flush=True)


if __name__ == "__main__":
    main()
