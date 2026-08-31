#!/usr/bin/env python3
"""Render before/after centerline comparison slides for two representative cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


PROJECT = Path(__file__).resolve().parents[1]
AUDIT = PROJECT / "outputs/centerline_postview_audit_173"
V2 = PROJECT / "outputs/centerline_v2_full_173_20260828"
OUT = V2 / "compare_before_after"

RENDER_COLORS = {
    -1: (0.10, 0.10, 0.12),
    0: (0.88, 0.18, 0.16),
    1: (0.12, 0.45, 0.86),
    2: (0.10, 0.68, 0.38),
    3: (0.84, 0.46, 0.08),
}

CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def read_stl(path: Path) -> vtk.vtkPolyData:
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(reader.GetOutput())
    return out


def read_vtp(path: Path) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(reader.GetOutput())
    return out


def points_of(poly: vtk.vtkPolyData) -> np.ndarray:
    return vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)


def camera_from_points(pts: np.ndarray) -> dict:
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    diag = float(np.linalg.norm(np.ptp(pts, axis=0)))
    _, _, vt = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
    return {
        "center": center,
        "diag": diag,
        "long_axis": vt[0],
        "side_axis": vt[1],
        "depth_axis": vt[2],
    }


def add_surface(renderer: vtk.vtkRenderer, surface: vtk.vtkPolyData) -> None:
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(surface)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.72, 0.76, 0.80)
    actor.GetProperty().SetOpacity(0.20)
    renderer.AddActor(actor)


def add_tubes(renderer: vtk.vtkRenderer, poly: vtk.vtkPolyData, radius: float, colors: list[tuple[float, float, float]]) -> None:
    if poly.GetNumberOfCells() == 0:
        return
    for cell_id in range(poly.GetNumberOfCells()):
        ids = vtk.vtkIdList()
        poly.GetCellPoints(cell_id, ids)
        if ids.GetNumberOfIds() < 2:
            continue
        pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(ids.GetNumberOfIds())
        for i in range(ids.GetNumberOfIds()):
            pts.InsertNextPoint(poly.GetPoint(ids.GetId(i)))
            line.GetPointIds().SetId(i, i)
        lines.InsertNextCell(line)
        part = vtk.vtkPolyData()
        part.SetPoints(pts)
        part.SetLines(lines)
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(part)
        tube.SetRadius(radius)
        tube.SetNumberOfSides(12)
        tube.CappingOn()
        tube.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(tube.GetOutput())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*colors[cell_id % len(colors)])
        renderer.AddActor(actor)


def add_openings(renderer: vtk.vtkRenderer, openings: vtk.vtkPolyData, diag: float) -> None:
    xyz = points_of(openings)
    inlet = vtk_to_numpy(openings.GetPointData().GetArray("IsInlet")).astype(bool)
    outlet = vtk_to_numpy(openings.GetPointData().GetArray("OutletId")).astype(int)
    radii = vtk_to_numpy(openings.GetPointData().GetArray("EquivalentRadiusMm"))
    for index, point in enumerate(xyz):
        color = (0.55, 0.05, 0.75) if inlet[index] else RENDER_COLORS.get(int(outlet[index]), (0.2, 0.2, 0.2))
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(*[float(v) for v in point])
        sphere.SetRadius(max(float(radii[index]) * 0.22, diag * 0.006))
        sphere.SetThetaResolution(18)
        sphere.SetPhiResolution(18)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        renderer.AddActor(actor)


def render_view(actors_fn, cam: dict, path: Path, caption: str) -> None:
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    actors_fn(renderer)
    direction = cam["side_axis"] + 0.55 * cam["depth_axis"]
    direction = direction / max(np.linalg.norm(direction), 1.0e-12)
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*cam["center"])
    camera.SetPosition(*(cam["center"] + direction * cam["diag"] * 1.8))
    camera.SetViewUp(*cam["long_axis"])
    camera.ParallelProjectionOn()
    camera.SetParallelScale(cam["diag"] * 0.42)
    title = vtk.vtkTextActor()
    title.SetInput(caption)
    title.GetTextProperty().SetFontSize(18)
    title.GetTextProperty().SetColor(0.05, 0.05, 0.05)
    title.SetPosition(18, 18)
    renderer.AddActor2D(title)
    if hasattr(vtk, "vtkEGLRenderWindow"):
        window = vtk.vtkEGLRenderWindow()
    else:
        window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(720, 900)
    window.SetMultiSamples(0)
    window.AddRenderer(renderer)
    window.Render()
    image_filter = vtk.vtkWindowToImageFilter()
    image_filter.SetInput(window)
    image_filter.SetScale(1)
    image_filter.ReadFrontBufferOff()
    image_filter.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(image_filter.GetOutputPort())
    writer.Write()
    window.Finalize()


def compose_pair(before_png: Path, after_png: Path, out_path: Path, spec: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.gridspec import GridSpec
    from PIL import Image

    font_manager.fontManager.addfont(CJK_FONT)
    prop = font_manager.FontProperties(fname=CJK_FONT)
    left = np.asarray(Image.open(before_png).convert("RGB"))
    right = np.asarray(Image.open(after_png).convert("RGB"))
    fig = plt.figure(figsize=(12.4, 8.6), dpi=160)
    gs = GridSpec(2, 2, height_ratios=[0.13, 1.0], hspace=0.04, wspace=0.02, left=0.03, right=0.97, top=0.90, bottom=0.08)
    fig.suptitle(spec["title"], fontproperties=prop, fontsize=20, y=0.97)
    ax_h0 = fig.add_subplot(gs[0, 0])
    ax_h1 = fig.add_subplot(gs[0, 1])
    ax0 = fig.add_subplot(gs[1, 0])
    ax1 = fig.add_subplot(gs[1, 1])
    ax_h0.axis("off")
    ax_h1.axis("off")
    ax_h0.set_title("优化前", fontproperties=prop, fontsize=16, pad=8, color="#8B1E1E")
    ax_h1.set_title("优化后 · Centerline V2", fontproperties=prop, fontsize=16, pad=8, color="#1B5E3B")
    ax0.imshow(left)
    ax1.imshow(right)
    ax0.axis("off")
    ax1.axis("off")
    ax0.set_xlabel(spec["before_caption"], fontproperties=prop, fontsize=11, labelpad=8)
    ax1.set_xlabel(spec["after_caption"], fontproperties=prop, fontsize=11, labelpad=8)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


YANG_SPEC = {
    "title": "YANG_YU_QING-1：中心线取自错误 STL（整体脱离血管）",
    "before_caption": "开口到端点最大 116.6 mm · 壁面距离/局部半径 p95 = 29.1R · 壁面 >2R = 100%",
    "after_caption": "权威 STL + 固定单位 1000 · 开口距 0.41 mm · p95 = 1.45R · 壁面 >2R = 0.26%",
}
XIE_SPEC = {
    "title": "XIE_JIN_QUAN：旧算法漏支（Target not reached）",
    "before_caption": "5 开口只接到 4 个端点 · 旧算法隔离重跑仍失败 · 壁面距离/局部半径 p95 = 11.6R",
    "after_caption": "第 3 轮 clean + 种子内移后补齐第 5 支 · p95 = 1.58R · 壁面 >2R = 0.45%",
}


def render_yang_panels(out_dir: Path) -> tuple[Path, Path]:
    before_surface = read_stl(AUDIT / "cases/ILO/YANG_YU_QING-1/before/01_vessel_surface_audit_mm.stl")
    before_cl = read_vtp(AUDIT / "cases/ILO/YANG_YU_QING-1/before/03_centerline_polyline_audit_mm.vtp")
    after_surface = read_stl(V2 / "cases/ILO/YANG_YU_QING-1/before/surface_authoritative_mm.stl")
    after_cl = read_vtp(V2 / "cases/ILO/YANG_YU_QING-1/before/centerline_paths_mm.vtp")
    after_open = read_vtp(V2 / "cases/ILO/YANG_YU_QING-1/before/openings_mm.vtp")
    cam = camera_from_points(np.vstack([points_of(before_surface), points_of(before_cl)]))
    radius = max(cam["diag"] * 0.004, 0.35)
    before_png = out_dir / "_yang_before_iso.png"
    after_png = out_dir / "_yang_after_iso.png"
    render_view(
        lambda r: (add_surface(r, before_surface), add_tubes(r, before_cl, radius, [RENDER_COLORS[-1], *RENDER_COLORS.values()])),
        cam,
        before_png,
        "ILO/YANG_YU_QING-1  BEFORE\nwrong STL  endpoints=5  wall>2R=100%",
    )
    render_view(
        lambda r: (add_surface(r, after_surface), add_tubes(r, after_cl, radius, [RENDER_COLORS[i] for i in range(4)]), add_openings(r, after_open, cam["diag"])),
        cam,
        after_png,
        "ILO/YANG_YU_QING-1  AFTER\npass  endpoints=5  branches=3  max end-open=0.41 mm",
    )
    return before_png, after_png


def render_xie_panels(out_dir: Path) -> tuple[Path, Path]:
    before_surface = read_stl(AUDIT / "cases/AAA/ruputer/XIE_JIN_QUAN/01_vessel_surface_audit_mm.stl")
    before_cl = read_vtp(AUDIT / "cases/AAA/ruputer/XIE_JIN_QUAN/03_centerline_polyline_audit_mm.vtp")
    after_surface = read_stl(V2 / "cases/AAA/ruputer/XIE_JIN_QUAN/surface_authoritative_mm.stl")
    after_cl = read_vtp(V2 / "cases/AAA/ruputer/XIE_JIN_QUAN/centerline_paths_mm.vtp")
    after_open = read_vtp(V2 / "cases/AAA/ruputer/XIE_JIN_QUAN/openings_mm.vtp")
    cam = camera_from_points(points_of(before_surface))
    radius = max(cam["diag"] * 0.004, 0.35)
    before_png = out_dir / "_xie_before_iso.png"
    after_png = out_dir / "_xie_after_iso.png"
    render_view(
        lambda r: (add_surface(r, before_surface), add_tubes(r, before_cl, radius, [RENDER_COLORS[-1], *RENDER_COLORS.values()])),
        cam,
        before_png,
        "AAA/ruputer/XIE_JIN_QUAN  BEFORE\nTarget not reached  endpoints=4  branches=2",
    )
    render_view(
        lambda r: (add_surface(r, after_surface), add_tubes(r, after_cl, radius, [RENDER_COLORS[i] for i in range(4)]), add_openings(r, after_open, cam["diag"])),
        cam,
        after_png,
        "AAA/ruputer/XIE_JIN_QUAN  AFTER\npass  endpoints=5  branches=3  max end-open=4.83 mm",
    )
    return before_png, after_png


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.compose_only:
        render_yang_panels(OUT)
        render_xie_panels(OUT)
    if args.render_only:
        print(OUT / "_yang_before_iso.png")
        print(OUT / "_yang_after_iso.png")
        print(OUT / "_xie_before_iso.png")
        print(OUT / "_xie_after_iso.png")
        return
    yang = OUT / "YANG_YU_QING-1_before_vs_after_iso.png"
    xie = OUT / "XIE_JIN_QUAN_before_vs_after_iso.png"
    compose_pair(OUT / "_yang_before_iso.png", OUT / "_yang_after_iso.png", yang, YANG_SPEC)
    compose_pair(OUT / "_xie_before_iso.png", OUT / "_xie_after_iso.png", xie, XIE_SPEC)
    print(yang)
    print(xie)


if __name__ == "__main__":
    main()
