#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出训练壁面采样的 ParaView 诊断包。

示例：
  python -m training_wss_min.tools.visualize_sampling \
    --run-dir training_wss_min/runs/r4_dev1_b0_tgtw_batchq_s1234 \
    --case slow/WU_FENG_YAN

产物放在 ``<run>/sampling_viz/<cohort>__<case>/``：
- ``vessel_surface.vtp``：原始 STL 三角面，作为半透明灰色背景；
- ``wall_points_all.vtp``：完整壁面点云，作为灰色小点；
- ``sampled_epoch_XXX.vtp``：该 epoch 实际送入训练的采样点，作为橙色大点；
- ``sampling_epochs.pvd``：多 epoch 时供 ParaView 时间控件播放；
- ``sampling_projection.png``：不打开 ParaView 也可查看的三视图预览。

这是采样位置诊断，不做 WSS/预测值插值，也不在 STL 上计算任何模型指标。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from training_wss_min import config as C
from training_wss_min import dataset as D


def _write_vtp(
    vertices: np.ndarray,
    out_path: Path,
    *,
    triangles: np.ndarray | None = None,
    scalars: Dict[str, np.ndarray] | None = None,
) -> None:
    """写入点云或三角面 VTP；只在导出时依赖 vtk。"""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(vertices, dtype=np.float64), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)

    if triangles is not None:
        cells = vtk.vtkCellArray()
        packed = np.hstack([
            np.full((len(triangles), 1), 3, dtype=np.int64),
            np.asarray(triangles, dtype=np.int64),
        ]).ravel()
        cells.SetCells(len(triangles), numpy_to_vtkIdTypeArray(packed, deep=True))
        poly.SetPolys(cells)
    else:
        verts = vtk.vtkCellArray()
        for i in range(len(vertices)):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
        poly.SetVerts(verts)

    for name, values in (scalars or {}).items():
        arr = numpy_to_vtk(np.ascontiguousarray(values, dtype=np.float32), deep=True)
        arr.SetName(name)
        poly.GetPointData().AddArray(arr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(poly)
    if writer.Write() != 1:
        raise RuntimeError(f"VTP 写入失败: {out_path}")


def _load_stl(stl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_path))
    reader.Update()
    poly = reader.GetOutput()
    if poly.GetNumberOfPoints() == 0 or poly.GetNumberOfPolys() == 0:
        raise RuntimeError(f"无法读取 STL 三角面: {stl_path}")
    vertices = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    cells = vtk_to_numpy(poly.GetPolys().GetData())
    # STL 三角面固定为 [3, i, j, k]。
    triangles = cells.reshape(-1, 4)[:, 1:4].astype(np.int64)
    return vertices, triangles


def _distance_summary(distances: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(distances)),
        "p50": float(np.percentile(distances, 50)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(np.max(distances)),
    }


def _bbox_diag(points: np.ndarray) -> float:
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def _write_projection(
    all_xyz: np.ndarray,
    sampled_xyz: np.ndarray,
    output: Path,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 使用配准后的毫米坐标；每张图的 x/y 比例锁定，防止血管外形被拉伸。
    views = ((0, 1, "X–Y"), (0, 2, "X–Z"), (1, 2, "Y–Z"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    # 静态投影里所有点会压到同一平面；采样点只略大于背景，才能同时读出灰色未采样点。
    background_size = max(0.7, 7000.0 / len(all_xyz))
    sampled_size = max(2.5, background_size * 3.0)
    labels = ("X (mm)", "Y (mm)", "Z (mm)")
    for ax, (a, b, name) in zip(axes, views):
        ax.scatter(all_xyz[:, a], all_xyz[:, b], s=background_size,
                   c="#B8B8B8", alpha=0.55, linewidths=0, label="all wall points")
        ax.scatter(sampled_xyz[:, a], sampled_xyz[:, b], s=sampled_size,
                   c="#FF4D00", alpha=0.92, linewidths=0, label="sampled")
        ax.set_title(name)
        ax.set_xlabel(labels[a]); ax.set_ylabel(labels[b]); ax.set_aspect("equal")
    axes[0].legend(loc="best", markerscale=1.2)
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_pvd(files: Iterable[Path], output: Path) -> None:
    rows = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for epoch, path in enumerate(files):
        rows.append(f'    <DataSet timestep="{epoch}" group="" part="0" file="{path.name}"/>')
    rows += ["  </Collection>", "</VTKFile>", ""]
    output.write_text("\n".join(rows), encoding="utf-8")


def _write_readme(
    output: Path,
    *,
    case_label: str,
    sampling: str,
    n_all: int,
    n_selected: int,
    epochs: int,
) -> None:
    dynamic_note = (
        "`sampling_epochs.pvd` 已包含多个 epoch，可通过 ParaView 的播放控件切换采样点。"
        if epochs > 1 else
        "该 run 使用固定 FPS，本包只包含 epoch 0；旋转视角即可观察空间覆盖。"
    )
    output.write_text(
        (
            f"# 训练采样 ParaView 诊断包\n\n"
            f"- 病例：`{case_label}`\n"
            f"- 采样策略：`{sampling}`\n"
            f"- 采样数量：`{n_selected} / {n_all}`（`{n_selected / n_all:.1%}`）\n\n"
            f"## ParaView 打开方式\n\n"
            f"1. 同时打开 `vessel_surface.vtp`、`wall_points_all.vtp` 和 `sampled_epoch_000.vtp`。\n"
            f"2. `vessel_surface.vtp`：`Representation = Surface`，设浅灰，`Opacity ≈ 0.20`。\n"
            f"3. `wall_points_all.vtp`：`Representation = Point Gaussian`，设灰色，点大小约 `2–3`。\n"
            f"4. `sampled_epoch_000.vtp`：`Representation = Point Gaussian`，设橙红 `#FF4D00`，点大小约 `7–10`。\n\n"
            f"{dynamic_note}\n\n"
            f"本包展示的是**精确的原始壁面采样点**；未对 `is_sampled` 做 Gaussian/IDW 插值，"
            f"因此不会把单个采样点扩散成面片色块。\n"
        ),
        encoding="utf-8",
    )


def export_sampling_viz(run_dir: Path, case_label: str, epochs: int) -> Path:
    if epochs < 1:
        raise ValueError("epochs 必须 >= 1")
    cfg = C.ExpConfig.from_json(run_dir / "config.json")
    stats_path = run_dir / "wss_global_stats.json"
    stats = D.load_wss_stats(stats_path if stats_path.is_file() else cfg.data.wss_stats_path)
    cases = D.load_partition(cfg.data.split_path, "train", stats, strict=True)
    try:
        case_index = next(i for i, c in enumerate(cases)
                          if f"{c['cohort'].removeprefix('AG/')}/{c['case']}" == case_label)
    except StopIteration as exc:
        available = ", ".join(
            f"{c['cohort'].removeprefix('AG/')}/{c['case']}" for c in cases[:5]
        )
        raise ValueError(f"case 不在该 run 的 train split: {case_label}; 例如: {available}") from exc

    case = cases[case_index]
    geom_seed = int(cfg.train.seed + 7919 * case_index)
    bundle = Path(case["bundle_path"])
    with np.load(bundle, allow_pickle=True) as data:
        wall_raw_xyz = data["wall_coords_raw"].astype(np.float64)
        coord_scale = float(data["coord_scale"])
        centroid = data["transform_centroid"].astype(np.float64)
        rotation = data["transform_rotation"].astype(np.float64)
        extent_mismatch = bool(data["unit_extent_mismatch"])
    if len(wall_raw_xyz) != len(case["pos"]):
        raise RuntimeError(
            f"坐标长度不一致: raw={len(wall_raw_xyz)} norm={len(case['pos'])}"
        )
    # 训练实际使用 wall_coords_norm。乘回 scale 后仍处于模型的刚性配准坐标系（单位 mm）。
    wall_xyz = case["pos"].astype(np.float64) * coord_scale

    cohort_rel = case["cohort"]
    stl = C.PROJECT_ROOT / "data_new" / cohort_rel / case["case"] / f"{case['case']}.stl"
    if not stl.is_file():
        raise FileNotFoundError(f"缺少对应 STL: {stl}")

    out_dir = run_dir / "sampling_viz" / case_label.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    stl_raw_xyz, stl_triangles = _load_stl(stl)
    # STL 的原生单位可能与 CFD 壁面坐标不同。大部分病例以 bbox 对角线比例即可精确换到
    # pipeline 的毫米坐标；随后应用 bundle 中保存的同一刚性变换，才能与训练点逐点叠加。
    stl_to_pipeline = _bbox_diag(wall_raw_xyz) / max(_bbox_diag(stl_raw_xyz), 1e-12)
    stl_xyz = (stl_raw_xyz * stl_to_pipeline - centroid) @ rotation
    _write_vtp(stl_xyz, out_dir / "vessel_surface.vtp", triangles=stl_triangles)
    _write_vtp(
        wall_xyz, out_dir / "wall_points_all.vtp",
        scalars={"point_index": np.arange(len(wall_xyz)), "is_sampled": np.zeros(len(wall_xyz))},
    )

    epoch_files: list[Path] = []
    idx0: np.ndarray | None = None
    for epoch in range(epochs):
        # 与 WSSMinDataset.__getitem__ 使用完全相同的随机种子与采样函数。
        if cfg.data.resample_each_epoch and cfg.data.sampling in (
            "random", "geom_weighted", "fps_multistart"
        ):
            seed = int(cfg.train.seed + 100003 * epoch + 7919 * case_index)
        else:
            seed = geom_seed
        idx = D.sample_indices(
            case, cfg.data, seed, epoch=epoch, case_index=case_index, run_seed=cfg.train.seed,
        )
        if idx0 is None:
            idx0 = idx
        path = out_dir / f"sampled_epoch_{epoch:03d}.vtp"
        _write_vtp(
            wall_xyz[idx], path,
            scalars={"point_index": idx, "is_sampled": np.ones(len(idx)), "epoch": np.full(len(idx), epoch)},
        )
        epoch_files.append(path)

    assert idx0 is not None
    _write_pvd(epoch_files, out_dir / "sampling_epochs.pvd")
    title = (f"{case_label}  |  {cfg.data.sampling}  |  "
             f"{len(idx0)} / {len(wall_xyz)} sampled ({len(idx0) / len(wall_xyz):.1%})")
    _write_projection(wall_xyz, wall_xyz[idx0], out_dir / "sampling_projection.png", title)

    # 最近 STL 顶点距离仅用于坐标叠加 QC，不是严格的点到三角面距离。
    from scipy.spatial import cKDTree
    tree = cKDTree(stl_xyz)
    all_dist = tree.query(wall_xyz, workers=-1)[0]
    selected_dist = tree.query(wall_xyz[idx0], workers=-1)[0]
    _write_readme(
        out_dir / "README_ParaView.md", case_label=case_label, sampling=cfg.data.sampling,
        n_all=len(wall_xyz), n_selected=len(idx0), epochs=epochs,
    )
    manifest = {
        "purpose": "exact training wall-point sampling visualization; no scalar interpolation",
        "run": cfg.name,
        "case": case_label,
        "partition": "train",
        "sampling": cfg.data.sampling,
        "wall_n_points": int(cfg.data.wall_n_points),
        "epochs_exported": list(range(epochs)),
        "seed": int(cfg.train.seed),
        "case_index_in_train_split": case_index,
        "source_bundle": str(bundle),
        "source_stl": str(stl),
        "coordinate_system": "bundle registration frame in mm; STL scale matched by bbox then transformed with bundle rigid transform",
        "stl_to_pipeline_scale": float(stl_to_pipeline),
        "unit_extent_mismatch": extent_mismatch,
        "stl_vertex_distance_qc_mm": {
            "all_wall_points": _distance_summary(all_dist),
            "sampled_epoch_000": _distance_summary(selected_dist),
            "note": "nearest STL vertex distance; used only to detect gross coordinate mismatch",
        },
        "files": {
            "surface": "vessel_surface.vtp",
            "all_wall_points": "wall_points_all.vtp",
            "sampled_epoch_000": "sampled_epoch_000.vtp",
            "epoch_collection": "sampling_epochs.pvd",
            "preview": "sampling_projection.png",
        },
    }
    (out_dir / "manifest_sampling_viz.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 WSS-min 训练壁面采样 ParaView 诊断包")
    parser.add_argument("--run-dir", required=True, help="包含 config.json 的 run 目录")
    parser.add_argument("--case", required=True, help="train split 内病例，如 slow/WU_FENG_YAN")
    parser.add_argument("--epochs", type=int, default=1, help="导出多少个 epoch 的采样点")
    args = parser.parse_args()
    output = export_sampling_viz(Path(args.run_dir), args.case, args.epochs)
    print(output)


if __name__ == "__main__":
    main()
