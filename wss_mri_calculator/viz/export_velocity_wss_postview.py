"""
velocity→WSS 结果 → postview 风格面片包（ParaView .vtp）
========================================================
流程对齐 postview-surface-viz：
  1) 全量壁面点算 pred WSS，写 __wall.csv（同点指标口径）
  2) Gaussian 回插到病例 STL → *__stl_mapped_wall.vtp
  3) 可选三联 PNG + 打开说明

用法
----
  cd wss_mri_calculator/viz
  python export_velocity_wss_postview.py \\
    --case-dir .../data_new/AG/slow/LIU_JIN_LIANG
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_VIZ = Path(__file__).resolve().parent
_SRC = _VIZ.parent / "src"
_REPO = _VIZ.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SRC))

import data_loader_cfd as dl
from calculate_wss_cfd import find_stl, resolve_step, run_case

REPO = _REPO
EXPLORE = REPO / "outputs/wss_pinn/audits/cfd_velocity_wss_explore"
CMAP_CANDIDATES = [
    REPO / "例子/04_最佳权重_最好最差病例_postview/slow__XU_YI_CAI__peak_wss/GNN_blue_white_red.xml",
    REPO / "tools/cfdpost_cloud_export/GNN_blue_white_red.xml",
]


def write_wall_csv(
    coords_mm: np.ndarray,
    truth_mag: np.ndarray,
    pred_mag: np.ndarray,
    path: Path,
    alpha: float,
) -> Path:
    err = pred_mag - truth_mag
    pred_scaled = alpha * pred_mag
    err_scaled = pred_scaled - truth_mag
    cfd_max = float(np.nanmax(truth_mag))
    cfd_max = max(cfd_max, 1e-12)
    header = (
        "x,y,z,wss_cfd,wss_pred,err_wss,abs_err_wss,"
        "wss_pred_scaled,err_wss_scaled,abs_err_wss_scaled,"
        "wss_cfd_over_cfd_max,wss_pred_over_cfd_max,"
        "err_wss_over_cfd_max,abs_err_wss_over_cfd_max"
    )
    rows = np.column_stack(
        [
            coords_mm,
            truth_mag,
            pred_mag,
            err,
            np.abs(err),
            pred_scaled,
            err_scaled,
            np.abs(err_scaled),
            truth_mag / cfd_max,
            pred_mag / cfd_max,
            (pred_mag - truth_mag) / cfd_max,
            np.abs(pred_mag - truth_mag) / cfd_max,
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header + "\n")
        np.savetxt(handle, rows, delimiter=",", fmt="%.8g")
    return path


def write_readme(out_dir: Path, case_name: str, report: dict, stl_name: str) -> None:
    text = f"""# 后处理打开说明（velocity→WSS · ParaView）

## 病例 / 算法

- 病例：`{report.get("case", case_name)}`
- 算法：Fluent 速度场 → 近壁剖面拟合 WSS（`calculate_wss_cfd.py`）
- 帧：peak_step={report.get("step")}（{report.get("step_source")}）
- 配置：K={report.get("neighbors")} deg={report.get("degree")} visc={report.get("viscosity_model")} normals={report.get("normals_mode")}
- 壁面点数：{report.get("n_valid")}（全量 ascii 壁面节点）
- 插值：Gaussian r=3 mm, sharpness=2, max_dist=3 mm（与 postview 默认一致）
- 坐标系：CFD ascii / STL 同系（mm）

## 指标（同点 CSV，非面片）

- raw R² = {float(report.get("raw_r2", float("nan"))):.3f}
- scaled R² = {float(report.get("scaled_r2", float("nan"))):.3f}（α={float(report.get("alpha", float("nan"))):.3f}）
- Spearman = {float(report.get("spearman", float("nan"))):.3f}
- 方向余弦 p50 = {float(report.get("direction_cosine_p50", float("nan"))):.3f}

正式数字只读 `_export/{case_name}__wall.csv` 或 `manifest_bundle.json`，**不要在面片 VTP 上算 R²**。

## 主文件

| 文件 | 用途 |
| --- | --- |
| **`{case_name}__surface_wall.vtp`** | ParaView 主面片（推荐） |
| `surface_gaussian/{case_name}__stl_mapped_wall.vtp` | 同上源文件 |
| `_export/{case_name}__wall.csv` | 同点 pred/truth |
| `plots/fig_wss_triptych.png` | CFD | Pred | Error 预览 |
| `GNN_blue_white_red.xml` | 蓝-白-红色标 |
| `{stl_name}` | 源 STL 副本 |

### VTP 标量

| 名 | 含义 |
| --- | --- |
| `wss_cfd` | Fluent 真值 WSS (Pa) |
| `wss_pred` | 速度算法 raw 预测 (Pa) |
| `err_wss` / `abs_err_wss` | pred − truth / 绝对值 |
| `wss_pred_scaled` | α·pred（全局幅值校正后） |
| `*_over_cfd_max` | 除以本例 CFD 壁面 max WSS |

## ParaView 步骤

1. Open `{case_name}__surface_wall.vtp` → Apply
2. Representation = **Surface**
3. Coloring：先看 `wss_cfd` 与 `wss_pred`（**共用同一 Data Range**）
4. 再看 `err_wss` 或 `wss_pred_scaled`
5. 可选导入 `GNN_blue_white_red.xml`
"""
    (out_dir / "README_后处理打开说明.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-dir",
        type=str,
        default=str(REPO / "data_new/AG/slow/LIU_JIN_LIANG"),
    )
    parser.add_argument("--sample-count", type=int, default=0)
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--normals", choices=["pca", "stl"], default="pca")
    parser.add_argument("--viscosity", choices=["carreau", "newton"], default="carreau")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-triptych", action="store_true")
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    case_name = case_dir.name
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else EXPLORE / "06_postview_vtp" / f"slow__{case_name}__peak_wss"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    export_dir = out_dir / "_export"
    surf_dir = out_dir / "surface_gaussian"
    plots_dir = out_dir / "plots"
    export_dir.mkdir(parents=True, exist_ok=True)
    surf_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    report = run_case(
        case_dir,
        None,
        args.sample_count,
        args.neighbors,
        args.degree,
        args.viscosity,
        False,
        args.seed,
        args.normals,
        prefer_peak=True,
        return_arrays=True,
    )
    arrays = report.pop("_arrays")
    step, _ = resolve_step(case_dir, None, prefer_peak=True)
    wall = dl.load_wall(dl.step_path(case_dir, step, "ascii"))
    n_wall = len(wall["coords_mm"])
    if 0 < args.sample_count < n_wall:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(n_wall, args.sample_count, replace=False))
    else:
        idx = np.arange(n_wall)

    csv_path = export_dir / f"{case_name}__wall.csv"
    write_wall_csv(
        wall["coords_mm"][idx],
        arrays["truth_mag"],
        arrays["pred_mag"],
        csv_path,
        float(report["alpha"]),
    )

    stl = find_stl(case_dir)
    stl_copy = out_dir / stl.name
    if not stl_copy.exists():
        shutil.copy2(stl, stl_copy)

    py = Path(sys.executable)
    map_script = REPO / "tools/cfdpost_cloud_export/map_to_stl_surface.py"
    scalars = (
        "wss_cfd,wss_pred,err_wss,abs_err_wss,"
        "wss_pred_scaled,err_wss_scaled,abs_err_wss_scaled,"
        "wss_cfd_over_cfd_max,wss_pred_over_cfd_max,"
        "err_wss_over_cfd_max,abs_err_wss_over_cfd_max"
    )
    report_json = surf_dir / f"{case_name}__mapping_report_wall.json"
    subprocess.run(
        [
            str(py),
            str(map_script),
            "--csv",
            str(csv_path),
            "--stl",
            str(stl),
            "--output-dir",
            str(surf_dir),
            "--method",
            "gaussian",
            "--radius",
            "3.0",
            "--sharpness",
            "2.0",
            "--max-dist",
            "3.0",
            "--scalars",
            scalars,
            "--report-json",
            str(report_json),
        ],
        check=True,
    )

    mapped_vtp = surf_dir / f"{case_name}__stl_mapped_wall.vtp"
    surface_vtp = out_dir / f"{case_name}__surface_wall.vtp"
    if mapped_vtp.exists():
        shutil.copy2(mapped_vtp, surface_vtp)

    for cmap in CMAP_CANDIDATES:
        if cmap.exists():
            shutil.copy2(cmap, out_dir / "GNN_blue_white_red.xml")
            break

    if not args.skip_triptych and mapped_vtp.exists():
        trip_script = REPO / "tools/cfdpost_cloud_export/plot_stl_mapped_triptych.py"
        trip_out = plots_dir / "fig_wss_triptych.png"
        subprocess.run(
            [
                str(py),
                str(trip_script),
                "--vtp",
                str(mapped_vtp),
                "--render",
                "surface",
                "--variable",
                "wss",
                "--output",
                str(trip_out),
                "--field-cmap",
                "GNN_BWR",
                "--err-cmap",
                "GNN_BWR",
                "--report-json",
                str(plots_dir / "fig_wss_triptych_report.json"),
            ],
            check=False,
        )

    write_readme(out_dir, case_name, report, stl.name)

    mapping = {}
    if report_json.exists():
        mapping = json.loads(report_json.read_text(encoding="utf-8"))

    bundle = {
        "case": report.get("case"),
        "step": report.get("step"),
        "algorithm": "velocity_to_wss_cfd",
        "metrics_pointcloud": {
            k: report[k]
            for k in (
                "raw_r2",
                "scaled_r2",
                "alpha",
                "spearman",
                "direction_cosine_p50",
                "n_valid",
                "coverage",
            )
            if k in report
        },
        "mapping_report_summary": {
            k: mapping.get(k)
            for k in ("valid_ratio", "n_points", "method", "radius", "max_dist")
            if k in mapping
        },
        "files": {
            "surface_wall_vtp": str(surface_vtp),
            "stl_mapped_vtp": str(mapped_vtp),
            "wall_csv": str(csv_path),
            "readme": str(out_dir / "README_后处理打开说明.md"),
        },
    }
    (out_dir / "manifest_bundle.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(bundle, indent=2, ensure_ascii=False))
    print(f"\nParaView 打开: {surface_vtp}")


if __name__ == "__main__":
    main()
