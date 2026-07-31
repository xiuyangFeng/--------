"""
诊断可视化：velocity→WSS vs Fluent 真值，并检查近壁采样尺度
============================================================
关注老师提到的两点：
  1) 采样最好落在边界层厚度量级，过深会穿出细支 / 混入主流；
  2) 细小血管处 K 近邻更容易「穿模」。

输出多面板图到 outputs/wss_pinn/audits/cfd_velocity_wss_explore/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

_VIZ = Path(__file__).resolve().parent
_SRC = _VIZ.parent / "src"
_REPO = _VIZ.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SRC))

import data_loader_cfd as dl
from calculate_wss_cfd import (
    evaluate,
    find_peak_step,
    find_stl,
    fit_wall_gradient,
    pca_wall_normals,
    resolve_step,
    stl_wall_normals,
    wss_from_gradient,
)

MM_TO_M = 1e-3


def load_local_radius_mm(case_dir: Path, wall_coords_mm: np.ndarray) -> np.ndarray | None:
    """若存在 data_wss_min bundle，则把 wall_local_radius 映射到当前壁面点。"""
    parts = case_dir.resolve().parts
    if "data_new" not in parts:
        return None
    idx = parts.index("data_new")
    bundle = Path(*parts[:idx]) / "data_wss_min" / Path(*parts[idx + 1 :]) / "bundle.npz"
    if not bundle.exists():
        return None
    with np.load(bundle, allow_pickle=False) as source:
        if "wall_local_radius" not in source.files or "wall_coords_raw" not in source.files:
            return None
        # bundle wall_coords_raw 通常是 mm；与 ascii 壁面同系
        raw = np.asarray(source["wall_coords_raw"], dtype=np.float64)
        radius = np.asarray(source["wall_local_radius"], dtype=np.float64)
        # 若 raw 量级像米，转到 mm
        if np.nanmedian(np.linalg.norm(raw, axis=1)) < 20:
            raw = raw * 1e3
    # 最近邻映射（壁面点顺序不一定一致）
    _, nn = cKDTree(raw).query(wall_coords_mm, k=1)
    return radius[nn].astype(np.float64)


def fit_with_eta_stats(
    wall_mm: np.ndarray,
    normals: np.ndarray,
    interior_mm: np.ndarray,
    velocity: np.ndarray,
    tree: cKDTree,
    neighbors: int,
    degree: int,
) -> dict[str, np.ndarray]:
    """在拟合同时记录每个壁面点用到的法向距离统计（mm）。"""
    k = min(int(neighbors), len(interior_mm))
    _, nb = tree.query(wall_mm, k=k)
    gradient = np.full((len(wall_mm), 3), np.nan)
    used = np.zeros(len(wall_mm), dtype=np.int32)
    eta_min = np.full(len(wall_mm), np.nan)
    eta_med = np.full(len(wall_mm), np.nan)
    eta_max = np.full(len(wall_mm), np.nan)
    eta_p90 = np.full(len(wall_mm), np.nan)
    frac_beyond_radius = np.full(len(wall_mm), np.nan)
    min_samples = degree + 3

    for row, (point, normal, idx) in enumerate(zip(wall_mm, normals, nb)):
        eta_m = ((interior_mm[idx] - point) @ normal) * MM_TO_M
        keep = eta_m > 1e-7
        if keep.sum() < min_samples:
            continue
        e = eta_m[keep]
        e_mm = e / MM_TO_M
        eta_min[row] = float(e_mm.min())
        eta_med[row] = float(np.median(e_mm))
        eta_max[row] = float(e_mm.max())
        eta_p90[row] = float(np.quantile(e_mm, 0.9))
        used[row] = len(e)

        v = velocity[idx][keep]
        v = v - np.outer(v @ normal, normal)
        design = np.column_stack([e**p for p in range(1, degree + 1)])
        try:
            coefficient, *_ = np.linalg.lstsq(design, v, rcond=None)
        except np.linalg.LinAlgError:
            continue
        gradient[row] = coefficient[0]

    return {
        "gradient": gradient,
        "used": used,
        "eta_min_mm": eta_min,
        "eta_med_mm": eta_med,
        "eta_max_mm": eta_max,
        "eta_p90_mm": eta_p90,
        "frac_beyond_radius": frac_beyond_radius,
    }


def make_figure(
    truth: np.ndarray,
    pred: np.ndarray,
    coords_mm: np.ndarray,
    eta_med: np.ndarray,
    eta_max: np.ndarray,
    local_radius: np.ndarray | None,
    report: dict,
    out_path: Path,
) -> Path:
    valid = np.isfinite(truth) & np.isfinite(pred) & np.isfinite(eta_med)
    y, p = truth[valid], pred[valid]
    err = p - y
    abs_err = np.abs(err)
    rel_err = abs_err / np.maximum(np.abs(y), 1e-6)
    eta_m = eta_med[valid]
    eta_x = eta_max[valid]
    xyz = coords_mm[valid]
    alpha = float(report["alpha"])

    # 主轴投影：用最大方差轴做 2D 展开，便于看空间分布
    centered = xyz - xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    uv = centered @ vt[:2].T

    fig = plt.figure(figsize=(14, 10), dpi=140)
    fig.suptitle(
        f"{report['case']}  peak step={report['step']}  "
        f"rawR²={report['raw_r2']:.3f}  scaledR²={report['scaled_r2']:.3f}  "
        f"Spearman={report['spearman']:.3f}  α={alpha:.3f}",
        fontsize=11,
    )

    # 1) scatter
    ax = fig.add_subplot(2, 3, 1)
    lim = float(max(y.max(), p.max(), 1e-6)) * 1.05
    ax.scatter(y, p, s=6, alpha=0.3, c="#1f77b4", edgecolors="none")
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.plot([0, lim], [0, lim / max(alpha, 1e-12)], color="#d62728", lw=1.2)
    ax.set_xlabel("CFD truth WSS (Pa)")
    ax.set_ylabel("Pred WSS (Pa)")
    ax.set_title("Truth vs Pred")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    # 2) abs error vs truth
    ax = fig.add_subplot(2, 3, 2)
    ax.scatter(y, abs_err, s=6, alpha=0.3, c="#ff7f0e", edgecolors="none")
    ax.set_xlabel("CFD truth WSS (Pa)")
    ax.set_ylabel("|pred − truth| (Pa)")
    ax.set_title("Absolute error vs truth")
    ax.grid(True, alpha=0.25)

    # 3) eta_med histogram vs typical BL scale
    ax = fig.add_subplot(2, 3, 3)
    ax.hist(eta_m, bins=40, color="#2ca02c", alpha=0.85)
    for mark, label in ((0.2, "0.2mm"), (0.5, "0.5mm"), (1.0, "1.0mm"), (1.5, "1.5mm")):
        ax.axvline(mark, color="k", ls="--", lw=0.8, alpha=0.6)
        ax.text(mark, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] else 1, label, rotation=90, fontsize=7, va="top")
    ax.set_xlabel("Median inward sample depth η (mm)")
    ax.set_ylabel("Wall point count")
    ax.set_title(
        f"Near-wall sample depth\n"
        f"η_med p50={np.median(eta_m):.2f}mm  p95={np.quantile(eta_m,0.95):.2f}mm  "
        f"η_max p95={np.quantile(eta_x,0.95):.2f}mm"
    )
    ax.grid(True, alpha=0.25)

    # 4) error vs sample depth
    ax = fig.add_subplot(2, 3, 4)
    ax.scatter(eta_m, rel_err, s=6, alpha=0.3, c="#9467bd", edgecolors="none")
    ax.set_xlabel("Median sample depth η (mm)")
    ax.set_ylabel("|pred−truth| / max(|truth|,1e-6)")
    ax.set_title("Relative error vs sample depth")
    ax.set_ylim(0, min(5.0, np.nanquantile(rel_err, 0.99) * 1.2))
    ax.grid(True, alpha=0.25)

    # 5) spatial map colored by abs error
    ax = fig.add_subplot(2, 3, 5)
    sc = ax.scatter(uv[:, 0], uv[:, 1], c=abs_err, s=5, cmap="magma", alpha=0.85)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="|err| Pa")
    ax.set_xlabel("PCA-u (mm)")
    ax.set_ylabel("PCA-v (mm)")
    ax.set_title("Spatial |error| (wall unfolded)")
    ax.set_aspect("equal", adjustable="datalim")

    # 6) local radius / small-vessel analysis
    ax = fig.add_subplot(2, 3, 6)
    if local_radius is not None:
        r = local_radius[valid]
        # 穿模风险代理：最深采样点超过局部半径
        risk = eta_x / np.maximum(r, 1e-6)
        sc = ax.scatter(r, abs_err, c=risk, s=8, cmap="coolwarm", alpha=0.7, vmin=0, vmax=2)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="η_max / local_radius")
        ax.set_xlabel("Local radius (bundle, mm)")
        ax.set_ylabel("|err| (Pa)")
        ax.set_title(
            f"Small-vessel risk\n"
            f"η_max>radius: {(risk > 1).mean()*100:.1f}% points"
        )
        ax.grid(True, alpha=0.25)
    else:
        ax.scatter(eta_x, abs_err, s=6, alpha=0.3, c="#8c564b", edgecolors="none")
        ax.set_xlabel("Max sample depth η_max (mm)")
        ax.set_ylabel("|err| (Pa)")
        ax.set_title("Error vs deepest sample (no local_radius)")
        ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-dir",
        type=str,
        default="/public/newhome/cy/Digital_twin/GNN/data_new/AG/slow/LIU_JIN_LIANG",
    )
    parser.add_argument("--sample-count", type=int, default=4000)
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--normals", choices=["pca", "stl"], default="pca")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="默认写入 explore/04_sampling_diagnostics/<病例名>/",
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    if not args.out_dir:
        args.out_dir = str(
            Path("/public/newhome/cy/Digital_twin/GNN/outputs/wss_pinn/audits/cfd_velocity_wss_explore")
            / "04_sampling_diagnostics"
            / case_dir.name
        )
    step, step_source = resolve_step(case_dir, None, prefer_peak=True)
    wall = dl.load_wall(dl.step_path(case_dir, step, "ascii"))
    interior = dl.load_interior(dl.step_path(case_dir, step, "ascii_in"))
    tree = cKDTree(interior["coords_mm"])

    if args.normals == "pca":
        normals = pca_wall_normals(wall["coords_mm"], interior["coords_mm"], tree)
    else:
        normals = stl_wall_normals(
            wall["coords_mm"], interior["coords_mm"], find_stl(case_dir), tree
        )

    n_wall = len(wall["coords_mm"])
    if 0 < args.sample_count < n_wall:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(n_wall, args.sample_count, replace=False))
    else:
        idx = np.arange(n_wall)

    stats = fit_with_eta_stats(
        wall["coords_mm"][idx],
        normals[idx],
        interior["coords_mm"],
        interior["velocity"],
        tree,
        neighbors=args.neighbors,
        degree=args.degree,
    )
    pred_vec = wss_from_gradient(stats["gradient"], "carreau")
    truth = wall["wss_mag"][idx]
    truth_vec = wall["wss_vec"][idx]
    pred = np.linalg.norm(pred_vec, axis=1)
    report = evaluate(truth, truth_vec, pred_vec)
    report.update(
        {
            "case": f"{case_dir.parent.name}/{case_dir.name}",
            "step": step,
            "step_source": step_source,
            "neighbors": args.neighbors,
            "degree": args.degree,
            "normals_mode": args.normals,
            "n_points": int(len(idx)),
            "eta_med_p50_mm": float(np.nanmedian(stats["eta_med_mm"])),
            "eta_med_p95_mm": float(np.nanquantile(stats["eta_med_mm"], 0.95)),
            "eta_max_p95_mm": float(np.nanquantile(stats["eta_max_mm"], 0.95)),
            "nearest_interior_p50_mm": float(np.nanmedian(stats["eta_min_mm"])),
        }
    )

    local_radius = load_local_radius_mm(case_dir, wall["coords_mm"])
    if local_radius is not None:
        r = local_radius[idx]
        risk = stats["eta_max_mm"] / np.maximum(r, 1e-6)
        valid = np.isfinite(risk) & np.isfinite(pred)
        report["local_radius_p50_mm"] = float(np.nanmedian(r))
        report["frac_eta_max_gt_radius"] = float(np.mean(risk[valid] > 1.0))
        # 细支（半径较小）上的 raw 相关性
        small = valid & (r < np.nanquantile(r, 0.2))
        large = valid & (r > np.nanquantile(r, 0.8))
        if small.sum() > 20 and large.sum() > 20:
            def _r2(a, b):
                a, b = a[np.isfinite(a) & np.isfinite(b)], b[np.isfinite(a) & np.isfinite(b)]
                var = np.sum((a - a.mean()) ** 2)
                return float(1 - np.sum((a - b) ** 2) / max(var, 1e-30))
            report["raw_r2_small_radius_q20"] = _r2(truth[small], pred[small])
            report["raw_r2_large_radius_q80"] = _r2(truth[large], pred[large])

    out_dir = Path(args.out_dir)
    stem = f"{case_dir.name}_peak{step}_diag"
    fig_path = make_figure(
        truth,
        pred,
        wall["coords_mm"][idx],
        stats["eta_med_mm"],
        stats["eta_max_mm"],
        local_radius[idx] if local_radius is not None else None,
        report,
        out_dir / f"{stem}.png",
    )
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n图: {fig_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
