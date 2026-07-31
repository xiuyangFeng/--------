"""
法向定向对比图：全局质心定向（原实现） vs 局部内部质心定向
============================================================
给老师核对用：在同一套壁面点 + STL 最近面法向上，只换定向规则，
画出带矢量箭头的血管图，并标出两者方向相反的点。

同时打印坐标系对齐诊断（wall↔STL 距离、bbox 重叠）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import cKDTree

_VIZ = Path(__file__).resolve().parent
_SRC = _VIZ.parent / "src"
_REPO = _VIZ.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SRC))

import data_loader_cfd as dl
from calculate_wss_cfd import _orient_inward, find_stl, resolve_step
from wss_pinn.physics.wall_shear import stl_face_geometry


def orient_centroid(normals: np.ndarray, wall_mm: np.ndarray) -> np.ndarray:
    """原实现：法向统一翻转到指向整个壁面几何质心。"""
    inward = wall_mm.mean(axis=0) - wall_mm
    sign = np.sign(np.einsum("ij,ij->i", inward, normals))
    sign[sign == 0] = 1.0
    n = normals * sign[:, None]
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def pca_uv(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = xyz - xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T, vt[:2]


def make_figure(
    wall_mm: np.ndarray,
    n_cent: np.ndarray,
    n_loc: np.ndarray,
    disagree: np.ndarray,
    meta: dict,
    out_path: Path,
    arrow_stride: int,
    arrow_len_mm: float,
) -> Path:
    uv, axes2 = pca_uv(wall_mm)
    # 把 3D 法向投到 PCA 平面，便于 2D 箭头阅读
    n_cent_2 = n_cent @ axes2.T
    n_loc_2 = n_loc @ axes2.T

    rng = np.random.default_rng(0)
    keep = np.arange(len(wall_mm))
    if arrow_stride > 1:
        keep = np.sort(rng.choice(len(wall_mm), size=max(200, len(wall_mm) // arrow_stride), replace=False))

    fig = plt.figure(figsize=(14.5, 10.5), dpi=150)
    fig.suptitle(
        f"{meta['case']}  step={meta['step']}  |  STL normals + orientation only\n"
        f"wall->STL dist p50={meta['wall_to_stl_p50_mm']:.3f}mm  p95={meta['wall_to_stl_p95_mm']:.3f}mm  "
        f"|  disagree(centroid vs local)={meta['disagree_frac']*100:.1f}%  "
        f"|  +side nearest-interior: centroid {meta['pos_side_centroid']*100:.1f}% / "
        f"local {meta['pos_side_local']*100:.1f}%",
        fontsize=11,
    )

    # ---- row 1: 2D PCA with arrows ----
    for col, (normals2, title, flip_mask) in enumerate(
        [
            (n_cent_2, "A. Original: orient toward global centroid", disagree),
            (n_loc_2, "B. Local: orient toward local interior centroid", np.zeros(len(wall_mm), dtype=bool)),
        ],
        start=1,
    ):
        ax = fig.add_subplot(2, 2, col)
        ax.scatter(uv[:, 0], uv[:, 1], s=2, c="#b0b0b0", alpha=0.35, linewidths=0)
        if flip_mask.any():
            ax.scatter(
                uv[flip_mask, 0],
                uv[flip_mask, 1],
                s=6,
                c="#d62728",
                alpha=0.55,
                linewidths=0,
                label=f"opposite to local ({flip_mask.mean()*100:.1f}%)",
            )
        # arrows: agree=blue, disagree=red
        ok = keep[~disagree[keep]] if flip_mask.any() else keep
        bad = keep[disagree[keep]] if flip_mask.any() else np.array([], dtype=int)
        if len(ok):
            ax.quiver(
                uv[ok, 0],
                uv[ok, 1],
                normals2[ok, 0],
                normals2[ok, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0 / arrow_len_mm,
                width=0.0022,
                color="#1f77b4",
                alpha=0.85,
                label="inward normal",
            )
        if len(bad):
            ax.quiver(
                uv[bad, 0],
                uv[bad, 1],
                normals2[bad, 0],
                normals2[bad, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0 / arrow_len_mm,
                width=0.0025,
                color="#d62728",
                alpha=0.9,
            )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("PCA-u (mm)")
        ax.set_ylabel("PCA-v (mm)")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.2)

    # ---- C: disagreement only ----
    ax = fig.add_subplot(2, 2, 3)
    ax.scatter(uv[:, 0], uv[:, 1], s=2, c="#dddddd", linewidths=0)
    if disagree.any():
        ax.scatter(
            uv[disagree, 0],
            uv[disagree, 1],
            s=10,
            c="#d62728",
            linewidths=0,
            label=f"disagree points {disagree.sum()} / {len(disagree)}",
        )
        # on disagree points: blue=local, red=centroid (should oppose)
        bad = keep[disagree[keep]]
        if len(bad):
            ax.quiver(
                uv[bad, 0],
                uv[bad, 1],
                n_loc_2[bad, 0],
                n_loc_2[bad, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0 / arrow_len_mm,
                width=0.0028,
                color="#1f77b4",
                alpha=0.95,
                label="local",
            )
            ax.quiver(
                uv[bad, 0],
                uv[bad, 1],
                n_cent_2[bad, 0],
                n_cent_2[bad, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0 / arrow_len_mm,
                width=0.0028,
                color="#d62728",
                alpha=0.95,
                label="centroid",
            )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("PCA-u (mm)")
    ax.set_ylabel("PCA-v (mm)")
    ax.set_title("C. Disagree points: red=centroid, blue=local (should oppose)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)

    # ---- D: 3D overview ----
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    sub = keep[:: max(1, len(keep) // 800)]
    ax.scatter(
        wall_mm[sub, 0],
        wall_mm[sub, 1],
        wall_mm[sub, 2],
        s=2,
        c=np.where(disagree[sub], "#d62728", "#7f7f7f"),
        alpha=0.55,
        linewidths=0,
    )
    # 3D arrow subsample
    arrow_idx = sub[:: max(1, len(sub) // 180)]
    ax.quiver(
        wall_mm[arrow_idx, 0],
        wall_mm[arrow_idx, 1],
        wall_mm[arrow_idx, 2],
        n_loc[arrow_idx, 0],
        n_loc[arrow_idx, 1],
        n_loc[arrow_idx, 2],
        length=arrow_len_mm,
        normalize=True,
        color="#1f77b4",
        linewidth=0.6,
        arrow_length_ratio=0.35,
    )
    # disagree: red arrows = centroid orientation
    bad3 = arrow_idx[disagree[arrow_idx]]
    if len(bad3):
        ax.quiver(
            wall_mm[bad3, 0],
            wall_mm[bad3, 1],
            wall_mm[bad3, 2],
            n_cent[bad3, 0],
            n_cent[bad3, 1],
            n_cent[bad3, 2],
            length=arrow_len_mm,
            normalize=True,
            color="#d62728",
            linewidth=0.8,
            arrow_length_ratio=0.35,
        )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("D. 3D: blue=local inward; red=centroid wrong side")
    # 等比例
    mins = wall_mm.min(0)
    maxs = wall_mm.max(0)
    centers = 0.5 * (mins + maxs)
    span = float((maxs - mins).max()) * 0.55
    ax.set_xlim(centers[0] - span, centers[0] + span)
    ax.set_ylim(centers[1] - span, centers[1] + span)
    ax.set_zlim(centers[2] - span, centers[2] + span)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
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
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--arrow-stride", type=int, default=12, help="箭头抽样步长，越大越稀疏")
    parser.add_argument("--arrow-len-mm", type=float, default=2.5)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="默认写入 explore/02_normal_orientation/<病例名>/",
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    if not args.out_dir:
        args.out_dir = str(
            Path("/public/newhome/cy/Digital_twin/GNN/outputs/wss_pinn/audits/cfd_velocity_wss_explore")
            / "02_normal_orientation"
            / case_dir.name
        )
    step, step_source = resolve_step(case_dir, args.step, prefer_peak=True)
    wall = dl.load_wall(dl.step_path(case_dir, step, "ascii"))
    interior = dl.load_interior(dl.step_path(case_dir, step, "ascii_in"))
    wall_mm = wall["coords_mm"]
    interior_mm = interior["coords_mm"]
    tree = cKDTree(interior_mm)

    stl_path = find_stl(case_dir)
    centers, face_normals = stl_face_geometry(stl_path, 1.0)
    dist, face_idx = cKDTree(centers).query(wall_mm, k=1)
    n_raw = face_normals[face_idx]
    n_raw = n_raw / np.maximum(np.linalg.norm(n_raw, axis=1, keepdims=True), 1e-12)

    n_cent = orient_centroid(n_raw, wall_mm)
    n_loc = _orient_inward(n_raw.copy(), wall_mm, interior_mm, tree, orient_k=32)
    disagree = np.einsum("ij,ij->i", n_cent, n_loc) < 0

    # 最近内部点落在法向正侧的比例（定向是否真的朝向管腔）
    _, nn = tree.query(wall_mm, k=1)
    delta = interior_mm[nn] - wall_mm
    pos_c = float((np.einsum("ij,ij->i", delta, n_cent) > 0).mean())
    pos_l = float((np.einsum("ij,ij->i", delta, n_loc) > 0).mean())

    meta = {
        "case": f"{case_dir.parent.name}/{case_dir.name}",
        "step": int(step),
        "step_source": step_source,
        "stl": str(stl_path),
        "n_wall": int(len(wall_mm)),
        "n_stl_faces": int(len(centers)),
        "wall_bbox_mm": {
            "min": wall_mm.min(0).tolist(),
            "max": wall_mm.max(0).tolist(),
        },
        "stl_bbox_mm": {
            "min": centers.min(0).tolist(),
            "max": centers.max(0).tolist(),
        },
        "wall_to_stl_p50_mm": float(np.median(dist)),
        "wall_to_stl_p95_mm": float(np.quantile(dist, 0.95)),
        "wall_to_stl_max_mm": float(dist.max()),
        "disagree_frac": float(disagree.mean()),
        "disagree_count": int(disagree.sum()),
        "pos_side_centroid": pos_c,
        "pos_side_local": pos_l,
        "same_coordinate_frame_ok": bool(np.quantile(dist, 0.95) < 3.0),
    }

    out_dir = Path(args.out_dir)
    stem = f"{case_dir.name}_peak{step}_normal_orient_compare"
    fig_path = make_figure(
        wall_mm,
        n_cent,
        n_loc,
        disagree,
        meta,
        out_dir / f"{stem}.png",
        arrow_stride=args.arrow_stride,
        arrow_len_mm=args.arrow_len_mm,
    )
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n图: {fig_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
