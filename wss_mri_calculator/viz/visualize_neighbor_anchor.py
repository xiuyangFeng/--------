"""
近壁采样锚定可视化：壁面点如何挂上 K 近邻内部单元
================================================
说明两件事：
  1) 评估点 = Fluent ascii 壁面节点（原始 CFD 壁面点；可全量或随机抽样）
  2) 梯度拟合样本 = ascii_in 内部单元中心的欧氏 K 近邻，再按内法向投影 η>0 保留

输出一张多面板图，给老师看「点从哪来、邻域怎么锚定」。
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
from calculate_wss_cfd import pca_wall_normals, resolve_step

MM_TO_M = 1e-3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-dir",
        type=str,
        default="/public/newhome/cy/Digital_twin/GNN/data_new/AG/slow/LIU_JIN_LIANG",
    )
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--n-focus", type=int, default=3, help="抽几个壁面点做局部放大")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="默认写入 explore/03_neighbor_anchor/<病例名>/",
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    if not args.out_dir:
        args.out_dir = str(
            Path("/public/newhome/cy/Digital_twin/GNN/outputs/wss_pinn/audits/cfd_velocity_wss_explore")
            / "03_neighbor_anchor"
            / case_dir.name
        )
    step, step_source = resolve_step(case_dir, None, prefer_peak=True)
    wall = dl.load_wall(dl.step_path(case_dir, step, "ascii"))
    interior = dl.load_interior(dl.step_path(case_dir, step, "ascii_in"))
    wall_mm = wall["coords_mm"]
    int_mm = interior["coords_mm"]
    tree = cKDTree(int_mm)
    normals = pca_wall_normals(wall_mm, int_mm, tree)

    rng = np.random.default_rng(args.seed)
    # 选 3 个焦点：一个高 WSS、一个中位、一个随机（覆盖不同近壁密度）
    order = np.argsort(wall["wss_mag"])
    focus = np.unique(
        [
            int(order[int(0.95 * (len(order) - 1))]),
            int(order[len(order) // 2]),
            int(rng.integers(0, len(wall_mm))),
        ]
    )[: args.n_focus]

    k = min(args.neighbors, len(int_mm))
    dists, nbs = tree.query(wall_mm[focus], k=k)

    # 全局 PCA 展开（背景）
    centered = wall_mm - wall_mm.mean(0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    uv_wall = centered @ vt[:2].T

    fig = plt.figure(figsize=(14.5, 10.2), dpi=150)
    fig.suptitle(
        f"{case_dir.parent.name}/{case_dir.name}  step={step} ({step_source})\n"
        f"Wall eval pts = CFD ascii wall nodes (n={len(wall_mm)}); "
        f"neighbors = ascii_in cell centers (n={len(int_mm)}), K={k}, keep eta>0",
        fontsize=11,
    )

    # ---- A: overview ----
    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(uv_wall[:, 0], uv_wall[:, 1], s=2, c="#c7c7c7", linewidths=0, label="all wall nodes")
    ax.scatter(
        uv_wall[focus, 0],
        uv_wall[focus, 1],
        s=60,
        c=["#d62728", "#ff7f0e", "#2ca02c"][: len(focus)],
        edgecolors="k",
        linewidths=0.6,
        zorder=5,
        label="focus wall pts",
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("A. Wall nodes (original CFD ascii); colored = focus")
    ax.set_xlabel("PCA-u (mm)")
    ax.set_ylabel("PCA-v (mm)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ---- B/C/D: local zoom per focus ----
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    labels = ["high-WSS", "median-WSS", "random"]
    summary = []
    for i, wi in enumerate(focus):
        ax = fig.add_subplot(2, 2, i + 2)
        p = wall_mm[wi]
        n = normals[wi]
        idx = nbs[i]
        pts = int_mm[idx]
        eta = (pts - p) @ n  # mm
        keep = eta > 1e-4
        drop = ~keep

        # local frame: normal vs a tangential axis in plane of (pts-p)
        # plot: horizontal = signed distance along normal (eta), vertical = lateral distance
        lateral = np.linalg.norm((pts - p) - eta[:, None] * n, axis=1)

        ax.axvline(0, color="k", lw=1.0, alpha=0.7)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        if drop.any():
            ax.scatter(
                eta[drop],
                lateral[drop],
                s=28,
                c="#999999",
                marker="x",
                label=f"drop eta<=0 ({drop.sum()})",
            )
        ax.scatter(
            eta[keep],
            lateral[keep],
            s=22,
            c=colors[i],
            alpha=0.85,
            edgecolors="none",
            label=f"keep eta>0 ({keep.sum()}/{k})",
        )
        ax.scatter([0], [0], s=80, c="k", marker="s", label="wall node", zorder=6)
        # normal direction indicator on eta axis
        ax.annotate(
            "",
            xy=(max(0.8, float(np.nanpercentile(eta[keep], 80)) if keep.any() else 0.8), 0),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.8),
        )
        ax.text(
            max(0.4, float(np.nanpercentile(eta[keep], 50)) if keep.any() else 0.4),
            0.05,
            "inward normal (+eta)",
            color="#1f77b4",
            fontsize=8,
        )

        ax.set_xlabel("eta = normal projection (mm)")
        ax.set_ylabel("lateral distance to normal ray (mm)")
        ax.set_title(
            f"{chr(66+i)}. Focus {labels[i] if i < len(labels) else i}: "
            f"WSS_truth={wall['wss_mag'][wi]:.2f}Pa\n"
            f"eta_keep p50={np.median(eta[keep]):.3f}mm  "
            f"max={eta[keep].max():.3f}mm"
            if keep.any()
            else f"{chr(66+i)}. Focus {i}: no positive-eta neighbors"
        )
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.25)
        # symmetric-ish x if negatives exist
        xmax = float(max(np.abs(eta).max(), 0.5))
        ax.set_xlim(-0.15 * xmax, 1.05 * xmax)

        summary.append(
            {
                "wall_index": int(wi),
                "label": labels[i] if i < len(labels) else f"focus_{i}",
                "wss_truth_pa": float(wall["wss_mag"][wi]),
                "k": int(k),
                "n_keep": int(keep.sum()),
                "n_drop": int(drop.sum()),
                "eta_keep_min_mm": float(eta[keep].min()) if keep.any() else None,
                "eta_keep_p50_mm": float(np.median(eta[keep])) if keep.any() else None,
                "eta_keep_max_mm": float(eta[keep].max()) if keep.any() else None,
                "euclid_nn_mm": float(dists[i, 0]),
            }
        )

    meta = {
        "case": f"{case_dir.parent.name}/{case_dir.name}",
        "step": int(step),
        "n_wall_total": int(len(wall_mm)),
        "n_interior_total": int(len(int_mm)),
        "note": (
            "Wall evaluation uses original CFD wall nodes from ascii/. "
            "Default CLI --sample-count 2000 randomly subsets them; 0 = all. "
            "Gradient samples are original interior cell centers from ascii_in/, "
            "selected as Euclidean KNN then filtered by eta>0."
        ),
        "focus": summary,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{case_dir.name}_peak{step}_neighbor_anchor"
    fig_path = out_dir / f"{stem}.png"
    json_path = out_dir / f"{stem}.json"
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig_path)
    plt.close(fig)
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n图: {fig_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
