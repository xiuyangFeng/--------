#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对齐可视化：确认解剖原点 + 统一视角（所有病例同一朝向）。

只算配准（分叉/兜底原点 + 中心线主导刚性旋转 + 逐病例归一化），不落 bundle，快。
读 split 的 included 病例（默认 split_AG_wss_min_v1）。

产出（outputs/wss_min/alignment_viz_anatomical_v2/）：
- 01_before_after_overlay.png  对齐前(仅去重心/未旋转) vs 对齐后 —— 全病例叠加，看是否统一朝向
- 02_aligned_ortho_overlay.png 对齐后三视图叠加 —— 看重心在原点、朝向一致
- 03_aligned_small_multiples.png 每病例一格 —— 逐个核查有无歪的
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体（系统 Noto Sans CJK），无则退回默认（标题会是方框，但不影响图形）
for _p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    if Path(_p).is_file():
        try:
            matplotlib.font_manager.fontManager.addfont(_p)
            matplotlib.rcParams["font.family"] = \
                matplotlib.font_manager.FontProperties(fname=_p).get_name()
            break
        except Exception:  # noqa: BLE001
            pass
matplotlib.rcParams["axes.unicode_minus"] = False

from . import config as C
from . import raw_io
from .preprocess import _resolve_unit_factor
from .registration import compute_transform

OUT_DIR = C.PROJECT_ROOT / "outputs" / "wss_min" / "alignment_viz_anatomical_v2"
PLOT_POINTS = 1500  # 每病例绘图下采样点数


def _load_included(split_path: Path):
    sp = json.loads(split_path.read_text())
    cases = sp.get("train_cases", []) + sp.get("val_cases", []) + sp.get("test_cases", [])
    # split 用 'fast/xxx'；本流程用 cohort 'AG/fast'
    return [("AG/" + c.split("/")[0], c.split("/")[1]) for c in cases], sp


def _aligned_wall(cohort_rel: str, case_name: str, cfg: C.PipelineConfig):
    """返回 (centered_only, aligned_norm, anomaly, transform)，前两者归一化到 ~[-1,1]。"""
    case_dir = C.raw_case_dir(cohort_rel, case_name)
    steps = raw_io.list_timesteps(case_dir)
    ref = steps[0]
    wall_native = raw_io.read_wall_geometry(case_dir, case_name, ref)
    cl = raw_io.read_centerline(case_dir)
    factor, anomaly, _ = _resolve_unit_factor(wall_native, cl, cfg.unit)
    wall = wall_native * factor

    reg_cfg = C.registration_for_case(cohort_rel, case_name, cfg.registration)
    T = compute_transform(wall, wall, cl, reg_cfg)
    centered = wall - T.centroid
    aligned = centered @ T.rotation
    scale = float(np.abs(aligned).max()) or 1.0
    return centered / scale, aligned / scale, anomaly, T


def _subsample(a, n, seed=0):
    if len(a) <= n:
        return a
    idx = np.random.default_rng(seed).choice(len(a), n, replace=False)
    return a[idx]


def run(split_name: str = "split_AG_wss_min_v1", tag: str | None = None):
    cfg = C.DEFAULT
    global OUT_DIR
    if tag:
        OUT_DIR = C.PROJECT_ROOT / "outputs" / "wss_min" / f"alignment_viz_{tag}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_path = C.PROJECT_ROOT / "training" / "splits" / f"{split_name}.json"
    cases, sp = _load_included(split_path)
    print(f"[viz] split={split_name} included={len(cases)}")

    centered_all, aligned_all, names, anomalies, transforms = [], [], [], [], []
    for i, (coh, cn) in enumerate(cases):
        try:
            c0, a0, anom, trans = _aligned_wall(coh, cn, cfg)
            centered_all.append(_subsample(c0, PLOT_POINTS, i))
            aligned_all.append(_subsample(a0, PLOT_POINTS, i))
            names.append(cn)
            anomalies.append(anom)
            transforms.append(trans)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {coh}/{cn}: {e}")
    print(f"[viz] processed {len(names)} cases")

    _fig_before_after(centered_all, aligned_all)
    _fig_ortho_overlay(aligned_all)
    _fig_small_multiples(aligned_all, names, anomalies, transforms)
    print(f"[viz] figures -> {OUT_DIR}")


# ---------------------------------------------------------------------------
def _fig_before_after(centered_all, aligned_all):
    fig = plt.figure(figsize=(14, 7))
    cmap = plt.cm.turbo
    for j, (data, title) in enumerate([
        (centered_all, "对齐前：仅平移到解剖原点，未旋转（各病例朝向不一）"),
        (aligned_all,  "对齐后：分叉原点 + trunk 主轴 + 分支平面横轴（统一朝向 +Z）"),
    ]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        for i, pc in enumerate(data):
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1, alpha=0.25,
                       color=cmap(i / max(len(data) - 1, 1)), linewidths=0)
        ax.set_title(title, fontsize=10, pad=10)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.view_init(elev=18, azim=-60)
    fig.suptitle("全病例壁面点云叠加（每病例一色）：解剖原点版对齐前 vs 对齐后", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "01_before_after_overlay.png", dpi=130)
    plt.close(fig)


def _fig_ortho_overlay(aligned_all):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    cmap = plt.cm.turbo
    planes = [(0, 2, "X", "Z (主轴)"), (1, 2, "Y", "Z (主轴)"), (0, 1, "X", "Y")]
    for ax, (a, b, la, lb) in zip(axes, planes):
        for i, pc in enumerate(aligned_all):
            ax.scatter(pc[:, a], pc[:, b], s=0.8, alpha=0.2,
                       color=cmap(i / max(len(aligned_all) - 1, 1)), linewidths=0)
        ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.5)
        ax.axvline(0, color="k", lw=0.6, ls="--", alpha=0.5)
        ax.set_xlabel(la); ax.set_ylabel(lb); ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{la}-{lb} 投影", pad=8)
    fig.suptitle("对齐后 三视图叠加（虚线=解剖原点）：分叉原点、朝向一致", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT_DIR / "02_aligned_ortho_overlay.png", dpi=130)
    plt.close(fig)


def _fig_small_multiples(aligned_all, names, anomalies, transforms):
    n = len(aligned_all)
    cols = min(9, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.9))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < n:
            pc = aligned_all[i]
            ax.scatter(pc[:, 0], pc[:, 2], s=0.6, c=pc[:, 2], cmap="viridis", linewidths=0)
            trans = transforms[i]
            flags = []
            if anomalies[i]:
                flags.append("单位")
            if trans.origin_kind != "bifurcation":
                flags.append(trans.origin_kind)
            # 左右轴回退（非默认 wall_branches）才告警
            if trans.roll_source in ("branches", "curvature"):
                flags.append(trans.roll_source)
            # 左右符号两锚冲突、待人工复核
            if not getattr(trans, "roll_sign_reliable", True):
                flags.append("翻转?")
            flag = ("  ⚠" + "/".join(flags)) if flags else ""
            ax.set_title(f"{names[i][:14]}{flag}", fontsize=6)
            ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("对齐后每病例 X-Z 投影（颜色=z 高度）：逐个核查朝向是否一致", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(OUT_DIR / "03_aligned_small_multiples.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="split_AG_wss_min_v1")
    ap.add_argument("--tag", default=None,
                    help="输出目录后缀：outputs/wss_min/alignment_viz_<tag>；缺省用 _anatomical_v2")
    args = ap.parse_args()
    run(args.split, args.tag)
