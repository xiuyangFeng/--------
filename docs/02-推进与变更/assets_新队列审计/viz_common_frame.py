#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同一框架下的对齐可视化（只读）。

关键区别于旧版：不再逐例按 max|坐标| 归一化，而是
  - 配准原点(分叉点)统一置于 (0,0)；
  - 每队列用一个**共同尺度** S=median(各例 98 分位 |坐标|_mm) 归一化，全格同尺；
  - 固定坐标范围，各格可直接比位置/大小/朝向。
另加自动翻转检测：分叉端(两髂支分叉、X 展开大)应在 +Z；若在 -Z 记 'INV'。
"""
from __future__ import annotations
import sys, re, math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
sys.path.insert(0, str(ROOT))
from pipeline_wss_min import raw_io, config as C
from pipeline_wss_min.preprocess import _resolve_unit_factor
from pipeline_wss_min.registration import compute_transform

SCR = Path("/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/"
           "c63e71da-acd6-4ca8-b8c0-20257aea34aa/scratchpad")
OUT = ROOT / "docs" / "02-推进与变更" / "assets_新队列审计" / "alignment_viz"
OUT.mkdir(parents=True, exist_ok=True)
PLOT_POINTS = 1500
AX = 1.9   # 固定坐标半径（共同尺度的倍数）

REG_EXCLUDE = {
    "ILO/WANG_LI_MIN-0/before", "ILO/WANG_LI_MIN-0/after",
    "ILO/ZHANG_MAO_JIN-0/before", "ILO/ZHANG_MAO_JIN-0/after",
    "AAA/ruputer/ZHOU_KE_XUN", "AAA/unruputer/LIU_WEN_QI",
    "ILO/WEI_QING_FENG-1/before", "ILO/XIE_ZHI_FU-0/after", "ILO/DONG_JIA_JU-1/after",
}
TI = pd.read_csv(SCR / "probe_registration_tiered.csv")
TI = TI[(TI["status"] == "ok") & (~TI["unit_id"].isin(REG_EXCLUDE))].copy()
DIAG = {r["unit_id"]: r for _, r in TI.iterrows()}
UNITS = TI["unit_id"].tolist()

GROUPS = {
    "AAA_ruputer":  [u for u in UNITS if u.startswith("AAA/ruputer")],
    "AAA_unruputer":[u for u in UNITS if u.startswith("AAA/unruputer")],
    "ILO_before":   [u for u in UNITS if u.startswith("ILO/") and u.endswith("/before")],
    "ILO_after":    [u for u in UNITS if u.startswith("ILO/") and u.endswith("/after")],
}


def pfx(cd):
    for f in sorted((cd / "ascii").iterdir()):
        m = re.match(r"(.+)-(\d+)$", f.name)
        if m:
            return m.group(1)


def aligned_mm(u):
    """对齐后坐标(mm)，分叉原点在 (0,0,0)。"""
    cd = ROOT / "data_new" / u
    steps = raw_io.list_timesteps(cd)
    wall = raw_io.read_wall_geometry(cd, pfx(cd), steps[0])
    cl = raw_io.read_centerline(cd)
    f, _, _ = _resolve_unit_factor(wall, cl, C.DEFAULT.unit)
    T = compute_transform(wall * f, wall * f, cl, C.DEFAULT.registration)
    return (wall * f - T.centroid) @ T.rotation


def sub(a, n, seed=0):
    if len(a) <= n:
        return a
    return a[np.random.default_rng(seed).choice(len(a), n, replace=False)]


def fork_is_inverted(a):
    """tilt-无关 + 抗瘤体翻转检测：沿病例 PCA 主轴取**很薄的末端切片(5%)**，
    在切片内测横向展开——分叉端(两髂支分开)展开大、单管/瘤体端小；瘤体在中段不在末梢。
    分叉端应在 +Z；在 -Z 且展开比>1.5 记翻转。"""
    c = a - a.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    v = vt[0]
    if v[2] < 0:
        v = -v
    t = c @ v

    def spread(mask):
        p = c[mask]
        if len(p) < 10:
            return 0.0
        p = p - p.mean(0)
        _, _, w = np.linalg.svd(p, full_matrices=False)
        proj = p @ w[0]
        return float(proj.max() - proj.min())

    hi = spread(t >= np.quantile(t, 0.95))
    lo = spread(t <= np.quantile(t, 0.05))
    fork_at_high = hi >= lo
    big, small = (hi, lo) if fork_at_high else (lo, hi)
    if small > 0 and big / small < 1.5:      # 展开差不显著 -> 不判翻转
        return False, hi, lo
    fork_z_sign = np.sign(v[2]) if fork_at_high else -np.sign(v[2])
    return bool(fork_z_sign < 0), hi, lo


def label(u):
    p = u.split("/")
    return f"{p[1][:12]}/{p[2][0]}" if u.startswith("ILO") else p[-1][:14]


def flags(u, inv):
    r = DIAG[u]; f = []
    if inv: f.append("INV")
    if str(r["unit_extent_mismatch"]) == "True": f.append("EXT")
    if str(r["roll_sign_reliable"]) != "True": f.append("FLIP?")
    if r["roll_source"] != "wall_branches": f.append("branch")
    return f


def render(group, units):
    clouds, S_list, labs, invs = [], [], [], []
    raw = []
    for i, u in enumerate(units):
        try:
            a = aligned_mm(u)
        except Exception as e:
            print(f"  skip {u}: {e}", flush=True); continue
        raw.append((u, a))
        S_list.append(np.percentile(np.abs(a), 98))
    S = float(np.median(S_list))  # 队列共同尺度
    for u, a in raw:
        inv, _, _ = fork_is_inverted(a)
        clouds.append(sub(a, PLOT_POINTS, 0) / S)
        labs.append(label(u)); invs.append((u, inv))
    n = len(clouds)
    cols = min(8, n) or 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.0))
    axes = np.atleast_1d(axes).ravel()
    inv_names = []
    for i, ax in enumerate(axes):
        if i < n:
            pc = clouds[i]; u, inv = invs[i]
            fl = flags(u, inv)
            if inv: inv_names.append(u)
            ax.axhline(0, color="0.8", lw=0.5, zorder=0)
            ax.axvline(0, color="0.8", lw=0.5, zorder=0)
            ax.scatter(pc[:, 0], pc[:, 2], s=0.6, c=pc[:, 2], cmap="viridis",
                       vmin=-AX, vmax=AX, linewidths=0)
            tt = ("*" + "/".join(fl)) if fl else ""
            ax.set_title(f"{labs[i]}{(' '+tt) if tt else ''}", fontsize=6,
                         color=("red" if fl else "black"))
            ax.set_xlim(-AX, AX); ax.set_ylim(-AX, AX); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{group}  COMMON-FRAME X-Z (origin=bifurcation at 0,0; shared scale S={S:.0f}mm; "
                 f"+Z=iliac fork side)  n={n}  [red=INV(flipped)/EXT/FLIP?/branch]", fontsize=9.5, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT / f"{group}_small_multiples.png", dpi=145); plt.close(fig)
    # ortho overlay 同一共同尺度
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
    cmap = plt.cm.turbo
    planes = [(0, 2, "X", "Z(main)"), (1, 2, "Y", "Z(main)"), (0, 1, "X", "Y")]
    for ax, (a_, b_, la, lb) in zip(axs, planes):
        for i, pc in enumerate(clouds):
            ax.scatter(pc[:, a_], pc[:, b_], s=0.7, alpha=0.18,
                       color=cmap(i / max(len(clouds) - 1, 1)), linewidths=0)
        ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.5)
        ax.axvline(0, color="k", lw=0.6, ls="--", alpha=0.5)
        ax.set_xlabel(la); ax.set_ylabel(lb); ax.set_aspect("equal")
        ax.set_xlim(-AX, AX); ax.set_ylim(-AX, AX); ax.set_title(f"{la}-{lb}", pad=6)
    fig.suptitle(f"{group}  COMMON-FRAME ortho overlay (shared scale, origin-centered)  n={n}", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / f"{group}_ortho_overlay.png", dpi=130); plt.close(fig)
    print(f"[viz] {group}: n={n} S={S:.0f}mm  INV={len(inv_names)} {inv_names}", flush=True)
    return inv_names


def main():
    allinv = []
    for g, units in GROUPS.items():
        allinv += render(g, units)
    print(f"\n[viz] 所有疑似翻转 INV: {allinv}")
    print(f"[viz] figures -> {OUT}")


if __name__ == "__main__":
    main()
