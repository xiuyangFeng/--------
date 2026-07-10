#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 AAA/ILO 可配准单元生成对齐可视化 QA（只读，不写 data_wss_min、不改 pipeline）。

复用 visualize_alignment._aligned_wall 同款逻辑：
  read wall geometry(native) -> factor -> compute_transform -> centered@R -> /max|.|
每组输出：逐病例 X-Z 投影小图（颜色=z 高度，翻转会显示为配色/朝向反转）+ 三视图叠加。
"""
from __future__ import annotations
import sys, re, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
sys.path.insert(0, str(ROOT))
from pipeline_wss_min import raw_io, config as C            # noqa: E402
from pipeline_wss_min.preprocess import _resolve_unit_factor  # noqa: E402
from pipeline_wss_min.registration import compute_transform   # noqa: E402

SCR = Path("/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/"
           "c63e71da-acd6-4ca8-b8c0-20257aea34aa/scratchpad")
OUT = ROOT / "docs" / "02-推进与变更" / "assets_新队列审计" / "alignment_viz"
OUT.mkdir(parents=True, exist_ok=True)
PLOT_POINTS = 1500

# 配准层排除的 9 单元（§6.5），从可视化中剔除
REG_EXCLUDE = {
    "ILO/WANG_LI_MIN-0/before", "ILO/WANG_LI_MIN-0/after",
    "ILO/ZHANG_MAO_JIN-0/before", "ILO/ZHANG_MAO_JIN-0/after",
    "AAA/ruputer/ZHOU_KE_XUN", "AAA/unruputer/LIU_WEN_QI",
    "ILO/WEI_QING_FENG-1/before", "ILO/XIE_ZHI_FU-0/after", "ILO/DONG_JIA_JU-1/after",
}

TI = pd.read_csv(SCR / "probe_registration_tiered.csv")
TI = TI[(TI["status"] == "ok") & (~TI["unit_id"].isin(REG_EXCLUDE))].copy()
UNITS = TI["unit_id"].tolist()
DIAG = {r["unit_id"]: r for _, r in TI.iterrows()}

GROUPS = {
    "AAA_ruputer":  [u for u in UNITS if u.startswith("AAA/ruputer")],
    "AAA_unruputer":[u for u in UNITS if u.startswith("AAA/unruputer")],
    "ILO_before":   [u for u in UNITS if u.startswith("ILO/") and u.endswith("/before")],
    "ILO_after":    [u for u in UNITS if u.startswith("ILO/") and u.endswith("/after")],
}


def prefix(cd: Path):
    for f in sorted((cd / "ascii").iterdir()):
        m = re.match(r"(.+)-(\d+)$", f.name)
        if m:
            return m.group(1)
    return None


def aligned_wall(unit_id: str):
    cd = ROOT / "data_new" / unit_id
    pfx = prefix(cd)
    steps = raw_io.list_timesteps(cd)
    ref = steps[0]
    wall_native = raw_io.read_wall_geometry(cd, pfx, ref)
    cl = raw_io.read_centerline(cd)
    factor, anom, _ = _resolve_unit_factor(wall_native, cl, C.DEFAULT.unit)
    wall = wall_native * factor
    T = compute_transform(wall, wall, cl, C.DEFAULT.registration)
    aligned = (wall - T.centroid) @ T.rotation
    scale = float(np.abs(aligned).max()) or 1.0
    return aligned / scale


def subsample(a, n, seed=0):
    if len(a) <= n:
        return a
    idx = np.random.default_rng(seed).choice(len(a), n, replace=False)
    return a[idx]


def short(u):
    p = u.split("/")
    return (p[-1] if p[-1] in ("before", "after") else p[-1])[:0] or ""


def label(u):
    parts = u.split("/")
    if u.startswith("ILO"):
        return f"{parts[1][:12]}/{parts[2][0]}"   # NAME/b or NAME/a
    return parts[-1][:14]


def flags(u):
    r = DIAG[u]
    f = []
    if str(r["unit_extent_mismatch"]) == "True": f.append("EXT")
    if str(r["unit_anomaly"]) == "True": f.append("UNIT")
    if r["origin_kind"] != "flow_divider": f.append(r["origin_kind"][:4])
    if r["roll_source"] != "wall_branches": f.append("branch")
    if str(r["roll_sign_reliable"]) != "True": f.append("FLIP?")
    if float(r["trunk_offset_frac"]) > 0.05: f.append("off")
    return f


def small_multiples(group, units):
    clouds, labels_, flags_ = [], [], []
    for i, u in enumerate(units):
        try:
            a = aligned_wall(u)
        except Exception as e:
            print(f"  skip {u}: {e}", flush=True); continue
        clouds.append(subsample(a, PLOT_POINTS, i))
        labels_.append(label(u)); flags_.append(flags(u))
    n = len(clouds)
    cols = min(8, n) or 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.0))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < n:
            pc = clouds[i]
            ax.scatter(pc[:, 0], pc[:, 2], s=0.6, c=pc[:, 2], cmap="viridis", linewidths=0)
            fl = flags_[i]
            tt = ("*" + "/".join(fl)) if fl else ""
            ax.set_title(f"{labels_[i]}{(' '+tt) if tt else ''}",
                         fontsize=6, color=("red" if fl else "black"))
            ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{group}  aligned X-Z projection (color=z; +Z=main axis, iliac branches at bottom)  "
                 f"n={n}  [red=has-flag: FLIP?/branch/EXT/UNIT/off]", fontsize=10, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = OUT / f"{group}_small_multiples.png"
    fig.savefig(p, dpi=145); plt.close(fig)
    print(f"[viz] {group}: {n} cases -> {p.name}", flush=True)
    return clouds


def ortho_overlay(group, clouds):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    cmap = plt.cm.turbo
    planes = [(0, 2, "X", "Z(main)"), (1, 2, "Y", "Z(main)"), (0, 1, "X", "Y")]
    for ax, (a, b, la, lb) in zip(axes, planes):
        for i, pc in enumerate(clouds):
            ax.scatter(pc[:, a], pc[:, b], s=0.8, alpha=0.2,
                       color=cmap(i / max(len(clouds) - 1, 1)), linewidths=0)
        ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.5)
        ax.axvline(0, color="k", lw=0.6, ls="--", alpha=0.5)
        ax.set_xlabel(la); ax.set_ylabel(lb); ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{la}-{lb}", pad=6)
    fig.suptitle(f"{group}  aligned ortho overlay (all cases, one color each; dashed=origin)  "
                 f"consistent frame => tight common shape", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = OUT / f"{group}_ortho_overlay.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"[viz] {group}: ortho -> {p.name}", flush=True)


def main():
    for group, units in GROUPS.items():
        print(f"\n=== {group}: {len(units)} units ===", flush=True)
        clouds = small_multiples(group, units)
        if clouds:
            ortho_overlay(group, clouds)
    print(f"\n[viz] all figures -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
