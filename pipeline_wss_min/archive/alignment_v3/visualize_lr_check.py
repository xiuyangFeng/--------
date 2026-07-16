#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""左右髂支朝向一致性 QA。

配准后 principal_axis_target='z' 下，新基 X 轴 = 左右轴（L-R）、Z 轴 = 主轴（入口->分叉->髂支）。
本脚本把每个病例的壁面点做同一套单位换算 + 刚性配准 + 逐病例归一化后，
取分叉下游髂支段，按 L-R 坐标（norm-x）符号上色，画到同一网格、同一视角：
- 若各病例朝向自洽，所有子图应呈现相同的左右配色（例如左髂一色、右髂一色）；
- 出现某个子图左右配色相反，即该病例朝向翻转，需复核。

支持对比新旧配准口径（--roll-source / --roll-sign-mode），方便验证优化效果。

用法：
    PY=/public/newhome/cy/.conda/envs/GNN/bin/python
    $PY -m pipeline_wss_min.archive.alignment_v3.visualize_lr_check \
      --split split_AG_wss_min_v1 --n 20
    # 旧口径对照
    $PY -m pipeline_wss_min.archive.alignment_v3.visualize_lr_check \
      --roll-source branches --roll-sign-mode world_axis --tag old
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import List, Tuple

import numpy as np

from pipeline_wss_min import config as C
from pipeline_wss_min import raw_io
from pipeline_wss_min.archive.alignment_v3 import legacy_registration_for_case
from pipeline_wss_min.preprocess import _resolve_unit_factor
from pipeline_wss_min.registration import compute_transform, _select_wall_branch_side


def _cases_from_split(split_name: str) -> List[str]:
    path = C.PROJECT_ROOT / "training" / "splits" / f"{split_name}.json"
    d = json.loads(path.read_text())
    return d["train_cases"] + d["val_cases"] + d["test_cases"]


def _aligned_wall(case: str, reg_cfg, use_case_overrides: bool = True) -> Tuple[np.ndarray, object] | Tuple[None, None]:
    """返回逐病例归一化后的壁面坐标 (N,3) 与 transform。失败返回 (None, None)。"""
    cohort, name = ("AG/" + case).rsplit("/", 1)
    case_dir = C.raw_case_dir(cohort, name)
    steps = raw_io.list_timesteps(case_dir)
    if not steps:
        return None, None
    wall_native = raw_io.read_wall_geometry(case_dir, name, steps[0])
    int_native = raw_io.read_interior_geometry(case_dir, name, steps[0])
    cl = raw_io.read_centerline(case_dir)
    factor, _, _ = _resolve_unit_factor(wall_native, cl, C.DEFAULT.unit)
    wall_pts = wall_native * factor
    int_pts = int_native * factor
    if use_case_overrides:
        reg_cfg = legacy_registration_for_case(cohort, name, reg_cfg)
    else:
        reg_cfg = replace(reg_cfg, frame_mode="legacy_centerline")
    T = compute_transform(wall_pts, int_pts, cl, reg_cfg)
    aln = T.apply_points(wall_pts)
    scale = float(np.abs(aln).max())
    scale = scale if scale > 1e-9 else 1.0
    return (aln / scale).astype(np.float32), T


def run(split_name: str, n_cases: int, tag: str, reg_cfg,
        down_frac: float, seed: int, cases_keep: List[str] | None = None,
        use_case_overrides: bool = True) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = _cases_from_split(split_name)
    if cases_keep:
        keep = set(cases_keep)
        cases = [c for c in cases if c in keep]
    rng = np.random.default_rng(seed)
    if not cases_keep and n_cases < len(cases):
        cases = [cases[i] for i in sorted(rng.choice(len(cases), n_cases, replace=False))]

    ncol = min(5, len(cases))
    nrow = math.ceil(len(cases) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.2 * nrow), squeeze=False)

    n_flag = 0
    for ax in axes.ravel():
        ax.axis("off")
    for k, case in enumerate(cases):
        ax = axes[k // ncol][k % ncol]
        ax.axis("on")
        coords, T = _aligned_wall(case, reg_cfg, use_case_overrides=use_case_overrides)
        if coords is None:
            ax.set_title(f"{case}\n(load fail)", fontsize=7)
            continue
        # 与正式配准保持一致：对分叉两侧各算横断面双峰分离度，取更像双髂支的一侧。
        best = _select_wall_branch_side(coords, np.array([0.0, 0.0, 1.0]), reg_cfg, min_sep=0.3)
        if best is None:
            ax.set_title(f"{case}\n(no downstream)", fontsize=7)
            continue
        down = best["mask"]
        x = coords[down, 0]                          # 左右坐标
        z = coords[:, 2]                              # 主轴
        zz = z[down]
        side = x >= 0
        # 冠状面视角：横轴 = 左右(x)，纵轴 = 主轴(z)
        ax.scatter(x[side], zz[side], s=1, c="#d62728", alpha=0.5, label="+X side")
        ax.scatter(x[~side], zz[~side], s=1, c="#1f77b4", alpha=0.5, label="-X side")
        rel = getattr(T, "roll_sign_reliable", True)
        if not rel:
            n_flag += 1
        flag = "" if rel else "  ⚠flip?"
        ax.set_title(f"{case}\n{T.roll_source}/{T.roll_sign_source}"
                     f" |cos|={T.roll_sign_cos:.2f} sep={best['sep']:.2f}{flag}", fontsize=6)
        ax.set_xlabel("L-R (norm x)", fontsize=6)
        ax.set_ylabel("main axis (norm z)", fontsize=6)
        ax.axvline(0, color="k", lw=0.4, ls="--")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=5)

    fig.suptitle(f"L-R iliac orientation QA  [{tag}]  "
                 f"roll_source={reg_cfg.roll_source} sign_mode={reg_cfg.roll_sign_mode}  "
                 f"overrides={'on' if use_case_overrides else 'off'}  "
                 f"unreliable={n_flag}/{len(cases)}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir = C.PROJECT_ROOT / "outputs" / "wss_min" / "lr_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"lr_side_grid_{tag}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[lr_check] {tag}: {len(cases)} cases, unreliable={n_flag} -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="split_AG_wss_min_v1")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--tag", default="new")
    ap.add_argument("--roll-source", default=None,
                    help="覆盖 roll_source（回退对照用，如 branches）")
    ap.add_argument("--roll-sign-mode", default=None,
                    help="覆盖 roll_sign_mode（回退对照用，如 world_axis）")
    ap.add_argument("--cases", default="",
                    help="逗号分隔病例列表，如 fast/A,slow/B；为空则按 --n 随机抽样")
    args = ap.parse_args()

    reg = C.DEFAULT.registration
    over = {}
    if args.roll_source:
        over["roll_source"] = args.roll_source
    if args.roll_sign_mode:
        over["roll_sign_mode"] = args.roll_sign_mode
    if over:
        reg = replace(reg, **over)
    cases_keep = [c.strip() for c in args.cases.split(",") if c.strip()]
    run(args.split, args.n, args.tag, reg, reg.downstream_wall_axis_frac, args.seed,
        cases_keep=cases_keep or None, use_case_overrides=not bool(over))


if __name__ == "__main__":
    main()
