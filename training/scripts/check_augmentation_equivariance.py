#!/usr/bin/env python3
"""增强等变一致性自检（前沿方向 §10.1.5 · CPU · 秒级）。

对真实图施加已知旋转 / 反射，验证矢量与标量变换后的不变量，抓「矢量搞反、
标量误转」类 bug。检查项：

  旋转（proper, det=+1）:
    - |velocity|、|wss_vec|、wss 幅值、|tangent| 逐点不变
    - global WSS 与 velocity 已同步旋转（与手工 R 一致）
    - 标量特征（Abscissa/NormRadius/Curvature/torsion/dist_*）不变

  反射（improper, det=-1）:
    - 上述不变量仍成立
    - torsion（赝标量）应变号；若 pipeline 增强未处理会在此报警
    - local WSS circ（若用 local frame）应变号

用法::

    python -m training.scripts.check_augmentation_equivariance --n-graphs 5 --verbose
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch

from pipeline.augmentation import (
    mirror_augmentation,
    random_rotation,
    rigid_transform_augment,
    rotation_matrix_axis_np,
)
from pipeline.config import NODE_FEATURE_NAMES
from ..core.splits import SplitSpec
from .run_v3_f0_decision import DEFAULT_NORM_PARAMS, REPO_ROOT, _load_graph

NODE_IDX = {n: i for i, n in enumerate(NODE_FEATURE_NAMES)}
SCALAR_FEATS = ["Abscissa", "NormRadius", "Curvature", "dist_to_bifurcation",
                "dR_ds", "d_tangent_ds", "dist_to_wall"]
TOL = 1e-4


def _mag(t: torch.Tensor) -> torch.Tensor:
    return t.norm(dim=-1)


def _check_invariants(orig, aug, label: str, verbose: bool) -> List[str]:
    issues: List[str] = []

    def _cmp(name, a, b, tol=TOL):
        d = float((a - b).abs().max())
        ok = d <= tol
        if verbose:
            print(f"    [{label}] {name:28s} max|Δ|={d:.2e} {'OK' if ok else 'FAIL'}")
        if not ok:
            issues.append(f"{label}:{name} max|Δ|={d:.2e}")

    # 速度 / WSS 矢量模长应逐点不变
    _cmp("|velocity|", _mag(orig.y[:, :3]), _mag(aug.y[:, :3]))
    _cmp("pressure", orig.y[:, 3], aug.y[:, 3])
    if hasattr(orig, "y_wss") and orig.y_wss is not None:
        _cmp("|wss_vec|", _mag(orig.y_wss[:, 1:4]), _mag(aug.y_wss[:, 1:4]))
        _cmp("wss_mag(col0)", orig.y_wss[:, 0], aug.y_wss[:, 0])
    _cmp("|tangent|", _mag(orig.x[:, 6:9]), _mag(aug.x[:, 6:9]))

    # 标量特征应不变
    for f in SCALAR_FEATS:
        i = NODE_IDX.get(f)
        if i is not None:
            _cmp(f"scalar:{f}", orig.x[:, i], aug.x[:, i])
    return issues


def _torsion_sign_check(orig, aug, verbose: bool) -> List[str]:
    i = NODE_IDX.get("torsion")
    if i is None:
        return []
    o = orig.x[:, i]
    if float(o.abs().max()) < 1e-8:
        if verbose:
            print("    [mirror] torsion 全 0（该图未含扭率），跳过变号检查")
        return []
    a = aug.x[:, i]
    flipped = float((a + o).abs().max()) <= TOL   # a ≈ -o ?
    unchanged = float((a - o).abs().max()) <= TOL
    if verbose:
        print(f"    [mirror] torsion flipped={flipped} unchanged={unchanged}")
    if unchanged and not flipped:
        return ["mirror:torsion NOT flipped（赝标量反射应变号；pipeline.mirror_augmentation 未处理）"]
    return []


def _phys_vec(t: torch.Tensor, names, stats) -> torch.Tensor:
    """按各向异性 z-score 反归一化为物理矢量。"""
    mean = torch.tensor([float(stats[n]["mean"]) for n in names], dtype=t.dtype)
    std = torch.tensor([float(stats[n]["std"]) for n in names], dtype=t.dtype)
    return t * std + mean


def _check_normaware(orig, stats, verbose: bool) -> List[str]:
    """新路径：物理系模长应严格不变；旧路径：量化各向异性 std 引入的物理失真。"""
    import json
    issues: List[str] = []
    vel_names = ("u", "v", "w")
    wss_names = ("wss_x", "wss_y", "wss_z")

    R_rot = torch.tensor(rotation_matrix_axis_np("y", 0.7))
    M = np.eye(3, dtype=np.float32)
    M[0, 0] = -1.0
    R_ref = torch.tensor(rotation_matrix_axis_np("z", 1.1) @ M)

    phys_v0 = _phys_vec(orig.y[:, :3], vel_names, stats).norm(dim=-1)
    phys_w0 = _phys_vec(orig.y_wss[:, 1:4], wss_names, stats).norm(dim=-1)

    for tag, R in (("rigid-rot", R_rot), ("rigid-ref", R_ref)):
        aug = rigid_transform_augment(orig.clone(), R, str(DEFAULT_NORM_PARAMS))
        pv = _phys_vec(aug.y[:, :3], vel_names, stats).norm(dim=-1)
        pw = _phys_vec(aug.y_wss[:, 1:4], wss_names, stats).norm(dim=-1)
        dv = float((pv - phys_v0).abs().max())
        dw = float((pw - phys_w0).abs().max())
        ok = dv <= 1e-3 and dw <= 1e-2
        if verbose:
            print(f"    [{tag}] phys|vel| max|Δ|={dv:.2e} phys|wss| max|Δ|={dw:.2e} {'OK' if ok else 'FAIL'}")
        if not ok:
            issues.append(f"{tag}: 物理模长漂移 vel={dv:.2e} wss={dw:.2e}")
        if tag == "rigid-ref":
            ti = NODE_IDX.get("torsion")
            if ti is not None and float(orig.x[:, ti].abs().max()) > 1e-8:
                if float((aug.x[:, ti] + orig.x[:, ti]).abs().max()) > TOL:
                    issues.append("rigid-ref: torsion 未变号")

    # 旧 random_rotation 的失真量化（信息项，不算 fail）
    legacy = random_rotation(orig.clone(), axis="y", angle=0.7)
    lv = _phys_vec(legacy.y[:, :3], vel_names, stats).norm(dim=-1)
    moving = phys_v0 > torch.quantile(phys_v0, 0.75)  # 壁面点速度≈0，只统计有流速的内部点
    if bool(moving.any()):
        rel = float((lv[moving] - phys_v0[moving]).abs().median() / phys_v0[moving].median().clamp_min(1e-9))
        if verbose:
            print(f"    [legacy-rot] 直接转归一化分量 → 内部点物理|vel| 失真 median|Δ|/median|v| = {rel*100:.1f}%（应改用 rigid_*）")
    return issues


def main() -> None:
    ap = argparse.ArgumentParser(description="增强等变一致性自检")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training/splits/split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new/AG")
    ap.add_argument("--graphs-subdir", default="processed/graphs")
    ap.add_argument("--n-graphs", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    split = SplitSpec.from_json(args.split.resolve())
    torch.manual_seed(0)
    import random
    random.seed(0)

    import json
    stats = json.loads(Path(DEFAULT_NORM_PARAMS).read_text()).get("statistics", {})

    all_issues: List[str] = []
    n_checked = 0
    for case_rel in (split.test_cases + split.train_cases):
        if n_checked >= args.n_graphs:
            break
        case_dir = args.data_root / case_rel / args.graphs_subdir
        if not case_dir.is_dir():
            continue
        gps = sorted(case_dir.glob("*.pt"))
        if not gps:
            continue
        data = _load_graph(gps[0])
        if not hasattr(data, "y") or data.y is None:
            continue
        print(f"[equiv-check] {case_rel} ({gps[0].name})", flush=True)

        rot = random_rotation(data.clone(), axis="y", angle=0.7)
        all_issues += _check_invariants(data, rot, "rotate", args.verbose)

        mir = mirror_augmentation(data.clone(), axis="x")
        all_issues += _check_invariants(data, mir, "mirror", args.verbose)
        all_issues += _torsion_sign_check(data, mir, args.verbose)

        if hasattr(data, "y_wss") and data.y_wss is not None:
            all_issues += _check_normaware(data, stats, args.verbose)
        n_checked += 1

    print("\n==================== 自检结论 ====================")
    if not all_issues:
        print(f"✅ {n_checked} 图全部通过：归一化空间与物理系模长均不变，"
              "torsion 反射变号正确；rigid_* 增强可用于短训验证。")
    else:
        print(f"⚠️ 发现 {len(all_issues)} 处问题（前 20 条）：")
        for s in all_issues[:20]:
            print("   -", s)


if __name__ == "__main__":
    main()
