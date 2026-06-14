#!/usr/bin/env python3
"""V3P 路径 I · Q0 oracle bundle（优化方案 §15.3 + I1 θ 审计 · 0 重训 · CPU）。

合并 I2-o / I20-o / I3-o / I14-o 前置 oracle：
  ① 标架投影统计 (τ_s, τ_θ) vs global wss_x/y
  ② 二次流能量比 τ_θ/|τ|
  ③ POD 系数 + 简化 HH 势低模的可预测性（病例级几何 → 系数 GBDT）
  ④ I1 θ-roll × POD 兼容性审计

产物：``outputs/field/f0_decision/v3p_i_oracle_bundle_<date>.json``
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from ..core.denylist import filter_case_names
from ..core.splits import SplitSpec
from pipeline.wall_unwrap.grid import compute_theta, load_norm_stats
from .run_v3_f0_decision import (
    DEFAULT_NORM_PARAMS,
    NODE_IDX,
    REPO_ROOT,
    X_TAN,
    _denorm_zscore,
    _load_graph,
    _r2_score,
    _safe_json_float,
    oracle_pod_2d,
)

WSS_X = 1
WSS_Y = 2
WSS_Z = 3


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _intrinsic_components(
    xyz: np.ndarray,
    tan: np.ndarray,
    wss_norm: np.ndarray,
    stats: Mapping[str, Dict[str, float]],
) -> Dict[str, np.ndarray]:
    """壁面 WSS 投影到 (e_s, e_θ, n) 内禀标架。"""
    wss_phys = np.stack(
        [
            _denorm_zscore(wss_norm[:, i], stats.get(n))
            for i, n in zip((WSS_X, WSS_Y, WSS_Z), ("wss_x", "wss_y", "wss_z"))
        ],
        axis=1,
    )
    t = _unit(tan)
    theta = compute_theta(xyz, tan)
    t_mean = t.mean(axis=0)
    t_mean = t_mean / (np.linalg.norm(t_mean) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(t_mean[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = ref - (ref @ t_mean) * t_mean
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(t_mean, e1)
    e_theta = _unit(np.tile(e2, (xyz.shape[0], 1)))
    e_s = t
    n = _unit(np.cross(e_s, e_theta))
    tau_s = np.sum(wss_phys * e_s, axis=1)
    tau_theta = np.sum(wss_phys * e_theta, axis=1)
    tau_n = np.sum(wss_phys * n, axis=1)
    return {
        "tau_s": tau_s,
        "tau_theta": tau_theta,
        "tau_n": tau_n,
        "wss_x": wss_phys[:, 0],
        "wss_y": wss_phys[:, 1],
        "wss_z": wss_phys[:, 2],
        "theta": theta,
    }


def _collect_wall_intrinsic(
    cases: Sequence[str],
    data_root: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    max_graphs_per_case: int,
) -> Dict[str, np.ndarray]:
    chunks: Dict[str, List[np.ndarray]] = {
        k: [] for k in ("tau_s", "tau_theta", "tau_n", "wss_x", "wss_y", "wss_z", "theta")
    }
    for case_rel in cases:
        case_dir = data_root / case_rel / "processed" / "graphs"
        if not case_dir.is_dir():
            continue
        for graph_path in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
            data = _load_graph(graph_path)
            x = data.x.numpy()
            y_wss = data.y_wss.numpy()
            wall = x[:, NODE_IDX["is_wall"]] > 0.5
            if int(np.sum(wall)) < 20:
                continue
            comp = _intrinsic_components(
                x[wall, :3].astype(np.float64),
                x[wall, X_TAN].astype(np.float64),
                y_wss[wall],
                stats,
            )
            for k, v in comp.items():
                chunks[k].append(v)
    return {k: np.concatenate(v) if v else np.empty(0) for k, v in chunks.items()}


def _frame_projection_stats(train: Mapping[str, np.ndarray], test: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    def _var_ratio(a: np.ndarray, b: np.ndarray) -> float:
        va, vb = float(np.var(a)), float(np.var(b))
        return va / vb if vb > 1e-12 else float("nan")

    stack = np.stack([train["tau_s"], train["tau_theta"], train["tau_n"]], axis=1)
    tau_n_ratio = float(np.mean(np.abs(train["tau_n"]) / (np.linalg.norm(stack, axis=1) + 1e-12)))

    results: Dict[str, Any] = {}
    for split_name, data in (("train", train), ("test", test)):
        results[f"{split_name}_var_tau_theta"] = float(np.var(data["tau_theta"]))
        results[f"{split_name}_var_wss_y"] = float(np.var(data["wss_y"]))
        results[f"{split_name}_var_tau_s"] = float(np.var(data["tau_s"]))
    results["var_ratio_tau_theta_over_wss_y_train"] = _var_ratio(train["tau_theta"], train["wss_y"])
    results["var_ratio_tau_theta_over_wss_y_test"] = _var_ratio(test["tau_theta"], test["wss_y"])
    results["tau_n_residual_fraction_mean"] = tau_n_ratio
    results["gate_tau_n_under_10pct"] = bool(tau_n_ratio < 0.10)
    results["gate_tau_theta_var_gt_wss_y"] = bool(
        results["var_ratio_tau_theta_over_wss_y_test"] > 1.05
    )
    return results


def _energy_ratio_stats(data: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    mag = np.sqrt(data["tau_s"] ** 2 + data["tau_theta"] ** 2 + data["tau_n"] ** 2)
    ratio = np.abs(data["tau_theta"]) / (mag + 1e-12)
    p95 = float(np.percentile(ratio, 95))
    dominant = float(np.mean(ratio > 0.35))
    return {
        "tau_theta_over_mag_mean": float(np.mean(ratio)),
        "tau_theta_over_mag_p95": p95,
        "fraction_cells_ratio_gt_0p35": dominant,
        "gate_secondary_flow_non_trivial": bool(dominant > 0.05),
    }


def _pod_coeff_predictability(
    split_path: Path,
    norm_stats: Mapping[str, Dict[str, float]],
    *,
    grid_s: int,
    n_sectors: int,
    n_modes: int,
    max_graphs_per_case: int,
) -> Dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    split = SplitSpec.from_json(split_path)
    data_root = REPO_ROOT / "data_new" / "AG"

    def _profiles_and_meta(cases: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        profiles: List[np.ndarray] = []
        meta: List[np.ndarray] = []
        for case_rel in cases:
            case_dir = data_root / case_rel / "processed" / "graphs"
            if not case_dir.is_dir():
                continue
            for graph_path in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
                data = _load_graph(graph_path)
                x = data.x.numpy()
                wall = x[:, NODE_IDX["is_wall"]] > 0.5
                if int(np.sum(wall)) < 20:
                    continue
                absc = x[wall, NODE_IDX["Abscissa"]]
                nr = x[wall, NODE_IDX["NormRadius"]]
                curv = x[wall, NODE_IDX["Curvature"]]
                gc = data.global_cond.view(-1).numpy() if hasattr(data, "global_cond") else np.zeros(6)
                from .run_v3_f0_decision import _collect_profiles_2d

                prof = _collect_profiles_2d(
                    [case_rel], norm_stats, grid_s=grid_s, n_sectors=n_sectors,
                    max_graphs_per_case=1,
                )
                if prof.shape[0] < 1:
                    continue
                profiles.append(prof[0])
                meta.append(
                    np.array(
                        [
                            float(np.mean(absc)),
                            float(np.std(absc)),
                            float(np.mean(nr)),
                            float(np.mean(curv)),
                            float(gc[1]),
                        ],
                        dtype=np.float64,
                    )
                )
        if not profiles:
            return np.empty((0, grid_s * n_sectors)), np.empty((0, 5))
        return np.stack(profiles), np.stack(meta)

    train_p, train_m = _profiles_and_meta(split.train_cases)
    test_p, test_m = _profiles_and_meta(split.test_cases)
    if train_p.shape[0] < 5 or test_p.shape[0] < 2:
        return {"error": "insufficient profiles for coefficient probe"}

    mean = train_p.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(train_p - mean, full_matrices=False)
    k = min(n_modes, vt.shape[0])
    basis = vt[:k]
    train_coef = (train_p - mean) @ basis.T
    test_coef = (test_p - mean) @ basis.T

    coef_r2: List[float] = []
    recon_r2: List[float] = []
    for j in range(k):
        reg = HistGradientBoostingRegressor(max_depth=4, max_iter=120, random_state=0)
        reg.fit(train_m, train_coef[:, j])
        pred_c = reg.predict(test_m)
        coef_r2.append(_r2_score(test_coef[:, j], pred_c))
    recon = (test_coef @ basis) + mean
    for i in range(test_p.shape[0]):
        recon_r2.append(_r2_score(test_p[i], recon[i]))

    return {
        "n_modes": k,
        "coef_r2_mean": _safe_json_float(float(np.mean(coef_r2))),
        "coef_r2_per_mode": [_safe_json_float(float(x)) for x in coef_r2],
        "recon_field_r2_mean": _safe_json_float(float(np.mean(recon_r2))),
        "gate_coef_predictable": bool(float(np.mean(coef_r2)) > 0.05),
        "gate_recon_from_input": bool(float(np.mean(recon_r2)) > 0.15),
    }


def _theta_roll_pod_audit(
    split_path: Path,
    norm_stats: Mapping[str, Dict[str, float]],
    *,
    grid_s: int,
    n_sectors: int,
    n_modes: int,
    roll_shifts: Sequence[int],
    max_graphs_per_case: int,
) -> Dict[str, Any]:
    from .run_v3_f0_decision import _collect_profiles_2d

    split = SplitSpec.from_json(split_path)
    cases = list(split.train_cases)[:12]
    base = _collect_profiles_2d(
        cases, norm_stats, grid_s=grid_s, n_sectors=n_sectors,
        max_graphs_per_case=max_graphs_per_case,
    )
    if base.shape[0] < 3:
        return {"error": "insufficient profiles for theta audit"}

    mean = base.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(base - mean, full_matrices=False)
    k = min(n_modes, vt.shape[0])
    basis = vt[:k]
    base_coef = (base - mean) @ basis.T

    shifts_out: List[Dict[str, Any]] = []
    for shift in roll_shifts:
        rolled = np.roll(base.reshape(base.shape[0], grid_s, n_sectors), shift, axis=2)
        rolled_flat = rolled.reshape(base.shape[0], -1)
        rolled_coef = (rolled_flat - mean) @ basis.T
        delta = float(np.mean(np.abs(rolled_coef - base_coef)))
        shifts_out.append({"theta_roll_sectors": int(shift), "mean_abs_coef_delta": delta})

    max_delta = max(s["mean_abs_coef_delta"] for s in shifts_out)
    return {
        "roll_shifts": shifts_out,
        "max_mean_abs_coef_delta": max_delta,
        "gate_roll_breaks_pod": bool(max_delta > 0.05),
        "i1b_roll_compatible_with_i3": bool(max_delta <= 0.05),
        "note": "I14 FFT-Poisson 对 θ 平移等变；roll 破坏 POD 对齐 → I3 与 I1-b 互斥",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P I-oracle bundle (§15.3)")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training/splits/split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new/AG")
    ap.add_argument("--norm-params", type=Path, default=DEFAULT_NORM_PARAMS)
    ap.add_argument("--grid-s", type=int, default=64)
    ap.add_argument("--n-sectors", type=int, default=4)
    ap.add_argument("--n-modes", type=int, default=20)
    ap.add_argument("--max-graphs-per-case", type=int, default=3)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    split = SplitSpec.from_json(args.split.resolve())
    data_root = args.data_root.resolve()
    stats = load_norm_stats(args.norm_params)
    active_train = filter_case_names(list(split.train_cases), data_root)
    active_test = filter_case_names(list(split.test_cases), data_root)

    train_w = _collect_wall_intrinsic(
        active_train, data_root, stats, max_graphs_per_case=args.max_graphs_per_case
    )
    test_w = _collect_wall_intrinsic(
        active_test, data_root, stats, max_graphs_per_case=args.max_graphs_per_case
    )

    report: Dict[str, Any] = {
        "label": "V3P-I-oracle-bundle",
        "date": date.today().isoformat(),
        "motivation": "G4-a/b No-Go 后 Q0：§15.3 oracle + I1 θ 审计",
        "split": str(args.split.resolve()),
        "n_train_wall_points": int(train_w["tau_s"].size),
        "n_test_wall_points": int(test_w["tau_s"].size),
        "product_1_frame_projection": _frame_projection_stats(train_w, test_w),
        "product_2_energy_ratio": {
            "train": _energy_ratio_stats(train_w),
            "test": _energy_ratio_stats(test_w),
        },
        "product_3_pod_predictability": _pod_coeff_predictability(
            args.split.resolve(),
            stats,
            grid_s=args.grid_s,
            n_sectors=args.n_sectors,
            n_modes=args.n_modes,
            max_graphs_per_case=args.max_graphs_per_case,
        ),
        "product_4_theta_roll_audit": _theta_roll_pod_audit(
            args.split.resolve(),
            stats,
            grid_s=args.grid_s,
            n_sectors=args.n_sectors,
            n_modes=args.n_modes,
            roll_shifts=(1, 2),
            max_graphs_per_case=args.max_graphs_per_case,
        ),
        "pod_2d_oracle_ref": oracle_pod_2d(
            args.split.resolve(),
            stats,
            grid_s=args.grid_s,
            sectors=[args.n_sectors],
            modes=[args.n_modes],
            max_graphs_per_case=args.max_graphs_per_case,
        ),
    }

    p1 = report["product_1_frame_projection"]
    p3 = report["product_3_pod_predictability"]
    p4 = report["product_4_theta_roll_audit"]
    report["recommendation"] = {
        "i2_pc_candidate": bool(p1.get("gate_tau_theta_var_gt_wss_y")),
        "i20_candidate": bool(report["product_2_energy_ratio"]["test"].get("gate_secondary_flow_non_trivial")),
        "i3_vs_i14": (
            "prefer_I14_if_roll_breaks_pod"
            if p4.get("gate_roll_breaks_pod")
            else "undecided"
        ),
        "next_gpu": (
            "I2-PC point cloud probe (1 seed)"
            if p1.get("gate_tau_theta_var_gt_wss_y")
            else "I6/I7 guided fix on FieldPointNeXt"
        ),
    }

    out = args.output or (
        REPO_ROOT / "outputs/field/f0_decision" / f"v3p_i_oracle_bundle_{date.today().strftime('%Y%m%d')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["recommendation"], indent=2, ensure_ascii=False))
    print(out)


if __name__ == "__main__":
    main()
