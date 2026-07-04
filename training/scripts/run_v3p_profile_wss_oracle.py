#!/usr/bin/env python3
"""V3P 剖面基 WSS Gate-0 oracle（前沿方向 §3.2 · 0 重训 · CPU）。

命题：不做逐点有限差分（历史 VelGrad/vel_diff 噪声放大 No-Go），而是对每个壁面点
沿估计内法向取近壁体点 (eta_i, u_t_i)，最小二乘拟合光滑剖面
``u_t(eta) = a1*eta + a2*eta^2``，解析求导得 ``WSS = mu * a1``。

判据（写入 f0_decision json）：
  - Go：cross 分量 calibrated R² > 0.10 且 profile 明显优于同点 finite-diff 基线；
  - weak：cross calibrated R² > 0.05；
  - 否则 No-Go（§3 可微 WSS head 不立项）。

与 Idea B（stream/cross oracle No-Go）的差异：
  1. 剖面拟合用多个近壁点 + 光滑基（vs 单点差分）；
  2. cross 方向用逐点估计法向 n̂ 构造局部 frame（vs case 级平均 normal）。

用法::

    python -m training.scripts.run_v3p_profile_wss_oracle --verbose
    python -m training.scripts.run_v3p_profile_wss_oracle --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..core.splits import SplitSpec
from ..core.utils import ensure_dir
from ._figure_utils import save_json
from .run_v3_f0_decision import (
    DEFAULT_NORM_PARAMS,
    NODE_IDX,
    REPO_ROOT,
    X_TAN,
    Y_VEL,
    _denorm_velocity,
    _denorm_zscore,
    _load_graph,
    _load_norm_stats,
    _r2_score,
    _safe_spearman,
)

MU_BLOOD = 3.5e-3  # Pa·s（与 platform oracle bundle ray proxy 同口径）
MM_TO_M = 1e-3
WSS_X, WSS_Y, WSS_Z = 1, 2, 3


def _safe_json(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, (np.floating, np.integer)):
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    return x


def _unit_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _fit_profile_a1(eta: np.ndarray, u_t: np.ndarray) -> Optional[float]:
    """u_t(eta) = a1*eta + a2*eta^2 最小二乘；返回 a1（du_t/deta|_0）。"""
    if eta.size < 3 or float(np.ptp(eta)) < 1e-9:
        return None
    A = np.stack([eta, eta**2], axis=1)
    try:
        coef, *_ = np.linalg.lstsq(A, u_t, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return float(coef[0])


def _graph_estimates(
    data,
    stats: Mapping[str, Dict[str, float]],
    *,
    k_neighbors: int,
    max_wall_points: int,
    rng: np.random.Generator,
) -> Optional[Dict[str, np.ndarray]]:
    """对单图返回逐壁面点的 GT 分量与 profile / finite-diff 估计。"""
    from scipy.spatial import cKDTree

    x = data.x.numpy()
    wall = x[:, NODE_IDX["is_wall"]] > 0.5
    interior = ~wall
    n_wall, n_int = int(wall.sum()), int(interior.sum())
    if n_wall < 50 or n_int < 50:
        return None

    pos_w = x[wall, :3].astype(np.float64)
    pos_i = x[interior, :3].astype(np.float64)
    tan_w = _unit_rows(x[wall, X_TAN].astype(np.float64))
    vel_i = _denorm_velocity(data.y.numpy()[interior, Y_VEL], stats)

    y_wss = data.y_wss.numpy()
    wss_vec = np.stack(
        [
            _denorm_zscore(y_wss[wall, i].astype(np.float64), stats.get(n))
            for i, n in zip((WSS_X, WSS_Y, WSS_Z), ("wss_x", "wss_y", "wss_z"))
        ],
        axis=1,
    )

    idx_all = np.arange(n_wall)
    if n_wall > max_wall_points:
        idx_all = rng.choice(n_wall, max_wall_points, replace=False)

    tree = cKDTree(pos_i)
    dists, nbrs = tree.query(pos_w[idx_all], k=k_neighbors)

    out: Dict[str, List[float]] = {
        k: []
        for k in (
            "gt_stream", "gt_cross", "gt_mag",
            "prof_stream", "prof_cross", "prof_mag",
            "fd_stream", "fd_cross", "fd_mag",
        )
    }

    for row, wi in enumerate(idx_all):
        nb = np.atleast_1d(nbrs[row])
        rel = pos_i[nb] - pos_w[wi]                      # (k,3) mm
        # 内法向估计：指向近壁体点的平均方向
        n_hat = rel.mean(axis=0)
        n_norm = float(np.linalg.norm(n_hat))
        if n_norm < 1e-9:
            continue
        n_hat = n_hat / n_norm
        # 局部 frame：stream = 切向去法向分量；cross = n × stream
        t_s = tan_w[wi] - float(tan_w[wi] @ n_hat) * n_hat
        t_norm = float(np.linalg.norm(t_s))
        if t_norm < 1e-6:
            continue
        t_s = t_s / t_norm
        t_c = np.cross(n_hat, t_s)

        eta = (rel @ n_hat) * MM_TO_M                    # m，投影距离
        valid = eta > 1e-7
        if int(valid.sum()) < 3:
            continue
        eta_v = eta[valid]
        v_nb = vel_i[nb][valid]
        v_t = v_nb - np.outer(v_nb @ n_hat, n_hat)       # 切向速度矢量
        u_s = v_t @ t_s
        u_c = v_t @ t_c

        a1_s = _fit_profile_a1(eta_v, u_s)
        a1_c = _fit_profile_a1(eta_v, u_c)
        if a1_s is None or a1_c is None:
            continue

        j0 = int(np.argmin(eta_v))                       # finite-diff 基线：最近单点差分
        fd_s = u_s[j0] / eta_v[j0]
        fd_c = u_c[j0] / eta_v[j0]

        out["gt_stream"].append(float(wss_vec[wi] @ t_s))
        out["gt_cross"].append(float(wss_vec[wi] @ t_c))
        out["gt_mag"].append(float(np.linalg.norm(wss_vec[wi])))
        out["prof_stream"].append(MU_BLOOD * a1_s)
        out["prof_cross"].append(MU_BLOOD * a1_c)
        out["prof_mag"].append(MU_BLOOD * math.hypot(a1_s, a1_c))
        out["fd_stream"].append(MU_BLOOD * fd_s)
        out["fd_cross"].append(MU_BLOOD * fd_c)
        out["fd_mag"].append(MU_BLOOD * math.hypot(fd_s, fd_c))

    if len(out["gt_stream"]) < 20:
        return None
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def _calibrated_r2(gt: np.ndarray, est: np.ndarray, calib_frac: float, rng: np.random.Generator) -> Tuple[float, float, float]:
    """返回 (raw_r2, calibrated_r2, spearman)。calibration=单标量线性 α·est+β（吸收单位/常数偏差）。"""
    raw = _r2_score(gt, est)
    sp = _safe_spearman(gt, est)
    n = gt.size
    perm = rng.permutation(n)
    cut = max(10, int(n * calib_frac))
    tr, te = perm[:cut], perm[cut:]
    if te.size < 10 or float(np.std(est[tr])) < 1e-15:
        return raw, raw, sp
    A = np.stack([est[tr], np.ones_like(est[tr])], axis=1)
    coef, *_ = np.linalg.lstsq(A, gt[tr], rcond=None)
    pred = coef[0] * est[te] + coef[1]
    return raw, _r2_score(gt[te], pred), sp


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P profile-basis WSS Gate-0 oracle")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training/splits/split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new/AG")
    ap.add_argument("--graphs-subdir", default="graphs")
    ap.add_argument("--norm-params", type=Path, default=DEFAULT_NORM_PARAMS)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/field/f0_decision")
    ap.add_argument("--date-suffix", default="")
    ap.add_argument("--subset", choices=("test", "train", "both"), default="both")
    ap.add_argument("--max-graphs-per-case", type=int, default=3)
    ap.add_argument("--max-wall-points", type=int, default=800, help="每图最多评估壁面点数")
    ap.add_argument("--k-neighbors", type=int, default=8)
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.max_graphs_per_case = 1
        args.max_wall_points = 200

    t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    split = SplitSpec.from_json(args.split.resolve())
    data_root = args.data_root.resolve()
    stats = _load_norm_stats(args.norm_params.resolve())
    suffix = args.date_suffix or date.today().strftime("%Y%m%d")

    cases: List[str] = []
    if args.subset in ("train", "both"):
        cases += list(split.train_cases)
    if args.subset in ("test", "both"):
        cases += list(split.test_cases)
    if args.smoke:
        cases = cases[:6]

    pooled: Dict[str, List[np.ndarray]] = {}
    per_case: Dict[str, Dict[str, Any]] = {}
    n_done = n_skip = 0

    for ci, case_rel in enumerate(cases, 1):
        case_dir = data_root / case_rel / "processed" / args.graphs_subdir
        if not case_dir.is_dir():
            print(f"[profile-oracle] [{ci}/{len(cases)}] SKIP no-graphs {case_rel}", flush=True)
            n_skip += 1
            continue
        tc = time.perf_counter()
        case_chunks: Dict[str, List[np.ndarray]] = {}
        n_graphs = 0
        for gp in sorted(case_dir.glob("*.pt"))[: args.max_graphs_per_case]:
            est = _graph_estimates(
                _load_graph(gp), stats,
                k_neighbors=args.k_neighbors,
                max_wall_points=args.max_wall_points,
                rng=rng,
            )
            if est is None:
                continue
            n_graphs += 1
            for k, v in est.items():
                case_chunks.setdefault(k, []).append(v)
        if not case_chunks:
            print(f"[profile-oracle] [{ci}/{len(cases)}] SKIP empty {case_rel}", flush=True)
            n_skip += 1
            continue
        case_arr = {k: np.concatenate(v) for k, v in case_chunks.items()}
        for k, v in case_arr.items():
            pooled.setdefault(k, []).append(v)
        cs_raw, cs_cal, cs_sp = _calibrated_r2(case_arr["gt_stream"], case_arr["prof_stream"], args.calib_frac, rng)
        cc_raw, cc_cal, cc_sp = _calibrated_r2(case_arr["gt_cross"], case_arr["prof_cross"], args.calib_frac, rng)
        per_case[case_rel] = {
            "n_points": int(case_arr["gt_stream"].size),
            "n_graphs": n_graphs,
            "stream_calibrated_r2": _safe_json(cs_cal),
            "cross_calibrated_r2": _safe_json(cc_cal),
            "cross_spearman": _safe_json(cc_sp),
        }
        n_done += 1
        print(
            f"[profile-oracle] [{ci}/{len(cases)}] DONE {case_rel} graphs={n_graphs} "
            f"pts={case_arr['gt_stream'].size} cross_calR2={cc_cal:.3f} "
            f"stream_calR2={cs_cal:.3f} ({time.perf_counter()-tc:.1f}s)",
            flush=True,
        )

    if not pooled:
        raise SystemExit("no usable cases")
    arr = {k: np.concatenate(v) for k, v in pooled.items()}

    metrics: Dict[str, Any] = {}
    for comp in ("stream", "cross", "mag"):
        gt = arr[f"gt_{comp}"]
        for method in ("prof", "fd"):
            raw, cal, sp = _calibrated_r2(gt, arr[f"{method}_{comp}"], args.calib_frac, rng)
            metrics[f"{method}_{comp}"] = {
                "raw_r2": _safe_json(raw),
                "calibrated_r2": _safe_json(cal),
                "spearman": _safe_json(sp),
            }

    cross_cal = metrics["prof_cross"]["calibrated_r2"] or float("-inf")
    fd_cross_cal = metrics["fd_cross"]["calibrated_r2"] or float("-inf")
    beats_fd = cross_cal > fd_cross_cal + 0.02
    go = cross_cal > 0.10 and beats_fd
    weak = (not go) and cross_cal > 0.05
    verdict = "go" if go else ("weak_no_go" if weak else "no_go")

    report = {
        "label": "v3p_profile_wss_oracle",
        "date": suffix,
        "context": "V3P · 剖面基近壁拟合 WSS oracle（前沿方向 §3.2 Gate-0）",
        "config": {
            "split": str(args.split),
            "data_root": str(data_root),
            "subset": args.subset,
            "mu_Pa_s": MU_BLOOD,
            "k_neighbors": args.k_neighbors,
            "max_graphs_per_case": args.max_graphs_per_case,
            "max_wall_points": args.max_wall_points,
            "profile_basis": "a1*eta + a2*eta^2",
            "smoke": args.smoke,
        },
        "n_cases_done": n_done,
        "n_cases_skipped": n_skip,
        "n_points_pooled": int(arr["gt_stream"].size),
        "metrics": metrics,
        "go_thresholds": {
            "cross_calibrated_r2_min": 0.10,
            "profile_vs_fd_margin": 0.02,
            "weak_cross_min": 0.05,
        },
        "profile_beats_finite_diff": bool(beats_fd),
        "verdict": verdict,
        "gpu_mapping": "前沿方向 §3 DifferentiableWSSHead (wss_output_mode=profile)",
        "recommendation": (
            "Go → 立项 §3 可微 WSS head seed1 短训；"
            "weak → 只作观察不立项；No-Go → §3 封口，横向信息不在近壁剖面内"
        ),
        "per_case": per_case,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }

    out_path = ensure_dir(args.output_dir.resolve()) / f"v3p_profile_wss_oracle_{suffix}.json"
    save_json(out_path, report)
    print(json.dumps({
        "output": str(out_path),
        "verdict": verdict,
        "prof_cross": metrics["prof_cross"],
        "fd_cross": metrics["fd_cross"],
        "prof_stream": metrics["prof_stream"],
        "prof_mag": metrics["prof_mag"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
