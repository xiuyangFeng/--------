#!/usr/bin/env python3
"""V3P M-E · E-B 近壁算子 oracle（0 重训 · CPU）。

命题：GT 近壁切向速度 u_t(δ) 在固定法向偏移 δ 处，能否经 WSS≈μ·u_t(δ)/δ
预测 GT WSS？对照工程方案 §5 E-B；规避 5276 direct vel_diff 失败模式。

产出 ``outputs/field/f0_decision/v3p_me_eb_nearwall_oracle_<date>.json``。

用法::

    python -m training.scripts.run_v3_me_eb_nearwall_oracle
    python -m training.scripts.run_v3_me_eb_nearwall_oracle --deltas-mm 0.5 1.0 1.5 2.0 --max-cases 6
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..core.denylist import filter_case_names
from ..core.splits import SplitSpec
from pipeline.config import NODE_FEATURE_NAMES, SAMPLING_CONFIG
from training.analysis.hemo import BLOOD_VISCOSITY
from .run_v3_f0_decision import (
    DEFAULT_NORM_PARAMS,
    NODE_IDX as F0_NODE_IDX,
    REPO_ROOT,
    Y_VEL,
    _denorm_velocity,
    _denorm_wss_mag,
    _estimate_normals_at,
    _load_graph,
    _load_norm_stats,
    _r2_score,
    _safe_pearson,
    _safe_spearman,
)

GO_PROXY_R2_MIN = 0.65
DEFAULT_DELTAS_MM = (0.5, 1.0, 1.5, 2.0)
BOUNDARY_THRESHOLD_MM = float(SAMPLING_CONFIG.get("boundary_threshold", 2.0))

NODE_IDX = {n: i for i, n in enumerate(NODE_FEATURE_NAMES)}
DIST_WALL_IDX = NODE_IDX["dist_to_wall"]


def _patient_family(case_rel: str) -> Optional[str]:
    name = case_rel.split("/")[-1]
    m = re.match(r"^(GUO|ZHANG|CHEN)", name, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _ut_at_delta_for_graph(
    data,
    stats: Mapping[str, Dict[str, float]],
    *,
    delta_mm: float,
    max_wall_per_graph: int,
    k_interior: int,
    cone_ratio: float,
    mu: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    x = data.x.numpy()
    y = data.y.numpy()
    y_wss = data.y_wss.numpy()
    wall = x[:, F0_NODE_IDX["is_wall"]] > 0.5
    interior = ~wall
    n_wall = int(np.sum(wall))
    n_int = int(np.sum(interior))
    if n_wall < 20 or n_int < 20:
        return None

    wall_xyz = x[wall, :3].astype(np.float64)
    int_xyz = x[interior, :3].astype(np.float64)
    int_vel = _denorm_velocity(y[interior, Y_VEL], stats)
    int_dtw = x[interior, DIST_WALL_IDX].astype(np.float64)
    wss_mag = _denorm_wss_mag(y_wss[wall, 0], stats)

    step = max(1, n_wall // max_wall_per_graph)
    q_idx = np.arange(0, n_wall, step)
    normals = _estimate_normals_at(wall_xyz, q_idx, k=16)

    nn_int = NearestNeighbors(n_neighbors=min(k_interior, n_int)).fit(int_xyz)
    _, idx_q = nn_int.kneighbors(wall_xyz[q_idx])

    proxies: List[float] = []
    gts: List[float] = []

    for r, wi in enumerate(q_idx):
        p = wall_xyz[wi]
        nb_idx = idx_q[r]
        offs = int_xyz[nb_idx] - p
        nhat = normals[r]
        nd_norm = offs @ nhat
        if np.mean(nd_norm) < 0:
            nhat = -nhat
            nd_norm = -nd_norm
        td = np.linalg.norm(offs - np.outer(nd_norm, nhat), axis=1)

        cone = (nd_norm > 1e-9) & (td <= cone_ratio * np.maximum(nd_norm, 1e-9))
        if not np.any(cone):
            continue
        dtw_c = int_dtw[nb_idx[cone]]
        # dist_to_wall 特征为物理 mm（见 pipeline/extract_features G04）
        jb = int(np.argmin(np.abs(dtw_c - delta_mm)))
        delta_mm_used = float(dtw_c[jb])
        if delta_mm_used <= 1e-6 or delta_mm_used > BOUNDARY_THRESHOLD_MM + 1e-6:
            continue
        vel_b = int_vel[nb_idx[cone][jb]]
        ut = vel_b - (vel_b @ nhat) * nhat
        ut_mag = float(np.linalg.norm(ut))
        proxy = mu * ut_mag / max(delta_mm_used * 1e-3, 1e-9)
        proxies.append(proxy)
        gts.append(float(wss_mag[wi]))

    if len(gts) < 10:
        return None
    return np.asarray(gts, dtype=np.float64), np.asarray(proxies, dtype=np.float64)


def _metrics(gt: np.ndarray, proxy: np.ndarray) -> Dict[str, Optional[float]]:
    mask = np.isfinite(gt) & np.isfinite(proxy) & (gt >= 0) & (proxy >= 0)
    if int(np.sum(mask)) < 10:
        return {"n": int(np.sum(mask)), "r2": None, "spearman": None, "pearson": None}
    g = gt[mask]
    p = proxy[mask]
    return {
        "n": int(g.size),
        "r2": round(_r2_score(g, p), 4),
        "spearman": round(_safe_spearman(g, p), 4) if _safe_spearman(g, p) is not None else None,
        "pearson": round(_safe_pearson(g, p), 4) if _safe_pearson(g, p) is not None else None,
    }


def _collect_samples(
    cases: Sequence[str],
    data_root: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    delta_mm: float,
    max_graphs_per_case: int,
    max_wall_per_graph: int,
    k_interior: int,
    cone_ratio: float,
    mu: float,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    all_gt: List[np.ndarray] = []
    all_proxy: List[np.ndarray] = []
    case_rows: List[Dict[str, Any]] = []

    for case_rel in cases:
        case_dir = data_root / case_rel / "processed" / "graphs"
        if not case_dir.is_dir():
            continue
        case_gt: List[np.ndarray] = []
        case_proxy: List[np.ndarray] = []
        n_graphs = 0
        for gp in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
            data = _load_graph(gp)
            out = _ut_at_delta_for_graph(
                data,
                stats,
                delta_mm=delta_mm,
                max_wall_per_graph=max_wall_per_graph,
                k_interior=k_interior,
                cone_ratio=cone_ratio,
                mu=mu,
            )
            if out is None:
                continue
            gt, proxy = out
            case_gt.append(gt)
            case_proxy.append(proxy)
            all_gt.append(gt)
            all_proxy.append(proxy)
            n_graphs += 1
        if not case_gt:
            continue
        cg = np.concatenate(case_gt)
        cp = np.concatenate(case_proxy)
        m = _metrics(cg, cp)
        case_rows.append({
            "case": case_rel,
            "family": _patient_family(case_rel),
            "n_graphs": n_graphs,
            **m,
        })
        if verbose:
            print(f"  δ={delta_mm}mm {case_rel}: R²={m.get('r2')} n={m.get('n')}", flush=True)

    if not all_gt:
        return np.array([]), np.array([]), case_rows
    return np.concatenate(all_gt), np.concatenate(all_proxy), case_rows


def _family_breakdown(case_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    families = ("GUO", "ZHANG", "CHEN")
    out: Dict[str, Any] = {}
    for fam in families:
        rows = [r for r in case_rows if r.get("family") == fam and r.get("r2") is not None]
        if not rows:
            out[fam] = {"n_cases": 0, "mean_r2": None}
            continue
        out[fam] = {
            "n_cases": len(rows),
            "mean_r2": round(float(np.mean([float(r["r2"]) for r in rows])), 4),
            "cases": [r["case"] for r in rows],
        }
    return out


def _delta_verdict(r2: Optional[float]) -> str:
    if r2 is None or not math.isfinite(float(r2)):
        return "insufficient_data"
    return "go" if float(r2) >= GO_PROXY_R2_MIN else "no_go"


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P M-E E-B 近壁算子 oracle")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training" / "splits" / "split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new" / "AG")
    ap.add_argument("--norm-params", type=Path, default=DEFAULT_NORM_PARAMS)
    ap.add_argument("--deltas-mm", type=float, nargs="+", default=list(DEFAULT_DELTAS_MM))
    ap.add_argument("--mu", type=float, default=BLOOD_VISCOSITY, help="动力粘度 Pa·s")
    ap.add_argument("--max-graphs-per-case", type=int, default=3)
    ap.add_argument("--max-wall-per-graph", type=int, default=1200)
    ap.add_argument("--max-cases", type=int, default=0, help="0=train+test 全例")
    ap.add_argument("--k-interior", type=int, default=40)
    ap.add_argument("--cone-ratio", type=float, default=1.0)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    split = SplitSpec.from_json(args.split.resolve())
    data_root = args.data_root.resolve()
    stats = _load_norm_stats(args.norm_params.resolve())

    train_cases = filter_case_names(split.train_cases, data_root)
    test_cases = filter_case_names(split.test_cases, data_root)
    cases = list(train_cases) + list(test_cases)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    print(
        f"E-B near-wall oracle: {len(cases)} cases · δ={args.deltas_mm} mm · μ={args.mu}",
        flush=True,
    )

    delta_results: Dict[str, Any] = {}
    best_delta: Optional[float] = None
    best_r2 = -math.inf

    t0 = time.perf_counter()
    for delta_mm in args.deltas_mm:
        print(f"δ={delta_mm} mm ...", flush=True)
        gt_tr, pr_tr, case_tr = _collect_samples(
            [c for c in cases if c in split.train_cases],
            data_root,
            stats,
            delta_mm=delta_mm,
            max_graphs_per_case=args.max_graphs_per_case,
            max_wall_per_graph=args.max_wall_per_graph,
            k_interior=args.k_interior,
            cone_ratio=args.cone_ratio,
            mu=args.mu,
            verbose=args.verbose,
        )
        gt_te, pr_te, case_te = _collect_samples(
            [c for c in cases if c in split.test_cases],
            data_root,
            stats,
            delta_mm=delta_mm,
            max_graphs_per_case=args.max_graphs_per_case,
            max_wall_per_graph=args.max_wall_per_graph,
            k_interior=args.k_interior,
            cone_ratio=args.cone_ratio,
            mu=args.mu,
            verbose=args.verbose,
        )
        m_train = _metrics(gt_tr, pr_tr)
        m_test = _metrics(gt_te, pr_te)
        m_pooled = _metrics(
            np.concatenate([gt_tr, gt_te]) if gt_tr.size and gt_te.size else (gt_tr if gt_tr.size else gt_te),
            np.concatenate([pr_tr, pr_te]) if pr_tr.size and pr_te.size else (pr_tr if pr_tr.size else pr_te),
        )
        pooled_r2 = m_pooled.get("r2")
        verdict = _delta_verdict(pooled_r2 if isinstance(pooled_r2, (int, float)) else None)
        if pooled_r2 is not None and math.isfinite(float(pooled_r2)) and float(pooled_r2) > best_r2:
            best_r2 = float(pooled_r2)
            best_delta = delta_mm

        delta_results[f"delta_{delta_mm:g}mm"] = {
            "delta_mm": delta_mm,
            "formula": "WSS_proxy = mu * u_t / delta_normal",
            "mu_pa_s": args.mu,
            "train": m_train,
            "test": m_test,
            "pooled": m_pooled,
            "verdict": verdict,
            "by_family_test": _family_breakdown(case_te),
            "cases_train": case_tr,
            "cases_test": case_te,
        }

    overall_go = best_r2 >= GO_PROXY_R2_MIN
    report: Dict[str, Any] = {
        "label": "V3P-ME-E-B-nearwall-oracle",
        "date": date.today().isoformat(),
        "split": str(args.split.resolve()),
        "data_root": str(data_root),
        "context": "V3P post5463 band · split_AG_v1 · GT-only · avoid 5276 vel_diff mode",
        "reference": {
            "engineering_plan": "V3P_工程化精度优化方案 §5 E-B",
            "o4_gradp_signal": "outputs/field/f0_decision/v3p_me_oracle_wave (|∇p| GBDT ~0.698)",
            "failed_mode": "5276 direct vel_diff (tang_normal z-score mismatch)",
        },
        "go_thresholds": {
            "proxy_r2_min": GO_PROXY_R2_MIN,
            "note": "pooled GT WSS vs μ·u_t(δ)/δ at best δ",
        },
        "best_delta_mm": best_delta,
        "best_pooled_r2": round(best_r2, 4) if math.isfinite(best_r2) else None,
        "phase0_gate": overall_go,
        "verdict": "go" if overall_go else "no_go",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "deltas": delta_results,
        "interpretation": (
            f"E-B {'PASS' if overall_go else 'REVIEW'}: best δ={best_delta} mm "
            f"pooled R²={best_r2:.4f} (Go ≥{GO_PROXY_R2_MIN}). "
            "若 No-Go → 优先 E-J |∇p| 桥接而非 direct vel_diff 重训。"
        ),
    }

    out = args.output or (
        REPO_ROOT / "outputs" / "field" / "f0_decision" / f"v3p_me_eb_nearwall_oracle_{date.today():%Y%m%d}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    # 精简 cases 列表写入（保留 family breakdown）
    slim = dict(report)
    for k, v in report["deltas"].items():
        slim["deltas"][k] = {kk: vv for kk, vv in v.items() if kk not in ("cases_train", "cases_test")}
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(slim, indent=2, ensure_ascii=False))
    print(out, flush=True)


if __name__ == "__main__":
    main()
