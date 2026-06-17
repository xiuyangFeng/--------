"""路径 G · S0 前置 oracle 编排（V3P · 0 重训 · CPU 后处理）。

把方案 §3 的 G0-a..e + 标签 QA 落成一次性可执行分析，产出
``outputs/field/f0_decision/v3p_g0_oracle_<date>.json``（不覆盖 F0 的
``v3_f0_oracle_v2.json``），并按 §3 触发判据 / §10 Go/No-Go 标注各 G 分支放行。

子项：
  - G0-a 坐标系负担量化 (Q1/Q2)：global vs local 探针 R²（复用 probe_linear_wss）
            + 跨 case 组间方差 + 全局旋转一致性诊断。
  - QA  标签/壁面/量纲质检 (Q3)：壁面点数、法向/基底有效率、物理层 wss_rad/wss 比、
            逐点 |rad|/|WSS| p95、WSS 物理 p95 离群、tangent 单位性 → denylist 候选。
  - G0-c 压力梯度 → WSS (Q5)：复用 oracle_pressure_gradient_wss 相关 + GBDT 上限。
  - G0-d 解析基线残差：per-case 缩放的 Poiseuille 形状基线，看残差方差下降比。
  - G0-e noise floor 复核：读已有 F0 json 或重算 AsymW 三 seed 带宽。

用法（V3P 默认）::

    python -m training.scripts.run_v3_g0_oracle \
        --max-graphs-per-case 3 --max-wall-per-graph 800
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np


class _S0Timer:
    """分阶段耗时日志（--verbose 时启用）。"""

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self._t0 = time.perf_counter()
        self.phases: List[Dict[str, object]] = []

    def log(self, msg: str) -> None:
        if not self.verbose:
            return
        elapsed = time.perf_counter() - self._t0
        print(f"[S0 +{elapsed:8.2f}s] {msg}", flush=True)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        self.log(f">>> {name}")
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.phases.append({"name": name, "seconds": round(dt, 3)})
            self.log(f"<<< {name} ({dt:.2f}s)")

    def summary(self) -> None:
        if not self.verbose or not self.phases:
            return
        total = time.perf_counter() - self._t0
        self.log("=== phase summary ===")
        for p in self.phases:
            sec = float(p["seconds"])
            pct = 100.0 * sec / total if total > 0 else 0.0
            self.log(f"  {p['name']}: {sec:.2f}s ({pct:.1f}%)")
        self.log(f"TOTAL: {total:.2f}s")

from ..core.splits import SplitSpec
from ..core.utils import ensure_dir
from ._figure_utils import load_json, save_json
from .run_v3_f0_decision import (
    DEFAULT_ASYMW_GLOB,
    DEFAULT_NORM_PARAMS,
    NODE_IDX,
    REPO_ROOT,
    X_TAN,
    Y_P,
    _denorm_wss_mag,
    _denorm_zscore,
    _discover_asymw_runs,
    _load_graph,
    _load_norm_stats,
    _r2_score,
    _safe_pearson,
    _safe_spearman,
    oracle_pressure_gradient_wss,
    todo_27a_seed_bandwidth,
)
from .probe_linear_wss import (
    DEFAULT_FEATURES,
    GLOBAL_COMPONENTS,
    LOCAL_COMPONENTS,
    collect_wall_dataset,
    run_probe,
)

# ----------------------------------------------------------------------------
# G0-a 旋转一致性诊断
# ----------------------------------------------------------------------------

def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """均匀随机 SO(3) 矩阵（QR 分解法）。"""
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def rotation_consistency(
    split_path: Path,
    *,
    data_root: Path,
    graphs_subdir: str,
    norm_stats: Mapping[str, Dict[str, float]],
    max_graphs_per_case: int,
    max_wall_per_graph: int,
    n_rot: int,
    seed: int,
    max_train_cases: Optional[int] = None,
    max_test_cases: Optional[int] = None,
    verbose: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """证明 global-frame 探针对整体旋转不一致（坐标系负担）。

    在未旋转 train 上拟合 Ridge（global wss_x/y/z），再把 test 的 tangent 特征与
    global WSS 目标整体旋转 R，用同一模型评 R²。R² 大幅下降 → 全局坐标表示需要
    「按朝向重学」，正是 local/等变表示要消除的负担。
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)
        elif verbose:
            print(f"[rotation] {msg}", flush=True)

    split = SplitSpec.from_json(split_path)
    feats = list(DEFAULT_FEATURES)
    tan_cols = [feats.index(c) for c in ("Tangent_X", "Tangent_Y", "Tangent_Z")]

    _log("collect train ...")
    t0 = time.perf_counter()
    tr = collect_wall_dataset(
        split, split.train_cases, data_root=data_root, graphs_subdir=graphs_subdir,
        feature_names=feats, norm_stats=norm_stats, target_frame="global",
        max_graphs_per_case=max_graphs_per_case, max_wall_per_graph=max_wall_per_graph,
        max_cases=max_train_cases, seed=seed, verbose=verbose, log_fn=log_fn,
    )
    _log(f"collect test ... (train {tr.X.shape[0]} pts in {time.perf_counter() - t0:.2f}s)")
    te = collect_wall_dataset(
        split, split.test_cases, data_root=data_root, graphs_subdir=graphs_subdir,
        feature_names=feats, norm_stats=norm_stats, target_frame="global",
        max_graphs_per_case=max_graphs_per_case, max_wall_per_graph=max_wall_per_graph,
        max_cases=max_test_cases, seed=seed + 1, verbose=verbose, log_fn=log_fn,
    )
    if tr.X.shape[0] < 50 or te.X.shape[0] < 20:
        return {"error": "insufficient samples"}

    t_fit = time.perf_counter()
    scaler = StandardScaler().fit(tr.X)
    models = {c: Ridge(alpha=1.0).fit(scaler.transform(tr.X), tr.targets[c]) for c in GLOBAL_COMPONENTS}
    _log(f"ridge fit {time.perf_counter() - t_fit:.2f}s")

    base_r2 = {c: _r2_score(te.targets[c], models[c].predict(scaler.transform(te.X))) for c in GLOBAL_COMPONENTS}

    rng = np.random.default_rng(seed + 7)
    Y = np.stack([te.targets[c] for c in GLOBAL_COMPONENTS], axis=1)  # (n,3)
    rot_r2: Dict[str, List[float]] = {c: [] for c in GLOBAL_COMPONENTS}
    t_rot = time.perf_counter()
    for i in range(n_rot):
        R = _random_rotation(rng)
        Xr = te.X.copy()
        Xr[:, tan_cols] = te.X[:, tan_cols] @ R.T
        Yr = Y @ R.T
        Xr_s = scaler.transform(Xr)
        for j, c in enumerate(GLOBAL_COMPONENTS):
            rot_r2[c].append(_r2_score(Yr[:, j], models[c].predict(Xr_s)))
        if verbose and (i == 0 or i == n_rot - 1):
            _log(f"rotation {i + 1}/{n_rot} done")

    _log(f"n_rot={n_rot} loop {time.perf_counter() - t_rot:.2f}s")
    out = {"baseline_test_r2": base_r2, "rotated_test_r2_mean": {}, "r2_drop_under_rotation": {}}
    for c in GLOBAL_COMPONENTS:
        m = float(np.mean(rot_r2[c])) if rot_r2[c] else float("nan")
        out["rotated_test_r2_mean"][c] = m
        out["r2_drop_under_rotation"][c] = float(base_r2[c] - m)
    out["n_rot"] = n_rot
    out["interpretation"] = (
        "global 探针在整体旋转后 R² 明显下降 ⇒ 表示对坐标系敏感（坐标系负担）；"
        "local/等变表示用旋转不变特征，理论上 R² 不随旋转变化"
    )
    return out


# ----------------------------------------------------------------------------
# 标签 QA (Q3)
# ----------------------------------------------------------------------------

# 反归一化后 WSS p95 低于此值时视为退化标签，不再计算 rad 比（避免除零爆炸）。
_WSS_P95_PHYS_MIN = 1e-3


def _percentiles(arr: np.ndarray, ps=(50, 95, 99)) -> Dict[str, float]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{p}": float(v) for p, v in zip(ps, np.percentile(arr, ps))}


def label_qa(
    split_path: Path,
    *,
    data_root: Path,
    graphs_subdir: str,
    norm_stats: Mapping[str, Dict[str, float]],
    max_graphs_per_case: int,
    thresholds: Mapping[str, float],
    max_cases: Optional[int] = None,
    verbose: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """逐 case 标签 / 壁面 / 量纲质检；产出 denylist 候选。"""
    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)
        elif verbose:
            print(f"[qa] {msg}", flush=True)

    split = SplitSpec.from_json(split_path)
    all_cases = list(split.train_cases) + list(split.val_cases) + list(split.test_cases)
    if max_cases is not None:
        all_cases = all_cases[:max_cases]
    rows: List[Dict[str, object]] = []
    _log(f"QA start: {len(all_cases)} cases, max_graphs={max_graphs_per_case}")

    for case_rel in all_cases:
        case_dir = data_root / case_rel / "processed" / graphs_subdir
        if not case_dir.is_dir():
            _log(f"  skip (no dir): {case_rel}")
            continue
        t_glob = time.perf_counter()
        graph_paths = sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]
        if not graph_paths:
            _log(f"  skip (no graphs, glob {time.perf_counter() - t_glob:.2f}s): {case_rel}")
            continue
        n_wall_list, nv_inv, bv_inv, rad_ratio_list, rad_frac_p95_list = [], [], [], [], []
        tan_unit_frac, wss_p95_list, wss_max_list = [], [], []
        for gp in graph_paths:
            t_load = time.perf_counter()
            data = _load_graph(gp)
            dt_load = time.perf_counter() - t_load
            _log(f"  {case_rel}/{gp.name}: load {dt_load:.2f}s")
            x = data.x.numpy()
            wall = x[:, NODE_IDX["is_wall"]] > 0.5
            n_wall = int(np.sum(wall))
            if n_wall < 20:
                continue
            n_wall_list.append(n_wall)
            tan = x[wall, X_TAN].astype(np.float64)
            tnorm = np.linalg.norm(tan, axis=1)
            tan_unit_frac.append(float(np.mean(np.abs(tnorm - 1.0) < 0.05)))

            y_wss = data.y_wss.numpy()
            wss_phys = _denorm_wss_mag(y_wss[wall, 0], norm_stats)
            wss_p95_list.append(float(np.nanpercentile(wss_phys, 95)))
            wss_max_list.append(float(np.nanmax(wss_phys)))

            yl = getattr(data, "y_wss_local", None)
            mask = getattr(data, "wss_local_mask", None)
            if yl is not None and mask is not None:
                m = mask.numpy()
                nv = m[wall, 0]
                bv = m[wall, 1]
                nv_inv.append(float(1.0 - np.mean(nv)))
                bv_inv.append(float(1.0 - np.mean(bv)))
                # 物理层口径（与 pipeline/local_wss.compute_qa_stats 一致）：
                # 反归一化 wss_rad / wss，不用 z-score 轴向分量做分母。
                yl_arr = yl.numpy()
                rad_phys = np.abs(
                    _denorm_zscore(yl_arr[wall, 3].astype(np.float64), norm_stats.get("wss_rad"))
                )
                wss_phys = _denorm_wss_mag(y_wss[wall, 0], norm_stats)
                wss_p95 = float(np.nanpercentile(wss_phys, 95))
                if wss_p95 < _WSS_P95_PHYS_MIN:
                    rad_ratio_list.append(float("nan"))
                    rad_frac_p95_list.append(float("nan"))
                else:
                    rad_p95 = float(np.nanpercentile(rad_phys, 95))
                    rad_ratio_list.append(rad_p95 / wss_p95)
                    valid = wss_phys >= _WSS_P95_PHYS_MIN
                    if np.any(valid):
                        per_pt_frac = rad_phys[valid] / wss_phys[valid]
                        rad_frac_p95_list.append(float(np.nanpercentile(per_pt_frac, 95)))
                    else:
                        rad_frac_p95_list.append(float("nan"))

        if not n_wall_list:
            continue
        row = {
            "case": case_rel,
            "pace_group": case_rel.split("/")[0] if "/" in case_rel else "unknown",
            "n_graphs": len(n_wall_list),
            "n_wall_mean": float(np.mean(n_wall_list)),
            "tangent_unit_frac": float(np.mean(tan_unit_frac)) if tan_unit_frac else None,
            "normal_invalid_rate": float(np.mean(nv_inv)) if nv_inv else None,
            "basis_invalid_rate": float(np.mean(bv_inv)) if bv_inv else None,
            "wss_rad_p95_over_wss_p95": float(np.nanmean(rad_ratio_list)) if rad_ratio_list else None,
            "wss_rad_frac_p95": float(np.nanmean(rad_frac_p95_list)) if rad_frac_p95_list else None,
            "wss_p95_phys": float(np.mean(wss_p95_list)),
            "wss_max_phys": float(np.max(wss_max_list)),
        }
        rows.append(row)

    # 量纲离群：WSS 物理 p95 的 robust z（中位数 + MAD）
    p95_vals = np.asarray([r["wss_p95_phys"] for r in rows], dtype=np.float64)
    med = float(np.median(p95_vals)) if p95_vals.size else float("nan")
    mad = float(np.median(np.abs(p95_vals - med))) if p95_vals.size else float("nan")
    scale = 1.4826 * mad if mad > 1e-12 else (np.std(p95_vals) + 1e-12)
    for r in rows:
        r["wss_p95_robust_z"] = float((r["wss_p95_phys"] - med) / scale) if scale > 0 else 0.0

    denylist_candidates: List[Dict[str, object]] = []
    for r in rows:
        reasons = []
        if r["normal_invalid_rate"] is not None and r["normal_invalid_rate"] > thresholds["normal_invalid_rate_max"]:
            reasons.append(f"normal_invalid_rate={r['normal_invalid_rate']:.3f}")
        if r["basis_invalid_rate"] is not None and r["basis_invalid_rate"] > thresholds["basis_invalid_rate_max"]:
            reasons.append(f"basis_invalid_rate={r['basis_invalid_rate']:.3f}")
        if r["wss_rad_p95_over_wss_p95"] is not None and r["wss_rad_p95_over_wss_p95"] > thresholds["rad_ratio_max"]:
            reasons.append(f"rad_ratio={r['wss_rad_p95_over_wss_p95']:.3f}")
        rad_frac_max = thresholds.get("rad_frac_p95_max")
        if (
            rad_frac_max is not None
            and r.get("wss_rad_frac_p95") is not None
            and r["wss_rad_frac_p95"] > rad_frac_max
        ):
            reasons.append(f"rad_frac_p95={r['wss_rad_frac_p95']:.3f}")
        if abs(r["wss_p95_robust_z"]) > thresholds["wss_p95_robust_z_max"]:
            reasons.append(f"wss_p95_robust_z={r['wss_p95_robust_z']:.2f}")
        if r["tangent_unit_frac"] is not None and r["tangent_unit_frac"] < thresholds["tangent_unit_frac_min"]:
            reasons.append(f"tangent_unit_frac={r['tangent_unit_frac']:.3f}")
        if r["wss_p95_phys"] < _WSS_P95_PHYS_MIN:
            reasons.append(f"wss_p95_degenerate={r['wss_p95_phys']:.2e}")
        if reasons:
            denylist_candidates.append({"case": r["case"], "reasons": reasons})

    def _mean(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r.get(key) is not None and math.isfinite(r[key])]
        return float(np.mean(vals)) if vals else None

    return {
        "n_cases": len(rows),
        "graphs_subdir": graphs_subdir,
        "rad_ratio_metric": "physical_denorm_p95",
        "thresholds": dict(thresholds),
        "aggregate": {
            "wss_p95_phys_median": med,
            "wss_p95_phys_mad": mad,
            "mean_normal_invalid_rate": _mean("normal_invalid_rate"),
            "mean_basis_invalid_rate": _mean("basis_invalid_rate"),
            "mean_rad_ratio": _mean("wss_rad_p95_over_wss_p95"),
            "mean_rad_frac_p95": _mean("wss_rad_frac_p95"),
            "mean_tangent_unit_frac": _mean("tangent_unit_frac"),
        },
        "n_denylist_candidates": len(denylist_candidates),
        "denylist_candidates": denylist_candidates,
        "cases": rows,
        "pass": len(denylist_candidates) == 0,
        "interpretation": (
            "denylist 候选为空 ⇒ 标签/壁面/量纲 QA 通过，可放行 G1/G2 训练；"
            "若有候选，先核查或剔除再开训（可能需新 split + 新表）"
        ),
    }


# ----------------------------------------------------------------------------
# G0-c 压力梯度 + 几何 → WSS 的 GBDT 上限
# ----------------------------------------------------------------------------

def _collect_pressure_geom(
    cases: Sequence[str],
    *,
    data_root: Path,
    graphs_subdir: str,
    norm_stats: Mapping[str, Dict[str, float]],
    max_graphs_per_case: int,
    max_wall_per_graph: int,
    k_wall: int,
    seed: int,
    max_cases: Optional[int] = None,
    verbose: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
):
    """每壁面点：[|∇p|, -dp/ds, NormRadius, Curvature, dist_to_bif, dR_ds, torsion] → WSS。"""
    from sklearn.neighbors import NearestNeighbors

    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)
        elif verbose:
            print(f"[g0c-geom] {msg}", flush=True)

    rng = np.random.default_rng(seed)
    X_rows, y_rows = [], []
    case_list = list(cases) if max_cases is None else list(cases)[:max_cases]
    _log(f"collect start: {len(case_list)} cases max_wall={max_wall_per_graph}")
    for case_rel in case_list:
        case_dir = data_root / case_rel / "processed" / graphs_subdir
        if not case_dir.is_dir():
            continue
        for gp in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
            t_load = time.perf_counter()
            data = _load_graph(gp)
            dt_load = time.perf_counter() - t_load
            x = data.x.numpy()
            wall = x[:, NODE_IDX["is_wall"]] > 0.5
            if int(np.sum(wall)) < 20:
                continue
            y = data.y.numpy()
            y_wss = data.y_wss.numpy()
            wall_xyz = x[wall, :3].astype(np.float64)
            p_wall = _denorm_zscore(y[wall, Y_P], norm_stats.get("p"))
            tan = x[wall, X_TAN].astype(np.float64)
            wss_mag = _denorm_wss_mag(y_wss[wall, 0], norm_stats)
            geom = np.stack([
                x[wall, NODE_IDX["Abscissa"]], x[wall, NODE_IDX["Curvature"]],
                x[wall, 4],  # NormRadius
                x[wall, 10], x[wall, 12], x[wall, 13],  # dist_to_bif, dR_ds, torsion
            ], axis=1).astype(np.float64)

            n_wall = wall_xyz.shape[0]
            # 先对查询壁面点下采样，再做 per-point kNN 最小二乘，避免逐点全量耗时
            if n_wall > max_wall_per_graph:
                q_idx = rng.choice(n_wall, size=max_wall_per_graph, replace=False)
            else:
                q_idx = np.arange(n_wall)
            k_eff = int(min(k_wall, n_wall - 1))
            t_knn = time.perf_counter()
            nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(wall_xyz)
            _, idx_nbr = nn.kneighbors(wall_xyz[q_idx])
            dt_knn = time.perf_counter() - t_knn
            dpds = np.full(q_idx.size, np.nan)
            gmag = np.full(q_idx.size, np.nan)
            t_lstsq = time.perf_counter()
            for r, wi in enumerate(q_idx):
                nb = idx_nbr[r, 1:]
                dx = wall_xyz[nb] - wall_xyz[wi]
                dp = p_wall[nb] - p_wall[wi]
                g, *_ = np.linalg.lstsq(dx, dp, rcond=None)
                t = tan[wi]
                tn = np.linalg.norm(t)
                if tn > 1e-9:
                    dpds[r] = float(g @ (t / tn))
                gmag[r] = float(np.linalg.norm(g))
            feat = np.column_stack([gmag, -dpds, geom[q_idx]])
            wss_q = wss_mag[q_idx]
            ok = np.all(np.isfinite(feat), axis=1) & np.isfinite(wss_q)
            n_ok = int(np.sum(ok))
            dt_lstsq = time.perf_counter() - t_lstsq
            _log(
                f"  {case_rel}/{gp.name}: load {dt_load:.2f}s n_wall={n_wall} "
                f"q={q_idx.size} knn {dt_knn:.2f}s lstsq {dt_lstsq:.2f}s ok={n_ok}"
            )
            if n_ok >= 10:
                X_rows.append(feat[ok])
                y_rows.append(wss_q[ok])
    if not X_rows:
        return np.empty((0, 8)), np.empty(0)
    return np.concatenate(X_rows, axis=0), np.concatenate(y_rows, axis=0)


def g0c_pressure_geom_gbdt(
    split_path: Path,
    *,
    data_root: Path,
    graphs_subdir: str,
    norm_stats: Mapping[str, Dict[str, float]],
    max_graphs_per_case: int,
    max_wall_per_graph: int,
    seed: int,
    max_train_cases: Optional[int] = None,
    max_test_cases: Optional[int] = None,
    verbose: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)
        elif verbose:
            print(f"[g0c-gbdt] {msg}", flush=True)

    split = SplitSpec.from_json(split_path)
    _log("collect train ...")
    t0 = time.perf_counter()
    Xtr, ytr = _collect_pressure_geom(
        split.train_cases, data_root=data_root, graphs_subdir=graphs_subdir, norm_stats=norm_stats,
        max_graphs_per_case=max_graphs_per_case, max_wall_per_graph=max_wall_per_graph, k_wall=12,
        seed=seed, max_cases=max_train_cases, verbose=verbose, log_fn=log_fn,
    )
    _log(f"collect test ... (train n={Xtr.shape[0]} in {time.perf_counter() - t0:.2f}s)")
    Xte, yte = _collect_pressure_geom(
        split.test_cases, data_root=data_root, graphs_subdir=graphs_subdir, norm_stats=norm_stats,
        max_graphs_per_case=max_graphs_per_case, max_wall_per_graph=max_wall_per_graph, k_wall=12,
        seed=seed + 1, max_cases=max_test_cases, verbose=verbose, log_fn=log_fn,
    )
    if Xtr.shape[0] < 50 or Xte.shape[0] < 20:
        return {"error": "insufficient samples", "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0])}
    t_fit = time.perf_counter()
    _log(f"HistGBDT fit n_train={Xtr.shape[0]} n_test={Xte.shape[0]} ...")
    reg = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=0,
    ).fit(Xtr, ytr)
    _log(f"HistGBDT fit done {time.perf_counter() - t_fit:.2f}s")
    pred = reg.predict(Xte)
    return {
        "features": ["grad_p_mag", "neg_dp_ds", "Abscissa", "Curvature", "NormRadius", "dist_to_bif", "dR_ds", "torsion"],
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "test_r2": _r2_score(yte, pred),
        "test_spearman": _safe_spearman(yte, pred),
        "interpretation": "GT |∇p|+几何 GBDT 预测 WSS 的 test R²>0.45 ⇒ G2-b 压力交叉注意力头值得做",
    }


# ----------------------------------------------------------------------------
# G0-d 解析基线残差方差下降
# ----------------------------------------------------------------------------

def g0d_residual_baseline(
    split_path: Path,
    *,
    data_root: Path,
    graphs_subdir: str,
    norm_stats: Mapping[str, Dict[str, float]],
    max_graphs_per_case: int,
    radius_exponent: float = 3.0,
    max_cases: Optional[int] = None,
    verbose: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Poiseuille 形状解析基线 WSS0 ∝ R^-p（per-case 缩放）的残差方差下降比。

    诚实边界：per-case 缩放 c 用 GT 最小二乘拟合（无量纲化的上界口径），属 oracle；
    若此上界都不降方差，固定 WSS₀ 残差头更不会有效。
    """
    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)
        elif verbose:
            print(f"[g0d] {msg}", flush=True)

    split = SplitSpec.from_json(split_path)
    case_rows: List[Dict[str, object]] = []
    pooled_var_full, pooled_var_resid = [], []

    test_cases = split.test_cases if max_cases is None else list(split.test_cases)[:max_cases]
    _log(f"G0-d start: {len(test_cases)} test cases")
    for case_rel in test_cases:
        case_dir = data_root / case_rel / "processed" / graphs_subdir
        if not case_dir.is_dir():
            continue
        wss_all, shape_all, curv_all = [], [], []
        for gp in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
            t_load = time.perf_counter()
            data = _load_graph(gp)
            if verbose:
                _log(f"  {case_rel}/{gp.name}: load {time.perf_counter() - t_load:.2f}s")
            x = data.x.numpy()
            wall = x[:, NODE_IDX["is_wall"]] > 0.5
            if int(np.sum(wall)) < 20:
                continue
            y_wss = data.y_wss.numpy()
            wss = _denorm_wss_mag(y_wss[wall, 0], norm_stats)
            r_eff = np.clip(x[wall, 4].astype(np.float64), 0.05, 1.0)  # NormRadius
            shape = r_eff ** (-radius_exponent)
            wss_all.append(wss)
            shape_all.append(shape)
            curv_all.append(x[wall, NODE_IDX["Curvature"]].astype(np.float64))
        if not wss_all:
            continue
        wss = np.concatenate(wss_all)
        shape = np.concatenate(shape_all)
        curv = np.concatenate(curv_all)
        denom = float(np.sum(shape * shape))
        c = float(np.sum(wss * shape) / denom) if denom > 1e-20 else 0.0
        wss0 = c * shape
        resid = wss - wss0
        var_full = float(np.var(wss))
        var_resid = float(np.var(resid))
        reduction = 1.0 - var_resid / var_full if var_full > 1e-20 else 0.0
        case_rows.append({
            "case": case_rel,
            "var_reduction": reduction,
            "resid_curv_spearman": _safe_spearman(resid, curv),
            "scale_c": c,
        })
        pooled_var_full.append(var_full)
        pooled_var_resid.append(var_resid)

    reductions = [r["var_reduction"] for r in case_rows]
    pooled_reduction = (
        1.0 - float(np.sum(pooled_var_resid)) / float(np.sum(pooled_var_full))
        if pooled_var_full and float(np.sum(pooled_var_full)) > 1e-20 else None
    )
    resid_struct = [abs(r["resid_curv_spearman"]) for r in case_rows
                    if r["resid_curv_spearman"] is not None and math.isfinite(r["resid_curv_spearman"])]
    return {
        "radius_exponent": radius_exponent,
        "n_cases": len(case_rows),
        "mean_var_reduction": float(np.mean(reductions)) if reductions else None,
        "pooled_var_reduction": pooled_reduction,
        "mean_abs_resid_curv_spearman": float(np.mean(resid_struct)) if resid_struct else None,
        "cases": case_rows,
        "interpretation": (
            "per-case 缩放 Poiseuille 基线（GT 拟合上界）：方差降 >30% 且残差仍与几何相关 "
            "⇒ G2-a1 固定 WSS₀ 残差头立项；否则固定残差线关闭（可转低保真 teacher）"
        ),
    }


# ----------------------------------------------------------------------------
# G0-e noise floor 复核
# ----------------------------------------------------------------------------

def g0e_noise_floor(
    *,
    field_root: Path,
    run_glob: str,
    existing_json: Path,
) -> Dict[str, object]:
    if existing_json.is_file():
        try:
            rep = load_json(existing_json)
            rec = rep.get("recommendation", {})
            if "asymw_bandwidth_mean" in rec:
                return {
                    "source": str(existing_json),
                    "asymw_bandwidth_mean": rec.get("asymw_bandwidth_mean"),
                    "asymw_bandwidth_ci95": rec.get("asymw_bandwidth_ci95"),
                    "note": "读取已有 F0 决策层（不重算）",
                }
        except Exception:
            pass
    runs = _discover_asymw_runs(field_root.resolve(), run_glob)
    if not runs:
        return {"error": "未找到 AsymW-a run 且无现成 F0 json"}
    seed = todo_27a_seed_bandwidth(runs)
    summ = seed.get("summary", {})
    return {
        "source": "recomputed",
        "asymw_bandwidth_mean": summ.get("mean"),
        "asymw_bandwidth_ci95": [summ.get("ci95_lo"), summ.get("ci95_hi")],
        "per_seed": seed.get("per_seed"),
    }


# ----------------------------------------------------------------------------
# 汇总 + Go/No-Go 门控
# ----------------------------------------------------------------------------

def _component_r2(frame_report: Dict[str, object], model: str, comp: str) -> Optional[float]:
    try:
        return float(frame_report["models"][model]["components"][comp]["test_r2"])
    except Exception:
        return None


def build_gates(report: Dict[str, object], *, go_margin: float) -> Dict[str, object]:
    g0a = report.get("g0a", {})
    glob = g0a.get("global_frame", {})
    loc = g0a.get("local_frame", {})

    def best_r2(frame_rep, comps):
        vals = []
        for m in ("ridge", "gbdt"):
            for c in comps:
                v = _component_r2(frame_rep, m, c)
                if v is not None:
                    vals.append(v)
        return max(vals) if vals else None

    global_xy = best_r2(glob, ["wss_x", "wss_y"]) if glob and "models" in glob else None
    local_cr = best_r2(loc, ["wss_circ", "wss_rad"]) if loc and "models" in loc else None
    g0a_delta = (local_cr - global_xy) if (local_cr is not None and global_xy is not None) else None
    g0a_pass = bool(g0a_delta is not None and g0a_delta >= go_margin)

    qa = report.get("qa", {})
    qa_pass = bool(qa.get("pass", False))

    g0c = report.get("g0c", {})
    g0c_pg = report.get("g0c_pressure_corr", {})
    g0c_r2 = g0c.get("test_r2")
    g0c_pass = bool(g0c_r2 is not None and g0c_r2 > 0.45)

    g0d = report.get("g0d", {})
    g0d_red = g0d.get("mean_var_reduction")
    g0d_struct = g0d.get("mean_abs_resid_curv_spearman")
    g0d_pass = bool(g0d_red is not None and g0d_red > 0.30
                    and g0d_struct is not None and g0d_struct >= 0.1)

    return {
        "go_margin": go_margin,
        "G0a_coordinate_burden": {
            "global_xy_best_r2": global_xy,
            "local_circ_rad_best_r2": local_cr,
            "delta_local_minus_global": g0a_delta,
            "threshold": go_margin,
            "pass": g0a_pass,
            "release": "G1-a0 VNHeadPlain" if g0a_pass else "hold; 若 local 也无结构则转 G3/G4/G5",
        },
        "QA_label_clean": {
            "pass": qa_pass,
            "n_denylist_candidates": qa.get("n_denylist_candidates"),
            "release": "G1/G2 可开训" if qa_pass else "先修标签/denylist（可能需新 split）",
        },
        "G0c_pressure_to_wss": {
            "gbdt_test_r2": g0c_r2,
            "pressure_corr_abs_spearman": (g0c_pg or {}).get("abs_spearman"),
            "threshold": 0.45,
            "pass": g0c_pass,
            "release": "G2-b pressure→WSS 交叉注意力头值得做" if g0c_pass else "G2-b 降级",
        },
        "G0d_residual_baseline": {
            "mean_var_reduction": g0d_red,
            "resid_geom_struct": g0d_struct,
            "threshold_reduction": 0.30,
            "pass": g0d_pass,
            "release": "G2-a1 固定 WSS₀ 残差头立项" if g0d_pass else "固定残差线关闭（转低保真 teacher）",
        },
        "G0e_noise_floor": report.get("g0e", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="路径 G · S0 前置 oracle 编排（V3P · 0 重训）")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training" / "splits" / "split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new" / "AG")
    ap.add_argument("--graphs-subdir", default="graphs")
    ap.add_argument("--graphs-subdir-local", default="graphs_local_v1")
    ap.add_argument("--norm-params", type=Path, default=DEFAULT_NORM_PARAMS)
    ap.add_argument("--max-graphs-per-case", type=int, default=3)
    ap.add_argument("--max-wall-per-graph", type=int, default=800)
    ap.add_argument("--qa-max-graphs", type=int, default=2)
    ap.add_argument("--oracle-max-cases", type=int, default=12)
    ap.add_argument("--oracle-max-graphs", type=int, default=1)
    ap.add_argument("--n-rot", type=int, default=8)
    ap.add_argument("--max-cases", type=int, default=None,
                    help="限制每子项的 case 数（自检/限时用；None=全量）")
    ap.add_argument("--go-margin", type=float, default=0.10)
    ap.add_argument("--local-source", choices=["precomputed", "knn"], default="precomputed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--field-root", type=Path, default=REPO_ROOT / "outputs" / "field")
    ap.add_argument("--run-glob", default=DEFAULT_ASYMW_GLOB)
    ap.add_argument("--existing-f0", type=Path,
                    default=REPO_ROOT / "outputs" / "field" / "f0_decision" / "v3_f0_oracle_v2.json")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "field" / "f0_decision")
    ap.add_argument("--output-name", default=None)
    ap.add_argument("--skip", nargs="*", default=[],
                    help="跳过的子项: g0a qa g0c g0d g0e rotation")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="打印分阶段耗时日志（自检排障用）")
    args = ap.parse_args()

    timer = _S0Timer(args.verbose)
    log_fn = timer.log

    split_path = args.split.resolve()
    data_root = args.data_root.resolve()
    timer.log(f"load norm_params {args.norm_params}")
    t_norm = time.perf_counter()
    norm_stats = _load_norm_stats(args.norm_params.resolve())
    timer.log(f"norm_params loaded ({time.perf_counter() - t_norm:.2f}s)")
    skip = set(args.skip)
    timer.log(
        f"config: max_cases={args.max_cases} max_graphs={args.max_graphs_per_case} "
        f"max_wall={args.max_wall_per_graph} oracle_cases={args.oracle_max_cases} "
        f"n_rot={args.n_rot} skip={sorted(skip) or 'none'}"
    )
    timer.log(
        f"thread env: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '(unset)')} "
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', '(unset)')} "
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', '(unset)')}"
    )

    report: Dict[str, object] = {
        "label": "V3P-G0-S0",
        "created": date.today().isoformat(),
        "split": str(split_path),
        "data_root": str(data_root),
        "norm_params": str(args.norm_params.resolve()),
        "config": {
            "max_graphs_per_case": args.max_graphs_per_case,
            "max_wall_per_graph": args.max_wall_per_graph,
            "go_margin": args.go_margin,
            "local_source": args.local_source,
        },
    }

    # G0-a 坐标系负担
    if "g0a" not in skip:
        with timer.phase("G0-a global frame probe"):
            glob_rep = run_probe(
                split_path=split_path, data_root=data_root,
                graphs_subdir_global=args.graphs_subdir, graphs_subdir_local=args.graphs_subdir_local,
                feature_names=DEFAULT_FEATURES, norm_stats=norm_stats, target_frame="global",
                models=["ridge", "gbdt"], max_graphs_per_case=args.max_graphs_per_case,
                max_wall_per_graph=args.max_wall_per_graph, local_source=args.local_source,
                max_train_cases=args.max_cases, max_test_cases=args.max_cases, seed=args.seed,
                verbose=args.verbose, log_fn=log_fn,
            )
        with timer.phase("G0-a local frame probe"):
            loc_rep = run_probe(
                split_path=split_path, data_root=data_root,
                graphs_subdir_global=args.graphs_subdir, graphs_subdir_local=args.graphs_subdir_local,
                feature_names=DEFAULT_FEATURES, norm_stats=norm_stats, target_frame="local",
                models=["ridge", "gbdt"], max_graphs_per_case=args.max_graphs_per_case,
                max_wall_per_graph=args.max_wall_per_graph, local_source=args.local_source,
                max_train_cases=args.max_cases, max_test_cases=args.max_cases, seed=args.seed,
                verbose=args.verbose, log_fn=log_fn,
            )
        g0a: Dict[str, object] = {"global_frame": glob_rep, "local_frame": loc_rep}
        if "rotation" not in skip:
            with timer.phase("G0-a rotation consistency"):
                g0a["rotation_consistency"] = rotation_consistency(
                    split_path, data_root=data_root, graphs_subdir=args.graphs_subdir,
                    norm_stats=norm_stats, max_graphs_per_case=args.max_graphs_per_case,
                    max_wall_per_graph=args.max_wall_per_graph, n_rot=args.n_rot, seed=args.seed,
                    max_train_cases=args.max_cases, max_test_cases=args.max_cases,
                    verbose=args.verbose, log_fn=log_fn,
                )
        report["g0a"] = g0a

    # QA 标签质检
    if "qa" not in skip:
        with timer.phase("QA label / wall / units"):
            report["qa"] = label_qa(
                split_path, data_root=data_root, graphs_subdir=args.graphs_subdir_local,
                norm_stats=norm_stats, max_graphs_per_case=args.qa_max_graphs, max_cases=args.max_cases,
                thresholds={
                    "normal_invalid_rate_max": 0.05,
                    "basis_invalid_rate_max": 0.10,
                    "rad_ratio_max": 0.30,
                    "rad_frac_p95_max": 0.35,
                    "wss_p95_robust_z_max": 4.0,
                    "tangent_unit_frac_min": 0.90,
                },
                verbose=args.verbose, log_fn=log_fn,
            )

    # G0-c 压力梯度 → WSS
    if "g0c" not in skip:
        with timer.phase("G0-c pressure-gradient correlation"):
            pg = oracle_pressure_gradient_wss(
                split_path, norm_stats,
                max_cases=args.oracle_max_cases, max_graphs_per_case=args.oracle_max_graphs,
                max_wall_per_graph=args.max_wall_per_graph,
                verbose=args.verbose, log_fn=log_fn,
            )
        pg_abs = max((abs(c) for c in [pg.get("mean_spearman_neg_dpds"), pg.get("mean_spearman_gradmag")]
                      if c is not None), default=None)
        report["g0c_pressure_corr"] = {
            "mean_spearman_neg_dpds": pg.get("mean_spearman_neg_dpds"),
            "mean_spearman_gradmag": pg.get("mean_spearman_gradmag"),
            "abs_spearman": pg_abs,
            "n_rows": pg.get("n_rows"),
        }
        with timer.phase("G0-c pressure+geom GBDT"):
            report["g0c"] = g0c_pressure_geom_gbdt(
                split_path, data_root=data_root, graphs_subdir=args.graphs_subdir, norm_stats=norm_stats,
                max_graphs_per_case=args.max_graphs_per_case, max_wall_per_graph=args.max_wall_per_graph,
                seed=args.seed, max_train_cases=args.max_cases, max_test_cases=args.max_cases,
                verbose=args.verbose, log_fn=log_fn,
            )

    # G0-d 残差基线
    if "g0d" not in skip:
        with timer.phase("G0-d analytic residual baseline"):
            report["g0d"] = g0d_residual_baseline(
                split_path, data_root=data_root, graphs_subdir=args.graphs_subdir, norm_stats=norm_stats,
                max_graphs_per_case=args.max_graphs_per_case, max_cases=args.max_cases,
                verbose=args.verbose, log_fn=log_fn,
            )

    # G0-e noise floor
    if "g0e" not in skip:
        with timer.phase("G0-e noise floor recheck"):
            report["g0e"] = g0e_noise_floor(
                field_root=args.field_root, run_glob=args.run_glob, existing_json=args.existing_f0.resolve(),
            )

    with timer.phase("build gates + save json"):
        report["gates"] = build_gates(report, go_margin=args.go_margin)
        out_dir = ensure_dir(args.output_dir.resolve())
        name = args.output_name or f"v3p_g0_oracle_{date.today().strftime('%Y%m%d')}.json"
        out_path = out_dir / name
        save_json(out_path, report)

    timer.summary()
    print("\n=== gates ===")
    print(json.dumps(report["gates"], indent=2, ensure_ascii=False))
    print(out_path)


if __name__ == "__main__":
    main()
