"""通用离线 WSS 探针（0 重训 · CPU 后处理）。

用 Ridge / 梯度提升（HistGBDT）在「壁面几何特征 → WSS 分量」上做线性 / 非线性
可学性探针，回答路径 G 的 G0-a「坐标系负担量化」(Q1/Q2)：

- ``global`` 目标系：``wss_x / wss_y / wss_z``（全局 xyz，受 case 朝向影响）。
- ``local`` 目标系：``wss_axial / wss_circ / wss_rad``（中心线局部坐标，旋转不变）。

判读：若 local ``circ/rad`` 的 test R² 明显高于 global ``x/y``（Δ≥0.10），说明
横向 WSS 在 local 表示下「本来可学」，只是被全局坐标系拆散——这是 G1 等变头的
立项硬前提。

本模块既可作脚本（``python -m training.scripts.probe_linear_wss``），也被
``run_v3_g0_oracle`` 复用。R² 对目标的仿射缩放不变，因此 global（z-score）与
local（物理）即使归一化口径不同，R² 仍可直接比较；跨 case 方差用「组间方差占比」
(η²) 这一无量纲量比较。
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple


def _vlog(verbose: bool, msg: str, *, log_fn: Optional[Callable[[str], None]] = None) -> None:
    if not verbose:
        return
    if log_fn is not None:
        log_fn(msg)
    else:
        print(f"[probe] {msg}", flush=True)

import numpy as np

from ..core.splits import SplitSpec
from ..core.utils import ensure_dir
from ._figure_utils import save_json
from .run_v3_f0_decision import (
    NODE_IDX,
    REPO_ROOT,
    DEFAULT_NORM_PARAMS,
    _denorm_zscore,
    _load_norm_stats,
    _r2_score,
    _safe_spearman,
    _load_graph,
)

# data.x 列名 → 索引（与 pipeline/config.py NODE_FEATURE_NAMES 一致）
FEATURE_NAME_TO_IDX: Dict[str, int] = {
    "x": 0, "y": 1, "z": 2,
    "Abscissa": 3, "NormRadius": 4, "Curvature": 5,
    "Tangent_X": 6, "Tangent_Y": 7, "Tangent_Z": 8,
    "is_wall": 9,
    "dist_to_bifurcation": 10, "branch_id": 11,
    "dR_ds": 12, "torsion": 13, "d_tangent_ds": 14, "dist_to_wall": 15,
}
# 旋转敏感（随全局旋转一起变换）的向量特征三元组
VECTOR_FEATURE_TRIPLES = {
    "coords": ("x", "y", "z"),
    "tangent": ("Tangent_X", "Tangent_Y", "Tangent_Z"),
}
DEFAULT_FEATURES = [
    "Abscissa", "NormRadius", "Curvature",
    "Tangent_X", "Tangent_Y", "Tangent_Z",
    "dist_to_bifurcation", "branch_id",
    "dR_ds", "torsion", "d_tangent_ds", "dist_to_wall",
]
GLOBAL_COMPONENTS = ["wss_x", "wss_y", "wss_z"]
LOCAL_COMPONENTS = ["wss_axial", "wss_circ", "wss_rad"]


@dataclass
class WallDataset:
    """壁面点池化数据集（跨 graph / case）。"""
    X: np.ndarray                       # (n, f)
    feature_names: List[str]
    groups: np.ndarray                  # (n,) case 索引
    case_names: List[str]
    targets: Dict[str, np.ndarray]      # name -> (n,)
    mag: np.ndarray                     # (n,) 物理 WSS 标量（>0），用于加权/筛选
    meta: Dict[str, object] = field(default_factory=dict)


def _project_local(
    wall_xyz: np.ndarray,
    int_xyz: np.ndarray,
    tangent: np.ndarray,
    wss_vec: np.ndarray,
    *,
    k_internal: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """在 graph 自身（coord_normalized）坐标系内把 WSS 矢量投影到 local frame。

    复用 pipeline/local_wss.py 的 kNN 法向估计 + 切/法向投影公式，但直接作用于
    图数组（坐标、切向、WSS 向量同在旋转后坐标系，量纲一致）。
    返回 axial/circ/rad 以及 basis_valid 掩码。
    """
    from scipy.spatial import cKDTree

    n = wall_xyz.shape[0]
    axial = np.full(n, np.nan)
    circ = np.full(n, np.nan)
    rad = np.full(n, np.nan)
    basis_valid = np.zeros(n, dtype=bool)
    if int_xyz.shape[0] < 1:
        return axial, circ, rad, basis_valid

    tree = cKDTree(int_xyz)
    kq = int(min(k_internal, int_xyz.shape[0]))
    for i in range(n):
        _, idxs = tree.query(wall_xyz[i], k=kq)
        idxs = np.atleast_1d(idxs)
        direction = wall_xyz[i] - int_xyz[idxs].mean(axis=0)
        nn = float(np.linalg.norm(direction))
        if nn <= 1e-10:
            continue
        n_hat = direction / nn
        t = tangent[i]
        tn = float(np.linalg.norm(t))
        if tn <= 1e-10:
            continue
        t_hat = t / tn
        b_raw = np.cross(n_hat, t_hat)
        b_norm = float(np.linalg.norm(b_raw))
        cos_par = abs(float(np.dot(n_hat, t_hat)))
        w = wss_vec[i]
        axial[i] = float(np.dot(w, t_hat))
        if b_norm <= 1e-10 or cos_par >= 0.95:
            continue
        b_hat = b_raw / b_norm
        circ[i] = float(np.dot(w, b_hat))
        rad[i] = float(np.dot(w, n_hat))
        basis_valid[i] = True
    return axial, circ, rad, basis_valid


def collect_wall_dataset(
    split: SplitSpec,
    cases: Sequence[str],
    *,
    data_root: Path,
    graphs_subdir: str,
    feature_names: Sequence[str],
    norm_stats: Mapping[str, Dict[str, float]],
    target_frame: str,
    max_graphs_per_case: int,
    max_wall_per_graph: int,
    local_source: str = "precomputed",
    max_cases: Optional[int] = None,
    seed: int = 0,
    verbose: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> WallDataset:
    """池化壁面点，构建特征矩阵 + 目标（global 或 local 系）。

    target_frame:
      - ``global``: 目标 = denorm(wss_x/y/z)（物理、旋转后坐标系）。
      - ``local`` : 目标 = wss_axial/circ/rad。``local_source``:
          ``precomputed`` 用 graphs_local_v1 的 ``y_wss_local``（STL 法向，精确）；
          ``knn`` 用图内 kNN 法向现场投影（不依赖 local 图）。
    """
    rng = np.random.default_rng(seed)
    feat_idx = [FEATURE_NAME_TO_IDX[f] for f in feature_names]
    X_rows: List[np.ndarray] = []
    groups: List[int] = []
    mag_rows: List[np.ndarray] = []
    comp_names = GLOBAL_COMPONENTS if target_frame == "global" else LOCAL_COMPONENTS
    tgt_rows: Dict[str, List[np.ndarray]] = {c: [] for c in comp_names}
    used_cases: List[str] = []

    case_list = list(cases) if max_cases is None else list(cases)[:max_cases]
    _vlog(verbose, (
        f"collect_wall_dataset start: frame={target_frame} subdir={graphs_subdir} "
        f"cases={len(case_list)}/{len(cases)} max_graphs={max_graphs_per_case} "
        f"max_wall={max_wall_per_graph} local_source={local_source}"
    ), log_fn=log_fn)
    t_collect = time.perf_counter()
    for case_rel in case_list:
        case_dir = data_root / case_rel / "processed" / graphs_subdir
        if not case_dir.is_dir():
            _vlog(verbose, f"  skip case (no dir): {case_rel}", log_fn=log_fn)
            continue
        t_glob = time.perf_counter()
        graph_paths_all = sorted(p for p in case_dir.glob("*.pt"))
        dt_glob = time.perf_counter() - t_glob
        if not graph_paths_all:
            _vlog(verbose, f"  skip case (no graphs, glob {dt_glob:.2f}s): {case_rel}", log_fn=log_fn)
            continue
        graph_paths = graph_paths_all[:max_graphs_per_case]
        _vlog(verbose, (
            f"  case {case_rel}: use {len(graph_paths)}/{len(graph_paths_all)} graphs "
            f"(glob {dt_glob:.2f}s)"
        ), log_fn=log_fn)
        gi = len(used_cases)
        case_used = False
        for gp in graph_paths:
            t_load = time.perf_counter()
            data = _load_graph(gp)
            dt_load = time.perf_counter() - t_load
            x = data.x.numpy()
            wall = x[:, NODE_IDX["is_wall"]] > 0.5
            n_wall_raw = int(np.sum(wall))
            if n_wall_raw < 20:
                _vlog(verbose, f"    skip graph {gp.name}: n_wall={n_wall_raw} (load {dt_load:.2f}s)", log_fn=log_fn)
                continue
            y_wss = data.y_wss.numpy()
            feats = x[np.ix_(wall, feat_idx)].astype(np.float64)
            mag = _denorm_zscore(y_wss[wall, 0].astype(np.float64), norm_stats.get("wss"))

            if target_frame == "global":
                comps = {
                    "wss_x": _denorm_zscore(y_wss[wall, 1].astype(np.float64), norm_stats.get("wss_x")),
                    "wss_y": _denorm_zscore(y_wss[wall, 2].astype(np.float64), norm_stats.get("wss_y")),
                    "wss_z": _denorm_zscore(y_wss[wall, 3].astype(np.float64), norm_stats.get("wss_z")),
                }
                valid = np.ones(int(np.sum(wall)), dtype=bool)
            else:
                if local_source == "precomputed" and getattr(data, "y_wss_local", None) is not None:
                    yl = data.y_wss_local.numpy()
                    comps = {
                        "wss_axial": yl[wall, 1].astype(np.float64),
                        "wss_circ": yl[wall, 2].astype(np.float64),
                        "wss_rad": yl[wall, 3].astype(np.float64),
                    }
                    if getattr(data, "wss_local_mask", None) is not None:
                        m = data.wss_local_mask.numpy()
                        valid = m[wall, 1] > 0.5
                    else:
                        valid = np.isfinite(comps["wss_circ"])
                else:
                    wall_xyz = x[wall, :3].astype(np.float64)
                    int_xyz = x[~wall, :3].astype(np.float64)
                    tan = x[wall, 6:9].astype(np.float64)
                    wss_vec = y_wss[wall, 1:4].astype(np.float64)
                    ax, ci, ra, bv = _project_local(wall_xyz, int_xyz, tan, wss_vec)
                    comps = {"wss_axial": ax, "wss_circ": ci, "wss_rad": ra}
                    valid = bv

            valid = valid & np.all(np.isfinite(feats), axis=1)
            for c in comp_names:
                valid = valid & np.isfinite(comps[c])
            n_valid = int(np.sum(valid))
            if n_valid < 10:
                _vlog(verbose, (
                    f"    skip graph {gp.name}: valid={n_valid} (load {dt_load:.2f}s, "
                    f"n_wall={n_wall_raw})"
                ), log_fn=log_fn)
                continue

            idx = np.where(valid)[0]
            if idx.size > max_wall_per_graph:
                idx = rng.choice(idx, size=max_wall_per_graph, replace=False)

            _vlog(verbose, (
                f"    ok graph {gp.name}: load {dt_load:.2f}s n_wall={n_wall_raw} "
                f"valid={n_valid} sampled={idx.size}"
            ), log_fn=log_fn)
            X_rows.append(feats[idx])
            mag_rows.append(mag[idx])
            for c in comp_names:
                tgt_rows[c].append(comps[c][idx])
            groups.append(np.full(idx.size, gi, dtype=np.int64))
            case_used = True
        if case_used:
            used_cases.append(case_rel)

    dt_collect = time.perf_counter() - t_collect
    n_pts = sum(r.shape[0] for r in X_rows) if X_rows else 0
    _vlog(verbose, (
        f"collect_wall_dataset done: {n_pts} points from {len(used_cases)} cases "
        f"in {dt_collect:.2f}s"
    ), log_fn=log_fn)

    if not X_rows:
        return WallDataset(
            X=np.empty((0, len(feature_names))), feature_names=list(feature_names),
            groups=np.empty(0, dtype=np.int64), case_names=[],
            targets={c: np.empty(0) for c in comp_names}, mag=np.empty(0),
            meta={"target_frame": target_frame, "n_points": 0},
        )

    X = np.concatenate(X_rows, axis=0)
    groups_arr = np.concatenate(groups, axis=0)
    mag_arr = np.concatenate(mag_rows, axis=0)
    targets = {c: np.concatenate(tgt_rows[c], axis=0) for c in comp_names}
    return WallDataset(
        X=X, feature_names=list(feature_names), groups=groups_arr,
        case_names=used_cases, targets=targets, mag=mag_arr,
        meta={
            "target_frame": target_frame, "n_points": int(X.shape[0]),
            "n_cases": len(used_cases), "local_source": local_source,
        },
    )


def _fit_eval(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, *, model: str,
) -> Dict[str, object]:
    if model == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X_tr)
        reg = Ridge(alpha=1.0).fit(scaler.transform(X_tr), y_tr)
        pred = reg.predict(scaler.transform(X_te))
    elif model == "gbdt":
        from sklearn.ensemble import HistGradientBoostingRegressor
        reg = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=None,
            max_leaf_nodes=31, l2_regularization=1.0, random_state=0,
        ).fit(X_tr, y_tr)
        pred = reg.predict(X_te)
    else:
        raise ValueError(f"unknown model {model}")
    return {
        "test_r2": _r2_score(y_te, pred),
        "test_spearman": _safe_spearman(y_te, pred),
        "_pred": pred,
    }


def _eta_squared_between_cases(values: np.ndarray, groups: np.ndarray) -> float:
    """组间方差占比 η² = SS_between / SS_total（无量纲，衡量「跨 case 朝向负担」）。"""
    if values.size < 2:
        return float("nan")
    grand = float(np.mean(values))
    ss_tot = float(np.sum((values - grand) ** 2))
    if ss_tot <= 1e-20:
        return float("nan")
    ss_between = 0.0
    for g in np.unique(groups):
        v = values[groups == g]
        if v.size == 0:
            continue
        ss_between += v.size * (float(np.mean(v)) - grand) ** 2
    return float(ss_between / ss_tot)


def _vector_angle_error_deg(
    comps: Sequence[str], preds: Mapping[str, np.ndarray], trues: Mapping[str, np.ndarray],
) -> Optional[float]:
    if not all(c in preds for c in comps):
        return None
    p = np.stack([preds[c] for c in comps], axis=1)
    t = np.stack([trues[c] for c in comps], axis=1)
    pn = np.linalg.norm(p, axis=1)
    tn = np.linalg.norm(t, axis=1)
    ok = (pn > 1e-9) & (tn > 1e-9)
    if int(np.sum(ok)) < 10:
        return None
    cos = np.sum(p[ok] * t[ok], axis=1) / (pn[ok] * tn[ok])
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.median(np.degrees(np.arccos(cos))))


def run_probe(
    *,
    split_path: Path,
    data_root: Path,
    graphs_subdir_global: str,
    graphs_subdir_local: str,
    feature_names: Sequence[str],
    norm_stats: Mapping[str, Dict[str, float]],
    target_frame: str,
    models: Sequence[str],
    max_graphs_per_case: int,
    max_wall_per_graph: int,
    local_source: str,
    max_train_cases: Optional[int] = None,
    max_test_cases: Optional[int] = None,
    seed: int = 0,
    verbose: bool = False,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """对单一 target_frame（global / local）跑 Ridge+GBDT 探针，报告每分量 R²。"""
    split = SplitSpec.from_json(split_path)
    subdir = graphs_subdir_global if target_frame == "global" else graphs_subdir_local
    _vlog(verbose, f"run_probe({target_frame}) train collect ...", log_fn=log_fn)
    t0 = time.perf_counter()
    tr = collect_wall_dataset(
        split, split.train_cases, data_root=data_root, graphs_subdir=subdir,
        feature_names=feature_names, norm_stats=norm_stats, target_frame=target_frame,
        max_graphs_per_case=max_graphs_per_case, max_wall_per_graph=max_wall_per_graph,
        local_source=local_source, max_cases=max_train_cases, seed=seed,
        verbose=verbose, log_fn=log_fn,
    )
    _vlog(verbose, f"run_probe({target_frame}) test collect ...", log_fn=log_fn)
    te = collect_wall_dataset(
        split, split.test_cases, data_root=data_root, graphs_subdir=subdir,
        feature_names=feature_names, norm_stats=norm_stats, target_frame=target_frame,
        max_graphs_per_case=max_graphs_per_case, max_wall_per_graph=max_wall_per_graph,
        local_source=local_source, max_cases=max_test_cases, seed=seed + 1,
        verbose=verbose, log_fn=log_fn,
    )
    comp_names = GLOBAL_COMPONENTS if target_frame == "global" else LOCAL_COMPONENTS
    if tr.X.shape[0] < 50 or te.X.shape[0] < 20:
        return {"error": "insufficient samples", "n_train": int(tr.X.shape[0]), "n_test": int(te.X.shape[0])}

    per_model: Dict[str, object] = {}
    for model in models:
        t_fit = time.perf_counter()
        _vlog(verbose, f"run_probe({target_frame}) fit model={model} ...", log_fn=log_fn)
        comp_metrics: Dict[str, object] = {}
        preds: Dict[str, np.ndarray] = {}
        trues: Dict[str, np.ndarray] = {}
        for c in comp_names:
            t_comp = time.perf_counter()
            _vlog(verbose, f"run_probe({target_frame}) fit model={model} component={c} ...", log_fn=log_fn)
            res = _fit_eval(tr.X, tr.targets[c], te.X, te.targets[c], model=model)
            _vlog(verbose, (
                f"run_probe({target_frame}) model={model} component={c} "
                f"done in {time.perf_counter() - t_comp:.2f}s"
            ), log_fn=log_fn)
            preds[c] = res.pop("_pred")
            trues[c] = te.targets[c]
            res["between_case_eta2"] = _eta_squared_between_cases(te.targets[c], te.groups)
            comp_metrics[c] = res
        angle = _vector_angle_error_deg(comp_names, preds, trues)
        per_model[model] = {
            "components": comp_metrics,
            "vector_angle_error_median_deg": angle,
        }
        _vlog(verbose, f"run_probe({target_frame}) model={model} done in {time.perf_counter() - t_fit:.2f}s", log_fn=log_fn)

    _vlog(verbose, f"run_probe({target_frame}) total {time.perf_counter() - t0:.2f}s", log_fn=log_fn)
    return {
        "target_frame": target_frame,
        "graphs_subdir": subdir,
        "local_source": local_source if target_frame == "local" else None,
        "n_train_points": int(tr.X.shape[0]),
        "n_test_points": int(te.X.shape[0]),
        "n_train_cases": len(tr.case_names),
        "n_test_cases": len(te.case_names),
        "feature_names": list(feature_names),
        "models": per_model,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="离线 WSS 可学性探针（global vs local）")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training" / "splits" / "split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new" / "AG")
    ap.add_argument("--graphs-subdir-global", default="graphs")
    ap.add_argument("--graphs-subdir-local", default="graphs_local_v1")
    ap.add_argument("--target-frame", choices=["global", "local", "both"], default="both")
    ap.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    ap.add_argument("--models", nargs="+", default=["ridge", "gbdt"])
    ap.add_argument("--local-source", choices=["precomputed", "knn"], default="precomputed")
    ap.add_argument("--max-graphs-per-case", type=int, default=3)
    ap.add_argument("--max-wall-per-graph", type=int, default=800)
    ap.add_argument("--max-train-cases", type=int, default=None, help="限制 train case 数（自检用）")
    ap.add_argument("--max-test-cases", type=int, default=None, help="限制 test case 数（自检用）")
    ap.add_argument("--norm-params", type=Path, default=DEFAULT_NORM_PARAMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    norm_stats = _load_norm_stats(args.norm_params.resolve())
    frames = ["global", "local"] if args.target_frame == "both" else [args.target_frame]
    report: Dict[str, object] = {
        "split": str(args.split.resolve()),
        "data_root": str(args.data_root.resolve()),
        "frames": {},
    }
    for frame in frames:
        report["frames"][frame] = run_probe(
            split_path=args.split.resolve(), data_root=args.data_root.resolve(),
            graphs_subdir_global=args.graphs_subdir_global,
            graphs_subdir_local=args.graphs_subdir_local,
            feature_names=args.features, norm_stats=norm_stats, target_frame=frame,
            models=args.models, max_graphs_per_case=args.max_graphs_per_case,
            max_wall_per_graph=args.max_wall_per_graph, local_source=args.local_source,
            max_train_cases=args.max_train_cases, max_test_cases=args.max_test_cases,
            seed=args.seed,
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output is not None:
        ensure_dir(args.output.parent)
        save_json(args.output.resolve(), report)
        print(args.output)


if __name__ == "__main__":
    main()
