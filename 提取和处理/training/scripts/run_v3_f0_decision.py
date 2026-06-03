"""Path F · Phase 0 decision layer (TODO-27a / 53 / 32 / oracle).

Zero-retrain analysis on existing AsymW-a checkpoints and evaluation artifacts.
Outputs a single JSON report under ``outputs/field/f0_decision/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..analysis.stats import bootstrap_ci, summarize_seeds
from ..core.splits import SplitSpec
from ..core.utils import ensure_dir
from ._figure_utils import load_json, save_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASYMW_GLOB = (
    "field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed*"
)
NODE_IDX = {
    "x": 0, "y": 1, "z": 2,
    "Abscissa": 3, "NormRadius": 4, "Curvature": 5,
    "Tangent_X": 6, "Tangent_Y": 7, "Tangent_Z": 8,
    "is_wall": 9, "dist_to_wall": 15,
}
# data.y 列：u, v, w, p（见 pipeline/convert_to_graph.py）
Y_VEL = slice(0, 3)
Y_P = 3
# data.x Tangent_X/Y/Z（中心线切向 = 轴向参考）
X_TAN = slice(6, 9)
DEFAULT_NORM_PARAMS = REPO_ROOT / "data_new" / "normalization_params_global.json"


def _load_norm_stats(path: Path) -> Dict[str, Dict[str, float]]:
    """读取归一化统计量；oracle 需在物理量纲下算相关/重建。"""
    if not path.is_file():
        return {}
    params = load_json(path)
    return params.get("statistics", {})


def _denorm_zscore(arr: np.ndarray, stat: Optional[Mapping[str, float]]) -> np.ndarray:
    """z-score 反变换 original = normalized * std + mean。stat 缺失则原样返回。"""
    if not stat:
        return arr
    return arr * float(stat.get("std", 1.0)) + float(stat.get("mean", 0.0))


def _denorm_velocity(vel: np.ndarray, stats: Mapping[str, Dict[str, float]]) -> np.ndarray:
    """逐分量反归一化速度（u/v/w std 各异，否则矢量模会被各向异性扭曲）。"""
    out = vel.astype(np.float64).copy()
    for j, name in enumerate(("u", "v", "w")):
        out[:, j] = _denorm_zscore(out[:, j], stats.get(name))
    return out


def _denorm_wss_mag(wss_norm: np.ndarray, stats: Mapping[str, Dict[str, float]]) -> np.ndarray:
    """反归一化 WSS 标量（物理上 ≥0）；旧实现对归一化值取 abs 会打乱排序。"""
    return _denorm_zscore(wss_norm.astype(np.float64), stats.get("wss"))


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return _safe_pearson(ra, rb)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-20:
        return 1.0 if ss_res <= 1e-20 else 0.0
    return 1.0 - ss_res / ss_tot


def _safe_json_float(value: float) -> Optional[float]:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _discover_asymw_runs(field_root: Path, pattern: str) -> List[Path]:
    runs = sorted(field_root.glob(pattern))
    by_seed: Dict[int, Path] = {}
    for run_dir in runs:
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            continue
        summary = load_json(summary_path)
        if summary.get("exp_id") != "V3P-Main-01-PW-AsymW-a":
            continue
        seed = int(summary.get("seed", -1))
        if seed < 0:
            continue
        prev = by_seed.get(seed)
        if prev is None or run_dir.name > prev.name:
            by_seed[seed] = run_dir
    return [by_seed[s] for s in sorted(by_seed)]


def _eval_paths(run_dir: Path) -> Dict[str, Path]:
    base = run_dir / "evaluation" / "test_best_wss_model"
    return {
        "wss_direct": base / "wss_direct" / "wss_credibility_summary.json",
        "wss_case_metrics": base / "wss_direct" / "wss_case_metrics.csv",
        "wss_sample_metrics": base / "wss_direct" / "wss_sample_metrics.csv",
        "wss_pa": base / "wss_clinical_pa" / "wss_pa_summary.json",
        "manifest": run_dir / "predictions_test_best_wss" / "manifest.json",
        "summary": run_dir / "summary.json",
    }


def todo_27a_seed_bandwidth(runs: Sequence[Path]) -> Dict[str, object]:
    by_seed: Dict[int, float] = {}
    rows: List[Dict[str, object]] = []
    for run_dir in runs:
        summary = load_json(run_dir / "summary.json")
        seed = int(summary["seed"])
        test_best = summary.get("test_metrics_best_wss") or summary.get("test_metrics") or {}
        r2 = float(test_best.get("wss_r2_wss", float("nan")))
        by_seed[seed] = r2
        rows.append({
            "seed": seed,
            "run_dir": str(run_dir),
            "wss_r2_wss": r2,
            "best_wss_epoch": summary.get("best_wss_epoch"),
        })
    return {
        "metric": "wss_r2_wss",
        "checkpoint": "best_wss_model",
        "per_seed": rows,
        "summary": summarize_seeds(by_seed),
    }


def _bootstrap_case_pooled_r2(
    case_rows: Sequence[Mapping[str, str]],
    *,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, object]:
    usable = []
    for row in case_rows:
        r2_raw = row.get("wss_mag_r2", "")
        n_raw = row.get("n_wall_samples", row.get("n_wall", ""))
        if not r2_raw or not n_raw:
            continue
        r2 = float(r2_raw)
        n = int(float(n_raw))
        if not math.isfinite(r2) or n <= 0:
            continue
        usable.append({"case_name": row.get("case_name", ""), "r2": r2, "n": n})

    if len(usable) < 2:
        return {"error": "insufficient case rows for bootstrap", "n_cases": len(usable)}

    rng = np.random.default_rng(seed)
    n_cases = len(usable)
    pooled_r2 = np.empty(n_bootstrap, dtype=np.float64)
    mean_case_r2 = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n_cases, size=n_cases)
        sample = [usable[j] for j in idx]
        mean_case_r2[i] = float(np.mean([s["r2"] for s in sample]))
        weights = np.asarray([s["n"] for s in sample], dtype=np.float64)
        pooled_r2[i] = float(np.average([s["r2"] for s in sample], weights=weights))

    def _ci(arr: np.ndarray) -> Dict[str, float]:
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return {
            "mean": float(np.mean(arr)),
            "ci95_lo": float(lo),
            "ci95_hi": float(hi),
        }

    return {
        "n_cases": n_cases,
        "n_bootstrap": n_bootstrap,
        "pooled_r2": _ci(pooled_r2),
        "unweighted_mean_case_r2": _ci(mean_case_r2),
        "note": "case resampling with replacement; pooled R² = n-weighted mean of per-case R²",
    }


def todo_27a_bootstrap(runs: Sequence[Path], *, n_bootstrap: int, seed: int) -> Dict[str, object]:
    out: Dict[str, object] = {"per_run": []}
    for run_dir in runs:
        summary = load_json(run_dir / "summary.json")
        case_csv = _eval_paths(run_dir)["wss_case_metrics"]
        rows = _read_csv_rows(case_csv)
        result = _bootstrap_case_pooled_r2(rows, n_bootstrap=n_bootstrap, seed=seed + int(summary["seed"]))
        result["seed"] = int(summary["seed"])
        result["case_metrics_csv"] = str(case_csv)
        out["per_run"].append(result)
    return out


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = int(np.sum(y_true))
    neg = int(y_true.size - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty(y_true.size, dtype=np.float64)
    sorted_scores = y_score[order]
    i = 0
    rank = 1.0
    while i < y_true.size:
        j = i + 1
        while j < y_true.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (rank + rank + (j - i) - 1)
        ranks[order[i:j]] = avg_rank
        rank += j - i
        i = j
    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    return (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def todo_53_clinical_extended(run_dir: Path) -> Dict[str, object]:
    paths = _eval_paths(run_dir)
    credibility = load_json(paths["wss_direct"]) if paths["wss_direct"].is_file() else {}
    pa = load_json(paths["wss_pa"]) if paths["wss_pa"].is_file() else {}
    sample_rows = _read_csv_rows(paths["wss_sample_metrics"])

    auc_rows: List[Dict[str, object]] = []
    for frac in (0.05, 0.10):
        thr_key = f"top{int(round(frac * 100))}"
        labels: List[np.ndarray] = []
        scores: List[np.ndarray] = []
        for row in sample_rows:
            p95_true = row.get("p95_true")
            p95_pred = row.get("p95_pred")
            mean_true = row.get("mean_true")
            mean_pred = row.get("mean_pred")
            if not p95_true or not mean_true:
                continue
            thr = float(p95_true) * frac + float(mean_true) * (1.0 - frac)
            true_key = f"{thr_key}_overlap"
            if true_key not in row:
                continue
            yt = np.asarray([float(row["mean_true"]), float(row["p95_true"])], dtype=np.float64)
            yp = np.asarray([float(row["mean_pred"]), float(row["p95_pred"])], dtype=np.float64)
            labels.append((yt >= thr).astype(np.int64))
            scores.append(yp)
        if labels:
            y_all = np.concatenate(labels)
            s_all = np.concatenate(scores)
            auc_rows.append({
                "definition": f"sample-level high-WSS proxy top{int(round(frac * 100))}%",
                "auc": _binary_auc(y_all, s_all),
                "n_points": int(y_all.size),
            })

    overlap = credibility.get("high_wss_overlap") or []
    iou_summary = {
        "top5_jaccard": next((o.get("mean_jaccard") for o in overlap if abs(float(o.get("top_fraction", -1)) - 0.05) < 1e-9), None),
        "top10_jaccard": next((o.get("mean_jaccard") for o in overlap if abs(float(o.get("top_fraction", -1)) - 0.10) < 1e-9), None),
        "top10_dice": credibility.get("quick_view", {}).get("top10_mean_dice"),
    }

    profile_proxy = {
        "case_mean_spearman_norm": credibility.get("case_summary", {}).get("mean_spearman"),
        "case_p95_spearman_norm": credibility.get("case_summary", {}).get("p95_spearman"),
        "case_p95_spearman_pa": next(
            (r.get("spearman") for r in pa.get("case_level", []) if r.get("metric") == "p95"),
            None,
        ),
        "note": "centerline profile Pearson deferred; proxy uses case-level order stats",
    }

    return {
        "run_dir": str(run_dir),
        "point_r2": credibility.get("quick_view", {}).get("point_r2"),
        "high_wss_iou": iou_summary,
        "high_wss_auc_proxy": auc_rows,
        "profile_proxy": profile_proxy,
        "pa_case_level": pa.get("case_level", []),
    }


def todo_32_ensemble_status(runs: Sequence[Path]) -> Dict[str, object]:
    manifests = []
    missing = 0
    present = 0
    for run_dir in runs:
        manifest_path = run_dir / "predictions_test_best_wss" / "manifest.json"
        if not manifest_path.is_file():
            manifests.append({"seed": None, "manifest": str(manifest_path), "status": "missing_manifest"})
            continue
        manifest = load_json(manifest_path)
        items = manifest.get("items") or []
        n_missing = 0
        n_ok = 0
        for item in items[:20]:
            pred = Path(str(item.get("prediction_path", "")))
            if pred.is_file():
                n_ok += 1
            else:
                n_missing += 1
        sample_missing = n_missing > 0
        if sample_missing:
            missing += 1
        else:
            present += 1
        manifests.append({
            "seed": load_json(run_dir / "summary.json").get("seed"),
            "manifest": str(manifest_path),
            "num_predictions": len(items),
            "sample_check": {"ok": n_ok, "missing": n_missing},
            "predictions_available": not sample_missing,
        })

    repredict_cmds = []
    for run_dir in runs:
        cfg = run_dir / "config.snapshot.json"
        ckpt = run_dir / "best_wss_model.pt"
        out = run_dir / "predictions_test_best_wss"
        repredict_cmds.append(
            " ".join([
                "python -m training.scripts.predict_field",
                f"--config {cfg}",
                f"--checkpoint {ckpt}",
                "--subset test",
                f"--output {out}",
            ])
        )

    case_r2_by_seed: Dict[int, List[float]] = {}
    for run_dir in runs:
        seed = int(load_json(run_dir / "summary.json")["seed"])
        rows = _read_csv_rows(_eval_paths(run_dir)["wss_case_metrics"])
        vals = [float(r["wss_mag_r2"]) for r in rows if r.get("wss_mag_r2")]
        case_r2_by_seed[seed] = vals

    weak_upper_bound = None
    if len(case_r2_by_seed) >= 2:
        stacked = np.asarray(list(case_r2_by_seed.values()), dtype=np.float64)
        if stacked.shape[1] == stacked.shape[0] and stacked.shape[0] >= 2:
            weak_upper_bound = float(np.mean(np.max(stacked, axis=0)))

    return {
        "predictions_available_runs": present,
        "predictions_missing_runs": missing,
        "manifests": manifests,
        "weak_case_r2_oracle_upper_bound": weak_upper_bound,
        "note": "true point ensemble needs predictions_test_best_wss/*.pt; run repredict commands below",
        "repredict_commands": repredict_cmds,
    }


def _load_graph(path: Path):
    import torch
    return torch.load(path, map_location="cpu")


def _interp_profile(abscissa: np.ndarray, values: np.ndarray, grid_size: int) -> np.ndarray:
    order = np.argsort(abscissa)
    a = abscissa[order]
    v = values[order]
    if a.size < 2:
        return np.full(grid_size, np.nan, dtype=np.float64)
    uniq_a, idx = np.unique(a, return_index=True)
    uniq_v = v[idx]
    if uniq_a.size < 2:
        return np.full(grid_size, float(uniq_v[0]), dtype=np.float64)
    tgt = np.linspace(float(uniq_a.min()), float(uniq_a.max()), grid_size)
    return np.interp(tgt, uniq_a, uniq_v)


def _collect_profiles(
    split: SplitSpec,
    cases: Sequence[str],
    *,
    grid_size: int,
    max_graphs_per_case: int,
) -> Tuple[np.ndarray, List[str]]:
    profiles: List[np.ndarray] = []
    labels: List[str] = []
    for case_rel in cases:
        case_dir = REPO_ROOT / "data_new" / "AG" / case_rel / "processed" / "graphs"
        if not case_dir.is_dir():
            continue
        graph_paths = sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]
        for graph_path in graph_paths:
            data = _load_graph(graph_path)
            x = data.x.numpy()
            y_wss = data.y_wss.numpy()
            wall = x[:, NODE_IDX["is_wall"]] > 0.5
            if not np.any(wall):
                continue
            abscissa = x[wall, NODE_IDX["Abscissa"]]
            wss = y_wss[wall, 0]
            prof = _interp_profile(abscissa, wss, grid_size)
            if np.all(np.isfinite(prof)):
                profiles.append(prof)
                labels.append(f"{case_rel}:{graph_path.stem}")
    if not profiles:
        return np.empty((0, grid_size)), labels
    return np.stack(profiles, axis=0), labels


def oracle_pod(
    split_path: Path,
    *,
    grid_size: int,
    max_graphs_per_case: int,
    modes: Sequence[int],
) -> Dict[str, object]:
    split = SplitSpec.from_json(split_path)
    train_profiles, train_labels = _collect_profiles(
        split, split.train_cases, grid_size=grid_size, max_graphs_per_case=max_graphs_per_case,
    )
    test_profiles, test_labels = _collect_profiles(
        split, split.test_cases, grid_size=grid_size, max_graphs_per_case=max_graphs_per_case,
    )
    if train_profiles.shape[0] < 3 or test_profiles.shape[0] < 1:
        return {"error": "insufficient profiles", "n_train": int(train_profiles.shape[0]), "n_test": int(test_profiles.shape[0])}

    train_mean = np.mean(train_profiles, axis=0, keepdims=True)
    train_centered = train_profiles - train_mean
    _, _, vt = np.linalg.svd(train_centered, full_matrices=False)
    results = []
    for k in modes:
        k_eff = min(int(k), vt.shape[0])
        basis = vt[:k_eff]
        recon_train = (train_centered @ basis.T) @ basis + train_mean
        recon_test = ((test_profiles - train_mean) @ basis.T) @ basis + train_mean
        train_r2 = [_r2_score(train_profiles[i], recon_train[i]) for i in range(train_profiles.shape[0])]
        test_r2 = [_r2_score(test_profiles[i], recon_test[i]) for i in range(test_profiles.shape[0])]
        results.append({
            "n_modes": k_eff,
            "train_mean_r2": _safe_json_float(float(np.nanmean(train_r2))),
            "test_mean_r2": _safe_json_float(float(np.nanmean(test_r2))),
        })

    return {
        "grid_size": grid_size,
        "max_graphs_per_case": max_graphs_per_case,
        "n_train_profiles": int(train_profiles.shape[0]),
        "n_test_profiles": int(test_profiles.shape[0]),
        "modes": results,
        "interpretation": "high test R² ⇒ POD/low-rank surrogate may be viable (TODO-52); low ⇒ field too heterogeneous",
    }


def oracle_gt_wall_diff(
    split_path: Path,
    *,
    max_cases: int,
    max_graphs_per_case: int,
) -> Dict[str, object]:
    split = SplitSpec.from_json(split_path)
    case_rows = []
    for case_rel in split.test_cases[:max_cases]:
        case_dir = REPO_ROOT / "data_new" / "AG" / case_rel / "processed" / "graphs"
        if not case_dir.is_dir():
            continue
        for graph_path in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
            data = _load_graph(graph_path)
            x = data.x.numpy()
            y = data.y.numpy()
            y_wss = data.y_wss.numpy()
            wall = x[:, NODE_IDX["is_wall"]] > 0.5
            interior = ~wall
            if not np.any(wall) or not np.any(interior):
                continue
            wall_xyz = x[wall, :3]
            int_xyz = x[interior, :3]
            int_vel = y[interior, :3]
            wss_mag = np.abs(y_wss[wall, 0])
            # nearest interior point per wall node (subsample for speed)
            step = max(1, wall_xyz.shape[0] // 2000)
            approx = []
            gt = []
            for i in range(0, wall_xyz.shape[0], step):
                dist2 = np.sum((int_xyz - wall_xyz[i]) ** 2, axis=1)
                j = int(np.argmin(dist2))
                dn = float(np.sqrt(dist2[j]))
                if dn <= 1e-8:
                    continue
                du_dn = float(np.linalg.norm(int_vel[j] - y[i, :3])) / dn
                approx.append(du_dn)
                gt.append(float(wss_mag[i]))
            if len(gt) < 10:
                continue
            gt_arr = np.asarray(gt, dtype=np.float64)
            ap_arr = np.asarray(approx, dtype=np.float64)
            case_rows.append({
                "case": case_rel,
                "graph": graph_path.name,
                "n_wall_sampled": len(gt),
                "pearson": float(np.corrcoef(gt_arr, ap_arr)[0, 1]) if np.std(gt_arr) > 0 and np.std(ap_arr) > 0 else float("nan"),
                "spearman": float(np.corrcoef(np.argsort(gt_arr), np.argsort(ap_arr))[0, 1]),
            })

    pearsons = [r["pearson"] for r in case_rows if math.isfinite(r["pearson"])]
    return {
        "cases": case_rows,
        "mean_case_pearson": float(np.mean(pearsons)) if pearsons else None,
        "interpretation": "low correlation ⇒ TODO-30/33 diff-WSS branch unlikely to help without better normals/μ",
        "note": "naive baseline (full |Δvel|/euclidean, abs of normalized wss); 见 gt_wall_diff_v2 修正版",
    }


def _estimate_normals_at(
    wall_xyz: np.ndarray, query_idx: np.ndarray, *, k: int = 16
) -> np.ndarray:
    """对 query 壁面点用 kNN 壁面邻域 PCA 估法向（最小特征值方向）。

    返回未定向单位法向（朝向在调用处用最近内部点定为指向流体内部）。
    """
    from sklearn.neighbors import NearestNeighbors

    n = wall_xyz.shape[0]
    k_eff = int(min(k, max(2, n - 1)))
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(wall_xyz)
    _, idx = nn.kneighbors(wall_xyz[query_idx])
    normals = np.zeros((query_idx.size, 3), dtype=np.float64)
    for r in range(query_idx.size):
        nb = wall_xyz[idx[r, 1:]]
        c = nb - nb.mean(axis=0, keepdims=True)
        cov = c.T @ c
        w, v = np.linalg.eigh(cov)
        nrm = v[:, 0]
        norm = np.linalg.norm(nrm)
        normals[r] = nrm / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
    return normals


def oracle_gt_wall_diff_v2(
    split_path: Path,
    norm_stats: Mapping[str, Dict[str, float]],
    *,
    max_cases: int,
    max_graphs_per_case: int,
    max_wall_per_graph: int = 1500,
    k_interior: int = 40,
    cone_ratio: float = 1.0,
) -> Dict[str, object]:
    """修正版近壁差分 oracle：反归一化 + 切向投影 + 沿法向采样。

    变体（均与物理 WSS 标量做逐病例相关）：
    - naive            : |Δvel(最近内部点)| / 欧氏距离（≈旧实现，仅修反归一化）
    - tang_nearest     : 切向速度模(最近内部点) / 欧氏距离
    - tang_normal      : 切向速度模(沿法向 cone 内最近点) / 法向距离  ← 物理最贴近
    - tang_wssdir      : 速度在 GT WSS 方向投影 / 法向距离（半循环，作上限参考）
    """
    from sklearn.neighbors import NearestNeighbors

    split = SplitSpec.from_json(split_path)
    variants = ("naive", "tang_nearest", "tang_normal", "tang_wssdir")
    rows: List[Dict[str, object]] = []

    for case_rel in split.test_cases[:max_cases]:
        case_dir = REPO_ROOT / "data_new" / "AG" / case_rel / "processed" / "graphs"
        if not case_dir.is_dir():
            continue
        for graph_path in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
            data = _load_graph(graph_path)
            x = data.x.numpy()
            y = data.y.numpy()
            y_wss = data.y_wss.numpy()
            wall = x[:, NODE_IDX["is_wall"]] > 0.5
            interior = ~wall
            if int(np.sum(wall)) < 20 or int(np.sum(interior)) < 20:
                continue

            wall_xyz = x[wall, :3].astype(np.float64)
            int_xyz = x[interior, :3].astype(np.float64)
            int_vel = _denorm_velocity(y[interior, Y_VEL], norm_stats)
            wss_mag = _denorm_wss_mag(y_wss[wall, 0], norm_stats)
            wss_vec = y_wss[wall, 1:4].astype(np.float64)

            n_wall = wall_xyz.shape[0]
            step = max(1, n_wall // max_wall_per_graph)
            q_idx = np.arange(0, n_wall, step)
            normals = _estimate_normals_at(wall_xyz, q_idx, k=16)

            nn_int = NearestNeighbors(
                n_neighbors=int(min(k_interior, int_xyz.shape[0]))
            ).fit(int_xyz)
            dist_q, idx_q = nn_int.kneighbors(wall_xyz[q_idx])

            est = {v: [] for v in variants}
            gt = []
            for r, wi in enumerate(q_idx):
                p = wall_xyz[wi]
                nb_idx = idx_q[r]
                offs = int_xyz[nb_idx] - p
                nhat = normals[r]
                nd = offs @ nhat
                # 定向：法向指向内部（多数内部点应在正侧）
                if np.mean(nd) < 0:
                    nhat = -nhat
                    nd = -nd
                td = np.linalg.norm(offs - np.outer(nd, nhat), axis=1)

                # 最近内部点（naive / tang_nearest）
                j_near = int(np.argmin(dist_q[r]))
                vel_near = int_vel[nb_idx[j_near]]
                dn_near = float(dist_q[r][j_near])
                if dn_near <= 1e-9:
                    continue
                ut_near = vel_near - (vel_near @ nhat) * nhat

                # 沿法向 cone 内最近点
                cone = (nd > 1e-9) & (td <= cone_ratio * np.maximum(nd, 1e-9))
                gt.append(float(wss_mag[wi]))
                est["naive"].append(float(np.linalg.norm(vel_near)) / dn_near)
                est["tang_nearest"].append(float(np.linalg.norm(ut_near)) / dn_near)
                if np.any(cone):
                    kk = nb_idx[cone]
                    nd_c = nd[cone]
                    jb = int(np.argmin(nd_c))
                    nd_b = float(nd_c[jb])
                    vel_b = int_vel[kk[jb]]
                    ut_b = vel_b - (vel_b @ nhat) * nhat
                    est["tang_normal"].append(float(np.linalg.norm(ut_b)) / max(nd_b, 1e-9))
                    wv = wss_vec[wi]
                    wvn = np.linalg.norm(wv)
                    if wvn > 1e-12:
                        proj = abs(float(vel_b @ (wv / wvn)))
                        est["tang_wssdir"].append(proj / max(nd_b, 1e-9))
                    else:
                        est["tang_wssdir"].append(float("nan"))
                else:
                    est["tang_normal"].append(float("nan"))
                    est["tang_wssdir"].append(float("nan"))

            if len(gt) < 10:
                continue
            gt_arr = np.asarray(gt, dtype=np.float64)
            row: Dict[str, object] = {"case": case_rel, "graph": graph_path.name, "n_wall_sampled": len(gt)}
            for v in variants:
                ev = np.asarray(est[v], dtype=np.float64)
                mask = np.isfinite(ev) & np.isfinite(gt_arr)
                if int(np.sum(mask)) >= 10:
                    row[f"pearson_{v}"] = _safe_pearson(gt_arr[mask], ev[mask])
                    row[f"spearman_{v}"] = _safe_spearman(gt_arr[mask], ev[mask])
                else:
                    row[f"pearson_{v}"] = float("nan")
                    row[f"spearman_{v}"] = float("nan")
            rows.append(row)

    summary: Dict[str, object] = {}
    for v in variants:
        ps = [r[f"pearson_{v}"] for r in rows if math.isfinite(r.get(f"pearson_{v}", float("nan")))]
        ss = [r[f"spearman_{v}"] for r in rows if math.isfinite(r.get(f"spearman_{v}", float("nan")))]
        summary[v] = {
            "mean_pearson": float(np.mean(ps)) if ps else None,
            "mean_spearman": float(np.mean(ss)) if ss else None,
            "n_graphs": len(ps),
        }
    return {
        "variant_summary": summary,
        "n_rows": len(rows),
        "cases": rows,
        "interpretation": (
            "比较 tang_normal vs naive：若 tang_normal mean_pearson 显著为正（>~0.4），"
            "说明真实近壁差分能解释 WSS，TODO-30/33 值得开；naive 负相关是旧实现伪信号"
        ),
    }


def oracle_pressure_gradient_wss(
    split_path: Path,
    norm_stats: Mapping[str, Dict[str, float]],
    *,
    max_cases: int,
    max_graphs_per_case: int,
    max_wall_per_graph: int = 1500,
    k_wall: int = 12,
) -> Dict[str, object]:
    """壁面切向压力梯度 ∂p/∂s 与 WSS 的相关诊断（思路 2：压力→WSS 耦合）。

    每个壁面点用 kNN 壁面邻域最小二乘估 ∇p，再投影到中心线切向得 dp/ds；
    逐病例与物理 WSS 标量做 Pearson/Spearman。压力 R²≈0.96 已可靠，若相关明显
    则压力梯度可作为 WSS head 的强条件特征。
    """
    from sklearn.neighbors import NearestNeighbors

    split = SplitSpec.from_json(split_path)
    rows: List[Dict[str, object]] = []
    for case_rel in split.test_cases[:max_cases]:
        case_dir = REPO_ROOT / "data_new" / "AG" / case_rel / "processed" / "graphs"
        if not case_dir.is_dir():
            continue
        for graph_path in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
            data = _load_graph(graph_path)
            x = data.x.numpy()
            y = data.y.numpy()
            y_wss = data.y_wss.numpy()
            wall = x[:, NODE_IDX["is_wall"]] > 0.5
            if int(np.sum(wall)) < 20:
                continue
            wall_xyz = x[wall, :3].astype(np.float64)
            p_wall = _denorm_zscore(y[wall, Y_P], norm_stats.get("p"))
            tan = x[wall, X_TAN].astype(np.float64)
            wss_mag = _denorm_wss_mag(y_wss[wall, 0], norm_stats)

            n_wall = wall_xyz.shape[0]
            step = max(1, n_wall // max_wall_per_graph)
            q_idx = np.arange(0, n_wall, step)
            k_eff = int(min(k_wall, n_wall - 1))
            nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(wall_xyz)
            _, idx = nn.kneighbors(wall_xyz[q_idx])

            dpds = np.full(q_idx.size, np.nan, dtype=np.float64)
            gradmag = np.full(q_idx.size, np.nan, dtype=np.float64)
            for r, wi in enumerate(q_idx):
                nb = idx[r, 1:]
                dx = wall_xyz[nb] - wall_xyz[wi]
                dp = p_wall[nb] - p_wall[wi]
                # 最小二乘 ∇p：dp ≈ dx · g
                g, *_ = np.linalg.lstsq(dx, dp, rcond=None)
                t = tan[wi]
                tn = np.linalg.norm(t)
                if tn > 1e-9:
                    dpds[r] = float(g @ (t / tn))
                gradmag[r] = float(np.linalg.norm(g))

            gt = wss_mag[q_idx]
            mask = np.isfinite(dpds)
            if int(np.sum(mask)) < 10:
                continue
            rows.append({
                "case": case_rel,
                "graph": graph_path.name,
                "n_wall_sampled": int(np.sum(mask)),
                # 顺压梯度（dp/ds<0）通常对应高 WSS，用 -dp/ds 与 WSS 相关
                "pearson_neg_dpds": _safe_pearson(gt[mask], -dpds[mask]),
                "spearman_neg_dpds": _safe_spearman(gt[mask], -dpds[mask]),
                "pearson_gradmag": _safe_pearson(gt[mask], gradmag[mask]),
                "spearman_gradmag": _safe_spearman(gt[mask], gradmag[mask]),
            })

    def _mean(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if math.isfinite(r.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else None

    return {
        "n_rows": len(rows),
        "mean_pearson_neg_dpds": _mean("pearson_neg_dpds"),
        "mean_spearman_neg_dpds": _mean("spearman_neg_dpds"),
        "mean_pearson_gradmag": _mean("pearson_gradmag"),
        "mean_spearman_gradmag": _mean("spearman_gradmag"),
        "cases": rows,
        "interpretation": (
            "若 |mean correlation| 明显（>~0.3），压力梯度携带 WSS 空间信息，"
            "可把预测压力的壁面切向梯度喂入 wss_head 或加弱物理一致 loss（思路 2）"
        ),
    }


def _collect_profiles_2d(
    cases: Sequence[str],
    norm_stats: Mapping[str, Dict[str, float]],
    *,
    grid_s: int,
    n_sectors: int,
    max_graphs_per_case: int,
) -> np.ndarray:
    """构建 (Abscissa × θ) 2D 壁面 WSS 图，展平为向量。

    θ 由每图全局垂直平面投影得到（图内一致，跨病例仅近似一致 → 单边保守测试：
    2D 重建 R² 升高才有意义，降低可能由 θ 对齐不一致引起）。n_sectors=1 退化为
    周向平均 1D profile（与 oracle_pod 的 1D 口径自洽，可作 sanity check）。
    """
    out: List[np.ndarray] = []
    for case_rel in cases:
        case_dir = REPO_ROOT / "data_new" / "AG" / case_rel / "processed" / "graphs"
        if not case_dir.is_dir():
            continue
        for graph_path in sorted(case_dir.glob("*.pt"))[:max_graphs_per_case]:
            data = _load_graph(graph_path)
            x = data.x.numpy()
            y_wss = data.y_wss.numpy()
            wall = x[:, NODE_IDX["is_wall"]] > 0.5
            if int(np.sum(wall)) < 20:
                continue
            xyz = x[wall, :3].astype(np.float64)
            absc = x[wall, NODE_IDX["Abscissa"]].astype(np.float64)
            tan = x[wall, X_TAN].astype(np.float64)
            wss = _denorm_wss_mag(y_wss[wall, 0], norm_stats)

            t_mean = tan.mean(axis=0)
            tn = np.linalg.norm(t_mean)
            t_mean = t_mean / tn if tn > 1e-9 else np.array([0.0, 0.0, 1.0])
            ref = np.array([0.0, 0.0, 1.0]) if abs(t_mean[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            e1 = ref - (ref @ t_mean) * t_mean
            e1 = e1 / (np.linalg.norm(e1) + 1e-12)
            e2 = np.cross(t_mean, e1)
            c = xyz - xyz.mean(axis=0, keepdims=True)
            a1 = c @ e1
            a2 = c @ e2
            theta = np.arctan2(a2, a1)  # [-pi, pi]

            a_lo, a_hi = float(absc.min()), float(absc.max())
            if a_hi - a_lo < 1e-9:
                continue
            s_bin = np.clip(((absc - a_lo) / (a_hi - a_lo) * grid_s).astype(int), 0, grid_s - 1)
            th_bin = np.clip(((theta + math.pi) / (2 * math.pi) * n_sectors).astype(int), 0, n_sectors - 1)

            grid = np.full((grid_s, n_sectors), np.nan, dtype=np.float64)
            for si in range(grid_s):
                row_mask = s_bin == si
                if not np.any(row_mask):
                    continue
                row_mean = float(np.mean(wss[row_mask]))
                for ti in range(n_sectors):
                    cell = row_mask & (th_bin == ti)
                    grid[si, ti] = float(np.mean(wss[cell])) if np.any(cell) else row_mean
            if np.any(np.isnan(grid)):
                col = np.nanmean(grid)
                grid = np.where(np.isnan(grid), col, grid)
            out.append(grid.reshape(-1))
    if not out:
        return np.empty((0, grid_s * n_sectors))
    return np.stack(out, axis=0)


def oracle_pod_2d(
    split_path: Path,
    norm_stats: Mapping[str, Dict[str, float]],
    *,
    grid_s: int,
    sectors: Sequence[int],
    modes: Sequence[int],
    max_graphs_per_case: int,
) -> Dict[str, object]:
    """2D (Abscissa × θ) 壁面 WSS 场 POD 重建上限；修正 1D profile 的周向混叠。"""
    split = SplitSpec.from_json(split_path)
    results: List[Dict[str, object]] = []
    for n_sec in sectors:
        train = _collect_profiles_2d(
            split.train_cases, norm_stats, grid_s=grid_s, n_sectors=n_sec,
            max_graphs_per_case=max_graphs_per_case,
        )
        test = _collect_profiles_2d(
            split.test_cases, norm_stats, grid_s=grid_s, n_sectors=n_sec,
            max_graphs_per_case=max_graphs_per_case,
        )
        if train.shape[0] < 3 or test.shape[0] < 1:
            results.append({"n_sectors": n_sec, "error": "insufficient profiles"})
            continue
        mean = train.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(train - mean, full_matrices=False)
        per_mode = []
        for k in modes:
            k_eff = int(min(k, vt.shape[0]))
            basis = vt[:k_eff]
            recon = ((test - mean) @ basis.T) @ basis + mean
            r2 = [_r2_score(test[i], recon[i]) for i in range(test.shape[0])]
            per_mode.append({"n_modes": k_eff, "test_mean_r2": _safe_json_float(float(np.nanmean(r2)))})
        results.append({
            "n_sectors": n_sec,
            "dims": grid_s * n_sec,
            "n_train": int(train.shape[0]),
            "n_test": int(test.shape[0]),
            "modes": per_mode,
        })
    return {
        "grid_s": grid_s,
        "results": results,
        "interpretation": (
            "n_sectors=1 应≈ oracle_pod 1D；若 n_sectors=4/8 的 test R² 明显高于 1D，"
            "说明周向结构含可学信号，TODO-52 应做 2D 展开而非 1D（θ 跨病例仅近似对齐，"
            "为单边保守测试：升高才结论性）"
        ),
    }


def oracle_residual_stratification(run_dir: Path, split_path: Path) -> Dict[str, object]:
    split = SplitSpec.from_json(split_path)
    rows = _read_csv_rows(_eval_paths(run_dir)["wss_case_metrics"])
    bc_by_case: Dict[str, float] = {}
    for case_rel in split.test_cases:
        case_dir = REPO_ROOT / "data_new" / "AG" / case_rel / "processed" / "graphs"
        if not case_dir.is_dir():
            continue
        graph_path = next(iter(sorted(case_dir.glob("*.pt"))), None)
        if graph_path is None:
            continue
        data = _load_graph(graph_path)
        bc_by_case[case_rel] = float(data.global_cond.view(-1)[1].item())

    enriched = []
    for row in rows:
        case_name = row.get("case_name", "")
        case_rel = case_name if "/" in case_name else case_name
        r2_raw = row.get("wss_mag_r2", "")
        if not r2_raw:
            continue
        enriched.append({
            "case_name": case_name,
            "pace_group": case_name.split("/")[0] if "/" in case_name else "unknown",
            "bc_inlet": bc_by_case.get(case_name) or bc_by_case.get(case_rel),
            "wss_mag_r2": float(r2_raw),
        })

    def _group_stats(key_fn):
        groups: Dict[str, List[float]] = defaultdict(list)
        for item in enriched:
            key = key_fn(item)
            if key is None:
                continue
            groups[str(key)].append(item["wss_mag_r2"])
        return {
            k: {"n": len(v), "mean_r2": float(np.mean(v)), "std_r2": float(np.std(v)) if len(v) > 1 else 0.0}
            for k, v in sorted(groups.items())
        }

    bc_vals = [e["bc_inlet"] for e in enriched if e["bc_inlet"] is not None]
    bc_median = float(np.median(bc_vals)) if bc_vals else None
    return {
        "by_pace_group": _group_stats(lambda e: e["pace_group"]),
        "by_bc_inlet_median": _group_stats(
            lambda e: ("high" if e["bc_inlet"] is not None and bc_median is not None and e["bc_inlet"] >= bc_median else "low")
            if e["bc_inlet"] is not None and bc_median is not None else None
        ),
        "interpretation": "systematic group gaps ⇒ consider TODO-31/37; flat ⇒ point-level R² likely at ceiling",
    }


def _recommend_next_steps(report: Dict[str, object]) -> Dict[str, object]:
    seed_summary = report.get("todo_27a_seed", {}).get("summary", {})
    mean_r2 = float(seed_summary.get("mean", float("nan")))
    ci_lo = float(seed_summary.get("ci95_lo", mean_r2))
    ci_hi = float(seed_summary.get("ci95_hi", mean_r2))

    pod = report.get("oracle", {}).get("pod", {})
    pod_test = None
    if isinstance(pod.get("modes"), list) and pod["modes"]:
        pod_test = max(float(m.get("test_mean_r2", float("nan"))) for m in pod["modes"])

    # 旧 naive 变体（仅作对照，复现 −0.34）
    gt = report.get("oracle", {}).get("gt_wall_diff", {})
    gt_pearson_naive = gt.get("mean_case_pearson")
    # 修正版：取 tang_normal 变体作为判据
    gt_v2 = report.get("oracle", {}).get("gt_wall_diff_v2", {})
    gt_tn = gt_v2.get("variant_summary", {}).get("tang_normal", {}) or {}
    gt_pearson = gt_tn.get("mean_pearson")

    # 压力梯度 ↔ WSS（思路 2）
    pg = report.get("oracle", {}).get("pressure_gradient", {})
    pg_corrs = [pg.get("mean_spearman_neg_dpds"), pg.get("mean_spearman_gradmag")]
    pg_corr = max((abs(c) for c in pg_corrs if c is not None), default=None)

    # 2D POD vs 1D：周向是否含可学结构
    pod2d = report.get("oracle", {}).get("pod_2d", {})
    pod2d_1d = None
    pod2d_multi = None
    for res in pod2d.get("results", []) or []:
        modes = res.get("modes") or []
        if not modes:
            continue
        best = max((float(m.get("test_mean_r2", float("nan"))) for m in modes
                    if m.get("test_mean_r2") is not None), default=float("nan"))
        if res.get("n_sectors") == 1:
            pod2d_1d = best
        elif res.get("n_sectors", 0) >= 4:
            pod2d_multi = best if pod2d_multi is None else max(pod2d_multi, best)

    strat = report.get("oracle", {}).get("residual_stratification", {})
    pace = strat.get("by_pace_group", {})
    pace_gap = 0.0
    if len(pace) >= 2:
        means = [v["mean_r2"] for v in pace.values()]
        pace_gap = float(max(means) - min(means))

    triggers: List[str] = []
    blockers: List[str] = []

    if math.isfinite(ci_hi) and ci_hi <= 0.42:
        blockers.append("三 seed 带宽 CI 上界 ≤0.42：点级 R² 提升空间有限")
    if pace_gap >= 0.08:
        triggers.append("残差按 fast/slow 分层明显：优先 TODO-31（Cf/Re/Wo）或 TODO-37（FiLM BC）")
    if gt_pearson is not None and gt_pearson >= 0.4:
        triggers.append(f"GT 近壁差分(修正 tang_normal) 与 WSS 相关={gt_pearson:.2f}：评估 TODO-30/33")
    elif gt_pearson is not None:
        blockers.append(f"GT 近壁差分(修正)弱({gt_pearson:.2f})：TODO-30/33 降级（注意区分 oracle 实现与物理）")
    if pg_corr is not None and pg_corr >= 0.3:
        triggers.append(f"壁面压力梯度与 WSS 相关={pg_corr:.2f}：可做压力→WSS 耦合（思路 2）")
    if pod_test is not None and math.isfinite(pod_test) and pod_test >= 0.45:
        triggers.append("POD(1D) 重建 test R² 较高：TODO-52 候选（须看 2D）")
    elif pod_test is not None and math.isfinite(pod_test):
        blockers.append("POD(1D) 低秩上限一般：优先临床区域叙事 (TODO-53/42)")
    if (pod2d_1d is not None and pod2d_multi is not None
            and math.isfinite(pod2d_1d) and math.isfinite(pod2d_multi)
            and pod2d_multi - pod2d_1d >= 0.05):
        triggers.append(
            f"POD 2D(+周向) R² {pod2d_multi:.2f} > 1D {pod2d_1d:.2f}：TODO-52 应做 2D 展开")

    probe = None
    if any("TODO-30" in t for t in triggers):
        probe = "TODO-30/33 差分推 WSS / 边界层斜率（修正 oracle 已通过）"
    elif any("思路 2" in t for t in triggers):
        probe = "压力→WSS 耦合（思路 2，单变量 Probe）"
    elif any("TODO-52" in t for t in triggers):
        probe = "TODO-52 POD（优先 2D 展开）"
    elif any("TODO-31" in t for t in triggers):
        probe = "TODO-31 Cf/Re/Wo"
    elif any("TODO-37" in t for t in triggers):
        probe = "TODO-37 FiLM"
    elif blockers and not triggers:
        probe = "TODO-42 高 WSS 排序 loss / 临床叙事"
    else:
        probe = "TODO-42 区域叙事（默认；结构性证据不足时）"

    return {
        "asymw_bandwidth_mean": mean_r2,
        "asymw_bandwidth_ci95": [ci_lo, ci_hi],
        "gt_wall_diff_naive_pearson": gt_pearson_naive,
        "gt_wall_diff_v2_tang_normal_pearson": gt_pearson,
        "pressure_gradient_abs_spearman": pg_corr,
        "pod_1d_best_test_r2": pod_test,
        "pod_2d_1sector_r2": pod2d_1d,
        "pod_2d_multisector_r2": pod2d_multi,
        "triggers": triggers,
        "blockers": blockers,
        "recommended_single_probe": probe,
        "ensemble_status": report.get("todo_32", {}).get("predictions_missing_runs", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 Path F0 decision layer (0 retrain)")
    parser.add_argument("--field-root", type=Path, default=REPO_ROOT / "outputs" / "field")
    parser.add_argument("--run-glob", default=DEFAULT_ASYMW_GLOB)
    parser.add_argument("--split", type=Path, default=REPO_ROOT / "training" / "splits" / "split_AG_v1.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "field" / "f0_decision")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pod-grid-size", type=int, default=64)
    parser.add_argument("--pod-max-graphs", type=int, default=2)
    parser.add_argument("--pod-modes", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--norm-params", type=Path, default=DEFAULT_NORM_PARAMS,
                        help="归一化统计量 JSON（oracle 反归一化用）")
    parser.add_argument("--oracle-max-cases", type=int, default=8)
    parser.add_argument("--oracle-max-graphs", type=int, default=1)
    parser.add_argument("--pod2d-sectors", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--output-name", default="v3_f0_oracle_v2.json",
                        help="输出文件名（默认不覆盖原 v3_f0_asymw_decision.json）")
    args = parser.parse_args()

    runs = _discover_asymw_runs(args.field_root.resolve(), args.run_glob)
    if not runs:
        raise SystemExit(f"未找到 AsymW-a run: {args.field_root}/{args.run_glob}")

    out_dir = ensure_dir(args.output_dir.resolve())
    report: Dict[str, object] = {
        "label": "V3P-F0-AsymW-a",
        "split": str(args.split.resolve()),
        "runs": [str(r) for r in runs],
        "todo_27a_seed": todo_27a_seed_bandwidth(runs),
        "todo_27a_bootstrap": todo_27a_bootstrap(runs, n_bootstrap=args.n_bootstrap, seed=args.seed),
        "todo_53": [todo_53_clinical_extended(r) for r in runs],
        "todo_32": todo_32_ensemble_status(runs),
    }

    if not args.skip_oracle:
        norm_stats = _load_norm_stats(args.norm_params.resolve())
        split_path = args.split.resolve()
        report["oracle"] = {
            "pod": oracle_pod(
                split_path,
                grid_size=args.pod_grid_size,
                max_graphs_per_case=args.pod_max_graphs,
                modes=args.pod_modes,
            ),
            "pod_2d": oracle_pod_2d(
                split_path,
                norm_stats,
                grid_s=args.pod_grid_size,
                sectors=args.pod2d_sectors,
                modes=args.pod_modes,
                max_graphs_per_case=args.pod_max_graphs,
            ),
            "gt_wall_diff": oracle_gt_wall_diff(
                split_path,
                max_cases=args.oracle_max_cases,
                max_graphs_per_case=args.oracle_max_graphs,
            ),
            "gt_wall_diff_v2": oracle_gt_wall_diff_v2(
                split_path,
                norm_stats,
                max_cases=args.oracle_max_cases,
                max_graphs_per_case=args.oracle_max_graphs,
            ),
            "pressure_gradient": oracle_pressure_gradient_wss(
                split_path,
                norm_stats,
                max_cases=args.oracle_max_cases,
                max_graphs_per_case=args.oracle_max_graphs,
            ),
            "residual_stratification": oracle_residual_stratification(runs[0], split_path),
            "norm_params": str(args.norm_params.resolve()),
        }

    report["recommendation"] = _recommend_next_steps(report)
    out_path = out_dir / args.output_name
    save_json(out_path, report)
    print(json.dumps(report["recommendation"], indent=2, ensure_ascii=False))
    print(out_path)


if __name__ == "__main__":
    main()
