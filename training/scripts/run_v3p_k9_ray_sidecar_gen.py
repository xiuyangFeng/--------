#!/usr/bin/env python3
"""V3P K9 · 剖面射线训练目标 sidecar 生成器（前沿方向 §15.2 · Gate-0 Go 后落地）。

Gate-0 审计（5934 v1 / 5935 v2）已判 Go：k=16 全样本拟合的逐点 GT ``(a1_s, a1_c)``
覆盖率 1.000、split-half+Spearman-Brown 校正稳定性 0.84/0.88、cross 对齐
calibrated R² 0.704。本脚本把审计逻辑变成**目标生成器**：对每个图的每个壁面点
（不抽样）拟合 ``u_t(η)=a1·η+a2·η²``，连同局部 frame 存成逐图 sidecar。

sidecar（`<case>/processed/ray_targets_k{K}/<graph>.pt`，与图同 stem，节点对齐）::

    ray_a1     (N, 2) float32  [μ·a1_s, μ·a1_c]（Pa；无效行=0）
    ray_ts     (N, 3) float32  t̂_s（无效行/内部点 = 归一化 Tangent_XYZ 兜底）
    ray_tc     (N, 3) float32  t̂_c = n̂ × t̂_s（无效行/内部点 = 0，退化为 K7 口径）
    ray_valid  (N,)   bool     该行 a1 目标可用于 L_ray

frame 只依赖图几何（坐标 + is_wall + 中心线切向），推理期可得（E-J 纪律）；
a1 目标用 GT 速度拟合，只在训练期作辅助监督（L_ray），推理期零依赖。

汇总 JSON（f0_decision）额外给出 train-split 池化的 a1 统计，其中
``a1_scale_pooled_std``（两通道共享标准差）供训练配置 `ray_target_scale` 使用——
共享尺度不扭曲 (a1_s, a1_c) 的方向组成（K8 双分量头要按 frame 合成矢量）。

用法::

    python -m training.scripts.run_v3p_k9_ray_sidecar_gen --smoke
    python -m training.scripts.run_v3p_k9_ray_sidecar_gen --workers 8
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

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
    _load_graph,
    _load_norm_stats,
)
from .run_v3p_profile_wss_oracle import (
    MM_TO_M,
    MU_BLOOD,
    _fit_profile_a1,
    _safe_json,
    _unit_rows,
)

SIDECAR_VERSION = "k9_sidecar_v1"


def _graph_sidecar(
    data,
    stats: Mapping[str, Dict[str, float]],
    *,
    k_full: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """单图全壁面点拟合，返回节点对齐的 sidecar 张量；图不可用时返回 None。"""
    from scipy.spatial import cKDTree

    x = data.x.numpy()
    n_nodes = x.shape[0]
    wall = x[:, NODE_IDX["is_wall"]] > 0.5
    interior = ~wall
    if int(wall.sum()) < 50 or int(interior.sum()) < 50:
        return None

    tan_all = _unit_rows(x[:, X_TAN].astype(np.float64))
    pos_i = x[interior, :3].astype(np.float64)
    vel_i = _denorm_velocity(data.y.numpy()[interior, Y_VEL], stats)

    # 兜底 frame：t_s = 归一化中心线切向，t_c = 0（等价 K7 单标量口径的退化）
    ray_a1 = np.zeros((n_nodes, 2), dtype=np.float64)
    ray_ts = tan_all.copy()
    ray_tc = np.zeros((n_nodes, 3), dtype=np.float64)
    ray_valid = np.zeros(n_nodes, dtype=bool)

    wall_idx = np.flatnonzero(wall)
    tree = cKDTree(pos_i)
    _, nbrs = tree.query(x[wall_idx, :3].astype(np.float64), k=k_full)

    for row, wi in enumerate(wall_idx):
        nb = np.atleast_1d(nbrs[row])
        rel = pos_i[nb] - x[wi, :3].astype(np.float64)     # (k,3) mm
        n_hat = rel.mean(axis=0)
        n_norm = float(np.linalg.norm(n_hat))
        if n_norm < 1e-9:
            continue
        n_hat = n_hat / n_norm
        t_s = tan_all[wi] - float(tan_all[wi] @ n_hat) * n_hat
        t_norm = float(np.linalg.norm(t_s))
        if t_norm < 1e-6:
            continue
        t_s = t_s / t_norm
        t_c = np.cross(n_hat, t_s)

        eta = (rel @ n_hat) * MM_TO_M                       # m
        valid = eta > 1e-7
        if int(valid.sum()) < 3:
            # frame 可用但拟合样本不足：保留 frame，目标置无效
            ray_ts[wi] = t_s
            ray_tc[wi] = t_c
            continue
        eta_v = eta[valid]
        v_nb = vel_i[nb][valid]
        v_t = v_nb - np.outer(v_nb @ n_hat, n_hat)
        a1_s = _fit_profile_a1(eta_v, v_t @ t_s)
        a1_c = _fit_profile_a1(eta_v, v_t @ t_c)

        ray_ts[wi] = t_s
        ray_tc[wi] = t_c
        if a1_s is None or a1_c is None:
            continue
        ray_a1[wi, 0] = MU_BLOOD * a1_s
        ray_a1[wi, 1] = MU_BLOOD * a1_c
        ray_valid[wi] = True

    if int(ray_valid.sum()) < 20:
        return None
    return {
        "ray_a1": torch.from_numpy(ray_a1.astype(np.float32)),
        "ray_ts": torch.from_numpy(ray_ts.astype(np.float32)),
        "ray_tc": torch.from_numpy(ray_tc.astype(np.float32)),
        "ray_valid": torch.from_numpy(ray_valid),
        "n_wall": int(wall.sum()),
    }


def _process_case(
    case_rel: str,
    *,
    data_root: str,
    graphs_subdir: str,
    sidecar_subdir: str,
    norm_params: str,
    k_full: int,
    max_graphs: int,
    overwrite: bool,
) -> Dict[str, Any]:
    """处理单病例的全部图；供 multiprocessing.Pool 调用（进程内自载 norm stats）。"""
    stats = _load_norm_stats(Path(norm_params))
    case_dir = Path(data_root) / case_rel / "processed" / graphs_subdir
    out_dir = Path(data_root) / case_rel / "processed" / sidecar_subdir
    result: Dict[str, Any] = {
        "case": case_rel, "n_graphs": 0, "n_skipped_graphs": 0,
        "n_wall": 0, "n_valid": 0,
        "sum_s": 0.0, "sumsq_s": 0.0, "sum_c": 0.0, "sumsq_c": 0.0,
    }
    if not case_dir.is_dir():
        result["error"] = "no-graphs-dir"
        return result
    graph_paths = sorted(case_dir.glob("*.pt"))
    if max_graphs > 0:
        graph_paths = graph_paths[:max_graphs]
    out_dir.mkdir(parents=True, exist_ok=True)
    for gp in graph_paths:
        out_path = out_dir / gp.name
        if out_path.exists() and not overwrite:
            side = torch.load(out_path, map_location="cpu", weights_only=True)
        else:
            side = _graph_sidecar(_load_graph(gp), stats, k_full=k_full)
            if side is None:
                result["n_skipped_graphs"] += 1
                continue
            side["meta"] = {
                "version": SIDECAR_VERSION, "k_full": k_full,
                "mu_Pa_s": MU_BLOOD, "basis": "a1*eta + a2*eta^2",
                "n_wall": side.pop("n_wall"),
            }
            torch.save(side, out_path)
        valid = side["ray_valid"].numpy()
        a1 = side["ray_a1"].numpy()[valid].astype(np.float64)
        result["n_graphs"] += 1
        result["n_wall"] += int(side["meta"]["n_wall"])
        result["n_valid"] += int(valid.sum())
        result["sum_s"] += float(a1[:, 0].sum())
        result["sumsq_s"] += float((a1[:, 0] ** 2).sum())
        result["sum_c"] += float(a1[:, 1].sum())
        result["sumsq_c"] += float((a1[:, 1] ** 2).sum())
    return result


def _moments(n: int, s: float, sq: float) -> Dict[str, Optional[float]]:
    if n < 2:
        return {"mean": None, "std": None}
    mean = s / n
    var = max(sq / n - mean * mean, 0.0)
    return {"mean": _safe_json(mean), "std": _safe_json(float(np.sqrt(var)))}


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P K9 ray-target sidecar generator")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training/splits/split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new/AG")
    ap.add_argument("--graphs-subdir", default="graphs")
    ap.add_argument("--norm-params", type=Path, default=DEFAULT_NORM_PARAMS)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/field/f0_decision")
    ap.add_argument("--date-suffix", default="")
    ap.add_argument("--k-full", type=int, default=16)
    ap.add_argument("--max-graphs-per-case", type=int, default=0, help="0=全部图")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true", help="重算已存在的 sidecar")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.max_graphs_per_case = 2

    t0 = time.perf_counter()
    split = SplitSpec.from_json(args.split.resolve())
    train_cases = list(split.train_cases)
    # 训练/验证/测试三个 dataset 都要 frame——val_cases 不能漏（与审计脚本 train+test 口径不同）
    all_cases = train_cases + list(split.val_cases) + list(split.test_cases)
    if args.smoke:
        all_cases = all_cases[:4]
    sidecar_subdir = f"ray_targets_k{args.k_full}"
    suffix = args.date_suffix or date.today().strftime("%Y%m%d")

    worker = partial(
        _process_case,
        data_root=str(args.data_root.resolve()),
        graphs_subdir=args.graphs_subdir,
        sidecar_subdir=sidecar_subdir,
        norm_params=str(args.norm_params.resolve()),
        k_full=args.k_full,
        max_graphs=args.max_graphs_per_case,
        overwrite=args.overwrite,
    )
    results: List[Dict[str, Any]] = []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(worker, all_cases), 1):
                results.append(res)
                print(
                    f"[k9-sidecar] [{i}/{len(all_cases)}] {res['case']} graphs={res['n_graphs']} "
                    f"valid={res['n_valid']}/{res['n_wall']}", flush=True,
                )
    else:
        for i, case_rel in enumerate(all_cases, 1):
            res = worker(case_rel)
            results.append(res)
            print(
                f"[k9-sidecar] [{i}/{len(all_cases)}] {res['case']} graphs={res['n_graphs']} "
                f"valid={res['n_valid']}/{res['n_wall']}", flush=True,
            )

    by_case = {r["case"]: r for r in results}
    train_res = [by_case[c] for c in train_cases if c in by_case]
    n_tr = sum(r["n_valid"] for r in train_res)
    sum_s = sum(r["sum_s"] for r in train_res)
    sumsq_s = sum(r["sumsq_s"] for r in train_res)
    sum_c = sum(r["sum_c"] for r in train_res)
    sumsq_c = sum(r["sumsq_c"] for r in train_res)
    stats_s = _moments(n_tr, sum_s, sumsq_s)
    stats_c = _moments(n_tr, sum_c, sumsq_c)
    # 两通道共享尺度：pooled 二阶矩（不减均值，保持方向组成不被扭曲）
    pooled_std = (
        float(np.sqrt((sumsq_s + sumsq_c) / (2 * n_tr))) if n_tr >= 2 else None
    )

    n_wall_all = sum(r["n_wall"] for r in results)
    n_valid_all = sum(r["n_valid"] for r in results)
    report = {
        "label": "v3p_k9_ray_sidecar_gen",
        "date": suffix,
        "context": "V3P · K9 剖面射线目标 sidecar 生成（§15.2 · Gate-0 Go 5935_v2 后落地）",
        "config": {
            "split": str(args.split),
            "data_root": str(args.data_root),
            "sidecar_subdir": f"processed/{sidecar_subdir}",
            "k_full": args.k_full,
            "mu_Pa_s": MU_BLOOD,
            "profile_basis": "a1*eta + a2*eta^2",
            "max_graphs_per_case": args.max_graphs_per_case,
            "version": SIDECAR_VERSION,
            "smoke": args.smoke,
        },
        "n_cases": len(results),
        "n_graphs": sum(r["n_graphs"] for r in results),
        "n_skipped_graphs": sum(r["n_skipped_graphs"] for r in results),
        "coverage_valid_frac": _safe_json(n_valid_all / max(n_wall_all, 1)),
        "n_wall_points": n_wall_all,
        "n_valid_points": n_valid_all,
        "train_a1_stats_Pa": {
            "n_points": n_tr,
            "stream": stats_s,
            "cross": stats_c,
            "a1_scale_pooled_std": _safe_json(pooled_std),
            "note": "训练配置 data.ray_target_scale 取 a1_scale_pooled_std（两通道共享，保方向）",
        },
        "per_case": {
            r["case"]: {k: r[k] for k in ("n_graphs", "n_skipped_graphs", "n_wall", "n_valid")}
            for r in results
        },
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    out_path = ensure_dir(args.output_dir.resolve()) / f"v3p_k9_ray_sidecar_gen_{suffix}.json"
    save_json(out_path, report)
    print(json.dumps({
        "output": str(out_path),
        "n_graphs": report["n_graphs"],
        "coverage_valid_frac": report["coverage_valid_frac"],
        "train_a1_stats_Pa": report["train_a1_stats_Pa"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
