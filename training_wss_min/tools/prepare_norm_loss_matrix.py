#!/usr/bin/env python3
"""Prepare the Q2V normalization-口径 / tail-loss single-seed matrix.

单一自变量筛查矩阵，锚点为 Q2V-10477（pool2025 train138/test36，通用模型含 ILO）。
A0 复用已训练的 ``q2v_ilo_d3_pool2025_refit``（point-pooled train138 统计），本脚本
只生成新增 5 个 arm 所需的两份"平衡口径"目标统计与配置 + manifest：

- A1  casebal_mse            : A0 但目标统计改为 **逐病例等权** log mean/std
- A1b cohortbal_mse          : A0 但目标统计改为 **逐父系(AG/AAA/ILO)等权** log mean/std
- A2  casebal_cohortfeat     : A1 + cohort one-hot 输入特征（input_dim 6->9）
- A3  casebal_rawhuber       : A1 + 物理空间 raw-Huber 尾部项 λ=0.2
- A4  casebal_cohortfeat_rawhuber : A1 + cohort one-hot + raw-Huber λ=0.2

平衡口径定义（log 空间）：
  逐病例：mu = mean_i(m_i)；var = mean_i( v_i + (m_i-mu)^2 )，其中 m_i/v_i 为该病例
          log(WSS+eps) 的均值/总体方差；std=sqrt(var)。
  逐父系：先在父系内对病例等权求均值，再在父系间等权。
raw_percentiles 仍取 pooled（仅作 raw-Huber 的兜底；train.py 会用 train 病例 p90）。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data_wss_min"
FRAME = "stl_landmarks_v4"
EPS = 1e-6

POOL_SPLIT = REPO / "training/splits/split_AG_AAA_ILO_q2v_pool2025_train138_test36.json"
POOL_REFIT_STATS = DATA_ROOT / "fold_stats/q2v_ilo_20260718/q2v_pool2025_train138_global_stats.json"
A0_CONFIG = REPO / "training_wss_min/configs/pointnetpp_q2v_ilo_20260718/q2v_ilo_d3_pool2025_refit.json"
A0_RUN = REPO / "training_wss_min/runs/pointnetpp_q2v_ilo/outputs/q2v_ilo_d3_pool2025_refit"

STATS_DIR = DATA_ROOT / "fold_stats/norm_loss_20260720"
CASEBAL_STATS = STATS_DIR / "q2v_pool2025_train138_casebal_global_stats.json"
COHORTBAL_STATS = STATS_DIR / "q2v_pool2025_train138_cohortbal_global_stats.json"

CONFIG_DIR = REPO / "training_wss_min/configs/pointnetpp_norm_loss_20260720"
MANIFEST = REPO / "training_wss_min/preflight/norm_loss_matrix_prepared.json"

COHORT_FEATURES = ["cohort_ag", "cohort_aaa", "cohort_ilo"]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def family(unit: str) -> str:
    return unit.split("/", 1)[0]


def per_case_wss(units: list[str]) -> dict[str, dict]:
    """逐病例读取 peak-step 壁面 WSS，返回 log 均值/总体方差/点数/pooled 缓存。"""
    out: dict[str, dict] = {}
    n = len(units)
    for i, unit in enumerate(units, 1):
        bp = DATA_ROOT / unit / "bundle.npz"
        with np.load(bp, allow_pickle=False) as z:
            frame = str(np.asarray(z["transform_frame_version"]).item())
            if frame != FRAME:
                raise RuntimeError(f"wrong frame: {unit}={frame}")
            steps = z["steps"].tolist()
            peak = int(np.asarray(z["peak_step"]).item())
            wss = np.asarray(z["wall_wss"][steps.index(peak)], dtype=np.float64).ravel()
        if not np.isfinite(wss).all() or (wss < 0).any() or (wss <= 0).mean() > 0.01:
            raise RuntimeError(f"WSS gate failed while computing stats: {unit}")
        logs = np.log(np.clip(wss, 0, None) + EPS)
        out[unit] = {
            "family": family(unit),
            "n_points": int(wss.size),
            "log_mean": float(logs.mean()),
            "log_var": float(logs.var()),  # 总体方差（within-case）
            "wss": wss.astype(np.float32),
            "peak_step": peak,
            "bundle_path": str(bp.resolve()),
            "bundle_sha256": sha256(bp),
        }
        print(f"[per-case {i:03d}/{n:03d}] {unit} n={wss.size} logμ={out[unit]['log_mean']:.4f}",
              flush=True)
    return out


def case_balanced(per_case: dict[str, dict]) -> tuple[float, float]:
    ms = np.array([c["log_mean"] for c in per_case.values()], dtype=np.float64)
    vs = np.array([c["log_var"] for c in per_case.values()], dtype=np.float64)
    mu = float(ms.mean())
    var = float((vs + (ms - mu) ** 2).mean())
    return mu, float(np.sqrt(max(var, 1e-12)))


def cohort_balanced(per_case: dict[str, dict]) -> tuple[float, float, dict]:
    by: dict[str, list[dict]] = {}
    for c in per_case.values():
        by.setdefault(c["family"], []).append(c)
    fams = sorted(by)
    fam_mu = {f: float(np.mean([c["log_mean"] for c in by[f]])) for f in fams}
    mu = float(np.mean([fam_mu[f] for f in fams]))
    fam_var = {}
    for f in fams:
        ms = np.array([c["log_mean"] for c in by[f]], dtype=np.float64)
        vs = np.array([c["log_var"] for c in by[f]], dtype=np.float64)
        fam_var[f] = float((vs + (ms - mu) ** 2).mean())
    var = float(np.mean([fam_var[f] for f in fams]))
    audit = {"families": fams, "family_log_mean": fam_mu,
             "family_n_cases": {f: len(by[f]) for f in fams}}
    return mu, float(np.sqrt(max(var, 1e-12))), audit


def pooled_extras(per_case: dict[str, dict]) -> dict:
    values = np.concatenate([c["wss"].astype(np.float64) for c in per_case.values()])
    logs = np.log(np.clip(values, 0, None) + EPS)
    return {
        "n_points": int(values.size),
        "pooled_linear": {"mean": float(values.mean()), "std": float(values.std())},
        "pooled_log": {"mean": float(logs.mean()), "std": float(logs.std())},
        "raw_min": float(values.min()), "raw_max": float(values.max()),
        "raw_percentiles": {f"p{q}": float(np.percentile(values, q)) for q in (50, 90, 95, 99)},
    }


def contribution_by_family(per_case: dict[str, dict], total_points: int) -> dict:
    agg: dict[str, dict] = {}
    for c in per_case.values():
        d = agg.setdefault(c["family"], {"n_cases": 0, "n_points": 0})
        d["n_cases"] += 1
        d["n_points"] += c["n_points"]
    for f, d in agg.items():
        d["fraction"] = d["n_points"] / total_points
    return agg


def build_stats(units: list[str], per_case: dict[str, dict], balancing: str) -> dict:
    if balancing == "case_balanced":
        mu, std = case_balanced(per_case)
        bal_audit = {"method": "per-case equal weight"}
    elif balancing == "cohort_balanced":
        mu, std, bal_audit = cohort_balanced(per_case)
        bal_audit = {"method": "per-family(AG/AAA/ILO) equal weight", **bal_audit}
    else:
        raise ValueError(balancing)
    extras = pooled_extras(per_case)
    rows = [{"unit_id": u, "family": per_case[u]["family"], "n_points": per_case[u]["n_points"],
             "log_mean": per_case[u]["log_mean"], "log_var": per_case[u]["log_var"],
             "peak_step": per_case[u]["peak_step"], "bundle_path": per_case[u]["bundle_path"],
             "bundle_sha256": per_case[u]["bundle_sha256"]} for u in units]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "log_z",
        "eps": EPS,
        "required_frame_version": FRAME,
        "timesteps_scope": "peak",
        "statistics_scope": "train partition only",
        "balancing": balancing,
        "balancing_detail": bal_audit,
        "split_path": str(POOL_SPLIT.resolve()),
        "split_sha256": sha256(POOL_SPLIT),
        "n_cases": len(units),
        "n_points": extras["n_points"],
        # log.mean/std 为 normalize 实际使用的平衡口径量
        "log": {"mean": mu, "std": std},
        # linear/pooled_* 仅作记录，log_z 归一化不使用
        "linear": extras["pooled_linear"],
        "pooled_log": extras["pooled_log"],
        "raw_min": extras["raw_min"], "raw_max": extras["raw_max"],
        "raw_percentiles": extras["raw_percentiles"],
        "point_contribution_by_stratum": contribution_by_family(per_case, extras["n_points"]),
        "train_units": units,
        "bundle_manifest": rows,
    }


def clone_arm(base: dict, *, exp_id: str, note: str, stats_path: Path,
              add_cohort_feat: bool, raw_huber_lambda: float) -> dict:
    cfg = copy.deepcopy(base)
    cfg["name"] = f"pointnetpp_norm_loss/outputs/{exp_id}"
    cfg["notes"] = note
    cfg["data"]["split_path"] = str(POOL_SPLIT.resolve())
    cfg["data"]["wss_stats_path"] = str(stats_path.resolve())
    cfg["data"]["feature_stats_path"] = None
    feats = ["x", "y", "z", "abscissa_norm", "local_radius", "curvature"]
    if add_cohort_feat:
        feats = feats + list(COHORT_FEATURES)
    cfg["data"]["input_features"] = feats
    cfg["train"]["loss_raw_huber_lambda"] = raw_huber_lambda
    cfg["train"]["raw_huber_delta"] = 1.0
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO)
    args = ap.parse_args()
    if args.repo.resolve() != REPO.resolve():
        raise RuntimeError(f"pinned to {REPO}, got {args.repo}")
    for req in (POOL_SPLIT, POOL_REFIT_STATS, A0_CONFIG):
        if not req.is_file():
            raise FileNotFoundError(req)

    split = json.loads(POOL_SPLIT.read_text(encoding="utf-8"))
    units = list(split["train_cases"])
    if len(units) != 138:
        raise RuntimeError(f"expected pool2025 train138, got {len(units)}")

    per_case = per_case_wss(units)
    casebal = build_stats(units, per_case, "case_balanced")
    cohortbal = build_stats(units, per_case, "cohort_balanced")
    write_json(CASEBAL_STATS, casebal)
    write_json(COHORTBAL_STATS, cohortbal)
    print(f"[stats] casebal log(μ={casebal['log']['mean']:.4f}, σ={casebal['log']['std']:.4f}) "
          f"vs pooled(μ={casebal['pooled_log']['mean']:.4f}, σ={casebal['pooled_log']['std']:.4f})",
          flush=True)
    print(f"[stats] cohortbal log(μ={cohortbal['log']['mean']:.4f}, σ={cohortbal['log']['std']:.4f})",
          flush=True)

    base = json.loads(A0_CONFIG.read_text(encoding="utf-8"))
    arms = (
        ("norm_loss_a1_casebal_mse",
         "A1: A0 baseline but per-case equal-weight log-z target stats", CASEBAL_STATS, False, 0.0),
        ("norm_loss_a1b_cohortbal_mse",
         "A1b: A0 baseline but per-family(AG/AAA/ILO) equal-weight log-z target stats", COHORTBAL_STATS, False, 0.0),
        ("norm_loss_a2_casebal_cohortfeat",
         "A2: A1 casebal + cohort one-hot input features (input_dim 6->9)", CASEBAL_STATS, True, 0.0),
        ("norm_loss_a3_casebal_rawhuber",
         "A3: A1 casebal + physical-space raw-Huber tail term lambda=0.2", CASEBAL_STATS, False, 0.2),
        ("norm_loss_a4_casebal_cohortfeat_rawhuber",
         "A4: A1 casebal + cohort one-hot + raw-Huber tail lambda=0.2", CASEBAL_STATS, True, 0.2),
    )
    config_rows = []
    for exp_id, note, stats_path, add_feat, lam in arms:
        path = CONFIG_DIR / f"{exp_id}.json"
        write_json(path, clone_arm(base, exp_id=exp_id, note=note, stats_path=stats_path,
                                   add_cohort_feat=add_feat, raw_huber_lambda=lam))
        config_rows.append({"experiment_id": exp_id, "config": str(path.resolve()),
                            "sha256": sha256(path)})

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared",
        "protocol": "q2v_pool2025_norm_loss_single_seed_5arm",
        "anchor": "Q2V-10477 / q2v_ilo_d3_pool2025_refit (point-pooled train138)",
        "surface_metric_mode": "legacy_vertex",
        "single_variable_design": {
            "A1_vs_A0": "target norm 口径: point-pooled -> per-case equal weight",
            "A1b_vs_A1": "per-case -> per-family equal weight",
            "A2_vs_A1": "add cohort one-hot input feature",
            "A3_vs_A1": "add physical-space raw-Huber tail (lambda=0.2)",
            "A4_vs_A1": "cohort one-hot + raw-Huber (interaction)",
        },
        "a0_reference": {
            "experiment_id": "q2v_ilo_d3_pool2025_refit",
            "config": str(A0_CONFIG.resolve()),
            "config_sha256": sha256(A0_CONFIG),
            "run_dir": str(A0_RUN.resolve()),
            "target_stats": str(POOL_REFIT_STATS.resolve()),
            "target_stats_sha256": sha256(POOL_REFIT_STATS),
            "note": "already trained; reused as A0 baseline, not resubmitted",
        },
        "stats": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in (CASEBAL_STATS, COHORTBAL_STATS)],
        "configs": config_rows,
    }
    write_json(MANIFEST, payload)
    print(json.dumps({"status": "prepared", "n_arms": len(config_rows),
                      "manifest": str(MANIFEST)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
