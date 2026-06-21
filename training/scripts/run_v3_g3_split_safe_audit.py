#!/usr/bin/env python3
"""G3 Phase 0 · split-safe 与泄漏审计（V3P · 0 重训 · CPU）。

对照 [V3P_路径G_G3预训练方案.md] §5 六条检查 + 数据就绪性，产出
``outputs/field/f0_decision/v3p_g3_split_safe_audit_<date>.json``。

用法::

    python -m training.scripts.run_v3_g3_split_safe_audit
    python -m training.scripts.run_v3_g3_split_safe_audit --max-cases-per-split 20
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from ..core.denylist import denylist_hit, filter_case_names, skipped_case_names
from ..core.splits import SplitSpec
from pipeline.config import COORD_NORMALIZED_DIR
from pipeline.export_gap_preprocess_queue import PREPROCESS_DENYLIST

REPO_ROOT = Path(__file__).resolve().parents[2]

# SSL 预训练允许的特征（几何 + BC；不含 WSS/速度/压力场标签）
SSL_ALLOWED_NODE_FEATURES = {
    "x", "y", "z",
    "Abscissa", "NormRadius", "Curvature",
    "Tangent_X", "Tangent_Y", "Tangent_Z",
    "is_wall",
    "dist_to_wall", "dist_to_bifurcation", "branch_id",
    "dR_ds", "torsion", "d_tangent_ds",
}
SSL_FORBIDDEN_TARGETS = {
    "wss", "wss_x", "wss_y", "wss_z",
    "wss_axial", "wss_circ", "wss_rad",
    "u", "v", "w", "p", "vel_mag",
}

NORM_ZSCORE_FEATURES = ("Curvature",)
NORM_MINMAX_FEATURES = ("NormRadius",)
NORM_SKIP_FEATURES = ("x", "y", "z")  # keep_unchanged；coord_normalized 已处理，不在 statistics 中
NORM_REL_ERR_WARN = 0.05
NORM_REL_ERR_FAIL = 0.15
NORM_CLOSER_MARGIN = 0.002  # stored 距 all 比距 train 近多少则判泄漏


def _status(*, fail: bool, warn: bool) -> str:
    if fail:
        return "FAIL"
    if warn:
        return "WARN"
    return "PASS"


def _rel_err(a: float, b: float) -> float:
    denom = max(abs(b), 1e-12)
    return abs(a - b) / denom


def _case_assets(data_root: Path, case_rel: str) -> Dict[str, Any]:
    case_dir = data_root / case_rel
    merged = case_dir / "processed" / "merged"
    graphs = case_dir / "processed" / "graphs"
    stl = list(case_dir.glob("*.stl"))
    n_merged = len(list(merged.glob("*.csv"))) if merged.is_dir() else 0
    n_graphs = len(list(graphs.glob("*.pt"))) if graphs.is_dir() else 0
    coord = case_dir / "processed" / COORD_NORMALIZED_DIR.split("/")[-1]
    if not coord.is_dir():
        coord = case_dir / "processed" / "coord_normalized"
    n_coord = len(list(coord.glob("result_features_*.csv"))) if coord.is_dir() else 0
    return {
        "has_stl": bool(stl),
        "n_merged_csv": n_merged,
        "n_graphs": n_graphs,
        "n_coord_csv": n_coord,
        "geometry_ready": bool(stl) and n_merged > 0,
        "graphs_ready": n_graphs > 0,
        "coord_ready": n_coord > 0,
    }


def _sample_feature_moments(
    cases: Sequence[str],
    data_root: Path,
    *,
    features: Sequence[str],
    max_cases: Optional[int] = None,
    max_files_per_case: int = 2,
    max_rows_per_file: int = 4000,
    rng: np.random.Generator,
) -> Dict[str, Dict[str, float]]:
    """从 coord_normalized CSV 采样估计 mean/std（原始量纲，未 z-score）。"""
    selected = list(cases)
    if max_cases is not None and len(selected) > max_cases:
        selected = list(rng.choice(selected, size=max_cases, replace=False))

    buckets: Dict[str, List[float]] = {f: [] for f in features}
    n_files = 0
    for case_rel in selected:
        coord_dir = data_root / case_rel / "processed" / "coord_normalized"
        if not coord_dir.is_dir():
            continue
        csv_files = sorted(coord_dir.glob("result_features_*.csv"))
        if not csv_files:
            continue
        if len(csv_files) <= max_files_per_case:
            pick = csv_files
        else:
            idx = rng.choice(len(csv_files), size=max_files_per_case, replace=False)
            pick = [csv_files[i] for i in sorted(idx)]

        for csv_path in pick:
            try:
                import pandas as pd

                df = pd.read_csv(csv_path, usecols=lambda c: c in set(features))
            except Exception:
                continue
            n_files += 1
            if len(df) > max_rows_per_file:
                df = df.sample(n=max_rows_per_file, random_state=int(rng.integers(0, 2**31)))
            for feat in features:
                if feat not in df.columns:
                    continue
                vals = df[feat].to_numpy(dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                if vals.size:
                    buckets[feat].extend(vals.tolist())

    out: Dict[str, Dict[str, float]] = {}
    for feat, vals in buckets.items():
        if not vals:
            out[feat] = {"mean": float("nan"), "std": float("nan"), "n": 0}
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out[feat] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "n": int(arr.size),
        }
    out["_meta"] = {"n_cases": len(selected), "n_files": n_files}
    return out


def _load_stored_norm(norm_path: Path) -> Dict[str, Any]:
    raw = json.loads(norm_path.read_text(encoding="utf-8"))
    stats = raw.get("statistics", {})
    out: Dict[str, Any] = {"_raw_groups": raw.get("feature_groups", {})}
    for feat in NORM_ZSCORE_FEATURES:
        if feat in stats:
            out[feat] = {
                "mean": float(stats[feat]["mean"]),
                "std": float(stats[feat]["std"]),
            }
    for feat in NORM_MINMAX_FEATURES:
        if feat in stats:
            out[feat] = {
                "min": float(stats[feat]["min"]),
                "max": float(stats[feat]["max"]),
            }
    out["_wss"] = {
        k: {"mean": float(stats[k]["mean"]), "std": float(stats[k]["std"])}
        for k in ("wss", "wss_x", "wss_y", "wss_z")
        if k in stats
    }
    return out


def _collect_pipeline_stats(
    case_rels: Sequence[str],
    data_root: Path,
    *,
    restrict_to_train_list: bool,
) -> Dict[str, Dict[str, float]]:
    """调用 pipeline.normalize.collect_global_statistics（与产线一致）。"""
    from pipeline.normalize import collect_global_statistics

    case_dirs = [data_root / rel for rel in case_rels if (data_root / rel).is_dir()]
    if not case_dirs:
        return {}
    train_cases = list(case_rels) if restrict_to_train_list else None
    return collect_global_statistics(
        case_dirs,
        "processed/coord_normalized",
        train_cases=train_cases,
        data_root=data_root,
    )


def _compare_norm_blocks(
    stored: Mapping[str, Any],
    train_stats: Mapping[str, Dict[str, float]],
    all_stats: Mapping[str, Dict[str, float]],
) -> Tuple[List[Dict[str, Any]], bool, bool]:
    rows: List[Dict[str, Any]] = []
    any_fail = False
    any_warn = False

    for feat in NORM_SKIP_FEATURES:
        rows.append({
            "feature": feat,
            "status": "NOT_APPLICABLE",
            "note": "coordinates 组 keep_unchanged；SSL 直接用 coord_normalized 坐标，不经 global z-score",
        })

    for feat in NORM_ZSCORE_FEATURES:
        s = stored.get(feat, {})
        tr = train_stats.get(feat, {})
        al = all_stats.get(feat, {})
        if not s or not tr:
            rows.append({"feature": feat, "status": "WARN", "note": "缺少 stored 或 train 统计"})
            any_warn = True
            continue
        sm, ss = s["mean"], s["std"]
        trm, trs = tr["mean"], tr["std"]
        alm, als = al.get("mean", float("nan")), al.get("std", float("nan"))
        err_train_m = _rel_err(sm, trm)
        err_all_m = _rel_err(sm, alm) if np.isfinite(alm) else float("nan")
        err_train_s = _rel_err(ss, trs)
        closer_to_all = (
            np.isfinite(err_all_m)
            and (err_all_m + NORM_CLOSER_MARGIN) < err_train_m
        )
        row_fail = closer_to_all or err_train_m > NORM_REL_ERR_FAIL or err_train_s > NORM_REL_ERR_FAIL
        row_warn = (
            not row_fail
            and (err_train_m > NORM_REL_ERR_WARN or err_train_s > NORM_REL_ERR_WARN)
        )
        if row_fail:
            any_fail = True
        elif row_warn:
            any_warn = True
        rows.append(
            {
                "feature": feat,
                "metric": "z_score",
                "stored_mean": sm,
                "train_only_mean": trm,
                "all_active_mean": alm,
                "rel_err_train_mean": round(err_train_m, 6),
                "rel_err_all_mean": round(err_all_m, 6) if np.isfinite(err_all_m) else None,
                "rel_err_train_std": round(err_train_s, 6),
                "closer_to_all_than_train": closer_to_all,
                "status": _status(fail=row_fail, warn=row_warn),
            }
        )

    for feat in NORM_MINMAX_FEATURES:
        s = stored.get(feat, {})
        tr = train_stats.get(feat, {})
        al = all_stats.get(feat, {})
        if not s or not tr:
            rows.append({"feature": feat, "status": "WARN", "note": "缺少 stored 或 train 统计"})
            any_warn = True
            continue
        err_min = _rel_err(s["min"], tr["min"])
        err_max = _rel_err(s["max"], tr["max"])
        err_min_all = _rel_err(s["min"], al.get("min", float("nan"))) if al else float("nan")
        closer = np.isfinite(err_min_all) and (err_min_all + NORM_CLOSER_MARGIN) < err_min
        row_fail = closer or (
            err_min > NORM_REL_ERR_FAIL and err_max > NORM_REL_ERR_FAIL
        )
        row_warn = not row_fail and (err_min > NORM_REL_ERR_WARN or err_max > NORM_REL_ERR_WARN)
        if row_fail:
            any_fail = True
        elif row_warn:
            any_warn = True
        rows.append(
            {
                "feature": feat,
                "metric": "min_max",
                "stored_min": s["min"],
                "train_only_min": tr["min"],
                "rel_err_train_min": round(err_min, 6),
                "rel_err_train_max": round(err_max, 6),
                "closer_to_all_than_train": closer,
                "status": _status(fail=row_fail, warn=row_warn),
            }
        )
    return rows, any_fail, any_warn


def _check_split_disjoint(split: SplitSpec) -> Dict[str, Any]:
    tr, va, te = set(split.train_cases), set(split.val_cases), set(split.test_cases)
    overlap = {
        "train_val": sorted(tr & va),
        "train_test": sorted(tr & te),
        "val_test": sorted(va & te),
    }
    bad = any(overlap[k] for k in overlap)
    return {
        "id": 0,
        "name": "split_disjoint",
        "status": "FAIL" if bad else "PASS",
        "overlap": overlap,
        "counts": {"train": len(tr), "val": len(va), "test": len(te)},
    }


def _check_pretrain_scope(
    pretrain_cases: Sequence[str],
    test_cases: Sequence[str],
) -> Dict[str, Any]:
    overlap = sorted(set(pretrain_cases) & set(test_cases))
    return {
        "id": 1,
        "name": "pretrain_case_scope",
        "status": "FAIL" if overlap else "PASS",
        "n_pretrain_cases": len(pretrain_cases),
        "n_test_cases": len(test_cases),
        "test_in_pretrain": overlap,
        "rule": "SSL 预训练语料 = post-denylist train+val；test 零出现",
    }


def _check_norm_train_only(
    *,
    split: SplitSpec,
    data_root: Path,
    norm_path: Path,
    use_pipeline_collect: bool,
    max_cases_per_split: int,
    seed: int,
) -> Dict[str, Any]:
    active_train = filter_case_names(split.train_cases, data_root)
    active_val = filter_case_names(split.val_cases, data_root)
    active_test = filter_case_names(split.test_cases, data_root)
    all_active = active_train + active_val + active_test

    stored = _load_stored_norm(norm_path)
    split_notes = split.notes or ""
    denylist_renorm_skip = "不重算 normalization" in split_notes or "不重算 normalization_params" in split_notes

    if use_pipeline_collect:
        train_stats = _collect_pipeline_stats(active_train, data_root, restrict_to_train_list=True)
        all_stats = _collect_pipeline_stats(all_active, data_root, restrict_to_train_list=False)
        sampling_note = "pipeline.collect_global_statistics（全量 train / 全量 active）"
    else:
        rng = np.random.default_rng(seed)
        feats = NORM_ZSCORE_FEATURES + NORM_MINMAX_FEATURES
        train_stats = _sample_feature_moments(
            active_train, data_root, features=feats,
            max_cases=max_cases_per_split, rng=rng,
        )
        all_stats = _sample_feature_moments(
            all_active, data_root, features=feats,
            max_cases=max_cases_per_split, rng=rng,
        )
        sampling_note = f"CSV 采样 max_cases={max_cases_per_split}"

    feat_rows, any_fail, any_warn = _compare_norm_blocks(stored, train_stats, all_stats)
    all_feat_pass = all(r.get("status") == "PASS" for r in feat_rows if r.get("status") not in ("NOT_APPLICABLE",))
    if denylist_renorm_skip and not all_feat_pass:
        any_warn = True
    elif denylist_renorm_skip and all_feat_pass:
        any_warn = False

    leakage_fingerprint = any(r.get("closer_to_all_than_train") for r in feat_rows)
    if leakage_fingerprint:
        any_fail = True

    return {
        "id": 2,
        "name": "normalization_train_only",
        "status": _status(fail=any_fail, warn=any_warn and not any_fail),
        "norm_params": str(norm_path.resolve()),
        "split_notes_hint_no_renorm": denylist_renorm_skip,
        "collection_method": sampling_note,
        "features": feat_rows,
        "leakage_fingerprint": leakage_fingerprint,
        "ssl_action": (
            "① SSL 预训练：在 train-only 上 fit 几何统计（或重算 normalization）。"
            "② 微调 WSS：现网 JSON 含 denylist 前统计；post-denylist 公平对比前建议 "
            "`pipeline.normalize --train-split split_AG_v1` 重算。"
        ),
        "wss_stats_in_file": stored.get("_wss", {}),
    }


def _check_mask_sampling_policy() -> Dict[str, Any]:
    return {
        "id": 3,
        "name": "mask_sampling_policy",
        "status": "PASS",
        "mode": "design_time",
        "rule": "Point-MAE 类随机 mask 15–30% 点；禁止依赖 WSS 分布/误差图",
        "implementation": "待 Phase 1 ssl_tasks.py 落地时复审计",
    }


def _check_finetune_baseline(repo: Path) -> Dict[str, Any]:
    cfg = repo / "training/configs/field/generated/v3_pointcloud/V3P-G-Baseline-AsymW-a_seed1.json"
    ok = cfg.is_file()
    detail: Dict[str, Any] = {}
    if ok:
        raw = json.loads(cfg.read_text(encoding="utf-8"))
        detail = {
            "exp_id": raw.get("meta", {}).get("exp_id"),
            "split_file": raw.get("data", {}).get("split_file"),
            "path": str(cfg),
        }
    return {
        "id": 4,
        "name": "finetune_control_baseline",
        "status": "PASS" if ok else "FAIL",
        "detail": detail,
        "rule": "微调单变量：仅差 pretrained_ckpt；其余同 5439 配方",
    }


def _check_early_stop_val(repo: Path) -> Dict[str, Any]:
    cfg_path = repo / "training/configs/field/generated/v3_pointcloud/V3P-G-Baseline-AsymW-a_seed1.json"
    if not cfg_path.is_file():
        return {"id": 5, "name": "early_stop_val_only", "status": "FAIL", "note": "缺基线配置"}
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    optim = raw.get("optim", {})
    uses_val = optim.get("early_stopping_patience", 0) > 0
    return {
        "id": 5,
        "name": "early_stop_val_only",
        "status": "PASS" if uses_val else "WARN",
        "best_wss_checkpoint": "trainer 独立跟踪 val wss_r2_wss → best_wss_model.pt",
        "early_stopping_patience": optim.get("early_stopping_patience"),
        "val_score_ema_alpha": optim.get("val_score_ema_alpha"),
        "rule": "test 仅在最终 Probe 报告一次",
    }


def _check_synthetic_todo59() -> Dict[str, Any]:
    return {
        "id": 6,
        "name": "synthetic_low_fidelity_todo59",
        "status": "NOT_APPLICABLE",
        "rule": "TODO-58 首版不含合成/低保真 CFD",
    }


def _check_denylist_sync(
    split: SplitSpec,
    excluded_cases: Sequence[str],
    data_root: Path,
) -> Dict[str, Any]:
    missing_deny = [c for c in excluded_cases if not denylist_hit(c, data_root)]
    extra_hits = []
    for key in PREPROCESS_DENYLIST:
        if key.startswith("AG/"):
            rel = key[3:]
            if rel in split.train_cases or rel in split.val_cases or rel in split.test_cases:
                extra_hits.append(rel)
    status = "FAIL" if missing_deny or extra_hits else "PASS"
    return {
        "name": "denylist_split_sync",
        "status": status,
        "excluded_cases": list(excluded_cases),
        "excluded_not_in_denylist": missing_deny,
        "denylist_cases_still_in_split": extra_hits,
    }


def _check_geometry_pretrain_ready(
    pretrain_cases: Sequence[str],
    data_root: Path,
) -> Dict[str, Any]:
    rows = []
    not_ready = []
    for case in pretrain_cases:
        assets = _case_assets(data_root, case)
        rows.append({"case": case, **assets})
        if not (assets["geometry_ready"] and assets["coord_ready"] and assets["graphs_ready"]):
            not_ready.append(case)
    return {
        "name": "geometry_pretrain_ready",
        "status": "FAIL" if not_ready else "PASS",
        "n_pretrain": len(pretrain_cases),
        "n_not_ready": len(not_ready),
        "not_ready_cases": not_ready[:20],
    }


def _check_ssl_feature_policy() -> Dict[str, Any]:
    return {
        "name": "ssl_feature_policy",
        "status": "PASS",
        "allowed_node_features": sorted(SSL_ALLOWED_NODE_FEATURES),
        "forbidden_targets": sorted(SSL_FORBIDDEN_TARGETS),
        "rule": "预训练 loss 不得监督 forbidden 列；BC_Inlet 等全局条件可选",
    }


def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    split = SplitSpec.from_json(args.split.resolve())
    data_root = args.data_root.resolve()
    norm_path = args.norm_params.resolve()

    # excluded_cases from raw json (not on SplitSpec dataclass)
    split_raw = json.loads(args.split.read_text(encoding="utf-8"))
    split_excluded = split_raw.get("excluded_cases", [])

    active_train = filter_case_names(split.train_cases, data_root)
    active_val = filter_case_names(split.val_cases, data_root)
    active_test = filter_case_names(split.test_cases, data_root)
    pretrain_cases = active_train + active_val

    checks: List[Dict[str, Any]] = [
        _check_split_disjoint(split),
        _check_pretrain_scope(pretrain_cases, active_test),
        _check_norm_train_only(
            split=split,
            data_root=data_root,
            norm_path=norm_path,
            use_pipeline_collect=args.use_pipeline_collect,
            max_cases_per_split=args.max_cases_per_split,
            seed=args.seed,
        ),
        _check_mask_sampling_policy(),
        _check_finetune_baseline(REPO_ROOT),
        _check_early_stop_val(REPO_ROOT),
        _check_synthetic_todo59(),
    ]
    supplemental = [
        _check_denylist_sync(split, split_excluded, data_root),
        _check_geometry_pretrain_ready(pretrain_cases, data_root),
        _check_ssl_feature_policy(),
    ]

    statuses = [c["status"] for c in checks] + [s["status"] for s in supplemental]
    critical_fail = any(
        c["status"] == "FAIL"
        for c in checks
        if c["name"] in ("split_disjoint", "pretrain_case_scope")
    ) or any(s["status"] == "FAIL" for s in supplemental if s["name"] != "geometry_pretrain_ready")
    norm_check = next(c for c in checks if c["name"] == "normalization_train_only")
    overall = (
        "FAIL" if critical_fail
        else "WARN" if "FAIL" in statuses or "WARN" in statuses
        else "PASS"
    )
    phase0_ssl = not critical_fail and all(
        s["status"] == "PASS"
        for s in supplemental
        if s["name"] in ("geometry_pretrain_ready", "denylist_split_sync")
    ) and checks[1]["status"] == "PASS"

    skipped = sorted(skipped_case_names(
        list(split.train_cases) + list(split.val_cases) + list(split.test_cases),
        data_root,
    ))

    report = {
        "label": "V3P-G3-split-safe-audit",
        "created": str(date.today()),
        "split": str(args.split.resolve()),
        "data_root": str(data_root),
        "norm_params": str(norm_path),
        "overall_verdict": overall,
        "phase0_gate": overall == "PASS",
        "phase0_gate_ssl_pretrain": phase0_ssl,
        "phase0_gate_finetune_wss": overall == "PASS" and norm_check["status"] == "PASS",
        "case_counts": {
            "train_active": len(active_train),
            "val_active": len(active_val),
            "test_active": len(active_test),
            "pretrain_ssl": len(pretrain_cases),
            "denylist_skipped_in_split": len(skipped),
        },
        "checks": checks,
        "supplemental": supplemental,
        "interpretation": _interpret(overall, checks, supplemental),
    }
    return report


def _interpret(overall: str, checks: List[Dict[str, Any]], supplemental: List[Dict[str, Any]]) -> str:
    norm = next((c for c in checks if c["name"] == "normalization_train_only"), None)
    if overall == "PASS":
        return "Phase 0 全通过：可进入 G3 Phase 1 小子集 SSL 过拟合 + 微调 Probe。"
    if overall == "WARN":
        extra = ""
        if norm and norm.get("split_notes_hint_no_renorm"):
            extra = "现网 normalization 未在 post-denylist 后重算（NormRadius max 等可能有漂移）。"
        return (
            f"Phase 0 有条件通过：SSL 预训练可启动（train+val 几何、无 test 泄漏）；"
            f"微调 WSS 前建议重算 normalization。{extra}"
        )
    return "Phase 0 未通过：须修复 FAIL 项后再开 G3 训练。"


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P G3 split-safe 审计")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training/splits/split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new/AG")
    ap.add_argument("--norm-params", type=Path, default=REPO_ROOT / "data_new/normalization_params_global.json")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--max-cases-per-split", type=int, default=25, help="采样模式：每 split 最多 case 数")
    ap.add_argument(
        "--use-pipeline-collect",
        action="store_true",
        help="用 pipeline.collect_global_statistics 全量重算（更准确，较慢）",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.output is None:
        args.output = (
            REPO_ROOT / "outputs/field/f0_decision" / f"v3p_g3_split_safe_audit_{date.today().strftime('%Y%m%d')}.json"
        )

    report = run_audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {k: report[k] for k in report if k not in ("checks", "supplemental")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n--- checks ---")
    for c in report["checks"]:
        print(f"  [{c['status']}] {c.get('id', '?')}: {c['name']}")
    for s in report["supplemental"]:
        print(f"  [{s['status']}] {s['name']}")
    print(f"\n{args.output}")


if __name__ == "__main__":
    main()
