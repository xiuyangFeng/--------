#!/usr/bin/env python3
"""V3P L3 子集评估：CHEN / fast / failure_cluster 等病例级 WSS 指标。

对已有或新预测的 run 汇总 per-case wall WSS R²，并按 L3 子集池化对比
（如 I6-diag vs J4）。产物默认 ``outputs/field/f0_decision/v3p_l3_subset_eval_<date>.json``。

用法::

    python -m training.scripts.run_v3p_l3_subset_eval \\
        --run-dir outputs/field/field_v3_pointnext_i6diag_... \\
        --run-dir outputs/field/field_v3_pointnext_j4v3dlite_... \\
        --label I6-diag --label J4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.utils import ensure_dir
from ._figure_utils import load_json, load_manifest, load_prediction_payload, save_json
from .run_v3_f0_decision import (
    DEFAULT_NORM_PARAMS,
    REPO_ROOT,
    _denorm_wss_mag,
    _load_norm_stats,
    _r2_score,
)

DEFAULT_I6_RUN = (
    "outputs/field/field_v3_pointnext_i6diag_localpool_main01_geom_pw_asymw_a"
    "_wall13000_near2000_split_AG_v1_seed1_20260619_174001"
)
DEFAULT_J4_RUN = (
    "outputs/field/field_v3_pointnext_j4v3dlite_localpool_main01_geom_pw_asymw_a"
    "_wall13000_near2000_split_AG_v3lite_v1_seed1_20260630_131820"
)
DEFAULT_FAILURE_CLUSTER = (
    "fast/ZHANG_HAO",
    "fast/LIU_JUN_FENG",
    "fast/ZHANG_CHUN",
    "fast/LIU_LI_QUN",
    "fast/LI_ZHI_LIN",
)
L3_CHEN = "fast/CHEN_SHI_MING"


def _norm_v3p_case(case: str) -> str:
    """``AG/fast/FOO`` → ``fast/FOO``（J5 data_new split 兼容）。"""
    return case[3:] if case.startswith("AG/") else case


def _case_in_set(case: str, case_set: set[str]) -> bool:
    return case in case_set or _norm_v3p_case(case) in case_set


def _expand_case_set(cases: Sequence[str]) -> set[str]:
    out: set[str] = set()
    for c in cases:
        out.add(c)
        out.add(_norm_v3p_case(c))
        if not c.startswith("AG/") and "/" in c:
            out.add(f"AG/{c}")
    return out


def _resolve_run_dir(path: Path) -> Path:
    p = path if path.is_absolute() else REPO_ROOT / path
    if not p.is_dir():
        raise FileNotFoundError(f"run 目录不存在: {p}")
    return p.resolve()


def _predictions_dir(run_dir: Path, checkpoint: str) -> Path:
    if checkpoint in ("best_wss_model.pt", "best_wss"):
        return run_dir / "predictions_test_best_wss"
    if checkpoint in ("best_model.pt", "best_model"):
        return run_dir / "predictions_test"
    stem = Path(checkpoint).stem.replace("_model", "")
    return run_dir / f"predictions_test_{stem}"


def _ensure_predictions(run_dir: Path, *, checkpoint: str, force: bool) -> Path:
    pred_dir = _predictions_dir(run_dir, checkpoint)
    manifest = pred_dir / "manifest.json"
    if manifest.is_file() and not force:
        return pred_dir
    cfg = run_dir / "config.snapshot.json"
    ckpt = run_dir / (
        "best_wss_model.pt"
        if checkpoint.startswith("best_wss")
        else "best_model.pt"
    )
    if not ckpt.is_file():
        raise FileNotFoundError(f"缺少 checkpoint: {ckpt}")
    cmd = [
        sys.executable,
        "-m",
        "training.scripts.predict_field",
        "--config",
        str(cfg),
        "--checkpoint",
        str(ckpt),
        "--subset",
        "test",
        "--output",
        str(pred_dir),
    ]
    if force and pred_dir.is_dir():
        import shutil

        shutil.rmtree(pred_dir)
    print("[l3-eval] predict:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    return pred_dir


def _wall_wss_mag_phys(payload: Mapping[str, Any], stats: Mapping[str, Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    if "y_wss_true" not in payload or "y_wss_pred" not in payload:
        raise KeyError("预测文件缺少 y_wss_true / y_wss_pred")
    gt_n = payload["y_wss_true"].detach().cpu().numpy()[:, 0]
    pred_n = payload["y_wss_pred"].detach().cpu().numpy()[:, 0]
    gt = _denorm_wss_mag(gt_n, stats.get("wss"))
    pred = _denorm_wss_mag(pred_n, stats.get("wss"))
    return gt, pred


def _per_case_from_manifest(
    manifest_path: Path,
    stats: Mapping[str, Dict[str, float]],
) -> Dict[str, Dict[str, Any]]:
    man = load_manifest(manifest_path)
    by_case: Dict[str, Dict[str, List[float]]] = {}
    for item in man.get("items", []):
        pred_path = Path(str(item.get("prediction_path", "")))
        if not pred_path.is_absolute():
            pred_path = REPO_ROOT / pred_path
        if not pred_path.is_file():
            continue
        case = str(item.get("case_name", pred_path.stem))
        payload = load_prediction_payload(pred_path)
        gt, pred = _wall_wss_mag_phys(payload, stats)
        by_case.setdefault(case, {"gt": [], "pred": []})
        by_case[case]["gt"].extend(gt.tolist())
        by_case[case]["pred"].extend(pred.tolist())

    rows: Dict[str, Dict[str, Any]] = {}
    for case, d in by_case.items():
        gt = np.asarray(d["gt"], dtype=np.float64)
        pred = np.asarray(d["pred"], dtype=np.float64)
        if gt.size < 10:
            continue
        pace = case.split("/")[0] if "/" in case else "unknown"
        rows[case] = {
            "case": case,
            "pace": pace,
            "n_wall": int(gt.size),
            "wss_r2_wss": float(_r2_score(gt, pred)),
            "wss_rmse_wss": float(np.sqrt(np.mean((gt - pred) ** 2))),
            "wss_mae_wss": float(np.mean(np.abs(gt - pred))),
        }
    return rows


def _pool_subset_from_manifest(
    manifest_path: Path,
    stats: Mapping[str, Dict[str, float]],
    cases: Sequence[str],
) -> Optional[Dict[str, Any]]:
    case_set = _expand_case_set(cases)
    gt_all: List[float] = []
    pred_all: List[float] = []
    present: List[str] = []
    man = load_manifest(manifest_path)
    for item in man.get("items", []):
        case = str(item.get("case_name", ""))
        if not _case_in_set(case, case_set):
            continue
        pred_path = Path(str(item.get("prediction_path", "")))
        if not pred_path.is_absolute():
            pred_path = REPO_ROOT / pred_path
        if not pred_path.is_file():
            continue
        payload = load_prediction_payload(pred_path)
        gt, pred = _wall_wss_mag_phys(payload, stats)
        gt_all.extend(gt.tolist())
        pred_all.extend(pred.tolist())
        if case not in present:
            present.append(case)
    if len(gt_all) < 50:
        return None
    gt_arr = np.asarray(gt_all, dtype=np.float64)
    pred_arr = np.asarray(pred_all, dtype=np.float64)
    return {
        "n_cases": len(present),
        "cases": present,
        "n_wall": int(gt_arr.size),
        "wss_r2_wss": float(_r2_score(gt_arr, pred_arr)),
        "wss_rmse_wss": float(np.sqrt(np.mean((gt_arr - pred_arr) ** 2))),
    }


def _fast_cases(per_case: Mapping[str, Mapping[str, Any]]) -> List[str]:
    return sorted(c for c in per_case if _norm_v3p_case(c).startswith("fast/"))


def evaluate_run(
    run_dir: Path,
    label: str,
    stats: Mapping[str, Dict[str, float]],
    *,
    checkpoint: str,
    failure_cluster: Sequence[str],
    predict: bool,
    force_predict: bool,
) -> Dict[str, Any]:
    pred_dir = (
        _ensure_predictions(run_dir, checkpoint=checkpoint, force=force_predict)
        if predict
        else _predictions_dir(run_dir, checkpoint)
    )
    manifest = pred_dir / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"缺少 manifest: {manifest}（加 --predict）")

    per_case = _per_case_from_manifest(manifest, stats)
    fast_cases = _fast_cases(per_case)
    slow_cases = sorted(c for c in per_case if _norm_v3p_case(c).startswith("slow/"))

    subsets = {
        "L3_all_test": sorted(per_case.keys()),
        "L3_fast": fast_cases,
        "L3_slow": slow_cases,
        "L3_CHEN": [c for c in per_case if _norm_v3p_case(c) == L3_CHEN],
        "L3_failure_cluster": [c for c in per_case if _norm_v3p_case(c) in set(DEFAULT_FAILURE_CLUSTER)],
    }
    subset_metrics: Dict[str, Any] = {}
    for name, cases in subsets.items():
        if not cases:
            subset_metrics[name] = None
            continue
        pooled = _pool_subset_from_manifest(manifest, stats, cases)
        subset_metrics[name] = pooled
        if pooled and name == "L3_CHEN":
            subset_metrics[name]["per_case"] = {
                c: per_case[c] for c in cases if c in per_case
            }

    summary_path = run_dir / "summary.json"
    summary_wss = None
    if summary_path.is_file():
        sm = load_json(summary_path)
        key = "test_metrics_best_wss" if checkpoint.startswith("best_wss") else "test_metrics"
        summary_wss = sm.get(key, {}).get("wss_r2_wss")

    return {
        "label": label,
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "predictions_dir": str(pred_dir),
        "summary_test_wss_r2_wss": summary_wss,
        "n_test_cases": len(per_case),
        "per_case": sorted(per_case.values(), key=lambda r: r["case"]),
        "subsets": subset_metrics,
    }


def _compare_runs(runs: Sequence[Mapping[str, Any]], baseline_label: str) -> Dict[str, Any]:
    base = next((r for r in runs if r["label"] == baseline_label), runs[0] if runs else None)
    if base is None:
        return {}
    out: Dict[str, Any] = {}
    for subset_name in base.get("subsets", {}):
        b = base["subsets"].get(subset_name)
        if not b:
            continue
        row: Dict[str, Any] = {"baseline": baseline_label, "baseline_r2": b.get("wss_r2_wss")}
        for r in runs:
            if r["label"] == baseline_label:
                continue
            s = r.get("subsets", {}).get(subset_name)
            if not s:
                row[r["label"]] = None
                continue
            br2 = float(b["wss_r2_wss"])
            rr2 = float(s["wss_r2_wss"])
            row[r["label"]] = {
                "wss_r2_wss": rr2,
                "delta_vs_baseline": rr2 - br2,
            }
        out[subset_name] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P L3 子集评估（CHEN / fast / failure cluster）")
    ap.add_argument("--run-dir", type=Path, action="append", required=True)
    ap.add_argument("--label", action="append", default=[], help="与 --run-dir 一一对应")
    ap.add_argument("--baseline-label", default="I6-diag")
    ap.add_argument("--checkpoint", default="best_wss")
    ap.add_argument("--predict", action="store_true", help="缺失预测时自动 predict_field")
    ap.add_argument("--force-predict", action="store_true")
    ap.add_argument(
        "--audit",
        type=Path,
        default=REPO_ROOT / "outputs/field/f0_decision/v3p_v3d_lite_data_audit_20260630.json",
    )
    ap.add_argument("--norm-params", type=Path, default=DEFAULT_NORM_PARAMS)
    ap.add_argument("--date", default=date.today().strftime("%Y%m%d"))
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="默认 outputs/field/f0_decision/v3p_l3_subset_eval_<date>.json",
    )
    args = ap.parse_args()

    run_dirs = [_resolve_run_dir(p) for p in args.run_dir]
    labels = list(args.label)
    if not labels:
        labels = ["I6-diag", "J4"][: len(run_dirs)]
    if len(labels) != len(run_dirs):
        raise SystemExit("--run-dir 与 --label 数量须一致")

    audit = load_json(args.audit.resolve()) if args.audit.is_file() else {}
    failure_cluster = audit.get("ag_failure_cluster") or list(DEFAULT_FAILURE_CLUSTER)

    stats = _load_norm_stats(args.norm_params.resolve())
    runs = [
        evaluate_run(
            rd,
            lab,
            stats,
            checkpoint=args.checkpoint,
            failure_cluster=failure_cluster,
            predict=args.predict,
            force_predict=args.force_predict,
        )
        for rd, lab in zip(run_dirs, labels)
    ]

    out_path = (
        args.output.resolve()
        if args.output is not None
        else REPO_ROOT / "outputs/field/f0_decision" / f"v3p_l3_subset_eval_{args.date}.json"
    )
    ensure_dir(out_path.parent)
    report = {
        "label": "v3p_l3_subset_eval",
        "date": args.date,
        "context": "V3P · L3 CHEN/fast/failure_cluster subset WSS (wall magnitude R²)",
        "failure_cluster": failure_cluster,
        "baseline_label": args.baseline_label,
        "runs": runs,
        "comparison": _compare_runs(runs, args.baseline_label),
    }
    save_json(out_path, report)
    print(json.dumps({"output": str(out_path), "comparison": report["comparison"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
