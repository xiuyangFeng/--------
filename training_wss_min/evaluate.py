#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""完整点云评估（A 路核心）：稀疏训练的模型 -> 推理整条血管所有壁面点。

用法：
  python -m training_wss_min.evaluate --run-dir training_wss_min/runs/<name>
  python -m training_wss_min.evaluate --config training_wss_min/configs/<name>.json

产出（写入 run_dir/eval/）：
- metrics.json           整体 + 分区 + 逐病例指标（val & test）
- per_case_metrics.csv   逐病例一行，便于扫表/画点数-精度曲线
- heatmaps/<case>.png    每个 test 病例：真值 / 预测 / 误差 三联图（峰值收缩期）
`evaluate_partition` 同时被 train.py 用作 val 监控（make_plots=False）。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from . import config as C
from . import dataset as D
from . import metrics as M
from .models import build_model


@torch.no_grad()
def predict_case_norm(model, case: Dict, input_features, feat_stats, device: str) -> np.ndarray:
    """在完整点云上预测（标准化空间）。单例过大 OOM 时回退 CPU。"""
    batch = D.case_to_batch(case, input_features, feat_stats, device=device)
    try:
        out = model(batch["pos"], batch["x"], batch["batch"])
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        model_cpu = model.to("cpu")
        batch = D.case_to_batch(case, input_features, feat_stats, device="cpu")
        out = model_cpu(batch["pos"], batch["x"], batch["batch"])
        model.to(device)
    return out.detach().float().cpu().numpy()


def evaluate_partition(model, cases: List[Dict], cfg: C.ExpConfig, feat_stats: Dict,
                       wss_stats: Dict, device: str,
                       save_dir: Path | None = None, make_plots: bool = False) -> Dict:
    model.eval()
    pred_norm_by_case = []
    for case in cases:
        y_pred_norm = predict_case_norm(model, case, cfg.data.input_features, feat_stats, device)
        pred_norm_by_case.append(np.asarray(y_pred_norm, dtype=np.float64))

    true_norm_by_case = [np.asarray(c["y_norm"], dtype=np.float64) for c in cases]
    normalized = _evaluate_space(
        cases, true_norm_by_case, pred_norm_by_case,
        save_dir=save_dir, make_plots=make_plots, plot_space="normalized target",
    )
    normalized["normalization"] = {
        "mode": cfg.data.target_normalization,
        "case_metadata": {
            f"{c['cohort']}/{c['case']}": c.get("target_normalization_meta", {})
            for c in cases
        },
        "predictions_clipped_for_metrics": False,
    }

    if cfg.data.target_normalization == "case_max":
        result = dict(normalized)
        result["metric_space"] = "normalized_target"
        result["normalized"] = normalized
        return result

    pred_raw_by_case = []
    for pred_norm in pred_norm_by_case:
        pred_raw = D.denormalize_wss(pred_norm, wss_stats)
        if wss_stats.get("method") == "log_z":
            pred_raw = np.clip(pred_raw, 0, None)
        pred_raw_by_case.append(np.asarray(pred_raw, dtype=np.float64))
    true_raw_by_case = [np.asarray(c["y_raw"], dtype=np.float64) for c in cases]
    physical = _evaluate_space(
        cases, true_raw_by_case, pred_raw_by_case,
        save_dir=None, make_plots=False, plot_space="physical target",
    )
    physical["metric_space"] = "physical_target"
    physical["normalized"] = normalized
    return physical


def _evaluate_space(cases: List[Dict], true_by_case: List[np.ndarray],
                    pred_by_case: List[np.ndarray], *, save_dir: Path | None,
                    make_plots: bool, plot_space: str) -> Dict:
    per_case: Dict[str, Dict] = {}
    for case, y_true, y_pred in zip(cases, true_by_case, pred_by_case):
        reg = M.regional_metrics(case["pos"], case["local_radius"], y_true, y_pred)
        reg["calibration"] = M.calibration_metrics(y_true, y_pred)
        reg["distribution"] = M.distribution_metrics(y_true, y_pred)
        reg["hotspot"] = M.hotspot_localization_metrics(y_true, y_pred, case["pos"])
        per_case[f"{case['cohort']}/{case['case']}"] = reg
        if make_plots and save_dir is not None:
            _plot_case(case, y_true, y_pred, save_dir / "heatmaps", plot_space)

    if not true_by_case:
        raise ValueError("evaluate_partition received no cases")
    pt = np.concatenate(true_by_case)
    pp = np.concatenate(pred_by_case)
    return {
        "aggregate": M.aggregate_case_metrics(per_case),
        "field": M.basic_metrics(pt, pp),
        "field_casebalanced": M.casebalanced_field_metrics(true_by_case, pred_by_case),
        "calibration": M.calibration_metrics(pt, pp),
        "distribution": M.distribution_metrics(pt, pp),
        "hotspot": M.aggregate_hotspot_metrics(per_case),
        "regional_field": _regional_field(cases, true_by_case, pred_by_case),
        "per_case": per_case,
    }


def _regional_field(cases, pooled_true, pooled_pred) -> Dict:
    """把所有病例的点按分区 pool 起来算 field 级分区指标。"""
    acc = {"bifurcation": ([], []), "stenosis": ([], []), "high_wss": ([], [])}
    for case, yt, yp in zip(cases, pooled_true, pooled_pred):
        masks = M.region_masks(case["pos"], case["local_radius"], yt)
        for name, m in masks.items():
            if m.sum() > 0:
                acc[name][0].append(yt[m]); acc[name][1].append(yp[m])
    out = {}
    for name, (ts, ps) in acc.items():
        if ts:
            out[name] = M.basic_metrics(np.concatenate(ts), np.concatenate(ps))
    return out


def _plot_case(case: Dict, y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path,
               plot_space: str = "target"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    pos = case["pos"]
    # 投影到主轴(z) - x 平面看整条血管；点大小随点数自适应
    px, pz = pos[:, 0], pos[:, 2]
    err = np.abs(y_true - y_pred)
    vmin = float(np.percentile(y_true, 1))
    vmax = float(np.percentile(y_true, 99))
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0
    emax = float(np.percentile(err, 99)) or 1.0
    s = max(1.0, 4000.0 / len(pos))

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    for ax, val, title, lo, hi, cmap in [
        (axes[0], y_true, f"true ({plot_space})", vmin, vmax, "viridis"),
        (axes[1], y_pred, f"pred ({plot_space})", vmin, vmax, "viridis"),
        (axes[2], err, "|err|", 0.0, emax, "magma"),
    ]:
        sc = ax.scatter(pz, px, c=val, s=s, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title); ax.set_aspect("equal"); ax.set_xlabel("z"); ax.set_ylabel("x")
        fig.colorbar(sc, ax=ax, shrink=0.7)
    r2 = M.r2_score(y_true, y_pred)
    fig.suptitle(f"{case['cohort']}/{case['case']}  peak step {case['peak_step']}  "
                 f"field R2={r2:.3f}  n={len(pos)}")
    fig.savefig(out_dir / f"{case['cohort'].replace('/', '_')}__{case['case']}.png", dpi=90)
    plt.close(fig)


def write_reports(result_by_part: Dict[str, Dict], eval_dir: Path):
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "metrics.json").write_text(json.dumps(result_by_part, indent=2, ensure_ascii=False))
    rows = []
    for part, res in result_by_part.items():
        norm_res = res.get("normalized", res)
        for case, reg in norm_res["per_case"].items():
            row = {"partition": part, "case": case}
            for rname, rm in reg.items():
                if rname == "calibration":
                    for k in ("top10_pred_true_ratio", "p95_pred_true_ratio",
                              "p99_pred_true_ratio", "max_pred_true_ratio",
                              "calibration_slope"):
                        row[f"cal_{k}"] = round(rm.get(k, float("nan")), 4)
                    continue
                if rname == "hotspot":
                    for k in ("high_wss_mae", "top10_iou", "top10_precision",
                              "top10_recall", "spearman_all", "spearman_high_wss",
                              "peak_point_dist", "peak_point_dist_over_bbox",
                              "hotspot_centroid_dist", "hotspot_centroid_dist_over_bbox"):
                        row[f"hot_{k}"] = round(rm.get(k, float("nan")), 4)
                    continue
                if rname == "distribution":
                    for k, v in rm.items():
                        row[f"dist_{k}"] = round(v, 6) if isinstance(v, float) else v
                    continue
                row[f"{rname}_r2"] = round(rm.get("r2", float("nan")), 4)
                row[f"{rname}_mae"] = round(rm.get("mae", float("nan")), 6)
                row[f"{rname}_rmse"] = round(rm.get("rmse", float("nan")), 6)
                row[f"{rname}_nrmse"] = round(rm.get("nrmse_range", float("nan")), 4) \
                    if "nrmse_range" in rm else float("nan")
                row[f"{rname}_n"] = rm.get("n", 0)
            rows.append(row)
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(eval_dir / "per_case_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def checkpoint_filename(checkpoint: str) -> str:
    value = checkpoint.removesuffix(".pt")
    if value in {"best", "ckpt_best"}:
        return "ckpt_best.pt"
    if value in {"last", "ckpt_last"}:
        return "ckpt_last.pt"
    raise ValueError("checkpoint must be 'best' or 'last'")


def load_model_from_run(run_dir: Path, device: str, checkpoint: str = "best"):
    cfg = C.ExpConfig.from_json(run_dir / "config.json")
    feat_stats = json.loads((run_dir / "feature_stats.json").read_text())
    model = build_model(cfg.model, C.input_dim(cfg)).to(device)
    ckpt = torch.load(
        run_dir / checkpoint_filename(checkpoint), map_location=device, weights_only=False
    )
    model.load_state_dict(ckpt["model"])
    return cfg, feat_stats, model, ckpt


def load_wss_stats_for_run(run_dir: Path) -> Dict:
    stats_path = run_dir / "wss_global_stats.json"
    return D.load_wss_stats(stats_path if stats_path.is_file() else C.GLOBAL_STATS)


def parse_and_guard_partitions(value: str, allow_test: bool = False) -> List[str]:
    """Parse CLI partitions and require an explicit unlock before reading test."""
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("at least one partition is required")
    unknown = sorted(set(parts) - {"train", "val", "test"})
    if unknown:
        raise ValueError(f"unknown partitions: {', '.join(unknown)}")
    if len(parts) != len(set(parts)):
        raise ValueError("duplicate partitions are not allowed")
    if "test" in parts and not allow_test:
        raise PermissionError(
            "test access is locked during development; pass --allow-test only for the "
            "pre-authorized legacy test16 evaluation"
        )
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--partitions", type=str, default="val",
                    help="开发默认 val-only；test 还需 --allow-test 显式解锁")
    ap.add_argument("--allow-test", action="store_true",
                    help="显式解锁配置中已获批且锁定的 test partition")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--checkpoint", choices=("best", "last"), default="best")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="默认 run_dir/eval/ckpt_<best|last>，用于隔离 checkpoint 结果")
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.config:
        run_dir = C.ExpConfig.from_json(args.config).run_dir
    else:
        raise SystemExit("需要 --run-dir 或 --config")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, feat_stats, model, ckpt = load_model_from_run(run_dir, device, args.checkpoint)
    wss_stats = load_wss_stats_for_run(run_dir)
    eval_dir = Path(args.output_dir) if args.output_dir else run_dir / "eval" / f"ckpt_{args.checkpoint}"

    try:
        partitions = parse_and_guard_partitions(args.partitions, allow_test=args.allow_test)
    except (ValueError, PermissionError) as exc:
        ap.error(str(exc))

    result_by_part = {}
    for part in partitions:
        cases = D.load_partition(
            cfg.data.split_path, part, wss_stats, target=cfg.data.target,
            target_normalization=cfg.data.target_normalization,
            data_root=cfg.data.data_root,
            required_frame_version=cfg.data.required_frame_version,
        )
        res = evaluate_partition(model, cases, cfg, feat_stats, wss_stats, device,
                                 save_dir=eval_dir, make_plots=(not args.no_plots and part == "test"))
        result_by_part[part] = res
        report = res.get("normalized", res)
        agg, fld = report["aggregate"], report["field"]
        cb = report["field_casebalanced"]
        print(f"[{part}] cases={agg['n_cases']}  R2_casemean={agg['r2_casemean']:.4f}  "
              f"R2_field_raw={fld['r2']:.4f}  R2_field_casebalanced={cb['r2']:.4f}  "
              f"NRMSE_field={fld['nrmse_range']:.4f}  "
              f"MAE={fld['mae']:.4f}")
        for rname, rm in report["regional_field"].items():
            print(f"    {rname:12s} R2={rm['r2']:.4f} NRMSE={rm['nrmse_range']:.4f} n={rm['n']}")

    write_reports(result_by_part, eval_dir)
    print("eval 写入:", eval_dir)


if __name__ == "__main__":
    main()
