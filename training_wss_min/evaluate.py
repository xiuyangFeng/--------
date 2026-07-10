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
from .pointnext import build_model


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
    per_case: Dict[str, Dict] = {}
    pooled_true, pooled_pred = [], []
    for case in cases:
        y_pred_norm = predict_case_norm(model, case, cfg.data.input_features, feat_stats, device)
        y_pred_raw = np.clip(D.denormalize_wss(y_pred_norm, wss_stats), 0, None)
        y_true_raw = case["y_raw"].astype(np.float64)
        reg = M.regional_metrics(case["pos"], case["local_radius"], y_true_raw, y_pred_raw)
        reg["calibration"] = M.calibration_metrics(y_true_raw, y_pred_raw)
        per_case[f"{case['cohort']}/{case['case']}"] = reg
        pooled_true.append(y_true_raw)
        pooled_pred.append(y_pred_raw)
        if make_plots and save_dir is not None:
            _plot_case(case, y_true_raw, y_pred_raw, save_dir / "heatmaps")

    pt = np.concatenate(pooled_true)
    pp = np.concatenate(pooled_pred)
    result = {
        "aggregate": M.aggregate_case_metrics(per_case),
        "field": M.basic_metrics(pt, pp),
        "calibration": M.calibration_metrics(pt, pp),
        "regional_field": _regional_field(cases, pooled_true, pooled_pred),
        "per_case": per_case,
    }
    return result


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


def _plot_case(case: Dict, y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    pos = case["pos"]
    # 投影到主轴(z) - x 平面看整条血管；点大小随点数自适应
    px, pz = pos[:, 0], pos[:, 2]
    err = np.abs(y_true - y_pred)
    vmax = float(np.percentile(y_true, 99))
    emax = float(np.percentile(err, 99)) or 1.0
    s = max(1.0, 4000.0 / len(pos))

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    for ax, val, title, vm, cmap in [
        (axes[0], y_true, "true WSS", vmax, "viridis"),
        (axes[1], y_pred, "pred WSS", vmax, "viridis"),
        (axes[2], err, "|err|", emax, "magma"),
    ]:
        sc = ax.scatter(pz, px, c=np.clip(val, 0, vm), s=s, cmap=cmap, vmin=0, vmax=vm)
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
        for case, reg in res["per_case"].items():
            row = {"partition": part, "case": case}
            for rname, rm in reg.items():
                if rname == "calibration":
                    for k in ("top10_pred_true_ratio", "p95_pred_true_ratio",
                              "p99_pred_true_ratio", "max_pred_true_ratio",
                              "calibration_slope"):
                        row[f"cal_{k}"] = round(rm.get(k, float("nan")), 4)
                    continue
                row[f"{rname}_r2"] = round(rm.get("r2", float("nan")), 4)
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


def load_model_from_run(run_dir: Path, device: str):
    cfg = C.ExpConfig.from_json(run_dir / "config.json")
    feat_stats = json.loads((run_dir / "feature_stats.json").read_text())
    model = build_model(cfg.model, C.input_dim(cfg)).to(device)
    ckpt = torch.load(run_dir / "ckpt_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return cfg, feat_stats, model, ckpt


def load_wss_stats_for_run(run_dir: Path) -> Dict:
    stats_path = run_dir / "wss_global_stats.json"
    return D.load_wss_stats(stats_path if stats_path.is_file() else C.GLOBAL_STATS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--partitions", type=str, default="val,test")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.config:
        run_dir = C.ExpConfig.from_json(args.config).run_dir
    else:
        raise SystemExit("需要 --run-dir 或 --config")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, feat_stats, model, ckpt = load_model_from_run(run_dir, device)
    wss_stats = load_wss_stats_for_run(run_dir)
    eval_dir = run_dir / "eval"

    result_by_part = {}
    for part in args.partitions.split(","):
        cases = D.load_partition(cfg.data.split_path, part, wss_stats)
        res = evaluate_partition(model, cases, cfg, feat_stats, wss_stats, device,
                                 save_dir=eval_dir, make_plots=(not args.no_plots and part == "test"))
        result_by_part[part] = res
        agg, fld = res["aggregate"], res["field"]
        print(f"[{part}] cases={agg['n_cases']}  R2_casemean={agg['r2_casemean']:.4f}  "
              f"R2_field={fld['r2']:.4f}  NRMSE_field={fld['nrmse_range']:.4f}  "
              f"MAE={fld['mae']:.4f}")
        for rname, rm in res["regional_field"].items():
            print(f"    {rname:12s} R2={rm['r2']:.4f} NRMSE={rm['nrmse_range']:.4f} n={rm['n']}")

    write_reports(result_by_part, eval_dir)
    print("eval 写入:", eval_dir)


if __name__ == "__main__":
    main()
