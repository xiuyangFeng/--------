#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A3 Gate-0：第三轮 tgtw 三 seed 残差分位 + leave-one-case-out 校准（零新训练）。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from training_wss_min import dataset as D
from training_wss_min.evaluate import load_model_from_run, load_wss_stats_for_run, predict_case_norm


DEFAULT_RUNS = [
    "training_wss_min/runs/r3_clean_xyzgeom_tgtw_s1234",
    "training_wss_min/runs/r3_clean_xyzgeom_tgtw_s7",
    "training_wss_min/runs/r3_clean_xyzgeom_tgtw_s2025",
]


def _decile_residual(y_true, y_pred, eps=1e-6):
    yt = np.asarray(y_true, np.float64)
    yp = np.asarray(y_pred, np.float64)
    order = np.argsort(yt)
    yt, yp = yt[order], yp[order]
    n = len(yt)
    rows = []
    for d in range(10):
        lo, hi = int(d * n / 10), int((d + 1) * n / 10)
        if hi <= lo:
            continue
        t, p = yt[lo:hi], yp[lo:hi]
        raw_res = p - t
        log_res = np.log(np.clip(p, 0, None) + eps) - np.log(np.clip(t, 0, None) + eps)
        rows.append({
            "decile": d + 1,
            "n": int(hi - lo),
            "true_mean": float(t.mean()),
            "raw_res_mean": float(raw_res.mean()),
            "raw_under_frac": float((raw_res < 0).mean()),
            "log_res_mean": float(log_res.mean()),
            "top_like": d >= 8,
        })
    return rows


def _top10_ratio(yt, yp):
    thr = np.percentile(yt, 90)
    m = yt >= thr
    return float(yp[m].mean() / (yt[m].mean() + 1e-12))


def _r2(yt, yp):
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")


def loo_smearing(case_preds, eps=1e-6):
    """病例 leave-one-out：拟合全局 log-bias，看留出病例 top10/R2 是否改善。"""
    keys = list(case_preds.keys())
    results = []
    for hold in keys:
        # fit bias on others: mean(log y_true - log y_pred)
        biases = []
        for k, (yt, yp) in case_preds.items():
            if k == hold:
                continue
            biases.append(np.mean(np.log(yt + eps) - np.log(np.clip(yp, 0, None) + eps)))
        bias = float(np.mean(biases)) if biases else 0.0
        yt, yp = case_preds[hold]
        yp_cal = np.clip(yp, 0, None) * np.exp(bias)
        results.append({
            "case": hold,
            "bias": bias,
            "top10_before": _top10_ratio(yt, yp),
            "top10_after": _top10_ratio(yt, yp_cal),
            "r2_before": _r2(yt, yp),
            "r2_after": _r2(yt, yp_cal),
            "mae_before": float(np.mean(np.abs(yt - yp))),
            "mae_after": float(np.mean(np.abs(yt - yp_cal))),
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--partition", default="val")
    ap.add_argument("--out-dir", default="docs/02-推进与变更/assets_第四轮")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "device": device, "runs": {}}

    seed_support = []
    for rd in args.run_dirs:
        run_dir = Path(rd)
        cfg, feat_stats, model, ckpt = load_model_from_run(run_dir, device)
        wss_stats = load_wss_stats_for_run(run_dir)
        cases = D.load_partition(cfg.data.split_path, args.partition, wss_stats, strict=True)
        case_preds = {}
        yt_all, yp_all = [], []
        for case in cases:
            yp = np.clip(
                D.denormalize_wss(
                    predict_case_norm(model, case, cfg.data.input_features, feat_stats, device),
                    wss_stats),
                0, None,
            )
            yt = case["y_raw"].astype(np.float64)
            key = f"{case['cohort']}/{case['case']}"
            case_preds[key] = (yt, yp)
            yt_all.append(yt); yp_all.append(yp)
        yt_p = np.concatenate(yt_all); yp_p = np.concatenate(yp_all)
        deciles = _decile_residual(yt_p, yp_p)
        loo = loo_smearing(case_preds)
        top10_improve = [r["top10_after"] - r["top10_before"] for r in loo]
        r2_delta = [r["r2_after"] - r["r2_before"] for r in loo]
        mae_delta = [r["mae_after"] - r["mae_before"] for r in loo]
        # Gate: 留出病例 top10 中位改善 >0 且 R2/MAE 不显著劣化
        support = (
            float(np.median(top10_improve)) > 0.02
            and float(np.median(r2_delta)) >= -0.02
            and float(np.median(mae_delta)) <= 0.05 * float(np.median([r["mae_before"] for r in loo]))
        )
        # 尾部 decile 欠估是否单调增强
        under = [d["raw_under_frac"] for d in deciles]
        mono_under = all(under[i] <= under[i + 1] + 0.05 for i in range(len(under) - 1))
        seed_support.append(support)
        report["runs"][run_dir.name] = {
            "ckpt_epoch": int(ckpt.get("epoch", -1)),
            "deciles": deciles,
            "loo_summary": {
                "median_top10_delta": float(np.median(top10_improve)),
                "median_r2_delta": float(np.median(r2_delta)),
                "median_mae_delta": float(np.median(mae_delta)),
                "support_smearing": support,
                "mono_underest_by_decile": mono_under,
            },
            "loo_per_case": loo,
        }
        print(f"[{run_dir.name}] support={support} "
              f"medΔtop10={np.median(top10_improve):+.3f} "
              f"medΔr2={np.median(r2_delta):+.3f} mono_under={mono_under}")

    n_support = sum(1 for s in seed_support if s)
    report["gate0"] = {
        "n_seeds_support": n_support,
        "n_seeds": len(seed_support),
        "pass": n_support >= 2,
        "c3_nll_recommended": n_support >= 2,
        "note": "若简单 smearing 稳定改善尾部，优先可解释校准层；否则 NLL Jensen 依据不足",
    }
    out = out_dir / "a3_residual_calibration.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("Gate-0 pass:", report["gate0"]["pass"], "->", out)


if __name__ == "__main__":
    main()
