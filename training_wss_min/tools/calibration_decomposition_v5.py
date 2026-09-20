"""Post-hoc calibration decomposition for a trained WSS run (inference only).

Question answered: how much of the physical-space (Pa) gap between R² and the
affine-fit R² is explained by the log->Pa back-transform (Jensen bias) and by a
global scale error, versus per-case scale error and genuine pattern error?

Variants evaluated on test34 (all predictions from the same checkpoint):
  raw            exp(mu) as produced by the run (current reporting)
  jensen_train   exp(mu + s^2/2) with s = ln-space residual std fitted on train138 (deployment-legal)
  affine_log_tr  global affine recalibration in log space fitted on train138 (legal)
  affine_pa_tr   global affine recalibration in Pa fitted on train138 (legal)
  affine_pa_te   global affine fitted on test truth (oracle, = 'R² fit pooled')
  percase_pa_te  per-case affine fitted on test truth (oracle upper bound for scale fixes)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min.evaluate import load_model_from_run, load_wss_stats_for_run, predict_case_norm

ROOT = Path(__file__).resolve().parents[2]


def r2(y, p):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def r2_cb(ys, ps):
    gm = np.mean([y.mean() for y in ys])
    ss_res = np.mean([np.mean((y - p) ** 2) for y, p in zip(ys, ps)])
    ss_tot = np.mean([np.mean((y - gm) ** 2) for y in ys])
    return 1.0 - ss_res / ss_tot


def top_ratio(ys, ps, q=90.0):
    y = np.concatenate(ys); p = np.concatenate(ps)
    m = y >= np.percentile(y, q)
    return float(p[m].mean() / y[m].mean())


def p99_ratio(ys, ps):
    y = np.concatenate(ys); p = np.concatenate(ps)
    return float(np.percentile(p, 99) / np.percentile(y, 99))


def high_r2(ys, ps, pctl=90.0):
    masks = [y >= np.nanpercentile(y, pctl) for y in ys]
    y = np.concatenate([a[m] for a, m in zip(ys, masks)]); p = np.concatenate([a[m] for a, m in zip(ps, masks)])
    return r2(y, p)


def affine(x, y):
    a, b = np.polyfit(x, y, 1)
    return a, b


def summarize(name, ys, ps):
    return {"variant": name, "r2_cb": r2_cb(ys, ps), "r2_pooled": r2(np.concatenate(ys), np.concatenate(ps)),
            "case_mean_r2": float(np.mean([r2(y, p) for y, p in zip(ys, ps)])), "top10_ratio": top_ratio(ys, ps),
            "p99_ratio": p99_ratio(ys, ps), "high_wss_r2": high_r2(ys, ps)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, feat_stats, model, _ = load_model_from_run(run_dir, device, "best")
    model.eval()
    stats = load_wss_stats_for_run(run_dir)
    logstd, logmean, eps = stats["log"]["std"], stats["log"]["mean"], stats["eps"]
    def preds(partition):
        cases = D.load_partition(cfg.data.split_path, partition, stats, target=cfg.data.target, target_normalization=cfg.data.target_normalization,
                                 data_root=cfg.data.data_root, required_frame_version=cfg.data.required_frame_version)
        mus, ys, lvs = [], [], []
        for case in cases:
            out = np.asarray(predict_case_norm(model, case, cfg.data.input_features, feat_stats, device, cfg=cfg,
                                               return_all_channels=True), dtype=np.float64)
            if out.ndim == 2:
                mus.append(out[:, 0]); lvs.append(np.clip(out[:, 1], cfg.train.nll_logvar_min, cfg.train.nll_logvar_max))
            else:
                mus.append(out)
            ys.append(np.asarray(case["y_raw"], dtype=np.float64))
        return mus, ys, lvs
    with torch.no_grad():
        mu_tr, y_tr, _ = preds("train")
        mu_te, y_te, lv_te = preds("test")
    ln_tr = [np.log(np.clip(y, 0, None) + eps) for y in y_tr]
    znorm_tr = [(l - logmean) / logstd for l in ln_tr]
    resid = np.concatenate([z - m for z, m in zip(znorm_tr, mu_tr)])
    s_ln = float(resid.std() * logstd)
    a_log, b_log = affine(np.concatenate(mu_tr), np.concatenate(znorm_tr))  # z_true ~ a*mu + b (train)
    to_pa = lambda m: np.exp(m * logstd + logmean) - eps
    raw_tr = [np.clip(to_pa(m), 0, None) for m in mu_tr]
    a_pa, b_pa = affine(np.concatenate(raw_tr), np.concatenate(y_tr))  # y ~ a*pred + b (train)
    raw_te = [np.clip(to_pa(m), 0, None) for m in mu_te]
    variants = {
        "raw": raw_te,
        "jensen_train": [np.clip(to_pa(m + (s_ln ** 2 / 2) / logstd), 0, None) for m in mu_te],
        "affine_log_tr": [np.clip(to_pa(a_log * m + b_log), 0, None) for m in mu_te],
        "affine_pa_tr": [a_pa * p + b_pa for p in raw_te],
    }
    if lv_te:  # heteroscedastic head: per-point exp(mu + sigma_i^2/2), sigma_i in normalized units
        variants["jensen_pointwise(logvar head)"] = [np.clip(to_pa(m + 0.5 * np.exp(lv) * logstd), 0, None) for m, lv in zip(mu_te, lv_te)]
        out_sigma = float(np.mean([np.sqrt(np.exp(lv)).mean() for lv in lv_te]) * logstd)
    else:
        out_sigma = None
    a_te, b_te = affine(np.concatenate(raw_te), np.concatenate(y_te))
    variants["affine_pa_te(oracle)"] = [a_te * p + b_te for p in raw_te]
    pc = []
    for y, p in zip(y_te, raw_te):
        a, b = affine(p, y); pc.append(a * p + b)
    variants["percase_pa_te(oracle)"] = pc
    rows = [summarize(k, y_te, v) for k, v in variants.items()]
    out = {"run_dir": str(run_dir), "ln_residual_std_train": s_ln, "jensen_factor": math.exp(s_ln ** 2 / 2),
           "affine_log_train": {"a": a_log, "b": b_log}, "affine_pa_train": {"a": a_pa, "b": b_pa}, "affine_pa_test": {"a": a_te, "b": b_te}, "mean_predicted_sigma_ln(test)": out_sigma, "rows": rows}
    print(f"run={run_dir.name}  ln-resid std(train)={s_ln:.3f}  Jensen={out['jensen_factor']:.3f}  affine_log_train a={a_log:.3f} b={b_log:.3f}  affine_pa_train a={a_pa:.3f} b={b_pa:.3f}")
    print("| variant | R² cb | R² pooled | case mean | top10 ratio | p99 ratio | high-WSS R² |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        print(f"| {r['variant']} | {r['r2_cb']:.4f} | {r['r2_pooled']:.4f} | {r['case_mean_r2']:.4f} | {r['top10_ratio']:.3f} | {r['p99_ratio']:.3f} | {r['high_wss_r2']:.3f} |")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
