#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C5 CPU 增量 probe：radius_gradient 相对现有几何特征的交叉验证增益。

用 Ridge 在病例级 pooled 点上做 5-fold（按病例分组），比较
  base = [abscissa_norm, local_radius, curvature]
  + radius_gradient
的 R²。若无稳定增量则不进入 PointNeXt GPU 矩阵。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training_wss_min import config as C
from training_wss_min import dataset as D


def _pack(cases, keys):
    Xs, ys, groups = [], [], []
    for i, c in enumerate(cases):
        cols = [np.asarray(c[k], np.float64) for k in keys]
        X = np.stack(cols, axis=1)
        # 子采样加速
        n = len(X)
        if n > 2000:
            rng = np.random.default_rng(i + 7)
            idx = rng.choice(n, 2000, replace=False)
            X = X[idx]; y = c["y_norm"][idx]
        else:
            y = c["y_norm"]
        Xs.append(X); ys.append(y); groups.append(np.full(len(X), i))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(groups)


def eval_features(cases, keys):
    X, y, g = _pack(cases, keys)
    gkf = GroupKFold(n_splits=5)
    scores = []
    for tr, te in gkf.split(X, y, g):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        ss_res = np.sum((y[te] - pred) ** 2)
        ss_tot = np.sum((y[te] - y[te].mean()) ** 2)
        scores.append(1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan"))
    return float(np.nanmean(scores)), float(np.nanstd(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(C.PROJECT_ROOT / "training/splits/split_AG_wss_min_v2_dev1.json"))
    ap.add_argument("--stats", default=str(C.DATA_ROOT / "fold_stats/wss_stats_v2_dev1.json"))
    ap.add_argument("--out", default="docs/02-推进与变更/assets_第四轮/c5_radius_gradient_probe.json")
    args = ap.parse_args()

    wss = D.load_wss_stats(args.stats)
    cases = D.load_partition(args.split, "train", wss, strict=True)
    base = ["abscissa_norm", "local_radius", "curvature"]
    plus = base + ["radius_gradient"]
    m0, s0 = eval_features(cases, base)
    m1, s1 = eval_features(cases, plus)
    delta = m1 - m0
    go = delta >= 0.01
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": args.split,
        "base_keys": base,
        "plus_keys": plus,
        "base_r2_mean_std": [m0, s0],
        "plus_r2_mean_std": [m1, s1],
        "delta_r2": delta,
        "go_pointnext": go,
        "threshold": 0.01,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
