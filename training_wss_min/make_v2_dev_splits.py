#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A4：从 v1 的 train+val（61 例）生成 3 个 repeated-holdout 开发划分，并写 fold-specific WSS stats。

分层键：cohort(fast/slow)、病例 mean log-WSS、p99 WSS、点数。
test 不参与。每个 fold 的 stats 只读该 fold 的 train 病例。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from training_wss_min import config as C
from training_wss_min import dataset as D


def _case_strata(label: str, bundle_path: Path, eps: float = 1e-6) -> dict:
    subset, case = label.split("/", 1)
    with np.load(bundle_path, allow_pickle=True) as d:
        steps = d["steps"].tolist()
        si = steps.index(int(d["peak_step"]))
        wss = d["wall_wss"][si].astype(np.float64)
        n = len(d["wall_coords_norm"])
    return {
        "label": label,
        "cohort": subset,
        "n": n,
        "mean_log": float(np.mean(np.log(np.clip(wss, 0, None) + eps))),
        "p99": float(np.percentile(wss, 99)),
    }


def _stratified_holdout(rows, val_n: int, seed: int):
    rng = np.random.default_rng(seed)
    # 按 cohort 分层，再在 cohort 内按 mean_log 四分位分层抽样
    val = []
    train = []
    by_cohort = {}
    for r in rows:
        by_cohort.setdefault(r["cohort"], []).append(r)
    # 目标 val 比例
    target_frac = val_n / max(1, len(rows))
    for cohort, items in by_cohort.items():
        items = sorted(items, key=lambda x: x["mean_log"])
        n_val = max(1, int(round(len(items) * target_frac)))
        # 四分位各抽
        qs = np.array_split(np.arange(len(items)), 4)
        picked = []
        per_q = max(1, n_val // 4)
        for q in qs:
            if len(q) == 0:
                continue
            take = min(per_q, len(q))
            chosen = rng.choice(q, size=take, replace=False)
            picked.extend(int(i) for i in chosen)
        # 补齐/裁剪
        remaining = [i for i in range(len(items)) if i not in picked]
        while len(picked) < n_val and remaining:
            j = int(rng.choice(remaining))
            picked.append(j)
            remaining.remove(j)
        picked = picked[:n_val]
        for i, it in enumerate(items):
            (val if i in picked else train).append(it["label"])
    # 全局裁剪到精确 val_n
    rng.shuffle(val)
    if len(val) > val_n:
        extra = val[val_n:]
        val = val[:val_n]
        train.extend(extra)
    elif len(val) < val_n:
        need = val_n - len(val)
        rng.shuffle(train)
        val.extend(train[:need])
        train = train[need:]
    return sorted(train), sorted(val)


def compute_fold_stats(train_labels, data_root: Path, eps=1e-6) -> dict:
    n = 0
    n_zero = 0
    s_lin = ss_lin = 0.0
    s_log = ss_log = 0.0
    vmin, vmax = np.inf, -np.inf
    chunks = []
    for label in train_labels:
        subset, case = label.split("/", 1)
        p = data_root / "AG" / subset / case / "bundle.npz"
        with np.load(p, allow_pickle=True) as d:
            steps = d["steps"].tolist()
            si = steps.index(int(d["peak_step"]))
            wss = d["wall_wss"][si].astype(np.float64).ravel()
        wss = wss[np.isfinite(wss)]
        n += wss.size
        n_zero += int((wss <= 0).sum())
        s_lin += wss.sum(); ss_lin += (wss * wss).sum()
        lg = np.log(np.clip(wss, 0, None) + eps)
        s_log += lg.sum(); ss_log += (lg * lg).sum()
        vmin = min(vmin, float(wss.min())); vmax = max(vmax, float(wss.max()))
        chunks.append(wss.astype(np.float32))
    q = np.concatenate(chunks)
    mean_lin = s_lin / n
    std_lin = float(np.sqrt(max(ss_lin / n - mean_lin ** 2, 1e-12)))
    mean_log = s_log / n
    std_log = float(np.sqrt(max(ss_log / n - mean_log ** 2, 1e-12)))
    return {
        "method": "log_z",
        "eps": eps,
        "n_points": int(n),
        "n_cases": len(train_labels),
        "timesteps_scope": "peak",
        "zero_frac": float(n_zero / n),
        "linear": {"mean": mean_lin, "std": std_lin},
        "log": {"mean": mean_log, "std": std_log},
        "raw_min": vmin, "raw_max": vmax,
        "raw_percentiles": {
            "p50": float(np.percentile(q, 50)),
            "p90": float(np.percentile(q, 90)),
            "p95": float(np.percentile(q, 95)),
            "p99": float(np.percentile(q, 99)),
        },
        "partitions": ["train"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-split", default=str(C.DEFAULT_SPLIT))
    ap.add_argument("--out-dir", default=str(C.PROJECT_ROOT / "training" / "splits"))
    ap.add_argument("--stats-dir", default=str(C.DATA_ROOT / "fold_stats"))
    ap.add_argument("--val-n", type=int, default=8)
    ap.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    args = ap.parse_args()

    base = json.loads(Path(args.base_split).read_text())
    pool = list(base["train_cases"]) + list(base["val_cases"])
    assert len(pool) == 61, f"expected 61 train+val, got {len(pool)}"
    # 禁止 test 进入
    test_set = set(base.get("test_cases", []))
    assert not (set(pool) & test_set)

    data_root = C.DATA_ROOT
    rows = []
    for label in pool:
        subset, case = label.split("/", 1)
        p = data_root / "AG" / subset / case / "bundle.npz"
        if not p.is_file():
            raise FileNotFoundError(p)
        rows.append(_case_strata(label, p))

    out_dir = Path(args.out_dir)
    stats_dir = Path(args.stats_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    for i, seed in enumerate(args.seeds, start=1):
        train, val = _stratified_holdout(rows, args.val_n, seed)
        assert len(train) + len(val) == 61
        assert len(val) == args.val_n
        assert not (set(train) & set(val))
        assert not (set(train) | set(val)) & test_set
        split = {
            "split_version": f"split_AG_wss_min_v2_dev{i}",
            "source": "AG train+val from split_AG_wss_min_v1 (61 cases); test excluded",
            "pipeline": "pipeline_wss_min",
            "data_root": "data_wss_min/AG",
            "parent_split": "split_AG_wss_min_v1",
            "holdout_seed": seed,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "expected_counts": {"train": len(train), "val": len(val), "test": 0},
            "train_cases": train,
            "val_cases": val,
            "test_cases": [],
            "excluded_cases": base.get("excluded_cases", []),
            "pending_cases": base.get("pending_cases", []),
            "notes": "第四轮开发划分；WSS stats 必须用同 fold train 重算，禁止复用 v1 全局 53 例 stats",
        }
        spath = out_dir / f"split_AG_wss_min_v2_dev{i}.json"
        spath.write_text(json.dumps(split, indent=2, ensure_ascii=False) + "\n")
        stats = compute_fold_stats(train, data_root)
        stats["split_name"] = split["split_version"]
        stats["split_path"] = str(spath)
        stpath = stats_dir / f"wss_stats_v2_dev{i}.json"
        stpath.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
        print(f"dev{i}: train={len(train)} val={len(val)} "
              f"log_mean={stats['log']['mean']:.4f} -> {spath.name} / {stpath.name}")


if __name__ == "__main__":
    main()
