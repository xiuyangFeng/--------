#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第五轮 LC（learning curve）配置/划分/stats 生成器。

用法：
  python -m training_wss_min.make_configs_round5_lc --min-lr 2e-4 --seeds 1234 [--write]

产物（--write 时才落盘）：
  training/splits/split_AG_wss_min_v2_dev1_lc_<chain>_<size>.json  （train=子集, val=dev1 8 例）
  data_wss_min/fold_stats/wss_stats_v2_dev1_lc_<chain>_<size>.json （子集 train-only log_z stats）
  training_wss_min/configs/round5/lc/r5_lc_<chain>_<size>_s<seed>.json

设计（与 A0D 后 LC 卡一致）：
  * R=3 条独立嵌套链 chainA/chainB/chainC；每条 LC13 ⊂ LC26 ⊂ LC40 ⊂ LC53。
  * LC53 = 全部 53 例（三链公共终点）→ 复用既有 dev1 split/stats，不另建。
  * 硬分层 = cohort(fast/slow) × train-only peak-WSS 三分位(low/mid/high)。
  * case-drop 随机性只由 chain salt 决定（sha256），与 model seed 完全分离，不依赖 wall-clock/PYTHONHASHSEED。
  * 不使用任何 val/test 标签构造分层；不自动纳入 AAA/ILO/excluded/pending。
  feature stats 与 loss 分位（y_norm_q02/q98 等）由 train.py 在运行时按子集 train 自动重算，故此处只需生成 split + WSS stats。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from . import config as C
from .make_v2_dev_splits import compute_fold_stats

ROOT = C.PROJECT_ROOT
DEV1_SPLIT = ROOT / "training" / "splits" / "split_AG_wss_min_v2_dev1.json"
WSS_CSV = ROOT / "training_wss_min" / "runs" / "_round5" / "a0_micro" / "train_only_wss_candidates.csv"
SPLIT_DIR = ROOT / "training" / "splits"
STATS_DIR = C.DATA_ROOT / "fold_stats"
CFG_DIR = ROOT / "training_wss_min" / "configs" / "round5" / "lc"
DATA_ROOT = C.DATA_ROOT

CHAINS = ["chainA", "chainB", "chainC"]
SIZES = [13, 26, 40, 53]


def _load_inputs():
    d = json.loads(DEV1_SPLIT.read_text())
    train = list(d["train_cases"])
    val = list(d["val_cases"])
    assert len(train) == 53 and len(val) == 8, (len(train), len(val))
    wss = {}
    with open(WSS_CSV) as f:
        for row in csv.DictReader(f):
            wss[row["case"]] = float(row["wss_mean"])
    assert set(wss) == set(train), "WSS CSV 与 train 集不一致"
    return train, val, wss, d


def _strata(train, wss):
    vals = np.array([wss[c] for c in train], dtype=np.float64)
    p33, p66 = np.percentile(vals, 100 / 3), np.percentile(vals, 200 / 3)

    def tert(v):
        return "low" if v < p33 else ("mid" if v < p66 else "high")

    strata = {}
    for c in train:
        cohort = c.split("/", 1)[0]
        strata[c] = f"{cohort}/{tert(wss[c])}"
    return strata, float(p33), float(p66)


def _order_for_chain(train, strata, salt):
    """确定性分层交错排序：链内嵌套即取前缀。"""
    # 每个 stratum 内按 sha256(salt:case) 排序
    buckets: dict[str, list] = {}
    for c in train:
        buckets.setdefault(strata[c], []).append(c)
    for s in buckets:
        buckets[s].sort(key=lambda c: hashlib.sha256(f"{salt}:{c}".encode()).hexdigest())
    counts = {s: len(v) for s, v in buckets.items()}
    idx = {s: 0 for s in buckets}
    total = len(train)
    picked: dict[str, int] = {s: 0 for s in buckets}
    order = []
    for t in range(total):
        # deficit-greedy：选“已选 vs 目标比例”最亏欠且仍有剩余的 stratum
        best, best_key = None, None
        for s in buckets:
            if idx[s] >= counts[s]:
                continue
            deficit = (t + 1) * (counts[s] / total) - picked[s]
            key = (deficit, counts[s], s)  # 平局偏向更大 stratum，再按名字稳定
            if best_key is None or key > best_key:
                best_key, best = key, s
        order.append(buckets[best][idx[best]])
        idx[best] += 1
        picked[best] += 1
    assert len(order) == total and len(set(order)) == total
    return order


def _write_split(base, train_subset, val, path):
    d = dict(base)
    d["split_version"] = path.stem
    d["parent_split"] = "split_AG_wss_min_v2_dev1"
    d["train_cases"] = list(train_subset)
    d["val_cases"] = list(val)
    d["test_cases"] = []
    d["expected_counts"] = {"train": len(train_subset), "val": len(val), "test": 0}
    d["notes"] = ("第五轮 LC 子集；train=分层嵌套子集，val=dev1 固定 8 例；"
                  "WSS stats 用本子集 train 重算；不读 test/legacy。")
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False))


def _b1_config(name, notes, split_path, stats_path, seed, min_lr):
    return {
        "name": name, "notes": notes,
        "data": {
            "split_path": str(split_path),
            "wss_stats_path": str(stats_path),
            "wall_n_points": 2000, "sampling": "fps", "fps_pool_size": 8,
            "geom_weight_curv": 1.0, "geom_weight_invradius": 1.0,
            "geom_weight_bifurcation": 0.0,
            "resample_each_epoch": True, "rot_aug": False,
            "input_features": ["x", "y", "z", "abscissa_norm", "local_radius", "curvature"],
            "curvature_transform": "signed_log1p", "timesteps": "peak", "target": "wss",
            "num_workers": 4, "persistent_workers": False,
        },
        "model": {
            "name": "pointnext_s", "width": 32,
            "sa_ratios": [0.25, 0.25, 0.25, 0.25], "sa_radius": [0.05, 0.1, 0.2, 0.4],
            "sa_nsample": [16, 16, 16, 16], "sa_blocks": [1, 1, 1, 1],
            "invres_radius_scale": 1.0, "fp_knn": 3, "head_hidden": 64,
            "out_dim": 1, "dropout": 0.0,
        },
        "train": {
            "epochs": 160, "batch_cases": 8, "lr": 0.001, "weight_decay": 0.0001,
            "warmup_epochs": 10, "min_lr": min_lr, "grad_clip": 1.0, "seed": seed,
            "loss": "mse", "huber_delta": 1.0, "loss_geom_weight": False,
            "loss_weight_curv": 1.0, "loss_weight_invradius": 1.0,
            "loss_weight_target": True, "loss_weight_target_alpha": 2.0,
            "loss_weight_fixed_quantiles": True,
            "y_norm_q02": None, "y_norm_q98": None, "curv_q02": None, "curv_q98": None,
            "invr_q02": None, "invr_q98": None,
            "loss_raw_huber_lambda": 0.0, "raw_huber_delta": 1.0,
            "nll_logvar_min": -6.0, "nll_logvar_max": 2.0, "nll_logvar_reg": 0.0001,
            "amp": True, "eval_every": 10, "ckpt_metric": "val_selection_score",
            "selection_rule": "r4_composite_v1", "ckpt_top_k": 3,
            "early_stop_patience": 6, "min_epoch": 40, "log_every_steps": 0,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-lr", type=float, default=2e-4,
                    help="非饿死 LR 下限（A0E 校准后的协议）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234])
    ap.add_argument("--write", action="store_true", help="真正落盘；缺省只 dry-run 打印")
    args = ap.parse_args()

    train, val, wss, base = _load_inputs()
    strata, p33, p66 = _strata(train, wss)
    from collections import Counter
    print(f"tertile boundaries: p33={p33:.5f} p66={p66:.5f}")
    print("full-pool strata:", dict(Counter(strata.values())))

    if args.write:
        SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        STATS_DIR.mkdir(parents=True, exist_ok=True)
        CFG_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    # 预算 stats 缓存，避免重复读 bundle
    for chain in CHAINS:
        order = _order_for_chain(train, strata, chain)
        prev = None
        for size in SIZES:
            subset = order[:size]
            if prev is not None:
                assert set(prev).issubset(set(subset)), f"{chain} 非嵌套 {size}"
            prev = subset
            comp = Counter(strata[c] for c in subset)
            if size == 53:
                split_path = DEV1_SPLIT
                stats_path = STATS_DIR / "wss_stats_v2_dev1.json"
                tag = "all53(shared, reuse dev1)"
            else:
                split_path = SPLIT_DIR / f"split_AG_wss_min_v2_dev1_lc_{chain}_{size}.json"
                stats_path = STATS_DIR / f"wss_stats_v2_dev1_lc_{chain}_{size}.json"
                tag = "new"
                if args.write:
                    _write_split(base, subset, val, split_path)
                    st = compute_fold_stats(subset, DATA_ROOT)
                    st["split_name"] = split_path.stem
                    st["split_path"] = str(split_path)
                    stats_path.write_text(json.dumps(st, indent=2, ensure_ascii=False))
            for seed in args.seeds:
                name = f"r5_lc_{chain}_{size}_s{seed}"
                cfg = _b1_config(
                    name,
                    f"LC {chain} size={size} seed={seed}; B1 recipe; min_lr={args.min_lr}; "
                    f"val=dev1 8 例; strata={dict(comp)}",
                    split_path, stats_path, seed, args.min_lr,
                )
                cpath = CFG_DIR / f"{name}.json"
                if args.write:
                    cpath.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
                manifest.append({"chain": chain, "size": size, "seed": seed,
                                 "config": str(cpath), "strata": dict(comp),
                                 "shared53": size == 53})
            print(f"{chain} LC{size:<2} [{tag}] strata={dict(comp)} cohort="
                  f"{Counter(c.split('/',1)[0] for c in subset)}")

    if args.write:
        (CFG_DIR / "lc_manifest.json").write_text(
            json.dumps({"min_lr": args.min_lr, "seeds": args.seeds,
                        "runs": manifest}, indent=2, ensure_ascii=False))
        print(f"\nWROTE {len(manifest)} configs to {CFG_DIR}")
    else:
        print(f"\nDRY-RUN: would write {len(manifest)} configs (pass --write to persist)")


if __name__ == "__main__":
    main()
