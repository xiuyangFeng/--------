#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KNN / voxel / mean WSS 模板基线。

回答：统一坐标框架下的训练集平均热力图本身能达到多少 R2。
输出结构与深度模型 eval 对齐，落在 training_wss_min/runs/template_*_clean/eval/。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from . import config as C
from . import dataset as D
from . import metrics as M
from .evaluate import write_reports


def _stack_train(cases: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.concatenate([c["pos"] for c in cases], axis=0).astype(np.float32)
    y = np.concatenate([c["y_norm"] for c in cases], axis=0).astype(np.float32)
    return pos, y


def _result_from_predictions(cases: List[Dict], preds_raw: List[np.ndarray]) -> Dict:
    per_case: Dict[str, Dict] = {}
    pooled_true, pooled_pred = [], []
    for case, yp in zip(cases, preds_raw):
        yt = case["y_raw"].astype(np.float64)
        reg = M.regional_metrics(case["pos"], case["local_radius"], yt, yp)
        reg["calibration"] = M.calibration_metrics(yt, yp)
        per_case[f"{case['cohort']}/{case['case']}"] = reg
        pooled_true.append(yt); pooled_pred.append(yp)
    pt, pp = np.concatenate(pooled_true), np.concatenate(pooled_pred)
    return {
        "aggregate": M.aggregate_case_metrics(per_case),
        "field": M.basic_metrics(pt, pp),
        "calibration": M.calibration_metrics(pt, pp),
        "regional_field": _regional_field(cases, pooled_true, pooled_pred),
        "per_case": per_case,
    }


def _regional_field(cases, pooled_true, pooled_pred) -> Dict:
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


class TemplatePredictor:
    def __init__(self, method: str, train_cases: List[Dict], stats: Dict,
                 k: int = 16, voxel_res: int = 48):
        self.method = method
        self.stats = stats
        self.k = k
        self.voxel_res = voxel_res
        self.pos, self.y = _stack_train(train_cases)
        self.global_y = float(np.mean(self.y))
        self.nn = None
        self.voxels = {}
        if method == "knn":
            self.nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1).fit(self.pos)
        elif method == "voxel":
            self._build_voxels()
        elif method != "mean":
            raise ValueError(f"unsupported template method: {method}")

    def _voxel_key(self, pos: np.ndarray) -> np.ndarray:
        x = np.clip((pos + 1.0) * 0.5, 0.0, 0.999999)
        return np.floor(x * self.voxel_res).astype(np.int32)

    def _build_voxels(self):
        keys = self._voxel_key(self.pos)
        sums: Dict[tuple, float] = {}
        counts: Dict[tuple, int] = {}
        for key, y in zip(map(tuple, keys), self.y):
            sums[key] = sums.get(key, 0.0) + float(y)
            counts[key] = counts.get(key, 0) + 1
        self.voxels = {k: sums[k] / counts[k] for k in sums}

    def predict_norm(self, pos: np.ndarray) -> np.ndarray:
        if self.method == "mean":
            return np.full(len(pos), self.global_y, dtype=np.float32)
        if self.method == "knn":
            assert self.nn is not None
            _, idx = self.nn.kneighbors(pos)
            return self.y[idx].mean(axis=1).astype(np.float32)
        keys = self._voxel_key(pos)
        return np.asarray(
            [self.voxels.get(tuple(k), self.global_y) for k in keys],
            dtype=np.float32,
        )


def run_baseline(method: str, split_path: str, partitions: Tuple[str, ...],
                 k: int, voxel_res: int) -> Path:
    stats = D.load_wss_stats()
    train_cases = D.load_partition(split_path, "train", stats)
    predictor = TemplatePredictor(method, train_cases, stats, k=k, voxel_res=voxel_res)

    run_dir = C.RUNS_ROOT / f"template_{method}_clean"
    eval_dir = run_dir / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "name": run_dir.name,
        "method": method,
        "data": {
            "split_path": split_path,
            "wall_n_points": 0,
            "sampling": method,
            "input_features": ["x", "y", "z"],
        },
        "template": {"k": k, "voxel_res": voxel_res},
    }, indent=2, ensure_ascii=False))
    (run_dir / "wss_global_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    result_by_part = {}
    for part in partitions:
        cases = D.load_partition(split_path, part, stats)
        preds = [
            np.clip(D.denormalize_wss(predictor.predict_norm(c["pos"]), stats), 0, None)
            for c in cases
        ]
        result_by_part[part] = _result_from_predictions(cases, preds)
    write_reports(result_by_part, eval_dir)
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="mean,voxel,knn")
    ap.add_argument("--split-path", default=str(C.DEFAULT_SPLIT))
    ap.add_argument("--partitions", default="val,test")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--voxel-res", type=int, default=48)
    args = ap.parse_args()

    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        run_dir = run_baseline(
            method, args.split_path, tuple(args.partitions.split(",")),
            k=args.k, voxel_res=args.voxel_res,
        )
        print(f"[template] {method} -> {run_dir}")


if __name__ == "__main__":
    main()
