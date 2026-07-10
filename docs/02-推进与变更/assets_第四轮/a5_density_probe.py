#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A5 密度敏感性诊断（扩展版）。

在既有标准化口径之外，补齐：
1) 同索引 raw-space top10 mask / 幅值差；
2) 各层 radius(..., max_num_neighbors) 达到上限的 cap 比例。

用法：
  <GNN_env_python> docs/02-推进与变更/assets_第四轮/a5_density_probe.py \
      [--run-dirs ...] [--with-cap] [--out ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from training_wss_min import dataset as D  # noqa: E402
from training_wss_min.evaluate import (  # noqa: E402
    load_model_from_run, load_wss_stats_for_run, predict_case_norm,
)

DEFAULT_RUNS = [
    "training_wss_min/runs/r3_clean_xyzgeom_tgtw_s1234",
    "training_wss_min/runs/r3_clean_xyzgeom_tgtw_s7",
    "training_wss_min/runs/r3_clean_xyzgeom_tgtw_s2025",
]


class RadiusCapProbe:
    """Hook torch_geometric.nn.pool.radius 统计邻居数达上限比例。"""

    def __init__(self):
        self.records = []
        self._orig = None
        self._layer = 0

    def __enter__(self):
        import torch_geometric.nn.pool as pool
        self._orig = pool.radius
        probe = self

        def wrapped(*args, **kwargs):
            out = probe._orig(*args, **kwargs)
            max_n = kwargs.get("max_num_neighbors", None)
            if max_n is None and len(args) >= 6:
                max_n = args[5]
            if max_n is not None and out.numel() > 0:
                # pointnext: row, col = assign[0], assign[1]; scatter(..., row) 以 row 为 query
                if isinstance(out, torch.Tensor) and out.dim() == 2 and out.size(0) == 2:
                    row = out[0]
                    n_query = int(row.max().item()) + 1 if row.numel() else 0
                    if n_query > 0:
                        counts = torch.bincount(row, minlength=n_query).float()
                        cap_frac = float((counts >= float(max_n)).float().mean().item())
                        mean_deg = float(counts.mean().item())
                        probe.records.append({
                            "layer": probe._layer,
                            "max_num_neighbors": int(max_n),
                            "cap_frac": cap_frac,
                            "mean_degree": mean_deg,
                            "n_query": n_query,
                            "n_edges": int(out.size(1)),
                        })
                        probe._layer += 1
            return out

        pool.radius = wrapped
        # also patch local import site used by pointnext
        import training_wss_min.pointnext as pn
        if hasattr(pn, "radius"):
            self._pn_orig = pn.radius
            pn.radius = wrapped
        else:
            self._pn_orig = None
        return self

    def __exit__(self, *exc):
        import torch_geometric.nn.pool as pool
        if self._orig is not None:
            pool.radius = self._orig
        if self._pn_orig is not None:
            import training_wss_min.pointnext as pn
            pn.radius = self._pn_orig
        return False


def _top10_raw_metrics(y_true, raw_sub, raw_full):
    thr = np.percentile(y_true, 90.0)
    m = y_true >= thr
    if not m.any():
        return {}
    return {
        "top10_mae_sub_vs_full": float(np.mean(np.abs(raw_sub[m] - raw_full[m]))),
        "top10_mean_sub": float(raw_sub[m].mean()),
        "top10_mean_full": float(raw_full[m].mean()),
        "top10_mean_shift": float(raw_sub[m].mean() - raw_full[m].mean()),
        "top10_ratio_sub_over_full": float(raw_sub[m].mean() / (raw_full[m].mean() + 1e-12)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--partition", default="val")
    ap.add_argument("--n-points", type=int, default=2000)
    ap.add_argument("--with-cap", action="store_true", help="hook radius 统计 cap 比例（较慢）")
    ap.add_argument("--out", default=str(Path(__file__).parent / "a5_density_probe.csv"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    cap_rows = []
    meta = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": device, "partition": args.partition,
            "n_points": args.n_points, "with_cap": args.with_cap, "runs": {}}

    for rd in args.run_dirs:
        run_dir = Path(rd) if Path(rd).is_absolute() else REPO_ROOT / rd
        cfg, feat_stats, model, ckpt = load_model_from_run(run_dir, device)
        wss_stats = load_wss_stats_for_run(run_dir)
        cases = D.load_partition(cfg.data.split_path, args.partition, wss_stats, strict=True)
        model.eval()
        meta["runs"][run_dir.name] = {
            "split_path": cfg.data.split_path,
            "ckpt_epoch": int(ckpt.get("epoch", -1)),
            "train_seed": cfg.train.seed,
            "n_cases": len(cases),
        }
        for i, case in enumerate(cases):
            seed = cfg.train.seed + 7919 * i
            idx = D.farthest_point_sample(case["pos"], args.n_points, seed)
            feat = D.build_features(case, idx, cfg.data.input_features, feat_stats)
            with torch.no_grad():
                pred_sub = model(
                    torch.from_numpy(np.ascontiguousarray(case["pos"][idx])).to(device),
                    torch.from_numpy(feat).to(device),
                    torch.zeros(len(idx), dtype=torch.long, device=device),
                ).detach().float().cpu().numpy()
            if args.with_cap:
                probe = RadiusCapProbe()
                with probe, torch.no_grad():
                    _ = predict_case_norm(model, case, cfg.data.input_features,
                                          feat_stats, device)
                for rec in probe.records:
                    cap_rows.append({
                        "run": run_dir.name,
                        "case": f"{case['cohort']}/{case['case']}",
                        "mode": "full",
                        **rec,
                    })
                probe2 = RadiusCapProbe()
                with probe2, torch.no_grad():
                    _ = model(
                        torch.from_numpy(np.ascontiguousarray(case["pos"][idx])).to(device),
                        torch.from_numpy(feat).to(device),
                        torch.zeros(len(idx), dtype=torch.long, device=device),
                    )
                for rec in probe2.records:
                    cap_rows.append({
                        "run": run_dir.name,
                        "case": f"{case['cohort']}/{case['case']}",
                        "mode": "sub",
                        **rec,
                    })
                pred_full = predict_case_norm(model, case, cfg.data.input_features,
                                              feat_stats, device)[idx]
            else:
                pred_full = predict_case_norm(model, case, cfg.data.input_features,
                                              feat_stats, device)[idx]
            d = pred_sub - pred_full
            raw_sub = np.clip(D.denormalize_wss(pred_sub, wss_stats), 0, None)
            raw_full = np.clip(D.denormalize_wss(pred_full, wss_stats), 0, None)
            y_true_idx = case["y_raw"][idx].astype(np.float64)
            top10 = _top10_raw_metrics(y_true_idx, raw_sub, raw_full)
            row = {
                "run": run_dir.name, "case": f"{case['cohort']}/{case['case']}",
                "n_total": len(case["pos"]), "n_sub": len(idx),
                "mae_norm": float(np.mean(np.abs(d))),
                "rmse_norm": float(np.sqrt(np.mean(d ** 2))),
                "pearson_norm": float(np.corrcoef(pred_sub, pred_full)[0, 1]),
                "mean_shift_norm": float(np.mean(d)),
                "mae_raw": float(np.mean(np.abs(raw_sub - raw_full))),
                "p99_raw_sub": float(np.percentile(raw_sub, 99)),
                "p99_raw_full": float(np.percentile(raw_full, 99)),
            }
            row.update(top10)
            rows.append(row)
        rr = [r for r in rows if r["run"] == run_dir.name]
        print(f"[{run_dir.name}] cases={len(rr)}  "
              f"MAE_norm={np.mean([r['mae_norm'] for r in rr]):.4f}  "
              f"top10_mean_shift={np.mean([r.get('top10_mean_shift', np.nan) for r in rr]):+.4f}")

    out = Path(args.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta_path = out.with_name(out.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    if cap_rows:
        cap_out = out.with_name(out.stem + "_cap.csv")
        with open(cap_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cap_rows[0].keys()))
            w.writeheader(); w.writerows(cap_rows)
        print("写入 cap:", cap_out)
    print("写入:", out)


if __name__ == "__main__":
    main()
