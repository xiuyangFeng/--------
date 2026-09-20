"""Support-density robustness re-evaluation (inference only).

Deployment point clouds will be coarser than the CFD wall node set. This tool keeps the query set
(all wall nodes, with their precomputed features) fixed and draws the fixed-size encoder support
from a randomly thinned node pool (fraction 1.0 / 0.5 / 0.25 of the nodes), so it isolates the
sensitivity of the encoder to support-cloud density. Per-node features are NOT recomputed on the
thinned cloud (that needs the wss_v5 geometry program) - stated limitation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import surface as S
from training_wss_min.evaluate import load_model_from_run, load_wss_stats_for_run


def thin_case(case: dict, frac: float, rng: np.random.Generator) -> dict:
    n = len(case["pos"])
    keep = np.sort(rng.choice(n, size=max(int(round(n * frac)), 1), replace=False))
    out = {}
    for k, v in case.items():
        if k.startswith("_"):
            continue
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
            out[k] = v[keep]
        else:
            out[k] = v
    return out


def predict_with_pool(model, case, pool, cfg, feat_stats, device):
    support_n = int(cfg.data.support_n_points or cfg.data.wall_n_points)
    support_sampling = cfg.data.support_sampling or cfg.data.sampling
    seed = S.stable_seed(cfg.eval.support_seed, case["unit_id"], "eval_support")
    idx = D.sample_indices(pool, cfg.data, seed, n_points=support_n, sampling=support_sampling, stream="eval_support")
    sp = torch.from_numpy(np.ascontiguousarray(pool["pos"][idx])).to(device)
    sx = torch.from_numpy(D.build_features(pool, idx, cfg.data.input_features, feat_stats)).to(device)
    sb = torch.zeros(len(idx), dtype=torch.long, device=device)
    enc = model.encode_support(sp, sx, sb, unit_ids=[case["unit_id"]], epoch=0, global_seed=cfg.eval.support_seed, evaluation=True)
    chunk = int(cfg.eval.query_chunk_size) or len(case["pos"])
    outs = []
    for start in range(0, len(case["pos"]), chunk):
        q = np.arange(start, min(start + chunk, len(case["pos"])))
        qp = torch.from_numpy(np.ascontiguousarray(case["pos"][q])).to(device)
        qx = torch.from_numpy(D.build_features(case, q, cfg.data.input_features, feat_stats)).to(device)
        qb = torch.zeros(len(q), dtype=torch.long, device=device)
        outs.append(model.decode_query(enc, qp, qx, qb).detach().float().cpu())
    o = torch.cat(outs)
    if o.ndim == 2:
        o = o[:, 0]
    return o.numpy().astype(np.float64), len(idx), len(pool["pos"])


def r2(y, p):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def r2_cb(ys, ps):
    gm = np.mean([y.mean() for y in ys])
    return 1.0 - np.mean([np.mean((y - p) ** 2) for y, p in zip(ys, ps)]) / np.mean([np.mean((y - gm) ** 2) for y in ys])


def summarize(ys, ps):
    y = np.concatenate(ys); p = np.concatenate(ps)
    m = y >= np.percentile(y, 90)
    masks = [a >= np.nanpercentile(a, 90) for a in ys]
    yh = np.concatenate([a[k] for a, k in zip(ys, masks)]); ph = np.concatenate([a[k] for a, k in zip(ps, masks)])
    return {"r2_cb": r2_cb(ys, ps), "r2_pooled": r2(y, p), "case_mean_r2": float(np.mean([r2(a, b) for a, b in zip(ys, ps)])),
            "top10_ratio": float(p[m].mean() / y[m].mean()), "p99_ratio": float(np.percentile(p, 99) / np.percentile(y, 99)), "high_wss_r2": r2(yh, ph)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--fractions", default="1.0,0.5,0.25")
    ap.add_argument("--repeats", type=int, default=2, help="independent thinning draws per fraction (<1.0)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, feat_stats, model, _ = load_model_from_run(run_dir, device, "best")
    model.eval()
    stats = load_wss_stats_for_run(run_dir)
    cases = D.load_partition(cfg.data.split_path, "test", stats, target=cfg.data.target, target_normalization=cfg.data.target_normalization,
                             data_root=cfg.data.data_root, required_frame_version=cfg.data.required_frame_version)
    for c in cases:
        c.setdefault("unit_id", f"{c.get('cohort')}/{c.get('case')}")
    rows = []
    with torch.no_grad():
        for frac in [float(x) for x in args.fractions.split(",")]:
            reps = 1 if frac >= 1.0 else args.repeats
            for rep in range(reps):
                ys_raw, ps_raw, ys_n, ps_n, pool_sizes = [], [], [], [], []
                for ci, case in enumerate(cases):
                    rng = np.random.default_rng(10_000 * rep + ci + 1)
                    pool = case if frac >= 1.0 else thin_case(case, frac, rng)
                    pn, n_sup, n_pool = predict_with_pool(model, case, pool, cfg, feat_stats, device)
                    pool_sizes.append(n_pool)
                    ps_n.append(pn); ys_n.append(np.asarray(case["y_norm"], dtype=np.float64))
                    pr = D.denormalize_wss(pn, stats)
                    if stats.get("method") == "log_z":
                        pr = np.clip(pr, 0, None)
                    ps_raw.append(pr); ys_raw.append(np.asarray(case["y_raw"], dtype=np.float64))
                phys = summarize(ys_raw, ps_raw); norm = summarize(ys_n, ps_n)
                rows.append({"pool_fraction": frac, "repeat": rep, "median_pool_nodes": int(np.median(pool_sizes)), "support_n": n_sup,
                             "physical": phys, "normalized": norm})
                print(f"pool {frac:.2f} rep {rep}: pool~{int(np.median(pool_sizes))} nodes, support {n_sup} | phys R2cb {phys['r2_cb']:.4f} casemean {phys['case_mean_r2']:.4f} top10 {phys['top10_ratio']:.3f} p99 {phys['p99_ratio']:.3f} highR2 {phys['high_wss_r2']:.3f} | norm R2cb {norm['r2_cb']:.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"run_dir": str(run_dir), "rows": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
