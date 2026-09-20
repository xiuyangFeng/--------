#!/usr/bin/env python3
"""Did the bottleneck actually attend globally? Post-hoc attention analysis for a trained BT run.

For each evaluated case this reports, per bottleneck layer:
  * entropy of the attention rows (max = ln(n_tokens): uniform/global; near 0: collapsed to one token),
  * effective number of attended tokens exp(entropy),
  * attention-weighted mean distance between query and attended token, in mm, versus the mean pairwise distance
    of the tokens themselves (ratio ~1 means the attention is spatially indiscriminate, <<1 means it stayed local),
  * the learned residual scales gamma.

Implemented with a forward pre-hook so nothing in baseline_models.py has to change (and so it is safe to run
while other trainings are using that module).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min.evaluate import load_model_from_run, load_wss_stats_for_run
from training_wss_min import surface as S


def analyse(run_dir: Path, n_cases: int, device: str) -> dict:
    cfg, feat_stats, model, _ = load_model_from_run(run_dir, device, "best")
    model.eval()
    bt = getattr(model, "bottleneck", None)
    if bt is None:
        raise SystemExit(f"{run_dir.name} has no bottleneck module")
    if len(bt.layers) == 0:
        return {"run": run_dir.name, "note": "layers=0 (conditioning-only control): no attention to analyse",
                "gammas": {}}
    captured = []

    def pre_hook(module, args):
        captured.append(tuple(a.detach() for a in args))

    handles = [layer.register_forward_pre_hook(pre_hook) for layer in bt.layers]
    stats = load_wss_stats_for_run(run_dir)
    cases = D.load_partition(cfg.data.split_path, "test", stats, target=cfg.data.target,
                             target_normalization=cfg.data.target_normalization,
                             data_root=cfg.data.data_root,
                             required_frame_version=cfg.data.required_frame_version)[:n_cases]
    rows = []
    for case in cases:
        case.setdefault("unit_id", f"{case['cohort']}/{case['case']}")
        captured.clear()
        support_n = int(cfg.data.support_n_points or cfg.data.wall_n_points)
        seed = S.stable_seed(cfg.eval.support_seed, case["unit_id"], "eval_support")
        idx = D.sample_support_indices(case, cfg.data, seed, n_points=support_n,
                                       sampling=cfg.data.support_sampling or cfg.data.sampling, stream="eval_support")
        pos = torch.from_numpy(np.ascontiguousarray(case["pos"][idx])).to(device)
        x = torch.from_numpy(D.build_features(case, idx, cfg.data.input_features, feat_stats)).to(device)
        batch = torch.zeros(len(idx), dtype=torch.long, device=device)
        with torch.no_grad():
            model.encode_support(pos, x, batch, unit_ids=[case["unit_id"]], epoch=0,
                                 global_seed=cfg.eval.support_seed, evaluation=True)
        scale = float(case["coord_scale_scalar"])  # normalized coords -> mm
        for li, (layer, args) in enumerate(zip(bt.layers, captured)):
            dense, dense_pos, valid = args
            with torch.no_grad():
                h = layer.norm_attn(dense)
                b, n, _ = h.shape
                qkv = layer.qkv(h).reshape(b, n, 3, layer.heads, layer.head_dim)
                q, k, _ = qkv.unbind(dim=2)
                logits = torch.einsum("bihd,bjhd->bhij", q, k) / (layer.head_dim ** 0.5)
                if layer.geo_bias is not None:
                    delta = dense_pos[:, :, None, :] - dense_pos[:, None, :, :]
                    dist = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                    logits = logits + layer.geo_bias(torch.cat([delta, dist], dim=-1)).permute(0, 3, 1, 2)
                logits = logits.masked_fill(~valid[:, None, None, :], torch.finfo(logits.dtype).min)
                att = torch.softmax(logits, dim=-1)[0].float().cpu().numpy()  # (heads, n, n)
                pts = dense_pos[0][valid[0]].float().cpu().numpy() * scale
            m = int(valid[0].sum())
            att = att[:, :m, :m]
            att = att / np.clip(att.sum(-1, keepdims=True), 1e-12, None)
            ent = -(att * np.clip(att, 1e-12, None) * 0 + att * np.log(np.clip(att, 1e-12, None))).sum(-1)
            pd = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
            att_dist = (att * pd[None]).sum(-1)
            rows.append({
                "case": case["unit_id"], "layer": li, "n_tokens": m,
                "entropy": float(ent.mean()), "entropy_max": float(np.log(m)),
                "effective_tokens": float(np.exp(ent.mean())),
                "attended_mean_dist_mm": float(att_dist.mean()),
                "token_mean_pairwise_dist_mm": float(pd[np.triu_indices(m, 1)].mean()),
            })
    for h in handles:
        h.remove()
    gam = {k: float(v) for k, v in model.state_dict().items() if "bottleneck" in k and "gamma" in k}
    out = {"run": run_dir.name, "n_cases": len(cases), "per_case": rows, "gammas": gam}
    for li in sorted({r["layer"] for r in rows}):
        sel = [r for r in rows if r["layer"] == li]
        out[f"layer{li}_summary"] = {
            "n_tokens": sel[0]["n_tokens"],
            "entropy": float(np.mean([r["entropy"] for r in sel])),
            "entropy_max": sel[0]["entropy_max"],
            "effective_tokens": float(np.mean([r["effective_tokens"] for r in sel])),
            "attended_mean_dist_mm": float(np.mean([r["attended_mean_dist_mm"] for r in sel])),
            "token_mean_pairwise_dist_mm": float(np.mean([r["token_mean_pairwise_dist_mm"] for r in sel])),
            "reach_ratio": float(np.mean([r["attended_mean_dist_mm"] / max(r["token_mean_pairwise_dist_mm"], 1e-9) for r in sel])),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directory names under runs/v5_rerun_20260906/outputs")
    ap.add_argument("--n-cases", type=int, default=6)
    ap.add_argument("--out", default="training_wss_min/experiments/v5_rerun_20260906/bt_attention_analysis.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = C.PROJECT_ROOT / "training_wss_min" / "runs" / "v5_rerun_20260906" / "outputs"
    results = {}
    for name in args.runs:
        res = analyse(root / name, args.n_cases, device)
        results[name] = res
        print(f"== {name}")
        if "note" in res:
            print("   ", res["note"]); continue
        for k, v in res.items():
            if k.endswith("_summary"):
                print(f"   {k}: tokens={v['n_tokens']}  entropy {v['entropy']:.2f} / max {v['entropy_max']:.2f}"
                      f"  有效关注 token 数 {v['effective_tokens']:.1f}"
                      f"  注意力加权距离 {v['attended_mean_dist_mm']:.1f} mm vs token 平均间距 {v['token_mean_pairwise_dist_mm']:.1f} mm"
                      f"  → reach {v['reach_ratio']:.2f}")
        print("    gammas:", {k.split('layers.')[-1]: round(v, 4) for k, v in res["gammas"].items()})
    Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
