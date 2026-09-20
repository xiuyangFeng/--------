#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Did the new V6 modules actually activate?

A null result means two very different things depending on this file: "the mechanism does not
help here" or "this branch never grew away from its zero initialisation".  Every V6 module starts
as an exact no-op, so the learned magnitude is readable straight from the checkpoint, and the
per-case amplitude head can be reported in the physical unit that matters (a Pa scale factor).

    python -m training_wss_min.tools.v6_module_diagnostics [--runs DIR] [--out FILE]
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

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "training_wss_min/runs/v6_singleframe_20260909"
EXP = ROOT / "training_wss_min/experiments/v6_singleframe_20260909"


def tensor_report(t: torch.Tensor) -> dict:
    v = t.detach().float().flatten()
    return {"n": int(v.numel()), "abs_mean": float(v.abs().mean()), "abs_max": float(v.abs().max()),
            "l2": float(v.norm())}


def diagnose(run_dir: Path, device: str) -> dict:
    cfg, feat_stats, model, ckpt = load_model_from_run(run_dir, device, "best")
    state = ckpt["model"]
    out: dict = {"run": run_dir.name, "epoch": ckpt.get("epoch")}
    if cfg.model.local_branch:
        out["local_wall_branch"] = {
            "fuse_weight": tensor_report(state["local_wall_branch.fuse.weight"]),
            "fuse_bias": tensor_report(state["local_wall_branch.fuse.bias"]),
            "note": "zero-initialised; a non-zero fuse weight means the multi-scale branch was used",
        }
    if cfg.model.query_decoder == "local_attn":
        out["local_query_decoder"] = {
            "beta": float(state["local_query.beta"]),
            "beta_init": float(cfg.model.sep_beta_init),
            "weight_correction_last_layer": tensor_report(state["local_query.correction.2.weight"]),
            "residual_out": (tensor_report(state["local_query.res_out.weight"])
                             if "local_query.res_out.weight" in state else None),
            "note": "beta=2 with zero corrections == the parent 1/d^2 k-NN interpolation",
        }
    if cfg.model.case_scale_head:
        out["case_scale_head"] = {"last_layer": tensor_report(state["case_scale.mlp.2.weight"]),
                                  "bias": float(state["case_scale.mlp.2.bias"])}
        # the interesting number is the per-case Pa scale factor exp(bias * log_std)
        stats = load_wss_stats_for_run(run_dir)
        cases = D.load_partition(
            cfg.data.split_path, "test", stats, target=cfg.data.target,
            target_normalization=cfg.data.target_normalization, data_root=cfg.data.data_root,
            required_frame_version=cfg.data.required_frame_version,
            case_features_path=cfg.data.case_features_path,
            extra_point_features=C.v6_point_features(cfg),
            point_features_root=getattr(cfg.data, "point_features_root", None),
        )
        model.eval()
        scales = {}
        with torch.no_grad():
            for case in cases:
                seed = D.S.stable_seed(cfg.eval.support_seed, case["unit_id"], "eval_support")
                idx = D.sample_support_indices(
                    case, cfg.data, seed, n_points=int(cfg.data.support_n_points or cfg.data.wall_n_points),
                    sampling=cfg.data.support_sampling or cfg.data.sampling, stream="eval_support")
                pos = torch.from_numpy(np.ascontiguousarray(case["pos"][idx])).to(device)
                x = torch.from_numpy(D.build_features(case, idx, cfg.data.input_features, feat_stats)).to(device)
                batch = torch.zeros(len(idx), dtype=torch.long, device=device)
                encoded = model.encode_support(pos, x, batch, unit_ids=[case["unit_id"]], epoch=0,
                                               global_seed=cfg.eval.support_seed, evaluation=True)
                scales[case["unit_id"]] = float(encoded[4][0])
        bias = np.array(list(scales.values()))
        factor = np.exp(bias * float(stats["log"]["std"]))
        out["case_scale_head"].update({
            "per_case_bias_z": {"mean": float(bias.mean()), "sd": float(bias.std()),
                                "min": float(bias.min()), "max": float(bias.max())},
            "per_case_pa_factor": {"mean": float(factor.mean()), "sd": float(factor.std()),
                                   "min": float(factor.min()), "max": float(factor.max())},
            "per_case": {k: {"bias_z": v, "pa_factor": float(np.exp(v * float(stats["log"]["std"])))}
                         for k, v in scales.items()},
            "note": "a per-case additive bias in log_z space is a multiplicative Pa amplitude factor",
        })
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--out", default=str(EXP / "module_diagnostics.json"))
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reports = []
    for run_dir in sorted(Path(args.runs).iterdir()):
        if not (run_dir / "ckpt_best.pt").is_file():
            continue
        report = diagnose(run_dir, device)
        if len(report) > 2:          # something beyond run/epoch
            reports.append(report)
            print(json.dumps(report, ensure_ascii=False)[:400])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
