#!/usr/bin/env python3
"""生成 V5.1 老师 Temporal Transformer 探针矩阵（P0.12）。

四臂是 2×2 小型因子设计：是否使用冻结的峰值 X5D query_x，是否沿时间做
self-attention。所有臂共享同折 cache、峰值锚、T-null 基底、训练病例和 seed；
只写新配置/新产物目录，旧时间适配器配置逐位不变。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
NAME = "wss_time_transformer_v51_20260924"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
RUNS = ROOT / "training_wss_min/runs" / NAME
CACHE = ROOT / "training_wss_min/experiments/wss_time_adapter_v51_20260923/cache"
VIEW = ROOT / "data_wss_v5/views_v5_1/wss_min_view_v1"
OFF = ROOT / "training_wss_min/experiments/wss_time_ecc_20260918/offline"
FOLDS = (0, 1, 2)
SEED = 1234

ARMS = {
    "TT-warm": {"inputs": ["query_x", "raw", "peak_ln"], "temporal": "transformer",
                "title": "teacher full: frozen peak X5D query_x + raw geometry skip + temporal self-attention"},
    "TT-warm-noattn": {"inputs": ["query_x", "raw", "peak_ln"], "temporal": "frame_mlp",
                       "title": "attention ablation: same warm spatial inputs, frame-independent time MLP"},
    "TT-raw": {"inputs": ["raw", "peak_ln"], "temporal": "transformer",
                "title": "transfer ablation: raw geometry + peak anchor + temporal self-attention"},
    "TT-raw-noattn": {"inputs": ["raw", "peak_ln"], "temporal": "frame_mlp",
                      "title": "capacity/control: raw geometry + peak anchor + frame-independent time MLP"},
}


def config(arm, k, epochs, steps_per_epoch):
    spec = ARMS[arm]
    aid = f"{arm}_f{k}_s{SEED}"
    return {
        "schema": "time_transformer_v51_v1", "name": f"{NAME}/{aid}", "experiment": NAME,
        "arm": arm, "fold": k, "seed": SEED, "title": spec["title"],
        "cache_dir": str(CACHE / f"fold{k}"),
        "split_path": str(VIEW / "cv3_v51" / f"fold{k}.json"), "data_root": str(VIEW),
        "frame_stats_path": str(OFF / f"wss_frame_stats_fold{k}.json"),
        "waveform_path": str(OFF / "protocol_inlet_waveform_v51.json"),
        "out_dir": str(RUNS / aid), "peak_index": 21, "inputs": spec["inputs"],
        "architecture": {"temporal": spec["temporal"], "d_model": 64, "nhead": 4,
                         "num_layers": 2, "dim_feedforward": 128, "dropout": 0.0,
                         "spatial_hidden": 128, "decoder_hidden": [128, 128], "global_pool": "mean"},
        "train": {"epochs": epochs, "steps_per_epoch": steps_per_epoch, "batch_cases": 4,
                  "query_points": 1024, "train_pool_points": 8192, "stats_points": 4096,
                  "lr": 1e-3, "weight_decay": 1e-4, "warmup_steps": 100,
                  "min_lr_ratio": 0.01, "grad_clip": 1.0, "selection": "last",
                  "allow_tf32": True, "log_every_epochs": 5},
        "eval": {"partition": "test", "chunk_points": 4096},
        "notes": "V5.1 cv3 development probe; static geometry is encoded once via cache, no dynamic pressure/velocity input.",
    }


def main():
    global NAME, CONFIGS, EXP, RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps-per-epoch", type=int, default=30)
    ap.add_argument("--name", default=NAME, help="independent experiment directory name")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    NAME = args.name
    CONFIGS = ROOT / "training_wss_min/configs" / NAME
    EXP = ROOT / "training_wss_min/experiments" / NAME
    RUNS = ROOT / "training_wss_min/runs" / NAME

    def dump(path, obj):
        if path.exists() and not args.force:
            raise FileExistsError(f"{path} exists (use --force)")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")

    arms = []
    for k in FOLDS:
        for arm in ARMS:
            aid = f"{arm}_f{k}_s{SEED}"
            dump(CONFIGS / f"{aid}.json", config(arm, k, args.epochs, args.steps_per_epoch))
            arms.append({"id": aid, "config": f"{aid}.json", "done_marker": str(RUNS / aid / "metrics.json")})
    matrix = {
        "schema_version": 1, "experiment": NAME, "created": "2026-09-24",
        "question": "Does true temporal self-attention add value beyond a frame-independent time MLP, and does frozen peak X5D query_x add value beyond raw geometry, under the same peak anchor and V5.1 protocol?",
        "protocol": "V5.1 cv3 folds 0-2, seed 1234, four arms, one run per arm; 60 epochs x 30 steps; full 2x2 attention x spatial-input factorial",
        "references": {"probe_report": str(ROOT / "training_wss_min/experiments/wss_time_adapter_v51_20260923/report.json"),
                       "stage2": str(ROOT / "training_wss_min/experiments/wss_time_ecc_20260918/stage2_report_best.json")},
        "execution": "node04 direct launch outside Slurm, one explicitly free A100 GPU; queue records GPU UUID/PID",
        "arms": arms,
        "gates": {"G0": "peak frame anchor max abs error <= 1e-9 and finite metrics",
                  "A1": "TT-warm vs TT-warm-noattn: same spatial input, attention increment (three folds, descriptive)",
                  "A2": "TT-warm vs TT-raw: frozen peak representation increment (three folds, descriptive)",
                  "U": "cycle/trough/TAWSS, log-cycle, peak timing, temporal correlation and amplitude; no deployment gate in this development probe"},
    }
    dump(CONFIGS / "matrix.json", matrix)
    print(f"wrote {len(arms)} configs + matrix.json to {CONFIGS}")


if __name__ == "__main__":
    main()
