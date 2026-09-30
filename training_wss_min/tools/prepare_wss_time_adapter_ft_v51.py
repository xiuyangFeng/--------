#!/usr/bin/env python3
"""生成导师方案第二步「解冻版时间适配器」v5.1 配置（矩阵 P0.11，2026-09-23）。

两臂 × cv3 三折 × seed 1234（用户：跑一次，不补种子）：
  TL-warm    时间几何支路 = 同折 X5D_v51_f{k} checkpoint 初始化；只解冻 fp + local_wall_branch（lr 1e-4），时间头 lr 1e-3
  TL-random  完全同构、同锚、同损失、同可训练模块与日程；几何支路随机初始化（同种子）
共同：冻结峰值锚 = P0.10 的同折 cache（wss_time_adapter_v51_20260923/cache/fold{k}）；头输入 query_x ⊕ peak_ln
（不用 patch_context：P0.10 显示它过拟合——这是参考了 v5.1 留出折的开发选择）；无密度增广；选模 last。
用法：python -m training_wss_min.tools.prepare_wss_time_adapter_ft_v51 [--epochs 150] [--pilot] [--force]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
NAME = "wss_time_adapter_ft_v51_20260923"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
RUNS = ROOT / "training_wss_min/runs" / NAME
PROBE_EXP = ROOT / "training_wss_min/experiments/wss_time_adapter_v51_20260923"
VIEW = ROOT / "data_wss_v5/views_v5_1/wss_min_view_v1"
OFF = ROOT / "training_wss_min/experiments/wss_time_ecc_20260918/offline"
DONOR = ROOT / "training_wss_min/runs/wss_v51_wave2a_20260916"
FOLDS, SEED = (0, 1, 2), 1234
ARMS = {"TL-warm": ("donor", "teacher step 2: time geometry branch initialised from the same-fold X5D checkpoint; fp + local_wall_branch unfrozen (lr 1e-4)"),
        "TL-random": ("random", "attribution control: identical to TL-warm except the time geometry branch is randomly initialised")}


def arm_config(arm, k, epochs, out_dir, extra_train=None):
    init, title = ARMS[arm]
    return {"schema": "time_adapter_finetune_v1", "name": f"{NAME}/{out_dir.name}", "experiment": NAME, "arm": arm,
            "fold": k, "seed": SEED, "title": title,
            "donor_run": str(DONOR / f"X5D_v51_f{k}_s{SEED}"), "donor_checkpoint": "best", "geometry_init": init,
            "trainable_modules": ["fp", "local_wall_branch"],
            "cache_dir": str(PROBE_EXP / "cache" / f"fold{k}"), "split_path": str(VIEW / f"cv3_v51/fold{k}.json"),
            "data_root": str(VIEW), "frame_stats_path": str(OFF / f"wss_frame_stats_fold{k}.json"),
            "waveform_path": str(OFF / "protocol_inlet_waveform_v51.json"), "out_dir": str(out_dir), "peak_index": 21,
            "phase_features": ["q_norm", "dq_norm", "t_sin", "t_cos"],
            "head": {"inputs": ["query_x", "peak_ln"], "geom_hidden": 128, "time_hidden": 64, "decoder_hidden": [128, 128],
                     "zero_init_output": True},
            "train": {"epochs": epochs, "batch_cases": 8, "query_points": 2048, "lr_head": 1e-3, "lr_geometry": 1e-4,
                      "weight_decay": 1e-4, "warmup_epochs": 5, "min_lr_ratio": 0.01, "grad_clip": 1.0, "selection": "last",
                      "num_workers": 4, "input_stats_batches": 4, "allow_tf32": True, **(extra_train or {})}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--pilot", action="store_true", help="only write a 2-epoch TL-warm fold-0 timing pilot")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    def dump(path: Path, obj):
        if path.exists() and not args.force:
            raise FileExistsError(f"{path} exists (use --force)")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")

    if args.pilot:
        path = EXP / "pilot" / "TL-warm_f0_pilot.json"
        dump(path, arm_config("TL-warm", 0, 2, EXP / "pilot" / "TL-warm_f0_pilot"))
        print("wrote", path)
        return
    arms = []
    for k in FOLDS:
        for arm in ARMS:
            aid = f"{arm}_f{k}_s{SEED}"
            dump(CONFIGS / f"{aid}.json", arm_config(arm, k, args.epochs, RUNS / aid))
            arms.append({"id": aid, "kind": "finetune", "arm": arm, "fold": k, "seed": SEED, "config": f"{aid}.json",
                         "done_marker": str(RUNS / aid / "metrics.json")})
    dump(CONFIGS / "matrix.json", {
        "schema_version": 1, "experiment": NAME, "created": "2026-09-23",
        "question": "teacher idea step 2: does unfreezing the late X5D layers (fp + local wall branch) under a peak-anchored time head beat T-null / C-raw / T0, and does the pretrained initialisation matter (TL-warm vs TL-random)?",
        "protocol": f"cv3_v51 folds 0-2, seed 1234, one run per arm, {args.epochs} epochs; anchor/base identical to P0.10",
        "references": {"probe_report": str(PROBE_EXP / "report.json"),
                       "stage2": str(ROOT / "training_wss_min/experiments/wss_time_ecc_20260918/stage2_report_best.json")},
        "execution": "node04 2 x A100 over ssh (outside Slurm), frozen copy GNN_timeadapter_ft_frozen_20260923",
        "cache_jobs": [], "arms": arms})
    print(f"wrote {len(arms)} arm configs + matrix.json to {CONFIGS}")


if __name__ == "__main__":
    main()
