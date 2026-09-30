#!/usr/bin/env python3
"""生成「峰值预训练 → 时间适配器」v5.1 可行性探针的配置（矩阵 P0.10，2026-09-23）。

每折一个 cache 配置（同折冻结 X5D_v51_f{k}_s1234 峰值 checkpoint），六臂 × cv3 三折 × seed 1234：
  Tnull    无头，复现 D2 T-null(B_scale)（工程验收）
  P-lin    岭回归逐帧读出，输入 query_x ⊕ patch_context ⊕ peak_ln
  P-mlp    导师草图（几何投影 + 独立时间 encoder → 融合 → decoder），同上输入   ← 主臂
  P-mlp-qx 同 P-mlp，去掉 patch_context（病例向量）
  C-scalar 同 P-mlp 结构，输入只有 peak_ln（D9：隐藏特征 vs 峰值标量）
  C-raw    同 P-mlp 结构，输入 27 维原始几何 ⊕ peak_ln（预训练表示 vs 原始几何）
全部锚定：ln τ̂ = T-null + R(h,φ(t)) − R(h,φ(t_peak))。只写新目录，不改任何已有配置。
用法：python -m training_wss_min.tools.prepare_wss_time_adapter_v51 [--force]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
NAME = "wss_time_adapter_v51_20260923"
CONFIGS = ROOT / "training_wss_min/configs" / NAME
EXP = ROOT / "training_wss_min/experiments" / NAME
RUNS = ROOT / "training_wss_min/runs" / NAME
VIEW = ROOT / "data_wss_v5/views_v5_1/wss_min_view_v1"
OFF = ROOT / "training_wss_min/experiments/wss_time_ecc_20260918/offline"
DONOR = ROOT / "training_wss_min/runs/wss_v51_wave2a_20260916"
FOLDS, SEED = (0, 1, 2), 1234

ARMS = {
    "Tnull": ("T-null(B_scale) reproduction through the probe pipeline (no head)", {"type": "none", "inputs": []}),
    "P-lin": ("anchored per-frame ridge readout on frozen query_x + patch_context + peak_ln",
              {"type": "ridge", "inputs": ["query_x", "patch_context", "peak_ln"]}),
    "P-mlp": ("anchored teacher head (geometry projection + time encoder -> fusion -> decoder) on frozen query_x + patch_context + peak_ln",
              {"type": "mlp", "inputs": ["query_x", "patch_context", "peak_ln"]}),
    "P-mlp-qx": ("P-mlp without the per-case patch_context vector", {"type": "mlp", "inputs": ["query_x", "peak_ln"]}),
    "C-scalar": ("same head, peak_ln only (does the hidden representation add beyond the peak scalar)",
                 {"type": "mlp", "inputs": ["peak_ln"]}),
    "C-raw": ("same head, 27 raw standardised inputs + peak_ln (pretrained representation vs raw geometry)",
              {"type": "mlp", "inputs": ["raw", "peak_ln"]}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing config files")
    args = ap.parse_args()
    CONFIGS.mkdir(parents=True, exist_ok=True)
    written = []

    def dump(path: Path, obj):
        if path.exists() and not args.force:
            raise FileExistsError(f"{path} exists (use --force)")
        path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
        written.append(path.name)

    cache_jobs, arms = [], []
    for k in FOLDS:
        donor = DONOR / f"X5D_v51_f{k}_s{SEED}"
        cache_dir = EXP / "cache" / f"fold{k}"
        dump(CONFIGS / f"cache_f{k}.json", {
            "schema": "time_adapter_cache_v1", "name": f"{NAME}/cache_f{k}",
            "donor_run": str(donor), "checkpoint": "best",
            "split_path": str(VIEW / f"cv3_v51/fold{k}.json"), "partitions": ["train", "test"],
            "out_dir": str(cache_dir),
            "reference_predictions": str(donor / "eval/ckpt_best/predictions/test"),
            "notes": "frozen fold donor in eval mode; fixed_support + full_cloud_query; held-out peak predictions checked against the stored eval",
        })
        cache_jobs.append({"id": f"cache_f{k}", "config": f"cache_f{k}.json", "fold": k, "done_marker": str(cache_dir / "manifest.json")})
        for arm, (title, head) in ARMS.items():
            aid = f"{arm}_f{k}_s{SEED}"
            dump(CONFIGS / f"{aid}.json", {
                "schema": "time_adapter_probe_v1", "name": f"{NAME}/{aid}", "experiment": NAME, "arm": arm,
                "fold": k, "seed": SEED, "title": title,
                "cache_dir": str(cache_dir), "split_path": str(VIEW / f"cv3_v51/fold{k}.json"), "data_root": str(VIEW),
                "frame_stats_path": str(OFF / f"wss_frame_stats_fold{k}.json"),
                "waveform_path": str(OFF / "protocol_inlet_waveform_v51.json"),
                "out_dir": str(RUNS / aid), "peak_index": 21,
                "phase_features": ["q_norm", "dq_norm", "t_sin", "t_cos"],
                "head": {**head, "anchor": True},
            })
            arms.append({"id": aid, "arm": arm, "fold": k, "seed": SEED, "config": f"{aid}.json",
                         "depends_on": f"cache_f{k}", "done_marker": str(RUNS / aid / "metrics.json")})
    dump(CONFIGS / "matrix.json", {
        "schema_version": 1, "experiment": NAME, "created": "2026-09-23",
        "question": "teacher idea P0.8: does a frozen peak-trained X5D representation support an anchored time adapter on v5.1 cv3?",
        "protocol": "cv3_v51 folds 0-2 (train 90/90/92, held-out 46/46/44), seed 1234, one run per arm; donor = same-fold X5D_v51_f{k}_s1234 ckpt_best; frame stats / waveform = wss_time_ecc_20260918 offline",
        "references": {"stage2": str(ROOT / "training_wss_min/experiments/wss_time_ecc_20260918/stage2_report_best.json"),
                       "d2_tnull": str(OFF / "d2_tnull.json")},
        "execution": "node04 2 x A100 over ssh (outside Slurm), frozen copy GNN_timeadapter_frozen_20260923",
        "cache_jobs": cache_jobs, "arms": arms,
    })
    print(f"wrote {len(written)} files to {CONFIGS}")


if __name__ == "__main__":
    main()
