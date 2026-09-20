#!/usr/bin/env python3
"""Per-component R2 for the V5 velocity run, in two frames.

(a) anatomical-frame components u/v/w as stored by the evaluator, and
(b) the hemodynamically meaningful local decomposition: axial (along the centerline tangent),
    radial (outward from the centerline) and circumferential (tangent x radial).
The second one separates bulk flow from secondary flow, which the frame components mix together.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min.evaluate import load_model_from_run, load_wss_stats_for_run, predict_case_norm
from wss_v5 import contract as WC


def r2(y, p):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def r2_cb(ys, ps):
    gm = np.mean([y.mean() for y in ys])
    return 1.0 - np.mean([np.mean((y - p) ** 2) for y, p in zip(ys, ps)]) / np.mean([np.mean((y - gm) ** 2) for y in ys])


def local_frame(case_id: str, snapshot_root: Path, rotation: np.ndarray, radial_aligned: np.ndarray):
    """axial/radial/circumferential unit vectors per interior cell, in the bundle's aligned frame."""
    with h5py.File(WC.case_dir(case_id, snapshot_root) / "case.h5", "r") as h5:
        cols = json.loads(h5["geometry"].attrs["atlas_columns"])
        table = h5["geometry/atlas_table"][()]
        row = h5["volume_static/atlas_row"][()].astype(np.int64)
    tan = table[:, [cols.index("tangent_x"), cols.index("tangent_y"), cols.index("tangent_z")]][row]
    tan = tan / np.clip(np.linalg.norm(tan, axis=1, keepdims=True), 1e-12, None)
    axial = tan @ rotation.T                       # vectors rotate the same way the view rotated them
    radial = radial_aligned.astype(np.float64)
    # re-orthogonalise radial against axial, then circumferential = axial x radial
    radial = radial - np.sum(radial * axial, axis=1, keepdims=True) * axial
    n = np.linalg.norm(radial, axis=1, keepdims=True)
    ok = (n[:, 0] > 1e-6)
    radial = np.where(n > 1e-6, radial / np.clip(n, 1e-12, None), 0.0)
    circ = np.cross(axial, radial)
    return axial, radial, circ, ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r5v_velocity_qad_s1234")
    ap.add_argument("--snapshot-root", default=str(WC.SNAPSHOT_ROOT))
    ap.add_argument("--out", default="training_wss_min/experiments/v5_rerun_20260906/velocity_component_breakdown.json")
    args = ap.parse_args()
    run_dir = C.PROJECT_ROOT / "training_wss_min" / "runs" / "v5_rerun_20260906" / "outputs" / args.run
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, feat_stats, model, _ = load_model_from_run(run_dir, device, "best")
    model.eval()
    stats = load_wss_stats_for_run(run_dir)
    mean = np.asarray(stats["velocity"]["mean"], dtype=np.float64)
    std = np.asarray(stats["velocity"]["std"], dtype=np.float64)
    cases = D.load_partition(cfg.data.split_path, "test", stats, target=cfg.data.target,
                             target_normalization=cfg.data.target_normalization,
                             data_root=cfg.data.data_root,
                             required_frame_version=cfg.data.required_frame_version)
    acc = {k: {"t": [], "p": []} for k in ("u", "v", "w", "axial", "radial", "circ", "speed")}
    with torch.no_grad():
        for case in cases:
            case.setdefault("unit_id", f"{case['cohort']}/{case['case']}")
            out = np.asarray(predict_case_norm(model, case, cfg.data.input_features, feat_stats, device,
                                               cfg=cfg, return_all_channels=True), dtype=np.float64)
            rows = D.query_rows(case)
            pred = out * std + mean
            true = np.asarray(case["y_raw"], dtype=np.float64)[rows]
            with np.load(Path(cfg.data.data_root) / case["unit_id"] / "volume.npz") as vz:
                radial_aligned = vz["vol_radial_aligned"]
            with np.load(case["bundle_path"], allow_pickle=True) as bz:
                rotation = bz["transform_rotation"].astype(np.float64)
            axial, radial, circ, ok = local_frame(case["unit_id"], Path(args.snapshot_root), rotation, radial_aligned)
            for i, k in enumerate(("u", "v", "w")):
                acc[k]["t"].append(true[:, i]); acc[k]["p"].append(pred[:, i])
            for k, basis in (("axial", axial), ("radial", radial), ("circ", circ)):
                acc[k]["t"].append(np.sum(true[ok] * basis[ok], axis=1))
                acc[k]["p"].append(np.sum(pred[ok] * basis[ok], axis=1))
            acc["speed"]["t"].append(np.linalg.norm(true, axis=1)); acc["speed"]["p"].append(np.linalg.norm(pred, axis=1))
    res = {}
    for k, d in acc.items():
        res[k] = {"r2_casebalanced": r2_cb(d["t"], d["p"]),
                  "r2_pooled": r2(np.concatenate(d["t"]), np.concatenate(d["p"])),
                  "rmse": float(np.sqrt(np.mean((np.concatenate(d["t"]) - np.concatenate(d["p"])) ** 2))),
                  "true_std": float(np.concatenate(d["t"]).std()),
                  "true_absmean": float(np.abs(np.concatenate(d["t"])).mean())}
    print("=== 解剖系分量（评估器口径）")
    print("| 分量 | R² 病例等权 | R² pooled | RMSE (m/s) | 真值 sd (m/s) |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for k in ("u", "v", "w"):
        r = res[k]; print(f"| {k} | {r['r2_casebalanced']:.4f} | {r['r2_pooled']:.4f} | {r['rmse']:.4f} | {r['true_std']:.4f} |")
    print("\n=== 局部血流坐标分量（沿中心线切向 / 径向 / 周向）")
    print("| 分量 | R² 病例等权 | R² pooled | RMSE (m/s) | 真值 sd (m/s) | 真值 |·| 均值 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for k, lab in (("axial", "轴向（主流）"), ("radial", "径向（二次流）"), ("circ", "周向（二次流）")):
        r = res[k]; print(f"| {lab} | {r['r2_casebalanced']:.4f} | {r['r2_pooled']:.4f} | {r['rmse']:.4f} | {r['true_std']:.4f} | {r['true_absmean']:.4f} |")
    r = res["speed"]; print(f"\n速度幅值 |u|：R² 病例等权 {r['r2_casebalanced']:.4f}，pooled {r['r2_pooled']:.4f}，RMSE {r['rmse']:.4f} m/s")
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
