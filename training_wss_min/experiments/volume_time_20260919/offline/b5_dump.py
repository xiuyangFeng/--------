"""Dump the peak-frame volume data path for a few cases, so two code trees can be compared bit-for-bit.

旧 v5 视图（`data_wss_v5/views/`）已被 v5.1 取代并从磁盘删除，PF6_s1234 / VF6_s1234
两个体场旧 run 无法重评。改用同样严格的替代口径：拿**改动前的冻结代码树**
（`GNN_time_frozen_20260918`，在 WSS 时间改动之后、体场时间改动之前）与当前树，
在同一批 v5.1 病例上走 `load_partition` 的峰值帧体场路径，逐字节比较所有数组与折训练侧特征统计量。

    python -m training_wss_min.experiments.volume_time_20260919.offline.b5_dump <out.npz> <target>

两棵树各跑一次（cd 到各自树根），再用 b5_compare.py 比。
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np

from training_wss_min import config as C
from training_wss_min import dataset as D

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
VIEW = ROOT / "data_wss_v5/views_v5_1/wss_min_view_v1"
FLOW = ROOT / "data_wss_v5/views_v5_1/wss_min_flowref_v1"
STATS = ROOT / "training_wss_min/experiments/volume_time_20260919/offline/volume_stats_fold0.json"
SPLIT = VIEW / "cv3_v51/fold0.json"
FEATURES = ("x", "y", "z", "abscissa_norm", "local_radius", "curvature", "log_local_radius", "rho",
            "theta_sin", "theta_cos", "dr_ds", "dist_to_junction_mm", "dist_to_endpoint_mm", "end_zone",
            "nx_aligned", "ny_aligned", "nz_aligned", "dist_to_wall_mm",
            "log_q_branch_murray", "log_tau0_murray")
N_CASES = 4


def main() -> None:
    out = Path(sys.argv[1])
    target = sys.argv[2]
    stats = D.load_wss_stats(str(STATS))
    extra = tuple(f for f in FEATURES if f in C.SIDECAR_FEATURE_KEYS)
    cases = D.load_partition(str(SPLIT), "test", stats, strict=True, target=target,
                             data_root=str(VIEW), required_frame_version="v5_atlas_frame_v1",
                             extra_point_features=extra, point_features_root=[str(FLOW)])
    cases = sorted(cases, key=lambda c: f"{c['cohort']}/{c['case']}")[:N_CASES]
    blob: dict[str, np.ndarray] = {}
    for case in cases:
        cid = f"{case['cohort']}/{case['case']}".replace("/", "__")
        for key, value in sorted(case.items()):
            if isinstance(value, np.ndarray) and value.dtype.kind in "fiub" and value.ndim <= 2:
                blob[f"{cid}|{key}"] = value.astype(np.float64)
            elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                blob[f"{cid}|{key}"] = np.array([float(value)], dtype=np.float64)
    feats = D.compute_feature_stats(cases, FEATURES, "signed_log1p")   # x/y/z 走坐标归一化，不在表里
    blob["_feature_stats"] = np.array([[feats[k]["mean"], feats[k]["std"]] for k in sorted(feats)],
                                      dtype=np.float64)
    blob["_feature_stats_keys"] = np.array([zlib.crc32(k.encode()) for k in sorted(feats)], dtype=np.float64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **blob)
    print(out, len(cases), "cases,", len(blob), "arrays, target", target, flush=True)


if __name__ == "__main__":
    main()
