#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AAA/ILO 新队列的 v4 STL 解剖坐标架全量只读审计。

仅扫描数据层已通过的 F_clean_pass / E_dense_clean 单元；不读取或修复缺文件、WSS
全零、百万点混合区。输出逐例关键点、方向、置信度、STL 版本和 centerline 平移
修复证据，作为正式预处理前的几何质量门。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from collections import Counter

import numpy as np
import pandas as pd

from pipeline_wss_min import config as C, raw_io, surface_io
from pipeline_wss_min.new_cohorts.common import (
    ASSET_DIR, CLASSIFIED, PASS_CATEGORIES, split_unit,
)
from pipeline_wss_min.preprocess import _resolve_unit_factor
from pipeline_wss_min.registration import compute_transform


OUT_CSV = ASSET_DIR / "landmark_frame_v4_audit.csv"
OUT_JSON = ASSET_DIR / "landmark_frame_v4_summary.json"


def audit_case(unit_id: str, cohort: str, case_name: str) -> dict:
    case_dir = C.raw_case_dir(cohort, case_name)
    steps = raw_io.list_timesteps(case_dir)
    wall_native = raw_io.read_wall_geometry(case_dir, case_name, steps[0])
    centerline = raw_io.read_centerline(case_dir)
    factor, unit_anomaly, extent_mismatch = _resolve_unit_factor(
        wall_native, centerline, C.DEFAULT.unit)
    wall_mm = wall_native * factor
    selected = surface_io.select_case_surface(case_dir, wall_mm, factor)
    transform = compute_transform(
        wall_mm, wall_mm, centerline, C.DEFAULT.registration,
        anatomy_pts=selected.points_mm, anatomy_source="original_stl",
    )
    landmark_world = np.vstack([
        transform.landmark_trunk_point,
        transform.landmark_left_point,
        transform.landmark_right_point,
    ])
    landmark_aligned = transform.apply_points(landmark_world)
    det = float(np.linalg.det(transform.rotation))
    trunk_z, left_z, right_z = landmark_aligned[:, 2]
    lr_world_dx = float(
        transform.landmark_left_point[0] - transform.landmark_right_point[0])
    iliac_down = bool(left_z < 0 and right_z < 0)
    trunk_up = bool(trunk_z > 0)
    frame_pass = bool(
        transform.landmark_source == "original_stl"
        and trunk_up and iliac_down and lr_world_dx > 0
        and abs(det - 1.0) <= 1e-6
        and transform.roll_sign_reliable
    )
    return {
        "unit_id": unit_id,
        "status": "ok",
        "frame_pass": frame_pass,
        "frame_version": transform.frame_version,
        "landmark_source": transform.landmark_source,
        "stl_path": str(selected.path.relative_to(C.PROJECT_ROOT)),
        "stl_n_candidates": selected.n_candidates,
        "stl_scale_to_mm": selected.scale_to_mm,
        "stl_match_score": selected.match_score,
        "unit_factor": factor,
        "unit_anomaly": unit_anomaly,
        "unit_extent_mismatch": extent_mismatch,
        "origin_kind": transform.origin_kind,
        "main_axis_source": transform.main_axis_source,
        "fork_spread_ratio": transform.fork_spread_ratio,
        "lr_separation": transform.lr_separation,
        "roll_sign_reliable": transform.roll_sign_reliable,
        "rotation_det": det,
        "trunk_z_mm": float(trunk_z),
        "left_iliac_z_mm": float(left_z),
        "right_iliac_z_mm": float(right_z),
        "trunk_up_ok": trunk_up,
        "iliac_down_ok": iliac_down,
        "lr_world_dx_mm": lr_world_dx,
        "lr_world_x_ok": bool(lr_world_dx > 0),
        "centerline_translation_candidate": transform.centerline_translation_candidate,
        "centerline_translation_applied": transform.centerline_translation_applied,
        "centerline_offset_diag_frac": transform.centerline_offset_diag_frac,
        "centerline_shift_x_mm": float(transform.centerline_translation_mm[0]),
        "centerline_shift_y_mm": float(transform.centerline_translation_mm[1]),
        "centerline_shift_z_mm": float(transform.centerline_translation_mm[2]),
        "centerline_repair_p50_mm": transform.centerline_repair_p50_mm,
        "centerline_repair_p90_mm": transform.centerline_repair_p90_mm,
        "reason": "" if frame_pass else "frame_gate_failed",
    }


def audit_unit(unit_id: str) -> dict:
    cohort, case_name = split_unit(unit_id)
    return audit_case(unit_id, cohort, case_name)


def _safe_audit_unit(unit_id: str) -> dict:
    try:
        return audit_unit(unit_id)
    except Exception as exc:  # noqa: BLE001 - 全量审计不中断
        return {
            "unit_id": unit_id, "status": "error", "frame_pass": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="调试时只跑前 N 例")
    ap.add_argument("--workers", type=int, default=4, help="并行只读病例数")
    args = ap.parse_args()

    classified = pd.read_csv(CLASSIFIED)
    unit_ids = classified[classified["cat"].isin(PASS_CATEGORIES)]["unit_id"].tolist()
    if args.limit is not None:
        unit_ids = unit_ids[:args.limit]
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_safe_audit_unit, unit_id): unit_id for unit_id in unit_ids}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(f"[{i:3d}/{len(unit_ids)}] {str(row.get('frame_pass', False)):5s} "
                  f"{row['unit_id']} {row.get('reason', '')}", flush=True)

    # 网络文件系统偶发短读不应被误判为病例失败；并行阶段的 I/O error 串行重试一次。
    for i, row in enumerate(rows):
        if row.get("status") == "error":
            retry = _safe_audit_unit(row["unit_id"])
            rows[i] = retry
            print(f"[retry] {str(retry.get('frame_pass', False)):5s} {retry['unit_id']} "
                  f"{retry.get('reason', '')}", flush=True)

    order = {unit_id: i for i, unit_id in enumerate(unit_ids)}
    rows.sort(key=lambda row: order[row["unit_id"]])

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    ok = df[df["status"] == "ok"]
    repairs = ok[ok["centerline_translation_applied"] == True]["unit_id"].tolist() \
        if "centerline_translation_applied" in ok else []
    failures = df[df["frame_pass"] != True]["unit_id"].tolist()
    summary = {
        "scope": "AAA/ILO data-layer-clean units only",
        "n_total": len(df),
        "n_ok": int((df["status"] == "ok").sum()),
        "n_frame_pass": int((df["frame_pass"] == True).sum()),
        "n_frame_fail": len(failures),
        "frame_fail_units": failures,
        "n_centerline_translation_repaired": len(repairs),
        "centerline_translation_repaired_units": repairs,
        "origin_kind_counts": dict(Counter(ok.get("origin_kind", []))),
        "stl_source_counts": dict(Counter(ok.get("landmark_source", []))),
        "outputs": {"csv": str(OUT_CSV), "json": str(OUT_JSON)},
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
