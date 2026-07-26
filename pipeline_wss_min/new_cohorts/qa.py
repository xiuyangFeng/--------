#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""活动 AAA/ILO-before bundle 的 STL 关键点 v4 终检门。

AAA 沿用历史候选，ILO 只检查 before-only 终审白名单，不读取或修改 AG 产物。
逐例核对 report/bundle 完整性、原始 STL 来源、右手刚性旋转，以及关键点在落盘
坐标架中的 +Z 主干 / -Z 双髂支 / +X 原始 STL 世界方向语义。
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from pipeline_wss_min import config as C
from pipeline_wss_min.new_cohorts.common import candidate_units, split_unit
from pipeline_wss_min.new_cohorts.ilo_before import load_ilo_before_whitelist


OUT_DIR = C.OUT_ROOT / "pipeline_reports"
OUT_CSV = OUT_DIR / "new_cohorts_v4_bundle_qa.csv"
OUT_JSON = OUT_DIR / "new_cohorts_v4_bundle_qa.json"
EXPECTED_REPAIRS = {
    "AAA/ruputer/ZHOU_KE_XUN",
    "AAA/unruputer/LIU_WEN_QI",
}


def active_units() -> list[str]:
    aaa = [unit for unit in candidate_units() if unit.startswith("AAA/")]
    return aaa + load_ilo_before_whitelist()


def _scalar(z, key: str):
    value = np.asarray(z[key])
    return value.item() if value.ndim == 0 else value


def check_unit(unit_id: str) -> dict:
    cohort, case_name = split_unit(unit_id)
    case_dir = C.out_case_dir(cohort, case_name)
    report_path = case_dir / "report.json"
    bundle_path = case_dir / "bundle.npz"
    row = {
        "unit_id": unit_id,
        "qa_pass": False,
        "report_exists": report_path.is_file(),
        "bundle_exists": bundle_path.is_file(),
        "reason": "",
    }
    if not report_path.is_file() or not bundle_path.is_file():
        row["reason"] = "missing_report_or_bundle"
        return row

    report = json.loads(report_path.read_text())
    reasons = []
    if report.get("status") != "ok":
        reasons.append("report_not_ok")
    if report.get("frame_version") != "stl_landmarks_v4":
        reasons.append("wrong_frame_version")
    if report.get("landmark_source") != "original_stl":
        reasons.append("not_original_stl")
    if not report.get("nodenumber_alignment_ok", False):
        reasons.append("nodenumber_alignment_failed")
    if report.get("n_steps", 0) <= 0 or report.get("n_wall", 0) <= 0:
        reasons.append("empty_geometry_or_steps")
    if not np.isfinite(report.get("wss_raw_min", np.nan)) or not np.isfinite(
        report.get("wss_raw_max", np.nan)
    ):
        reasons.append("nonfinite_wss_range")
    if report.get("wss_raw_max", 0.0) <= 0:
        reasons.append("nonpositive_wss")

    try:
        with np.load(bundle_path, allow_pickle=False) as z:
            required = {
                "transform_frame_version", "transform_landmark_source",
                "transform_centroid", "transform_rotation",
                "transform_landmark_trunk_point", "transform_landmark_left_point",
                "transform_landmark_right_point",
                "transform_centerline_translation_applied", "original_stl_path",
                "wall_coords_norm", "wall_wss",
            }
            missing = sorted(required.difference(z.files))
            if missing:
                reasons.append("missing_bundle_keys:" + ",".join(missing))
            else:
                rotation = np.asarray(z["transform_rotation"], dtype=float)
                centroid = np.asarray(z["transform_centroid"], dtype=float)
                landmarks = np.vstack([
                    z["transform_landmark_trunk_point"],
                    z["transform_landmark_left_point"],
                    z["transform_landmark_right_point"],
                ]).astype(float)
                aligned = (landmarks - centroid) @ rotation
                det = float(np.linalg.det(rotation))
                trunk_z, left_z, right_z = aligned[:, 2]
                world_lr_dx = float(landmarks[1, 0] - landmarks[2, 0])
                repair = bool(_scalar(z, "transform_centerline_translation_applied"))
                row.update({
                    "frame_version": str(_scalar(z, "transform_frame_version")),
                    "landmark_source": str(_scalar(z, "transform_landmark_source")),
                    "rotation_det": det,
                    "trunk_z_mm": float(trunk_z),
                    "left_iliac_z_mm": float(left_z),
                    "right_iliac_z_mm": float(right_z),
                    "lr_world_dx_mm": world_lr_dx,
                    "centerline_translation_applied": repair,
                    "n_wall_bundle": int(z["wall_coords_norm"].shape[0]),
                    "n_steps_bundle": int(z["wall_wss"].shape[0]),
                })
                if row["frame_version"] != "stl_landmarks_v4":
                    reasons.append("bundle_wrong_frame_version")
                if row["landmark_source"] != "original_stl":
                    reasons.append("bundle_not_original_stl")
                if abs(det - 1.0) > 1e-6:
                    reasons.append("rotation_not_right_handed")
                if not (trunk_z > 0 and left_z < 0 and right_z < 0):
                    reasons.append("axial_landmark_sign_failed")
                if world_lr_dx <= 0:
                    reasons.append("lr_world_x_sign_failed")
                if row["n_wall_bundle"] != int(report["n_wall"]):
                    reasons.append("wall_count_mismatch")
                if row["n_steps_bundle"] != int(report["n_steps"]):
                    reasons.append("step_count_mismatch")
    except Exception as exc:  # noqa: BLE001 - 终检需汇总所有坏包
        reasons.append(f"bundle_read_error:{type(exc).__name__}:{exc}")

    row["reason"] = ";".join(reasons)
    row["qa_pass"] = not reasons
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="批处理运行中可先写汇总，不因缺包退出非零")
    args = ap.parse_args()

    units = active_units()
    rows = [check_unit(unit_id) for unit_id in units]
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    actual_repairs = set(
        df[df.get("centerline_translation_applied") == True]["unit_id"].tolist()
        if "centerline_translation_applied" in df else []
    )
    missing = df[df["reason"] == "missing_report_or_bundle"]["unit_id"].tolist()
    failed = df[(df["qa_pass"] != True) & (df["reason"] != "missing_report_or_bundle")][
        "unit_id"
    ].tolist()
    summary = {
        "scope": "active AAA + ILO-before final whitelist; AG and ILO-after excluded",
        "n_expected": len(units),
        "n_qa_pass": int((df["qa_pass"] == True).sum()),
        "n_missing": len(missing),
        "n_failed": len(failed),
        "missing_units": missing,
        "failed_units": failed,
        "centerline_translation_repaired_units": sorted(actual_repairs),
        "repair_set_matches_geometry_audit": actual_repairs == EXPECTED_REPAIRS,
        "unexpected_repairs": sorted(actual_repairs - EXPECTED_REPAIRS),
        "expected_repairs_not_yet_present": sorted(EXPECTED_REPAIRS - actual_repairs),
        "outputs": {"csv": str(OUT_CSV), "json": str(OUT_JSON)},
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed or (missing and not args.allow_incomplete):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
