#!/usr/bin/env python3
"""ILO 术前队列的身份、数据、WSS 与 v4 bundle 终审工具。

本模块把 ``ILO/*/before`` 作为唯一活动 ILO 作用域。历史 after 原始数据仍可保留，
但不会进入白名单或预处理入口。生成的 manifest 是后续增量接入的真源。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from pipeline_wss_min import config as C, raw_io
from pipeline_wss_min.new_cohorts.common import ASSET_DIR, CLASSIFIED, FRAME_AUDIT


SCHEMA_VERSION = 1
FRAME_VERSION = "stl_landmarks_v4"
LANDMARK_SOURCE = "original_stl"
PASS_CATEGORIES = frozenset({"F_clean_pass", "E_dense_clean"})
EXACT_AAA_PREFERENCE = frozenset({
    "GUO_AI_JUN", "SHEN_CHUN_WANG", "ZHANG_MAO_JIN",
})
BASELINE_REPORT_DIR = C.OUT_ROOT / "pipeline_reports" / "ilo_before_review_20260717"
REPORT_ARCHIVE_ROOT = (
    C.OUT_ROOT / "pipeline_reports" / "_archive" / "ilo_before_review_history_20260717"
)
DEFAULT_REPORT_DIR = C.OUT_ROOT / "pipeline_reports" / "ilo_before_final_approved_20260717"
DEFAULT_WHITELIST = DEFAULT_REPORT_DIR / "ilo_before_whitelist.json"
REFERENCE_QUALITY_AUDIT = (
    C.OUT_ROOT / "pipeline_reports" / "v4_cutover_20260715_1921"
    / "v4_training_quality_audit.json"
)
REFERENCE_FINAL_WHITELIST = (
    C.OUT_ROOT / "pipeline_reports" / "v4_cutover_20260715_1921"
    / "v4_final_whitelist.json"
)
NEAR_NAME_THRESHOLD = 0.88
GEOM_TOL = 1e-6
NORM_TOL = 2e-5
MANUAL_REVIEW_EXCLUSIONS = {
    "DONG_KE_QIN": "DONG_KE_QING",
    "WEI_QING_FENG": "WEI_QING_FENG",
    "ZHANG_WAN_ZENG": "ZHANG_WAN_ZENG",
    "LIU_CHUN_YANG": "LIU_CHUN_YANG",
    "LIU_YUN_ZHANG": "LIU_YUN_ZHANG",
    "LI_YU_GANG": "LI_YU_GANG",
    "WANG_LI_MIN": "WANG_LI_MIN",
    "YANG_QING_REN": "YANG_QING_REN",
}
MANUAL_REVIEW_ROUNDS = {
    patient: (3 if patient in {"WANG_LI_MIN", "YANG_QING_REN"} else 2)
    for patient in MANUAL_REVIEW_EXCLUSIONS
}
EXPECTED_INCLUDED_BEFORE = 41
EXPECTED_EXCLUDED_BEFORE = 20


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["unit_id"]
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_patient_name(value: str) -> str:
    """规范化拼音姓名；仅去掉 ILO 目录末尾的 ``-0/-1`` 身份后缀。"""
    value = re.sub(r"-(?:0|1)$", "", value.strip().upper())
    return re.sub(r"[^A-Z0-9]+", "_", value).strip("_")


def compact_patient_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", normalize_patient_name(value))


def patient_from_unit(unit_id: str) -> str:
    parts = unit_id.split("/")
    if parts[0] == "ILO" and len(parts) == 3:
        return normalize_patient_name(parts[1])
    if parts[0] in {"AAA", "AG"} and len(parts) == 3:
        return normalize_patient_name(parts[2])
    raise ValueError(f"非法病例 ID: {unit_id}")


def is_ilo_before_unit(unit_id: str) -> bool:
    parts = unit_id.split("/")
    return len(parts) == 3 and parts[0] == "ILO" and parts[2] == "before"


def validate_active_preprocess_unit(unit_id: str) -> None:
    if unit_id.startswith("ILO/") and not is_ilo_before_unit(unit_id):
        raise ValueError(f"活动 ILO 预处理只允许 before，拒绝: {unit_id}")


def _iter_cohort_units() -> dict[str, list[str]]:
    units: dict[str, list[str]] = {"ILO": [], "AAA": [], "AG": []}
    ilo_root = C.RAW_ROOT / "ILO"
    if ilo_root.is_dir():
        units["ILO"] = [
            f"ILO/{path.name}/before"
            for path in sorted(ilo_root.iterdir()) if path.is_dir()
        ]
    for cohort, groups in {"AAA": ("ruputer", "unruputer"), "AG": ("fast", "slow")}.items():
        for group in groups:
            root = C.RAW_ROOT / cohort / group
            if root.is_dir():
                units[cohort].extend(
                    f"{cohort}/{group}/{path.name}"
                    for path in sorted(root.iterdir()) if path.is_dir()
                )
    return units


def source_stat_fingerprint(case_dir: Path) -> tuple[str, int, int]:
    """用路径、大小、mtime 构建低成本增量指纹，不读取大型场文件正文。"""
    selected: list[Path] = []
    for rel in (
        C.RAW_LAYOUT["wall_ascii_dir"], C.RAW_LAYOUT["centerline_csv"],
        C.RAW_LAYOUT["inlet_waveform"],
    ):
        path = case_dir / rel
        if path.is_dir():
            selected.extend(sorted(item for item in path.iterdir() if item.is_file()))
        elif path.is_file():
            selected.append(path)
    waveform = case_dir / C.RAW_LAYOUT["inlet_waveform"]
    if waveform.parent.is_dir():
        selected.extend(sorted(waveform.parent.glob(f"{waveform.stem}_*{waveform.suffix}")))
    selected.extend(sorted(case_dir.glob("*.stl")))
    rows = []
    total = 0
    for path in sorted(set(selected)):
        stat = path.stat()
        total += stat.st_size
        rows.append((str(path.relative_to(case_dir)), stat.st_size, stat.st_mtime_ns))
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(rows), total


def _load_previous_fingerprints(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["unit_id"]: row.get("source_stat_fingerprint", "")
        for row in payload.get("inventory", [])
    }


def _quality_reason(category: str) -> str:
    return {
        "A_empty": "empty_no_wall_export",
        "B_no_centerline": "missing_centerline",
        "C_million_mixed_zone": "million_point_mixed_zone_zero_wss",
        "D_zeroed_wss": "zeroed_wss",
    }.get(category, f"data_quality_category:{category}")


def build_inventory(report_dir: Path = DEFAULT_REPORT_DIR) -> dict:
    report_dir = report_dir.resolve()
    classified = pd.read_csv(CLASSIFIED).set_index("unit_id", drop=False)
    frame = pd.read_csv(FRAME_AUDIT).set_index("unit_id", drop=False)
    cohort_units = _iter_cohort_units()
    aaa_by_name: dict[str, list[str]] = {}
    ag_by_name: dict[str, list[str]] = {}
    for unit in cohort_units["AAA"]:
        aaa_by_name.setdefault(patient_from_unit(unit), []).append(unit)
    for unit in cohort_units["AG"]:
        ag_by_name.setdefault(patient_from_unit(unit), []).append(unit)

    previous = _load_previous_fingerprints(report_dir / "ilo_before_manifest.json")
    inventory: list[dict] = []
    exclusions: list[dict] = []
    included: list[str] = []
    exact_overlaps: list[dict] = []

    for unit in cohort_units["ILO"]:
        patient = patient_from_unit(unit)
        case_dir = C.RAW_ROOT / unit
        source_fp, source_n, source_bytes = source_stat_fingerprint(case_dir)
        category = str(classified.loc[unit, "cat"]) if unit in classified.index else "missing_audit"
        frame_pass = bool(frame.loc[unit, "frame_pass"]) if unit in frame.index else False
        duplicate_aaa = sorted(aaa_by_name.get(patient, []))
        duplicate_ag = sorted(ag_by_name.get(patient, []))
        reasons: list[str] = []
        if category not in PASS_CATEGORIES:
            reasons.append(_quality_reason(category))
        if category in PASS_CATEGORIES and not frame_pass:
            reasons.append("v4_frame_gate_failed")
        if duplicate_aaa:
            reasons.append("prefer_AAA_exclude_ILO")
            exact_overlaps.append({
                "normalized_patient_id": patient,
                "ilo_unit_id": unit,
                "aaa_unit_ids": duplicate_aaa,
                "ag_unit_ids": duplicate_ag,
                "policy": "prefer_AAA_exclude_ILO",
                "ilo_data_category": category,
                "ilo_frame_pass": frame_pass,
            })
        if patient in MANUAL_REVIEW_EXCLUSIONS:
            reasons.append(
                f"manual_visual_review_exclude_20260717_round{MANUAL_REVIEW_ROUNDS[patient]}"
            )
        row = {
            "unit_id": unit,
            "patient_dir": unit.split("/")[1],
            "normalized_patient_id": patient,
            "phase": "before",
            "data_category": category,
            "data_pass": category in PASS_CATEGORIES,
            "frame_pass": frame_pass,
            "exact_duplicate_aaa": "|".join(duplicate_aaa),
            "exact_duplicate_ag": "|".join(duplicate_ag),
            "decision": "include" if not reasons else "exclude",
            "exclusion_reasons": "|".join(reasons),
            "manual_review_submitted_name": MANUAL_REVIEW_EXCLUSIONS.get(patient, ""),
            "source_stat_fingerprint": source_fp,
            "source_file_count": source_n,
            "source_total_bytes": source_bytes,
            "incremental_state": (
                "new" if unit not in previous else
                "unchanged" if previous[unit] == source_fp else "changed"
            ),
        }
        inventory.append(row)
        if reasons:
            exclusions.append({
                "unit_id": unit,
                "normalized_patient_id": patient,
                "reasons": reasons,
                "data_category": category,
                "aaa_preferred_units": duplicate_aaa,
                "manual_review_submitted_name": MANUAL_REVIEW_EXCLUSIONS.get(patient, ""),
            })
        else:
            included.append(unit)

    near_matches: list[dict] = []
    for ilo_unit in cohort_units["ILO"]:
        ilo_name = patient_from_unit(ilo_unit)
        for cohort, other_units in (("AAA", cohort_units["AAA"]), ("AG", cohort_units["AG"])):
            for other in other_units:
                other_name = patient_from_unit(other)
                if ilo_name == other_name:
                    continue
                score = SequenceMatcher(
                    None, compact_patient_name(ilo_name), compact_patient_name(other_name)
                ).ratio()
                if score >= NEAR_NAME_THRESHOLD:
                    near_matches.append({
                        "ilo_unit_id": ilo_unit,
                        "other_cohort": cohort,
                        "other_unit_id": other,
                        "ilo_normalized_name": ilo_name,
                        "other_normalized_name": other_name,
                        "similarity": round(score, 6),
                        "decision": "manual_watch_only",
                    })

    inventory.sort(key=lambda row: row["unit_id"])
    included.sort()
    exclusions.sort(key=lambda row: row["unit_id"])
    exact_overlaps.sort(key=lambda row: row["ilo_unit_id"])
    near_matches.sort(key=lambda row: (-row["similarity"], row["ilo_unit_id"], row["other_unit_id"]))

    counts = {
        "raw_ilo_patients": len(cohort_units["ILO"]),
        "before_units": len(inventory),
        "included_before": len(included),
        "excluded_before": len(exclusions),
        "exact_aaa_overlap": len(exact_overlaps),
        "near_name_watch": len(near_matches),
        "manual_visual_excluded": len(MANUAL_REVIEW_EXCLUSIONS),
        "after_units_in_active_manifest": 0,
    }
    if (
        counts["before_units"] != 61
        or counts["included_before"] != EXPECTED_INCLUDED_BEFORE
        or counts["excluded_before"] != EXPECTED_EXCLUDED_BEFORE
    ):
        raise RuntimeError(f"ILO before 清单与已签方案不一致: {counts}")
    if {row["normalized_patient_id"] for row in exact_overlaps} != EXACT_AAA_PREFERENCE:
        raise RuntimeError("精确重名集合与已签方案不一致")

    generated = utc_now()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "scope": "ILO before only; postoperative after is forbidden",
        "identity_policy": "prefer_AAA_exclude_ILO",
        "manual_review_round": 3,
        "manual_review_status": "final_approved",
        "manual_review_policy": "exclude_user_selected_cases_keep_derived_before_for_audit",
        "source_fingerprint_method": "sha256(relative_path,size,mtime_ns)",
        "counts": counts,
        "included_units": included,
        "excluded_units": [row["unit_id"] for row in exclusions],
        "inventory": inventory,
    }
    whitelist = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "required_frame_version": FRAME_VERSION,
        "required_landmark_source": LANDMARK_SOURCE,
        "phase": "before",
        "identity_policy": "prefer_AAA_exclude_ILO",
        "manual_review_round": 3,
        "manual_review_status": "final_approved",
        "manual_geometry_review_approved": True,
        "training_promotion_authorized": False,
        "included_units": included,
        "counts": {"ILO_before": len(included), "ILO_after": 0},
    }
    exclusion_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "counts": {"unique_excluded_before": len(exclusions)},
        "excluded_units": exclusions,
    }
    overlap_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "normalization": "uppercase; separators->_; strip ILO -0/-1 suffix",
        "policy": "prefer_AAA_exclude_ILO",
        "exact_overlaps": exact_overlaps,
        "near_matches": near_matches,
        "manual_name_aliases": [
            {
                "submitted_name": submitted,
                "resolved_normalized_patient_id": patient,
                "resolution": "unique_workspace_match",
            }
            for patient, submitted in MANUAL_REVIEW_EXCLUSIONS.items()
            if submitted != patient
        ],
    }
    manual_review_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "review_round": 3,
        "review_status": "final_approved",
        "decision": "exclude_from_active_ILO_before_whitelist",
        "final_active_count": EXPECTED_INCLUDED_BEFORE,
        "submitted_to_resolved": [
            {
                "submitted_name": submitted,
                "resolved_normalized_patient_id": patient,
                "unit_id": next(
                    row["unit_id"] for row in inventory
                    if row["normalized_patient_id"] == patient
                ),
                "reason": (
                    f"manual_visual_review_exclude_20260717_round{MANUAL_REVIEW_ROUNDS[patient]}"
                ),
                "exclusion_review_round": MANUAL_REVIEW_ROUNDS[patient],
                "name_resolution": (
                    "unique_workspace_match" if submitted != patient else "exact_normalized_match"
                ),
            }
            for patient, submitted in MANUAL_REVIEW_EXCLUSIONS.items()
        ],
    }

    _write_json_atomic(report_dir / "ilo_before_manifest.json", manifest)
    _write_json_atomic(report_dir / "ilo_before_whitelist.json", whitelist)
    _write_json_atomic(report_dir / "ilo_before_exclusions.json", exclusion_payload)
    _write_json_atomic(report_dir / "ilo_before_name_overlaps.json", overlap_payload)
    _write_json_atomic(
        report_dir / "ilo_before_manual_review_decisions_final.json",
        manual_review_payload,
    )
    for evidence_name in (
        "ilo_after_derived_deletion_manifest.json",
        "ilo_after_derived_deletion_result.json",
    ):
        candidates = (
            BASELINE_REPORT_DIR / evidence_name,
            REPORT_ARCHIVE_ROOT / BASELINE_REPORT_DIR.name / evidence_name,
        )
        source = next((path for path in candidates if path.is_file()), None)
        if source is not None:
            evidence = json.loads(source.read_text(encoding="utf-8"))
            evidence["evidence_copied_from"] = str(source)
            if evidence_name.endswith("_result.json"):
                evidence["manifest"] = str(
                    report_dir / "ilo_after_derived_deletion_manifest.json"
                )
            _write_json_atomic(report_dir / evidence_name, evidence)
    _write_csv_atomic(report_dir / "ilo_before_inventory.csv", inventory)
    _write_csv_atomic(report_dir / "ilo_before_near_name_watch.csv", near_matches)
    return manifest


def load_ilo_before_whitelist(path: Path = DEFAULT_WHITELIST) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"ILO-before 白名单不存在，先运行 inventory: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    units = list(payload.get("included_units", []))
    if any(not is_ilo_before_unit(unit) for unit in units):
        raise RuntimeError("ILO-before 白名单混入非 before 单元")
    if len(units) != len(set(units)):
        raise RuntimeError("ILO-before 白名单存在重复 ID")
    return units


def _scalar(z: np.lib.npyio.NpzFile, key: str):
    value = np.asarray(z[key])
    return value.item() if value.ndim == 0 else value


def _quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_p{label}": float(np.percentile(values, q))
        for q, label in ((0, "0"), (50, "50"), (90, "90"), (95, "95"),
                         (99, "99"), (99.9, "99_9"), (100, "100"))
    }


def wss_hard_failures(wss: np.ndarray, steps: np.ndarray) -> list[str]:
    """返回 ILO/AG/AAA 共用数值硬门禁的稳定错误码。"""
    failures: list[str] = []
    values = np.asarray(wss)
    step_ids = np.asarray(steps)
    if not np.isfinite(values).all():
        failures.append("wss_nan_or_inf")
    with np.errstate(invalid="ignore"):
        if (values < 0).any():
            failures.append("negative_wss")
        if float((values <= 0).mean()) > 0.01:
            failures.append("nonpositive_wss_over_1pct")
        if values.ndim == 2 and len(values) == len(step_ids):
            zero_steps = step_ids[np.all(values <= 0, axis=1)].tolist()
            if zero_steps:
                failures.append("all_zero_timesteps:" + ",".join(map(str, zero_steps)))
    return failures


def _reference_limits() -> dict[str, float]:
    if not REFERENCE_QUALITY_AUDIT.is_file():
        return {}
    rows = json.loads(REFERENCE_QUALITY_AUDIT.read_text(encoding="utf-8"))["rows"]
    accepted = [row for row in rows if row.get("quality_decision") == "include"]
    return {
        "all_wss_p100_max": max(float(row["all_wss_p100"]) for row in accepted),
        "peak_wss_p100_max": max(float(row["peak_wss_p100"]) for row in accepted),
        "temporal_spike_max": max(float(row["temporal_step_max_spike_ratio"]) for row in accepted),
        "inlet_abs_max_median": float(np.median([
            float(row["inlet_waveform_abs_max"])
            for row in accepted if row.get("inlet_waveform_abs_max") is not None
        ])),
    }


def audit_bundle(unit_id: str, reference: dict[str, float]) -> dict:
    validate_active_preprocess_unit(unit_id)
    case_dir = C.OUT_ROOT / unit_id
    raw_case_dir = C.RAW_ROOT / unit_id
    bundle_path = case_dir / "bundle.npz"
    report_path = case_dir / "report.json"
    row: dict = {
        "unit_id": unit_id,
        "bundle_path": str(bundle_path),
        "report_path": str(report_path),
        "qa_pass": False,
        "hard_failures": "",
        "soft_flags": "",
    }
    hard: list[str] = []
    watch: list[str] = []
    if not bundle_path.is_file():
        hard.append("missing_bundle")
    if not report_path.is_file():
        hard.append("missing_report")
    if hard:
        row["hard_failures"] = "|".join(hard)
        return row

    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = {
        "steps", "peak_step", "coord_scale", "wall_nodenumber", "wall_coords_raw",
        "wall_coords_norm", "wall_wss", "wall_pressure", "transform_centroid",
        "transform_rotation", "transform_frame_version", "transform_landmark_source",
        "transform_landmark_trunk_point", "transform_landmark_left_point",
        "transform_landmark_right_point",
    }
    with np.load(bundle_path, allow_pickle=False) as z:
        missing = sorted(required - set(z.files))
        if missing:
            hard.append("missing_bundle_keys:" + ",".join(missing))
        else:
            steps = np.asarray(z["steps"], dtype=np.int64)
            peak = int(_scalar(z, "peak_step"))
            ids = np.asarray(z["wall_nodenumber"], dtype=np.int64)
            raw = np.asarray(z["wall_coords_raw"])
            norm = np.asarray(z["wall_coords_norm"])
            wss = np.asarray(z["wall_wss"])
            pressure = np.asarray(z["wall_pressure"])
            centroid = np.asarray(z["transform_centroid"], dtype=np.float64)
            rotation = np.asarray(z["transform_rotation"], dtype=np.float64)
            landmarks = np.vstack([
                z["transform_landmark_trunk_point"],
                z["transform_landmark_left_point"],
                z["transform_landmark_right_point"],
            ]).astype(np.float64)
            scale = float(_scalar(z, "coord_scale"))
            frame = str(_scalar(z, "transform_frame_version"))
            landmark_source = str(_scalar(z, "transform_landmark_source"))

            if frame != FRAME_VERSION:
                hard.append(f"wrong_frame:{frame}")
            if landmark_source != LANDMARK_SOURCE:
                hard.append(f"wrong_landmark_source:{landmark_source}")
            if len(np.unique(ids)) != len(ids):
                hard.append("duplicate_node_ids")
            if raw.shape != (len(ids), 3) or norm.shape != (len(ids), 3):
                hard.append("coordinate_shape_mismatch")
            if wss.shape != (len(steps), len(ids)):
                hard.append("wss_shape_mismatch")
            if pressure.shape != (len(steps), len(ids)):
                hard.append("pressure_shape_mismatch")
            if not np.isfinite(raw).all() or not np.isfinite(norm).all():
                hard.append("coords_nan_or_inf")
            hard.extend(wss_hard_failures(wss, steps))
            nonpositive = float((wss <= 0).mean())
            zero_steps = steps[np.all(wss <= 0, axis=1)].tolist()
            if peak not in steps.tolist():
                hard.append("peak_step_not_exported")
                peak_index = 0
            else:
                peak_index = steps.tolist().index(peak)
            peak_wss = wss[peak_index]
            step_max = np.max(wss, axis=1)

            waveform = raw_io.read_inlet_waveform(raw_case_dir)
            candidates = {int(step): waveform[int(step)] for step in steps if int(step) in waveform}
            expected_peak = max(candidates, key=candidates.get) if candidates else None
            if expected_peak is None:
                hard.append("missing_inlet_waveform_for_exported_steps")
            elif expected_peak != peak:
                hard.append(f"peak_waveform_mismatch:{peak}!={expected_peak}")

            aligned_landmarks = (landmarks - centroid) @ rotation
            trunk_z, left_z, right_z = aligned_landmarks[:, 2]
            det = float(np.linalg.det(rotation))
            orth_err = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
            lr_world_dx = float(landmarks[1, 0] - landmarks[2, 0])
            reconstruction = ((raw.astype(np.float64) - centroid) @ rotation) / scale
            recon_err = float(np.max(np.abs(reconstruction - norm)))
            norm_abs_max = float(np.max(np.abs(norm)))
            if rotation.shape != (3, 3) or orth_err > GEOM_TOL:
                hard.append("rotation_not_orthonormal")
            if abs(det - 1.0) > GEOM_TOL:
                hard.append("rotation_not_right_handed")
            if not (trunk_z > 0 and left_z < 0 and right_z < 0):
                hard.append("axial_landmark_sign_failed")
            if lr_world_dx <= 0:
                hard.append("lr_world_x_sign_failed")
            if not np.isfinite(scale) or scale <= 0:
                hard.append("invalid_coord_scale")
            if recon_err > NORM_TOL:
                hard.append("normalization_reconstruction_failed")
            if abs(norm_abs_max - 1.0) > NORM_TOL:
                hard.append("normalized_coords_not_unit_scaled")
            if not report.get("nodenumber_alignment_ok", False):
                hard.append("report_node_alignment_failed")
            if int(report.get("wall_coord_mismatch_n_steps", 0) or 0) > 0:
                hard.append("wall_coordinate_mismatch")
            if not report.get("peak_from_waveform", False):
                hard.append("report_peak_not_from_waveform")

            all_q = _quantiles(wss.reshape(-1), "all_wss")
            peak_q = _quantiles(peak_wss, "peak_wss")
            temporal_spike = float(step_max.max() / max(float(np.median(step_max)), 1e-12))
            inlet_values = np.asarray(list(waveform.values()), dtype=np.float64)
            inlet_abs_max = float(np.max(np.abs(inlet_values))) if len(inlet_values) else math.nan
            if reference:
                if all_q["all_wss_p100"] > reference["all_wss_p100_max"]:
                    watch.append("all_wss_max_above_AG_AAA_reference")
                if peak_q["peak_wss_p100"] > reference["peak_wss_p100_max"]:
                    watch.append("peak_wss_max_above_AG_AAA_reference")
                if temporal_spike > reference["temporal_spike_max"]:
                    watch.append("temporal_spike_above_AG_AAA_reference")
                if inlet_abs_max > 10.0 * reference["inlet_abs_max_median"]:
                    watch.append("inlet_waveform_scale_watch")
            soft = []
            if report.get("centerline_translation_applied"):
                soft.append("SHIFT")
            origin = str(report.get("origin_kind", ""))
            if "stl_bifurcation" in origin:
                soft.append("STL-O")
            if "axial_guard" in origin:
                soft.append("AX")
            if report.get("unit_anomaly") or report.get("unit_extent_mismatch"):
                soft.append("UNIT")
            if int(report.get("wall_coord_spatial_remap_n_steps", 0) or 0) > 0:
                soft.append("COORD-REMAP")

            row.update({
                "frame_version": frame,
                "landmark_source": landmark_source,
                "n_steps": int(len(steps)),
                "n_nodes": int(len(ids)),
                "peak_step": peak,
                "expected_peak_step": expected_peak,
                "nonpositive_wss_frac": nonpositive,
                "negative_wss_count": int((wss < 0).sum()),
                "nonfinite_wss_count": int((~np.isfinite(wss)).sum()),
                "all_zero_timestep_count": len(zero_steps),
                "temporal_step_max_spike_ratio": temporal_spike,
                "pressure_min": float(np.min(pressure)),
                "pressure_max": float(np.max(pressure)),
                "pressure_mean": float(np.mean(pressure)),
                "pressure_std": float(np.std(pressure)),
                "inlet_waveform_exists": bool(waveform),
                "inlet_waveform_abs_max": inlet_abs_max,
                "rotation_det": det,
                "rotation_orthogonality_error": orth_err,
                "trunk_z_mm": float(trunk_z),
                "left_iliac_z_mm": float(left_z),
                "right_iliac_z_mm": float(right_z),
                "lr_world_dx_mm": lr_world_dx,
                "coord_scale_mm": scale,
                "norm_abs_max": norm_abs_max,
                "normalization_reconstruction_max_abs_error": recon_err,
                "centerline_translation_applied": bool(report.get("centerline_translation_applied")),
                "soft_flags": "|".join(soft),
                "wss_watch_flags": "|".join(watch),
                **all_q,
                **peak_q,
            })
    row["hard_failures"] = "|".join(dict.fromkeys(hard))
    row["qa_pass"] = not hard
    return row


def run_bundle_audit(report_dir: Path = DEFAULT_REPORT_DIR) -> dict:
    report_dir = report_dir.resolve()
    units = load_ilo_before_whitelist(report_dir / "ilo_before_whitelist.json")
    reference = _reference_limits()
    rows: list[dict] = []
    for index, unit in enumerate(units, 1):
        row = audit_bundle(unit, reference)
        rows.append(row)
        print(
            f"[{index:02d}/{len(units)}] {'PASS' if row['qa_pass'] else 'FAIL'} {unit} "
            f"{row.get('hard_failures', '')}",
            flush=True,
        )
    failed = [row for row in rows if not row["qa_pass"]]
    watch = [row for row in rows if row.get("wss_watch_flags")]
    repairs = [row["unit_id"] for row in rows if row.get("centerline_translation_applied")]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "scope": "ILO before whitelist only",
        "counts": {
            "expected": len(units),
            "qa_pass": len(rows) - len(failed),
            "qa_fail": len(failed),
            "wss_watch": len(watch),
            "after_units": 0,
        },
        "failed_units": [row["unit_id"] for row in failed],
        "watch_units": [row["unit_id"] for row in watch],
        "centerline_translation_repaired_units": repairs,
        "reference_limits": reference,
        "rows": rows,
    }
    _write_json_atomic(report_dir / "ilo_before_wss_geometry_audit.json", summary)
    _write_csv_atomic(report_dir / "ilo_before_wss_geometry_audit.csv", rows)
    if len(units) != EXPECTED_INCLUDED_BEFORE or failed:
        raise RuntimeError(
            f"ILO-before 终审未达到 {EXPECTED_INCLUDED_BEFORE}/{EXPECTED_INCLUDED_BEFORE}: "
            f"expected={len(units)} failed={len(failed)}"
        )
    if repairs:
        raise RuntimeError(f"centerline 平移修复集合异常: {repairs}")
    return summary


def discover_after_derived(out_root: Path = C.OUT_ROOT) -> tuple[list[Path], list[Path]]:
    ilo_root = out_root.resolve() / "ILO"
    targets: list[Path] = []
    after_dirs: list[Path] = []
    if not ilo_root.is_dir():
        return targets, after_dirs
    for directory in sorted(ilo_root.glob("*/after")):
        if not directory.is_dir():
            continue
        resolved = directory.resolve()
        if resolved.parent.parent != ilo_root:
            raise RuntimeError(f"非法 after 目录边界: {resolved}")
        after_dirs.append(resolved)
        for name in ("bundle.npz", "report.json"):
            path = resolved / name
            if path.is_file():
                targets.append(path)
    return targets, after_dirs


def delete_after_derived(
    report_dir: Path = DEFAULT_REPORT_DIR,
    *,
    execute: bool = False,
) -> dict:
    report_dir = report_dir.resolve()
    targets, after_dirs = discover_after_derived()
    rows: list[dict] = []
    total_bytes = 0
    for index, path in enumerate(targets, 1):
        size = path.stat().st_size
        total_bytes += size
        print(f"[sha {index:03d}/{len(targets)}] {path}", flush=True)
        rows.append({
            "path": str(path),
            "relative_path": str(path.relative_to(C.PROJECT_ROOT)),
            "size_bytes": size,
            "sha256": sha256(path),
        })
    before = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "decision": "direct_delete_derived_after_keep_raw_sources",
        "scope": "data_wss_min/ILO/*/after/{bundle.npz,report.json}",
        "raw_after_root_preserved": str(C.RAW_ROOT / "ILO"),
        "counts": {
            "after_dirs": len(after_dirs),
            "bundle_files": sum(path.name == "bundle.npz" for path in targets),
            "report_files": sum(path.name == "report.json" for path in targets),
            "target_files": len(targets),
            "total_bytes": total_bytes,
        },
        "files": rows,
        "execute_requested": execute,
    }
    manifest_path = report_dir / "ilo_after_derived_deletion_manifest.json"
    _write_json_atomic(manifest_path, before)

    if execute:
        if len(after_dirs) != 55 or len(targets) != 110:
            raise RuntimeError(
                f"拒绝删除：目标数量与已签范围不一致 dirs={len(after_dirs)} files={len(targets)}"
            )
        if sum(path.name == "bundle.npz" for path in targets) != 55:
            raise RuntimeError("拒绝删除：bundle 数量不是 55")
        if sum(path.name == "report.json" for path in targets) != 55:
            raise RuntimeError("拒绝删除：report 数量不是 55")
        for path in targets:
            path.unlink()
        removed_dirs: list[str] = []
        retained_nonempty_dirs: list[str] = []
        for directory in after_dirs:
            try:
                directory.rmdir()
                removed_dirs.append(str(directory))
            except OSError:
                retained_nonempty_dirs.append(str(directory))

        remaining_targets, remaining_dirs = discover_after_derived()
        raw_after_dirs = [
            path for path in (C.RAW_ROOT / "ILO").glob("*/after") if path.is_dir()
        ]
        result = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_now(),
            "manifest": str(manifest_path),
            "deleted_files": len(targets),
            "deleted_bytes": total_bytes,
            "removed_empty_after_dirs": removed_dirs,
            "retained_nonempty_after_dirs": retained_nonempty_dirs,
            "remaining_after_bundle_report_files": len(remaining_targets),
            "remaining_processed_after_dirs": len(remaining_dirs),
            "raw_after_dirs_preserved": len(raw_after_dirs),
            "verification_pass": (
                len(remaining_targets) == 0
                and len(remaining_dirs) == 0
                and len(raw_after_dirs) == 61
            ),
        }
        _write_json_atomic(report_dir / "ilo_after_derived_deletion_result.json", result)
        if not result["verification_pass"]:
            raise RuntimeError(f"after 删除后验证失败: {result}")
        return result
    return before


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("inventory", "qa", "delete-after", "all"), default="all"
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--execute-delete-after", action="store_true")
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()

    result: dict = {}
    if args.stage in {"inventory", "all"}:
        result["inventory"] = build_inventory(report_dir)["counts"]
    if args.stage in {"qa", "all"}:
        result["qa"] = run_bundle_audit(report_dir)["counts"]
    if args.stage in {"delete-after", "all"}:
        deletion = delete_after_derived(
            report_dir, execute=args.execute_delete_after,
        )
        result["delete_after"] = deletion.get("counts", deletion)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
