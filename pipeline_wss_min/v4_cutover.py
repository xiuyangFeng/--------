#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AG/AAA v4 staging 重建、硬 QA、数据清单与旧版快照清单。

本模块不会执行活动目录切换。``preprocess`` 只写显式 ``--out-root``；``qa``
同时检查 staging AG 与现有 AAA bundle；``snapshot-manifest`` 只读旧 AG 产物并
生成 SHA-256 清单。真正的目录提升必须等用户看完可视化后单独执行。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from . import config as C
from . import preprocess, reporting
from .new_cohorts.common import candidate_units, split_unit


FRAME_VERSION = "stl_landmarks_v4"
LANDMARK_SOURCE = "original_stl"
AG_PARTITIONS = ("train", "val", "test")
AG_EXCLUDED_OR_PENDING = frozenset({
    "AG/fast/PENG_JI_MING",
    "AG/slow/LIN_SHU_TIAN",
    "AG/slow/LIU_XI_QUAN",
    "AG/slow/NIE_QUAN_ZHONG",
    "AG/slow/SUN_WEN_QING",
    "AG/slow/TE_JIN_WANG",
    "AG/slow/WANG_BAO_SHAN",
    "AG/slow/WEI_JUN_WEN",
    "AG/slow/ZHANG_HUAN_LI",
    "AG/slow/ZHAO_XIU_XUAN",
})
GEOM_TOL = 1e-6
NORM_TOL = 2e-5


def _scalar(z, key: str):
    value = np.asarray(z[key])
    return value.item() if value.ndim == 0 else value


def ag_units(split_name: str = C.DEFAULT_SPLIT_NAME) -> list[str]:
    units = [f"AG/{label}" for label in C.split_case_labels(split_name, AG_PARTITIONS)]
    leaked = set(units) & AG_EXCLUDED_OR_PENDING
    if leaked:
        raise RuntimeError(f"AG excluded/pending units leaked into v4 staging: {sorted(leaked)}")
    if split_name == C.DEFAULT_SPLIT_NAME and len(units) != 77:
        raise RuntimeError(f"formal AG v4 split must contain exactly 77 units, got {len(units)}")
    return units


def aaa_units() -> list[str]:
    return [unit for unit in candidate_units() if unit.startswith("AAA/")]


def unit_parts(unit_id: str) -> tuple[str, str]:
    parts = unit_id.split("/")
    if parts[0] == "AG" and len(parts) == 3:
        return f"AG/{parts[1]}", parts[2]
    return split_unit(unit_id)


def bundle_path(data_root: str | Path, unit_id: str) -> Path:
    return Path(data_root).resolve() / unit_id / "bundle.npz"


def report_path(data_root: str | Path, unit_id: str) -> Path:
    return Path(data_root).resolve() / unit_id / "report.json"


def soft_flags(report: dict) -> list[str]:
    flags: list[str] = []
    if report.get("centerline_translation_applied"):
        flags.append("SHIFT")
    origin = str(report.get("origin_kind", ""))
    if "stl_bifurcation" in origin:
        flags.append("STL-O")
    if "axial_guard" in origin:
        flags.append("AX")
    if report.get("unit_anomaly") or report.get("unit_extent_mismatch"):
        flags.append("UNIT")
    if float(report.get("wall_crop_frac", 0.0) or 0.0) > 0.05:
        flags.append("CROP")
    if int(report.get("wall_coord_spatial_remap_n_steps", 0) or 0) > 0:
        flags.append("COORD-REMAP")
    elif int(report.get("wall_coord_mismatch_n_steps", 0) or 0) > 0:
        flags.append("COORD-TS")
    return flags


def check_v4_bundle(
    unit_id: str,
    data_root: str | Path,
    *,
    enforce_ag_physical_limits: bool,
    fallback_root: str | Path | None = None,
) -> dict:
    """检查实际落盘 bundle，而不是从原始几何临时重算。"""
    bpath = bundle_path(data_root, unit_id)
    rpath = report_path(data_root, unit_id)
    if fallback_root is not None and (not bpath.is_file() or not rpath.is_file()):
        fallback_bundle = bundle_path(fallback_root, unit_id)
        fallback_report = report_path(fallback_root, unit_id)
        if fallback_bundle.is_file() and fallback_report.is_file():
            bpath, rpath = fallback_bundle, fallback_report
    row = {
        "unit_id": unit_id,
        "bundle_path": str(bpath),
        "report_path": str(rpath),
        "bundle_exists": bpath.is_file(),
        "report_exists": rpath.is_file(),
        "qa_pass": False,
        "reason": "",
    }
    reasons: list[str] = []
    report: dict = {}
    if not bpath.is_file():
        reasons.append("missing_bundle")
    if not rpath.is_file():
        reasons.append("missing_report")
    if reasons:
        row["reason"] = ";".join(reasons)
        return row

    try:
        report = json.loads(rpath.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"report_read_error:{type(exc).__name__}:{exc}")
        row["reason"] = ";".join(reasons)
        return row

    if report.get("status") != "ok":
        reasons.append("report_not_ok")
    if report.get("frame_version") != FRAME_VERSION:
        reasons.append("report_wrong_frame_version")
    if report.get("landmark_source") != LANDMARK_SOURCE:
        reasons.append("report_wrong_landmark_source")
    if not report.get("nodenumber_alignment_ok", False):
        reasons.append("nodenumber_alignment_failed")
    if enforce_ag_physical_limits and int(report.get("wall_coord_mismatch_n_steps", 0) or 0) > 0:
        reasons.append("wall_coord_mismatch")
    if enforce_ag_physical_limits and report.get("unit_anomaly"):
        reasons.append("unit_anomaly")
    if enforce_ag_physical_limits and report.get("unit_extent_mismatch"):
        reasons.append("unit_extent_mismatch")

    required = {
        "steps", "peak_step", "coord_scale", "transform_centroid", "transform_rotation",
        "transform_frame_version", "transform_landmark_source",
        "transform_landmark_trunk_point", "transform_landmark_left_point",
        "transform_landmark_right_point", "wall_coords_norm", "wall_coords_raw",
        "wall_nodenumber", "wall_wss", "original_stl_path", "original_stl_match_score",
    }
    try:
        with np.load(bpath, allow_pickle=False) as z:
            missing = sorted(required.difference(z.files))
            if missing:
                reasons.append("missing_bundle_keys:" + ",".join(missing))
            else:
                frame_version = str(_scalar(z, "transform_frame_version"))
                landmark_source = str(_scalar(z, "transform_landmark_source"))
                rotation = np.asarray(z["transform_rotation"], dtype=np.float64)
                centroid = np.asarray(z["transform_centroid"], dtype=np.float64)
                landmarks = np.vstack([
                    z["transform_landmark_trunk_point"],
                    z["transform_landmark_left_point"],
                    z["transform_landmark_right_point"],
                ]).astype(np.float64)
                raw = np.asarray(z["wall_coords_raw"], dtype=np.float64)
                norm = np.asarray(z["wall_coords_norm"], dtype=np.float64)
                scale = float(_scalar(z, "coord_scale"))
                wss = np.asarray(z["wall_wss"], dtype=np.float64)
                steps = np.asarray(z["steps"]).tolist()
                peak_step = int(_scalar(z, "peak_step"))

                orth_err = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
                det = float(np.linalg.det(rotation))
                aligned_lm = (landmarks - centroid) @ rotation
                trunk_z, left_z, right_z = aligned_lm[:, 2]
                lr_world_dx = float(landmarks[1, 0] - landmarks[2, 0])
                reconstructed = ((raw - centroid) @ rotation) / scale
                recon_err = float(np.max(np.abs(reconstructed - norm))) if len(norm) else np.inf
                norm_abs_max = float(np.max(np.abs(norm))) if len(norm) else np.nan
                norm_min = np.min(norm, axis=0) if len(norm) else np.full(3, np.nan)
                norm_max = np.max(norm, axis=0) if len(norm) else np.full(3, np.nan)
                finite_wss = np.isfinite(wss)
                finite_norm = np.isfinite(norm)
                peak_index = steps.index(peak_step)
                peak_wss = wss[peak_index]
                peak_finite = np.isfinite(peak_wss)
                all_zero_frac = float((wss[finite_wss] <= 0).mean()) if finite_wss.any() else np.nan
                peak_zero_frac = float((peak_wss[peak_finite] <= 0).mean()) if peak_finite.any() else np.nan
                peak_p90 = float(np.nanpercentile(peak_wss, 90))

                row.update({
                    "frame_version": frame_version,
                    "landmark_source": landmark_source,
                    "rotation_det": det,
                    "orthogonality_max_abs_error": orth_err,
                    "trunk_z_mm": float(trunk_z),
                    "left_iliac_z_mm": float(left_z),
                    "right_iliac_z_mm": float(right_z),
                    "lr_world_dx_mm": lr_world_dx,
                    "coord_scale_mm": scale,
                    "normalization_reconstruction_max_abs_error": recon_err,
                    "norm_abs_max": norm_abs_max,
                    "norm_x_min": float(norm_min[0]), "norm_x_max": float(norm_max[0]),
                    "norm_y_min": float(norm_min[1]), "norm_y_max": float(norm_max[1]),
                    "norm_z_min": float(norm_min[2]), "norm_z_max": float(norm_max[2]),
                    "n_wall": int(norm.shape[0]),
                    "n_steps": int(wss.shape[0]),
                    "peak_zero_frac": peak_zero_frac,
                    "all_zero_frac": all_zero_frac,
                    "peak_p90": peak_p90,
                    "original_stl_path": str(_scalar(z, "original_stl_path")),
                    "original_stl_match_score": float(_scalar(z, "original_stl_match_score")),
                    "soft_flags": "|".join(soft_flags(report)),
                })

                if frame_version != FRAME_VERSION:
                    reasons.append("bundle_wrong_frame_version")
                if landmark_source != LANDMARK_SOURCE:
                    reasons.append("bundle_wrong_landmark_source")
                if rotation.shape != (3, 3) or orth_err > GEOM_TOL:
                    reasons.append("rotation_not_orthonormal")
                if abs(det - 1.0) > GEOM_TOL:
                    reasons.append("rotation_not_right_handed")
                if not (trunk_z > 0 and left_z < 0 and right_z < 0):
                    reasons.append("axial_landmark_sign_failed")
                if lr_world_dx <= 0:
                    reasons.append("lr_world_x_sign_failed")
                if not np.isfinite(scale) or scale <= 0:
                    reasons.append("invalid_coord_scale")
                if recon_err > NORM_TOL:
                    reasons.append("normalization_reconstruction_failed")
                if not finite_norm.all():
                    reasons.append("coords_nonfinite")
                if not np.isfinite(norm_abs_max) or norm_abs_max > 1.0 + NORM_TOL:
                    reasons.append("normalized_coords_out_of_bounds")
                if abs(norm_abs_max - 1.0) > NORM_TOL:
                    reasons.append("normalized_coords_not_unit_scaled")
                if not finite_wss.all():
                    reasons.append("wss_nonfinite")
                if peak_zero_frac > 0.01:
                    reasons.append("peak_zero_frac>0.01")
                if all_zero_frac > 0.01:
                    reasons.append("all_zero_frac>0.01")
                if peak_p90 <= 0:
                    reasons.append("peak_p90<=0")
                if len(np.unique(z["wall_nodenumber"])) != len(z["wall_nodenumber"]):
                    reasons.append("duplicate_wall_nodenumber")
                if wss.shape[1] != norm.shape[0]:
                    reasons.append("wall_wss_coord_count_mismatch")
                if enforce_ag_physical_limits and norm.shape[0] > 50000:
                    reasons.append("n_wall>50000")
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"bundle_read_error:{type(exc).__name__}:{exc}")

    row["reason"] = ";".join(dict.fromkeys(reasons))
    row["qa_pass"] = not reasons
    return row


def _write_rows(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    tmp_path = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, csv_path)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp_path, path)


def run_qa(
    ag_root: Path,
    aaa_root: Path,
    report_dir: Path,
    split_name: str = C.DEFAULT_SPLIT_NAME,
    strict: bool = True,
    aaa_fallback_root: Path | None = None,
) -> dict:
    expected_ag = ag_units(split_name)
    expected_aaa = aaa_units()
    rows = [
        check_v4_bundle(unit, ag_root, enforce_ag_physical_limits=True)
        for unit in expected_ag
    ]
    rows.extend(
        check_v4_bundle(
            unit, aaa_root, enforce_ag_physical_limits=False,
            fallback_root=aaa_fallback_root,
        )
        for unit in expected_aaa
    )
    ag_disk = sorted(
        str(p.parent.relative_to(ag_root)).replace(os.sep, "/")
        for p in (ag_root / "AG").glob("*/*/bundle.npz")
    ) if (ag_root / "AG").is_dir() else []
    expected_ag_set = set(expected_ag)
    extra_ag = sorted(set(ag_disk) - expected_ag_set)
    missing_ag = sorted(expected_ag_set - set(ag_disk))

    for row in rows:
        cohort, category, *_ = row["unit_id"].split("/")
        row["canonical_unit_id"] = row["unit_id"]
        row["cohort"] = cohort
        row["category"] = category
        row["exclusion_reason"] = "" if row.get("qa_pass") else row.get("reason", "")
        row["review_status"] = (
            "review_pending" if row.get("soft_flags") else "accepted_auto"
        ) if row.get("qa_pass") else "qa_failed"
        row["manual_signoff_status"] = (
            "pending" if row.get("soft_flags") else "not_required"
        ) if row.get("qa_pass") else "blocked_by_qa"

    ag_rows = [row for row in rows if row["unit_id"].startswith("AG/")]
    aaa_rows = [row for row in rows if row["unit_id"].startswith("AAA/")]
    flagged_aaa = [row["unit_id"] for row in aaa_rows if row.get("soft_flags")]
    flagged_ag = [row["unit_id"] for row in ag_rows if row.get("soft_flags")]
    failures = [row["unit_id"] for row in rows if not row.get("qa_pass")]
    summary = {
        "frame_version": FRAME_VERSION,
        "ag_root": str(ag_root.resolve()),
        "aaa_root": str(aaa_root.resolve()),
        "aaa_fallback_root": str(aaa_fallback_root.resolve()) if aaa_fallback_root else None,
        "split_name": split_name,
        "n_ag_expected": len(expected_ag),
        "n_ag_pass": sum(bool(row.get("qa_pass")) for row in ag_rows),
        "n_ag_review_pending": len(flagged_ag),
        "ag_review_pending_units": flagged_ag,
        "n_aaa_expected": len(expected_aaa),
        "n_aaa_pass": sum(bool(row.get("qa_pass")) for row in aaa_rows),
        "n_aaa_review_pending": len(flagged_aaa),
        "aaa_review_pending_units": flagged_aaa,
        "failures": failures,
        "ag_extra_bundle_units": extra_ag,
        "ag_missing_bundle_units": missing_ag,
        "ready_for_visual_review": not failures and not extra_ag and not missing_ag,
        "promotion_authorized": False,
        "promotion_blocker": "必须由用户查看 cutover 可视化后明确签核",
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, report_dir / "v4_dataset_manifest.csv")
    payload = {"summary": summary, "rows": rows}
    _write_json_atomic(report_dir / "v4_dataset_manifest.json", payload)
    _write_json_atomic(report_dir / "v4_cutover_qa_summary.json", summary)
    if strict and not summary["ready_for_visual_review"]:
        raise RuntimeError(
            f"v4 cutover QA failed: failures={len(failures)} "
            f"extra_ag={len(extra_ag)} missing_ag={len(missing_ag)}"
        )
    return payload


def finalize_review(report_dir: Path, decisions_path: Path) -> dict:
    """将用户的人工审核决定合并到 v4 manifest，不删除任何 bundle。"""
    manifest_path = report_dir / "v4_dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少 QA manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    if decisions.get("frame_version") != FRAME_VERSION:
        raise RuntimeError("人工审核决定的 frame_version 不是 stl_landmarks_v4")
    if not decisions.get("visual_review_signed_off"):
        raise RuntimeError("人工审核尚未签核")

    excluded_rows = decisions.get("excluded_units", [])
    excluded = {row["unit_id"]: row for row in excluded_rows}
    approved = set(decisions.get("approved_review_units", []))
    if set(excluded) & approved:
        raise RuntimeError("同一病例同时被批准和排除")
    rows = manifest.get("rows", [])
    known = {row["unit_id"] for row in rows}
    unknown = (set(excluded) | approved) - known
    if unknown:
        raise RuntimeError(f"人工审核包含未知病例: {sorted(unknown)}")

    auto_accept = bool(decisions.get("auto_accept_all_unflagged_except_exclusions"))
    pending = []
    qa_failures = []
    final_rows = []
    for source_row in rows:
        row = dict(source_row)
        unit = row["unit_id"]
        flagged = bool(row.get("soft_flags"))
        if not row.get("qa_pass"):
            row["final_status"] = "qa_failed"
            row["manual_signoff_status"] = "blocked_by_qa"
            row["final_eligible"] = False
            qa_failures.append(unit)
        elif unit in excluded:
            decision = excluded[unit]
            row["final_status"] = "excluded_by_user"
            row["manual_signoff_status"] = "excluded"
            row["final_eligible"] = False
            row["exclusion_reason"] = decision.get("reason", "user_visual_exclusion")
            row["human_decision_note"] = decision.get("note", "")
        elif unit in approved:
            row["final_status"] = "approved_by_user"
            row["manual_signoff_status"] = "approved"
            row["final_eligible"] = True
            row["exclusion_reason"] = ""
        elif not flagged and auto_accept:
            row["final_status"] = "accepted_auto"
            row["manual_signoff_status"] = "not_required"
            row["final_eligible"] = True
            row["exclusion_reason"] = ""
        else:
            row["final_status"] = "review_pending"
            row["manual_signoff_status"] = "pending"
            row["final_eligible"] = False
            pending.append(unit)
        final_rows.append(row)

    eligible = [row["unit_id"] for row in final_rows if row["final_eligible"]]
    eligible_ag = sorted(unit for unit in eligible if unit.startswith("AG/"))
    eligible_aaa = sorted(unit for unit in eligible if unit.startswith("AAA/"))
    expected_counts = decisions.get("expected_final_counts", {})
    if int(expected_counts.get("AG", -1)) != len(eligible_ag):
        raise RuntimeError(
            f"AG 终签数量不符: expected={expected_counts.get('AG')} actual={len(eligible_ag)}"
        )
    if int(expected_counts.get("AAA", -1)) != len(eligible_aaa):
        raise RuntimeError(
            f"AAA 终签数量不符: expected={expected_counts.get('AAA')} actual={len(eligible_aaa)}"
        )

    summary = {
        "frame_version": FRAME_VERSION,
        "decision_date": decisions.get("decision_date"),
        "visual_review_signed_off": True,
        "n_source_rows": len(final_rows),
        "n_ag_final_eligible": len(eligible_ag),
        "n_aaa_final_eligible": len(eligible_aaa),
        "n_excluded_by_user": len(excluded),
        "excluded_units": sorted(excluded),
        "n_approved_by_user": sum(
            row["final_status"] == "approved_by_user" for row in final_rows
        ),
        "n_review_pending": len(pending),
        "review_pending_units": pending,
        "qa_failures": qa_failures,
        "ready_for_promotion": not pending and not qa_failures,
        "promotion_authorized": False,
        "promotion_blocker": "终签已完成；需单独执行带 --confirm-reviewed 的原子切换",
    }
    payload = {"summary": summary, "rows": final_rows}
    whitelist = {
        "frame_version": FRAME_VERSION,
        "decision_date": decisions.get("decision_date"),
        "AG": eligible_ag,
        "AAA": eligible_aaa,
        "counts": {"AG": len(eligible_ag), "AAA": len(eligible_aaa)},
    }
    exclusions = {
        "frame_version": FRAME_VERSION,
        "decision_date": decisions.get("decision_date"),
        "manual_exclusions": excluded_rows,
        "inherited_ag_excluded_or_pending": sorted(AG_EXCLUDED_OR_PENDING),
    }
    _write_rows(final_rows, report_dir / "v4_final_dataset_manifest.csv")
    _write_json_atomic(report_dir / "v4_final_dataset_manifest.json", payload)
    _write_json_atomic(report_dir / "v4_final_whitelist.json", whitelist)
    _write_json_atomic(report_dir / "v4_final_exclusions.json", exclusions)
    _write_json_atomic(report_dir / "v4_manual_review_summary.json", summary)
    return payload


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_manifest(legacy_root: Path, output: Path, split_name: str) -> dict:
    # ``legacy-root`` historically means the data root, but the migration table
    # names ``data_wss_min/AG`` as the legacy bundle path.  Accept either spelling
    # and always store paths relative to the data root so promotion can verify them.
    if legacy_root.name == "AG" and any(legacy_root.glob("*/*/bundle.npz")):
        legacy_root = legacy_root.parent
    files = sorted(
        path for pattern in ("AG/*/*/bundle.npz", "AG/*/*/report.json")
        for path in legacy_root.glob(pattern)
    )
    rows = [{
        "relative_path": str(path.relative_to(legacy_root)),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    } for path in files]
    bundle_rows = [row for row in rows if row["relative_path"].endswith("bundle.npz")]
    included = set(ag_units(split_name))
    disk_units = {str(Path(row["relative_path"]).parent).replace(os.sep, "/") for row in bundle_rows}
    split_hash = hashlib.sha256(C.split_path(split_name).read_bytes()).hexdigest()
    payload = {
        "legacy_root": str(legacy_root.resolve()),
        "split_name": split_name,
        "split_sha256": split_hash,
        "n_files": len(rows),
        "n_bundles": len(bundle_rows),
        "n_reports": sum(row["relative_path"].endswith("report.json") for row in rows),
        "included_units": sorted(included),
        "nonincluded_bundle_units": sorted(disk_units - included),
        "files": rows,
    }
    _write_json_atomic(output, payload)
    return payload


def _verify_file_manifest(root: Path, manifest: dict) -> None:
    failures = []
    for row in manifest.get("files", []):
        path = root / row["relative_path"]
        if not path.is_file():
            failures.append(f"missing:{row['relative_path']}")
            continue
        if path.stat().st_size != int(row["size_bytes"]):
            failures.append(f"size:{row['relative_path']}")
            continue
        if _sha256(path) != row["sha256"]:
            failures.append(f"sha256:{row['relative_path']}")
    if failures:
        raise RuntimeError("legacy snapshot manifest verification failed: " + ",".join(failures[:10]))


def _case_file_hashes(case_dir: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for name in ("bundle.npz", "report.json"):
        path = case_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        rows[name] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    return rows


def _validate_promoted_ag(
    active_ag: Path, snapshot_ag: Path, staging_ag: Path,
    eligible_ag: list[str], legacy_manifest: dict,
) -> None:
    active_units = {
        f"AG/{path.parent.relative_to(active_ag)}".replace(os.sep, "/")
        for path in active_ag.glob("*/*/bundle.npz")
    }
    if active_units != set(eligible_ag):
        raise RuntimeError("promotion 后活动 AG 与最终白名单不一致")
    if len(list(active_ag.glob("*/*/report.json"))) != len(eligible_ag):
        raise RuntimeError("promotion 后活动 AG report 数量不一致")
    for unit in eligible_ag:
        row = check_v4_bundle(unit, active_ag.parent, enforce_ag_physical_limits=True)
        if not row["qa_pass"]:
            raise RuntimeError(f"promotion 后 hard QA 失败: {unit}: {row['reason']}")
    if (active_ag / "slow" / "WANG_DENG_FENG").exists():
        raise RuntimeError("WANG_DENG_FENG 不应存在于活动 AG")
    if not (active_ag / "fast" / "LI_ZHEN_SHAN" / "bundle.npz").is_file():
        raise RuntimeError("LI_ZHEN_SHAN 缺失于活动 AG")
    _verify_file_manifest(snapshot_ag.parent, legacy_manifest)
    if len(list(snapshot_ag.glob("*/*/bundle.npz"))) != 84:
        raise RuntimeError("旧 AG 快照不是84例")
    if len(list(staging_ag.glob("*/*/bundle.npz"))) != 77:
        raise RuntimeError("原始 AG staging 未保留77例")


def promote(
    staging_root: Path,
    active_root: Path,
    snapshot_root: Path,
    report_dir: Path,
    *,
    confirm_reviewed: bool,
) -> dict:
    """用户明确看图签核后才允许调用的同文件系统原子切换。"""
    if not confirm_reviewed:
        raise RuntimeError("缺少 --confirm-reviewed；禁止在用户可视化签核前切换活动 AG")
    qa_path = report_dir / "v4_manual_review_summary.json"
    whitelist_path = report_dir / "v4_final_whitelist.json"
    manifest_path = report_dir / "AG_legacy_v3_snapshot_manifest.json"
    if not qa_path.is_file() or not whitelist_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("缺少人工终签、最终白名单或旧版快照清单")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    whitelist = json.loads(whitelist_path.read_text(encoding="utf-8"))
    legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not qa.get("visual_review_signed_off") or not qa.get("ready_for_promotion"):
        raise RuntimeError("人工终签或最终 QA 未通过，禁止 promotion")
    if int(legacy_manifest.get("n_bundles", -1)) != 84:
        raise RuntimeError("旧版快照清单不是预期84个 bundle")

    eligible_ag = list(whitelist.get("AG", []))
    expected_active = int(whitelist.get("counts", {}).get("AG", -1))
    if expected_active != len(eligible_ag) or expected_active <= 0:
        raise RuntimeError("最终 AG 白名单数量不一致")
    if any(not unit.startswith("AG/") for unit in eligible_ag):
        raise RuntimeError("最终 AG 白名单含非 AG 病例")

    staging_ag = staging_root / "AG"
    active_ag = active_root / "AG"
    snapshot_ag = snapshot_root / "AG"
    if not staging_ag.is_dir() or not active_ag.is_dir():
        raise FileNotFoundError("staging AG 或活动 AG 目录不存在")
    if snapshot_ag.exists():
        raise FileExistsError(f"快照目标已存在: {snapshot_ag}")
    if staging_ag.stat().st_dev != active_ag.stat().st_dev:
        raise RuntimeError("staging 与活动 AG 不在同一文件系统，无法保证原子切换")
    staged_bundles = list(staging_ag.glob("*/*/bundle.npz"))
    active_bundles = list(active_ag.glob("*/*/bundle.npz"))
    if len(staged_bundles) != 77 or len(active_bundles) != 84:
        raise RuntimeError(
            f"目录数量不满足切换合同: staging={len(staged_bundles)} active={len(active_bundles)}"
        )
    _verify_file_manifest(active_root, legacy_manifest)

    # 保留原始77例 staging 不动；按终签白名单用硬链接物化一个
    # 同文件系统的候选 AG 目录，再用目录级 os.replace 原子提升。
    promotion_ag = staging_root / f".AG_promotion_{os.getpid()}.tmp"
    if promotion_ag.exists():
        raise FileExistsError(f"临时 promotion 目录已存在: {promotion_ag}")
    try:
        for unit in eligible_ag:
            rel = Path(unit).relative_to("AG")
            src_case = staging_ag / rel
            dst_case = promotion_ag / rel
            dst_case.mkdir(parents=True, exist_ok=False)
            for name in ("bundle.npz", "report.json"):
                src = src_case / name
                if not src.is_file():
                    raise FileNotFoundError(f"白名单病例缺文件: {src}")
                os.link(src, dst_case / name)
        materialized = {
            str(path.parent.relative_to(promotion_ag)).replace(os.sep, "/")
            for path in promotion_ag.glob("*/*/bundle.npz")
        }
        expected_rel = {str(Path(unit).relative_to("AG")) for unit in eligible_ag}
        if materialized != expected_rel:
            raise RuntimeError("物化的 AG promotion 目录与最终白名单不一致")
    except Exception:
        shutil.rmtree(promotion_ag, ignore_errors=True)
        raise

    snapshot_root.mkdir(parents=True, exist_ok=False)
    swapped = False
    try:
        os.replace(active_ag, snapshot_ag)
        os.replace(promotion_ag, active_ag)
        swapped = True
        _validate_promoted_ag(active_ag, snapshot_ag, staging_ag, eligible_ag, legacy_manifest)
        snapshot_manifest_copy = snapshot_root / "AG_legacy_v3_snapshot_manifest.json"
        _write_json_atomic(snapshot_manifest_copy, legacy_manifest)
        record = {
            "status": "promoted",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "frame_version": FRAME_VERSION,
            "staging_root": str(staging_root),
            "active_root": str(active_root),
            "snapshot_root": str(snapshot_root),
            "n_active_v4_bundles": expected_active,
            "n_active_v4_reports": expected_active,
            "n_legacy_snapshot_bundles": 84,
            "n_legacy_snapshot_reports": 84,
            "n_original_staging_bundles": 77,
            "final_ag_whitelist": eligible_ag,
            "excluded_from_staging": sorted(set(ag_units()) - set(eligible_ag)),
            "source_files": {
                "manual_review_summary": str(qa_path),
                "final_whitelist": str(whitelist_path),
                "legacy_manifest": str(manifest_path),
            },
            "source_sha256": {
                "manual_review_summary": _sha256(qa_path),
                "final_whitelist": _sha256(whitelist_path),
                "legacy_manifest": _sha256(manifest_path),
            },
            "user_visual_review_confirmed": True,
            "rollback": {
                "restore_from": str(snapshot_ag),
                "contract": "automatic before promotion_record commit; manual after commit",
            },
        }
        # 事务提交点：记录必须最后写入；此前任一步失败都会恢复旧活动目录。
        _write_json_atomic(report_dir / "promotion_record.json", record)
        return record
    except Exception:
        if swapped:
            failed_new = staging_root / f".AG_failed_promotion_{os.getpid()}.tmp"
            if failed_new.exists():
                shutil.rmtree(failed_new)
            if active_ag.exists():
                os.replace(active_ag, failed_new)
            if snapshot_ag.exists():
                os.replace(snapshot_ag, active_ag)
            shutil.rmtree(failed_new, ignore_errors=True)
            _verify_file_manifest(active_root, legacy_manifest)
        elif snapshot_ag.exists() and not active_ag.exists():
            os.replace(snapshot_ag, active_ag)
        shutil.rmtree(promotion_ag, ignore_errors=True)
        (report_dir / "promotion_record.json").unlink(missing_ok=True)
        (snapshot_root / "AG_legacy_v3_snapshot_manifest.json").unlink(missing_ok=True)
        if snapshot_root.is_dir() and not any(snapshot_root.iterdir()):
            snapshot_root.rmdir()
        raise


def promote_aaa_fix(
    fixes_root: Path, active_root: Path, snapshot_root: Path, record_path: Path,
    *, unit_id: str = "AAA/unruputer/HAN_JIAN_FU",
) -> dict:
    """原子替换一个 AAA 病例；后置验证或记录失败时恢复旧病例。"""
    src = fixes_root / unit_id
    active = active_root / unit_id
    snap = snapshot_root / unit_id
    if not src.is_dir() or not active.is_dir():
        raise FileNotFoundError(f"fix 或活动病例不存在: {src}, {active}")
    if snap.exists() or snapshot_root.exists():
        raise FileExistsError(f"AAA 病例快照目标已存在: {snapshot_root}")
    if src.stat().st_dev != active.stat().st_dev:
        raise RuntimeError("AAA fix 与活动病例不在同一文件系统")
    old_hashes = _case_file_hashes(active)
    fix_hashes = _case_file_hashes(src)
    candidate = active.parent / f".{active.name}_fix_{os.getpid()}.tmp"
    candidate.mkdir(parents=False, exist_ok=False)
    try:
        for name in ("bundle.npz", "report.json"):
            os.link(src / name, candidate / name)
        snapshot_root.mkdir(parents=True, exist_ok=False)
        snap.parent.mkdir(parents=True, exist_ok=True)
        os.replace(active, snap)
        swapped = False
        try:
            os.replace(candidate, active)
            swapped = True
            if _case_file_hashes(active) != fix_hashes:
                raise RuntimeError("活动 HAN_JIAN_FU 与 fixes staging 字节不一致")
            if _case_file_hashes(snap) != old_hashes:
                raise RuntimeError("HAN_JIAN_FU 病例快照哈希不一致")
            row = check_v4_bundle(unit_id, active_root, enforce_ag_physical_limits=False)
            if not row["qa_pass"]:
                raise RuntimeError(f"HAN_JIAN_FU hard QA 失败: {row['reason']}")
            report = json.loads((active / "report.json").read_text(encoding="utf-8"))
            remap_steps = [int(x) for x in report.get("wall_coord_spatial_remap_steps", [])]
            if int(report.get("wall_coord_spatial_remap_n_steps", 0)) != 1 or remap_steps != [1120]:
                raise RuntimeError("HAN_JIAN_FU 缺少唯一 step1120 COORD-REMAP 记录")
            if "COORD-REMAP" not in soft_flags(report):
                raise RuntimeError("HAN_JIAN_FU COORD-REMAP 软标记缺失")
            with np.load(active / "bundle.npz", allow_pickle=False) as z:
                n_steps = len(z["steps"])
                n_nodes = len(z["wall_nodenumber"])
                for key in ("wall_wss", "wall_pressure"):
                    if key in z.files and z[key].shape != (n_steps, n_nodes):
                        raise RuntimeError(f"HAN_JIAN_FU {key} shape 不对齐")
                for key in ("wall_coords_raw", "wall_coords_norm"):
                    if z[key].shape != (n_nodes, 3):
                        raise RuntimeError(f"HAN_JIAN_FU {key} shape 不对齐")
            record = {
                "status": "promoted", "unit_id": unit_id,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                "active_case": str(active), "fix_source": str(src),
                "snapshot_case": str(snap), "old_files": old_hashes,
                "fix_files": fix_hashes, "hard_qa_pass": True,
                "coord_remap": {"steps": [1120], "n_steps": 1},
                "array_alignment": {"n_steps": n_steps, "n_nodes": n_nodes, "shape_checks": "passed"},
                "rollback": {"restore_from": str(snap), "recorded_old_sha256": old_hashes},
            }
            _write_json_atomic(record_path, record)
            return record
        except Exception:
            failed = active.parent / f".{active.name}_failed_fix_{os.getpid()}.tmp"
            if swapped and active.exists():
                os.replace(active, failed)
            if snap.exists():
                os.replace(snap, active)
            shutil.rmtree(failed, ignore_errors=True)
            if _case_file_hashes(active) != old_hashes:
                raise RuntimeError("AAA fix 回滚后旧病例哈希复核失败")
            record_path.unlink(missing_ok=True)
            raise
    except Exception:
        shutil.rmtree(candidate, ignore_errors=True)
        if snapshot_root.is_dir():
            shutil.rmtree(snapshot_root, ignore_errors=True)
        raise


def preprocess_one(index: int, out_root: Path, split_name: str) -> dict:
    units = ag_units(split_name)
    if not 0 <= index < len(units):
        raise IndexError(f"index={index} 越界，AG正式病例数={len(units)}")
    unit = units[index]
    cohort, case_name = unit_parts(unit)
    reporting.setup_logging(f"ag_v4_{index:02d}_{case_name}")
    report = preprocess.preprocess_case(
        cohort, case_name, C.DEFAULT, out_root=out_root,
    )
    if report.get("frame_version") != FRAME_VERSION:
        raise RuntimeError(f"{unit} 未生成 {FRAME_VERSION}")
    if report.get("landmark_source") != LANDMARK_SOURCE:
        raise RuntimeError(f"{unit} 未使用原始 STL")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preprocess")
    p_pre.add_argument("--index", type=int, required=True)
    p_pre.add_argument("--out-root", type=Path, required=True)
    p_pre.add_argument("--split", default=C.DEFAULT_SPLIT_NAME)

    p_qa = sub.add_parser("qa")
    p_qa.add_argument("--ag-root", type=Path, required=True)
    p_qa.add_argument("--aaa-root", type=Path, default=C.OUT_ROOT)
    p_qa.add_argument("--aaa-fallback-root", type=Path, default=None)
    p_qa.add_argument("--report-dir", type=Path, required=True)
    p_qa.add_argument("--split", default=C.DEFAULT_SPLIT_NAME)
    p_qa.add_argument("--allow-incomplete", action="store_true")

    p_snap = sub.add_parser("snapshot-manifest")
    p_snap.add_argument(
        "--legacy-root", type=Path, default=C.OUT_ROOT,
        help="legacy data root or its AG directory (for example data_wss_min or data_wss_min/AG)",
    )
    p_snap.add_argument("--output", type=Path, required=True)
    p_snap.add_argument("--split", default=C.DEFAULT_SPLIT_NAME)

    p_review = sub.add_parser("finalize-review")
    p_review.add_argument("--report-dir", type=Path, required=True)
    p_review.add_argument("--decisions", type=Path, required=True)

    p_promote = sub.add_parser("promote")
    p_promote.add_argument("--staging-root", type=Path, required=True)
    p_promote.add_argument("--active-root", type=Path, default=C.OUT_ROOT)
    p_promote.add_argument("--snapshot-root", type=Path, required=True)
    p_promote.add_argument("--report-dir", type=Path, required=True)
    p_promote.add_argument("--confirm-reviewed", action="store_true")

    p_aaa = sub.add_parser("promote-aaa-fix")
    p_aaa.add_argument("--fixes-root", type=Path, required=True)
    p_aaa.add_argument("--active-root", type=Path, default=C.OUT_ROOT)
    p_aaa.add_argument("--snapshot-root", type=Path, required=True)
    p_aaa.add_argument("--record", type=Path, required=True)
    p_aaa.add_argument("--unit-id", default="AAA/unruputer/HAN_JIAN_FU")

    args = ap.parse_args()
    if args.command == "preprocess":
        result = preprocess_one(args.index, args.out_root.resolve(), args.split)
    elif args.command == "qa":
        result = run_qa(
            args.ag_root.resolve(), args.aaa_root.resolve(), args.report_dir.resolve(),
            args.split, strict=not args.allow_incomplete,
            aaa_fallback_root=(args.aaa_fallback_root.resolve() if args.aaa_fallback_root else None),
        )["summary"]
    elif args.command == "snapshot-manifest":
        result = snapshot_manifest(args.legacy_root.resolve(), args.output.resolve(), args.split)
    elif args.command == "finalize-review":
        result = finalize_review(
            args.report_dir.resolve(), args.decisions.resolve(),
        )["summary"]
    elif args.command == "promote":
        result = promote(
            args.staging_root.resolve(), args.active_root.resolve(),
            args.snapshot_root.resolve(), args.report_dir.resolve(),
            confirm_reviewed=args.confirm_reviewed,
        )
    else:
        result = promote_aaa_fix(
            args.fixes_root.resolve(), args.active_root.resolve(),
            args.snapshot_root.resolve(), args.record.resolve(), unit_id=args.unit_id,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
