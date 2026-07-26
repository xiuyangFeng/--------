#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""直接读取 bundle 的 AG/AAA v4 cutover 可视化审核包。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from . import config as C
from . import surface_io
from .v4_cutover import (
    aaa_units, ag_units, bundle_path, check_v4_bundle, report_path, soft_flags,
)


POINTS_PER_CASE = 1400


def _scalar(z, key: str):
    value = np.asarray(z[key])
    return value.item() if value.ndim == 0 else value


def _sample(points: np.ndarray, n: int = POINTS_PER_CASE) -> np.ndarray:
    points = np.asarray(points)
    if len(points) <= n:
        return points
    idx = np.linspace(0, len(points) - 1, n, dtype=np.int64)
    return points[idx]


def _bundle_item(data_root: Path, unit_id: str, fallback_root: Path | None = None) -> dict:
    path = bundle_path(data_root, unit_id)
    if not path.is_file() and fallback_root is not None:
        path = bundle_path(fallback_root, unit_id)
    with np.load(path, allow_pickle=False) as z:
        norm_full = np.asarray(z["wall_coords_norm"], dtype=np.float64)
        scale = float(_scalar(z, "coord_scale"))
        raw_full = np.asarray(z["wall_coords_raw"], dtype=np.float64)
        centroid = np.asarray(z["transform_centroid"], dtype=np.float64)
        rotation = np.asarray(z["transform_rotation"], dtype=np.float64)
        norm = _sample(norm_full)
        mm = norm * scale
        item = {
            "unit_id": unit_id,
            "norm": norm,
            "mm": mm,
            "raw": _sample(raw_full),
            "scale": scale,
            "centroid": centroid,
            "rotation": rotation,
            "frame_version": str(_scalar(z, "transform_frame_version"))
                if "transform_frame_version" in z.files else "legacy_v3_unversioned",
            "stl_path": str(_scalar(z, "original_stl_path"))
                if "original_stl_path" in z.files else "",
            "stl_scale": float(_scalar(z, "original_stl_scale_to_mm"))
                if "original_stl_scale_to_mm" in z.files else np.nan,
        }
        lm_keys = (
            "transform_landmark_trunk_point", "transform_landmark_left_point",
            "transform_landmark_right_point",
        )
        if all(key in z.files for key in lm_keys):
            lm_world = np.vstack([z[key] for key in lm_keys]).astype(np.float64)
            item["landmarks_world"] = lm_world
            item["landmarks_mm"] = (lm_world - centroid) @ rotation
        else:
            item["landmarks_world"] = None
            item["landmarks_mm"] = None
    rpath = path.with_name("report.json")
    report = json.loads(rpath.read_text(encoding="utf-8")) if rpath.is_file() else {}
    item["flags"] = soft_flags(report)
    item["report"] = report
    return item


def _limits(items: list[dict]) -> tuple[float, float]:
    clouds = np.vstack([item["mm"] for item in items])
    xy = float(np.nanpercentile(np.abs(clouds[:, :2]), 99.7))
    z = float(np.nanpercentile(np.abs(clouds[:, 2]), 99.7))
    round25 = lambda value: max(25.0, 25.0 * math.ceil(value / 25.0))
    return round25(xy), round25(z)


def _label(unit_id: str) -> str:
    return unit_id.split("/")[-1][:17]


def _needs_red(item: dict) -> bool:
    status = item.get("final_status")
    if status:
        return status in {"excluded_by_user", "review_pending", "qa_failed"}
    return bool(item.get("flags"))


def _review_suffix(item: dict) -> str:
    flags = item.get("flags", [])
    tokens = list(flags)
    status = item.get("final_status")
    if status == "excluded_by_user":
        tokens.append("EXCLUDE")
    elif status == "approved_by_user" and flags:
        tokens.append("APPROVED")
    return f" [{'|'.join(tokens)}]" if tokens else ""


def _small_multiples(items: list[dict], title: str, path: Path,
                     xy_lim: float, z_lim: float, cols: int = 8) -> None:
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.05, rows * 2.35), squeeze=False)
    for ax, item in zip(axes.ravel(), items):
        cloud = item["mm"]
        ax.scatter(cloud[:, 0], cloud[:, 2], c=cloud[:, 2], cmap="viridis",
                   vmin=-z_lim, vmax=z_lim, s=0.65, linewidths=0)
        lm = item.get("landmarks_mm")
        if lm is not None:
            ax.scatter(lm[0, 0], lm[0, 2], marker="^", s=20, c="#d62728")
            ax.scatter(lm[1, 0], lm[1, 2], marker="o", s=14, c="#ff7f0e")
            ax.scatter(lm[2, 0], lm[2, 2], marker="o", s=14, c="#1f77b4")
        ax.set_title(_label(item["unit_id"]) + _review_suffix(item), fontsize=6.2,
                     color="#d62728" if _needs_red(item) else "black")
        ax.axhline(0, color="0.72", lw=0.5)
        ax.axvline(0, color="0.72", lw=0.5)
        ax.set_xlim(-xy_lim, xy_lim)
        ax.set_ylim(-z_lim, z_lim)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _ortho_group_overlay(items: list[dict], title: str, path: Path,
                         xy_lim: float, z_lim: float) -> None:
    """20 例以内的物理坐标三视图叠加，保留逐例颜色和姓名图例。"""
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.5))
    planes = ((0, 2, "X", "Z"), (1, 2, "Y", "Z"), (0, 1, "X", "Y"))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, len(items))))
    handles: list[Line2D] = []
    labels: list[str] = []
    review_pending: list[bool] = []
    for item, color in zip(items, colors):
        cloud = item["mm"]
        for ax, (a, b, _la, _lb) in zip(axes, planes):
            ax.scatter(cloud[:, a], cloud[:, b], s=0.55, alpha=0.18,
                       color=color, linewidths=0)
        category, case_name = item["unit_id"].split("/")[1:]
        suffix = _review_suffix(item)
        labels.append(f"{category}/{case_name}{suffix}")
        review_pending.append(_needs_red(item))
        handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                              markerfacecolor=color, markeredgecolor="none"))

    for ax, (a, b, la, lb) in zip(axes, planes):
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_xlim(-(z_lim if a == 2 else xy_lim), z_lim if a == 2 else xy_lim)
        ax.set_ylim(-(z_lim if b == 2 else xy_lim), z_lim if b == 2 else xy_lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{la} (mm)")
        ax.set_ylabel(f"{lb} (mm)")
    legend = fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.82, 0.5),
                        fontsize=7.5, frameon=True, title="red = excluded/pending")
    for text, pending in zip(legend.get_texts(), review_pending):
        text.set_color("#d62728" if pending else "black")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.815, 0.95])
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _stratified_random_groups(items: list[dict], seed: int,
                              group_size: int = 20) -> list[list[dict]]:
    """按组容量分配红/黑病例后固定种子乱序，避免待审病例聚在一张图。"""
    rng = np.random.default_rng(seed)
    pending = [item for item in items if _needs_red(item)]
    passed = [item for item in items if not _needs_red(item)]
    rng.shuffle(pending)
    rng.shuffle(passed)

    capacities = [
        min(group_size, len(items) - start)
        for start in range(0, len(items), group_size)
    ]
    expected = np.asarray(capacities, dtype=np.float64) * len(pending) / len(items)
    pending_quota = np.floor(expected).astype(int)
    remainder = len(pending) - int(pending_quota.sum())
    tie_break = rng.random(len(capacities))
    priority = sorted(
        range(len(capacities)),
        key=lambda i: (-(expected[i] - pending_quota[i]), tie_break[i]),
    )
    for index in priority:
        if remainder == 0:
            break
        if pending_quota[index] < capacities[index]:
            pending_quota[index] += 1
            remainder -= 1

    groups: list[list[dict]] = []
    pending_cursor = passed_cursor = 0
    for capacity, n_pending in zip(capacities, pending_quota):
        n_passed = capacity - int(n_pending)
        group = (
            pending[pending_cursor:pending_cursor + int(n_pending)]
            + passed[passed_cursor:passed_cursor + n_passed]
        )
        pending_cursor += int(n_pending)
        passed_cursor += n_passed
        rng.shuffle(group)
        groups.append(group)
    if pending_cursor != len(pending) or passed_cursor != len(passed):
        raise RuntimeError("grouped20 stratified random allocation lost cases")
    return groups


def _grouped_20_review_figures(items: list[dict], cohort: str, out_dir: Path,
                               xy_lim: float, z_lim: float) -> list[dict]:
    """待审/过审分层随机混排，每 20 例生成正面和三视图。"""
    cohort_dir = out_dir / f"grouped20_{cohort}"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    seed = 20260716 if cohort == "AAA" else 20261716
    groups = _stratified_random_groups(items, seed=seed)
    rows: list[dict] = []
    n_groups = len(groups)
    start = 0
    for group_index, group in enumerate(groups, start=1):
        end = start + len(group)
        n_pending = sum(_needs_red(item) for item in group)
        stem = f"{cohort}_group{group_index:02d}_rank{start + 1:03d}-{end:03d}"
        front_path = cohort_dir / f"{stem}_front_common_mm_xz.png"
        overlay_path = cohort_dir / f"{stem}_common_mm_ortho_overlay.png"
        scope = (
            f"{cohort} v4 · group {group_index}/{n_groups} · rank {start + 1}-{end} · "
            f"red status {n_pending}/{len(group)} · red=excluded/pending, black=accepted"
        )
        _small_multiples(group, scope + " · front X-Z", front_path,
                         xy_lim, z_lim, cols=5)
        _ortho_group_overlay(group, scope + " · common physical frame",
                             overlay_path, xy_lim, z_lim)
        for position, item in enumerate(group, start=1):
            rows.append({
                "cohort": cohort,
                "group": group_index,
                "group_position": position,
                "display_order": start + position,
                "shuffle_seed": seed,
                "unit_id": item["unit_id"],
                "review_status": item.get(
                    "final_status", "review_pending" if item["flags"] else "passed"
                ),
                "soft_flags": "|".join(item["flags"]),
                "front_figure": str(front_path),
                "overlay_figure": str(overlay_path),
            })
        start = end
    return rows


def _matched_before_after(legacy_root: Path, ag_root: Path, units: list[str], path: Path) -> None:
    must = [
        "AG/fast/CHEN_SHI_MING", "AG/slow/WANG_DENG_FENG",
        "AG/slow/GONG_HUI_XIA", "AG/fast/ZHANG_LIANG",
    ]
    selected = []
    for unit in must + units[::max(1, len(units) // 10)]:
        if unit not in units:
            continue
        if unit not in selected and bundle_path(legacy_root, unit).is_file():
            selected.append(unit)
        if len(selected) == 12:
            break
    fig, axes = plt.subplots(len(selected), 2, figsize=(7.6, len(selected) * 2.25), squeeze=False)
    for row, unit in enumerate(selected):
        old = _bundle_item(legacy_root, unit)
        new = _bundle_item(ag_root, unit)
        lim = max(np.percentile(np.abs(old["mm"][:, [0, 2]]), 99.5),
                  np.percentile(np.abs(new["mm"][:, [0, 2]]), 99.5))
        lim = max(25.0, 25.0 * math.ceil(float(lim) / 25.0))
        for col, (item, subtitle) in enumerate(((old, "legacy bundle"), (new, "v4 staging bundle"))):
            ax = axes[row, col]
            cloud = item["mm"]
            ax.scatter(cloud[:, 0], cloud[:, 2], c=cloud[:, 2], cmap="viridis",
                       vmin=-lim, vmax=lim, s=0.8, linewidths=0)
            ax.axhline(0, color="0.72", lw=0.5); ax.axvline(0, color="0.72", lw=0.5)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{_label(unit)} · {subtitle}", fontsize=8)
    fig.suptitle("AG matched cases · legacy vs v4 staged bundle · front X-Z", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _overlay(ag: list[dict], aaa: list[dict], path: Path,
             xy_lim: float, z_lim: float, normalized: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    planes = ((0, 2, "X", "Z"), (1, 2, "Y", "Z"), (0, 1, "X", "Y"))
    key = "norm" if normalized else "mm"
    ag_cloud = np.vstack([item[key] for item in ag])
    aaa_cloud = np.vstack([item[key] for item in aaa])
    for ax, (a, b, la, lb) in zip(axes, planes):
        ax.scatter(ag_cloud[:, a], ag_cloud[:, b], s=0.3, alpha=0.055,
                   color="0.35", linewidths=0, label="AG v4")
        ax.scatter(aaa_cloud[:, a], aaa_cloud[:, b], s=0.3, alpha=0.055,
                   color="#d62728", linewidths=0, label="AAA v4")
        ax.axhline(0, color="k", lw=0.6, ls="--"); ax.axvline(0, color="k", lw=0.6, ls="--")
        if normalized:
            ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02)
            unit = "normalized"
        else:
            ax.set_xlim(-(z_lim if a == 2 else xy_lim), z_lim if a == 2 else xy_lim)
            ax.set_ylim(-(z_lim if b == 2 else xy_lim), z_lim if b == 2 else xy_lim)
            unit = "mm"
        ax.set_aspect("equal")
        ax.set_xlabel(f"{la} ({unit})"); ax.set_ylabel(f"{lb} ({unit})")
    axes[0].legend(markerscale=12, fontsize=9)
    scope = "bundle normalized coordinates" if normalized else "common physical frame (mm)"
    fig.suptitle(f"AG + AAA v4 · {scope} · +Z proximal / -Z iliac / +X patient-left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _flagged_page(item: dict, path: Path) -> None:
    stl_path = Path(item["stl_path"])
    stl = surface_io.read_stl_points(stl_path) * item["stl_scale"]
    stl = _sample(stl, 5000)
    wall_raw = item["raw"]
    repair_display = bool(item.get("report", {}).get("centerline_translation_applied"))
    if repair_display:
        # 与 registration v4 的 translation-only STL/centerline 修复保持一致；这里只
        # 重建有效 STL 的平移供审查显示，不改 bundle、原始 STL 或旋转/尺度。
        wall_center = 0.5 * (np.min(wall_raw, axis=0) + np.max(wall_raw, axis=0))
        stl_center = 0.5 * (np.min(stl, axis=0) + np.max(stl, axis=0))
        stl = stl + (wall_center - stl_center)
    centroid, rotation = item["centroid"], item["rotation"]
    stl_aln = (stl - centroid) @ rotation
    lm = item["landmarks_mm"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.4))
    axes[0].scatter(stl[:, 0], stl[:, 2], s=0.35, c="#1f77b4", alpha=0.35, label="selected STL")
    axes[0].scatter(wall_raw[:, 0], wall_raw[:, 2], s=0.5, c="#d62728", alpha=0.35, label="CFD wall")
    raw_title = (
        "effective world X-Z\ntranslated STL vs CFD wall"
        if repair_display else "raw world X-Z\nSTL vs CFD wall"
    )
    axes[0].set_title(raw_title); axes[0].legend(fontsize=7)
    for ax, (a, b, title) in zip(axes[1:4], ((0, 2, "aligned X-Z front"),
                                              (1, 2, "aligned Y-Z side"),
                                              (0, 1, "aligned X-Y top"))):
        ax.scatter(stl_aln[:, a], stl_aln[:, b], s=0.35, c="#1f77b4", alpha=0.4)
        ax.scatter(lm[:, a], lm[:, b], s=[28, 22, 22], c=["#d62728", "#ff7f0e", "#1f77b4"])
        ax.axhline(0, color="0.6", lw=0.6); ax.axvline(0, color="0.6", lw=0.6)
        ax.set_title(title)
    norm = item["norm"]
    axes[4].scatter(norm[:, 0], norm[:, 2], s=0.65, c=norm[:, 2], cmap="viridis", linewidths=0)
    axes[4].axhline(0, color="0.6", lw=0.6); axes[4].axvline(0, color="0.6", lw=0.6)
    axes[4].set_xlim(-1.02, 1.02); axes[4].set_ylim(-1.02, 1.02)
    axes[4].set_title("bundle normalized X-Z")
    for ax in axes:
        ax.set_aspect("equal"); ax.tick_params(labelsize=7)
    fig.suptitle(
        f"{item['unit_id']} · flags={'|'.join(item['flags'])} · scale={item['scale']:.2f} mm\n"
        f"STL={stl_path.name}", fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate(
    ag_root: Path,
    aaa_root: Path,
    legacy_root: Path,
    out_dir: Path,
    aaa_fallback_root: Path | None = None,
    review_manifest: Path | None = None,
    final_eligible_only: bool = False,
    training_whitelist: Path | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ag_ids, aaa_ids = ag_units(), aaa_units()
    final_status = {}
    if review_manifest is not None:
        review_payload = json.loads(review_manifest.read_text(encoding="utf-8"))
        final_status = {row["unit_id"]: row for row in review_payload.get("rows", [])}
        missing_status = (set(ag_ids) | set(aaa_ids)) - set(final_status)
        if missing_status:
            raise RuntimeError(f"review manifest 缺病例: {sorted(missing_status)}")
    if final_eligible_only:
        if review_manifest is None:
            raise ValueError("--final-eligible-only 必须同时提供 --review-manifest")
        ag_ids = [unit for unit in ag_ids if final_status[unit].get("final_eligible")]
        aaa_ids = [unit for unit in aaa_ids if final_status[unit].get("final_eligible")]
    if training_whitelist is not None:
        training_payload = json.loads(training_whitelist.read_text(encoding="utf-8"))
        training_ag = set(training_payload.get("AG", []))
        training_aaa = set(training_payload.get("AAA", []))
        selected = training_ag | training_aaa
        available = set(ag_ids) | set(aaa_ids)
        unexpected = selected - available
        if unexpected:
            raise RuntimeError(
                "training whitelist 含不在当前候选范围内的病例: "
                f"{sorted(unexpected)}"
            )
        ag_ids = [unit for unit in ag_ids if unit in training_ag]
        aaa_ids = [unit for unit in aaa_ids if unit in training_aaa]

    ag = [_bundle_item(ag_root, unit) for unit in ag_ids]
    aaa = [_bundle_item(aaa_root, unit, aaa_fallback_root) for unit in aaa_ids]
    if final_status:
        for item in ag + aaa:
            row = final_status[item["unit_id"]]
            item["final_status"] = row.get("final_status")
            item["manual_signoff_status"] = row.get("manual_signoff_status")
    aaa_r = [item for item in aaa if "/ruputer/" in item["unit_id"]]
    aaa_u = [item for item in aaa if "/unruputer/" in item["unit_id"]]
    xy_lim, z_lim = _limits(ag + aaa)

    if training_whitelist is not None:
        eligibility_scope = "training eligible only · "
    elif final_eligible_only:
        eligibility_scope = "final geometry eligible only · "
    else:
        eligibility_scope = ""
    _matched_before_after(legacy_root, ag_root, ag_ids, out_dir / "01_AG_v3_vs_v4_front_matched.png")
    _small_multiples(ag, f"AG v4 {eligibility_scope}n={len(ag)} · common physical frame (mm) · front X-Z",
                     out_dir / "02_AG_v4_all_common_mm_xz.png", xy_lim, z_lim)
    _small_multiples(aaa_r, f"AAA rupture v4 {eligibility_scope}n={len(aaa_r)} · common physical frame (mm) · front X-Z",
                     out_dir / "03_AAA_ruputer_v4_common_mm_xz.png", xy_lim, z_lim)
    _small_multiples(aaa_u, f"AAA unruptured v4 {eligibility_scope}n={len(aaa_u)} · common physical frame (mm) · front X-Z",
                     out_dir / "04_AAA_unruputer_v4_common_mm_xz.png", xy_lim, z_lim)
    _overlay(ag, aaa, out_dir / "05_AG_AAA_v4_common_mm_ortho_overlay.png",
             xy_lim, z_lim, normalized=False)
    _overlay(ag, aaa, out_dir / "06_AG_AAA_v4_bundle_norm_ortho_overlay.png",
             xy_lim, z_lim, normalized=True)

    grouped_rows = []
    grouped_rows.extend(_grouped_20_review_figures(
        aaa, "AAA", out_dir, xy_lim, z_lim,
    ))
    grouped_rows.extend(_grouped_20_review_figures(
        ag, "AG", out_dir, xy_lim, z_lim,
    ))
    pd.DataFrame(grouped_rows).to_csv(out_dir / "grouped20_review_index.csv", index=False)

    rows = [check_v4_bundle(unit, ag_root, enforce_ag_physical_limits=True) for unit in ag_ids]
    rows.extend(check_v4_bundle(
        unit, aaa_root, enforce_ag_physical_limits=False,
        fallback_root=aaa_fallback_root,
    ) for unit in aaa_ids)
    pd.DataFrame(rows).to_csv(out_dir / "07_frame_metrics.csv", index=False)
    (out_dir / "07_frame_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )

    flagged_aaa_dir = out_dir / "flagged_AAA_case_pages"
    flagged_aaa_dir.mkdir(exist_ok=True)
    flagged_aaa = [item for item in aaa if item["flags"]]
    for item in flagged_aaa:
        safe = item["unit_id"].replace("/", "__")
        _flagged_page(item, flagged_aaa_dir / f"{safe}.png")
    flagged_ag_dir = out_dir / "flagged_AG_case_pages"
    flagged_ag_dir.mkdir(exist_ok=True)
    flagged_ag = [item for item in ag if item["flags"]]
    for item in flagged_ag:
        safe = item["unit_id"].replace("/", "__")
        _flagged_page(item, flagged_ag_dir / f"{safe}.png")

    summary = {
        "ag_root": str(ag_root), "aaa_root": str(aaa_root),
        "aaa_fallback_root": str(aaa_fallback_root) if aaa_fallback_root else None,
        "legacy_root": str(legacy_root), "out_dir": str(out_dir),
        "n_ag": len(ag), "n_aaa": len(aaa),
        "final_eligible_only": final_eligible_only,
        "training_whitelist": str(training_whitelist) if training_whitelist else None,
        "n_aaa_ruputer": len(aaa_r), "n_aaa_unruputer": len(aaa_u),
        "n_flagged_aaa_pages": len(flagged_aaa),
        "flagged_aaa_units": [item["unit_id"] for item in flagged_aaa],
        "n_flagged_ag_pages": len(flagged_ag),
        "flagged_ag_units": [item["unit_id"] for item in flagged_ag],
        "grouped20_review_index": str(out_dir / "grouped20_review_index.csv"),
        "n_grouped20_aaa_figures": 2 * math.ceil(len(aaa) / 20),
        "n_grouped20_ag_figures": 2 * math.ceil(len(ag) / 20),
        "review_manifest": str(review_manifest) if review_manifest else None,
        "n_excluded_by_user": sum(
            item.get("final_status") == "excluded_by_user" for item in ag + aaa
        ),
        "excluded_units": [
            item["unit_id"] for item in ag + aaa
            if item.get("final_status") == "excluded_by_user"
        ],
        "grouped20_order": "stratified random mix of red-status and accepted cases",
        "grouped20_shuffle_seeds": {"AAA": 20260716, "AG": 20261716},
        "common_limits_mm": {"xy": xy_lim, "z": z_lim},
        "source": "bundle-only for all alignment plots; raw STL is read only for flagged case overlap panels",
        "promotion_authorized": False,
    }
    (out_dir / "visualization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag-root", type=Path, required=True)
    ap.add_argument("--aaa-root", type=Path, default=C.OUT_ROOT)
    ap.add_argument("--aaa-fallback-root", type=Path, default=None)
    ap.add_argument("--legacy-root", type=Path, default=C.OUT_ROOT)
    ap.add_argument("--review-manifest", type=Path, default=None)
    ap.add_argument("--final-eligible-only", action="store_true",
                    help="仅绘制最终 manifest 中 final_eligible=true 的病例")
    ap.add_argument("--training-whitelist", type=Path, default=None,
                    help="在当前候选范围内进一步仅绘制训练白名单病例")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    summary = generate(
        args.ag_root.resolve(), args.aaa_root.resolve(),
        args.legacy_root.resolve(), args.out_dir.resolve(),
        args.aaa_fallback_root.resolve() if args.aaa_fallback_root else None,
        args.review_manifest.resolve() if args.review_manifest else None,
        args.final_eligible_only,
        args.training_whitelist.resolve() if args.training_whitelist else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
