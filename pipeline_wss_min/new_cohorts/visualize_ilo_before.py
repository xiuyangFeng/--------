#!/usr/bin/env python3
"""生成 AG76 / AAA63 / ILO-before 人工复审活动集的共同框架可视化包。"""

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

from pipeline_wss_min import config as C
from pipeline_wss_min.new_cohorts.ilo_before import (
    BASELINE_REPORT_DIR,
    DEFAULT_REPORT_DIR,
    REFERENCE_FINAL_WHITELIST,
    REFERENCE_QUALITY_AUDIT,
    load_ilo_before_whitelist,
)
from pipeline_wss_min.visualize_v4_cutover import (
    _bundle_item,
    _flagged_page,
    _limits,
    _needs_red,
    _stratified_random_groups,
)


DEFAULT_OUT = (
    C.PROJECT_ROOT / "docs" / "02-推进与变更" / "assets_新队列审计"
    / "ilo_before_final_approved_20260717"
)
PALETTE = {"AG": "#666666", "AAA": "#C44E52", "ILO-before": "#3B6FB6"}
WSS_DISPLAY_MAX_PA = 300.0


def _scalar(z: np.lib.npyio.NpzFile, key: str):
    value = np.asarray(z[key])
    return value.item() if value.ndim == 0 else value


def _ilo_label(unit_id: str) -> str:
    return unit_id.split("/")[1][:18]


def _ilo_small_multiples(
    items: list[dict], title: str, path: Path, xy_lim: float, z_lim: float, *, cols: int = 7,
) -> None:
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.55), squeeze=False)
    for ax, item in zip(axes.ravel(), items):
        cloud = item["mm"]
        ax.scatter(
            cloud[:, 0], cloud[:, 2], c=cloud[:, 2], cmap="viridis",
            vmin=-z_lim, vmax=z_lim, s=0.7, linewidths=0,
        )
        landmarks = item.get("landmarks_mm")
        if landmarks is not None:
            ax.scatter(landmarks[0, 0], landmarks[0, 2], marker="^", s=20, c="#C44E52")
            ax.scatter(landmarks[1, 0], landmarks[1, 2], marker="o", s=14, c="#E39C37")
            ax.scatter(landmarks[2, 0], landmarks[2, 2], marker="o", s=14, c="#3B6FB6")
        suffix = f" [{'|'.join(item['flags'])}]" if item["flags"] else ""
        ax.set_title(
            _ilo_label(item["unit_id"]) + suffix,
            fontsize=6.5,
            color="#C44E52" if _needs_red(item) else "#222222",
        )
        ax.axhline(0, color="0.75", lw=0.5)
        ax.axvline(0, color="0.75", lw=0.5)
        ax.set_xlim(-xy_lim, xy_lim)
        ax.set_ylim(-z_lim, z_lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _attach_peak_wss(items: list[dict]) -> None:
    """按几何相同的确定性索引附加 peak WSS；bundle 中仍保留原始 Pa。"""
    for item in items:
        bundle_path = C.OUT_ROOT / item["unit_id"] / "bundle.npz"
        with np.load(bundle_path, allow_pickle=False) as z:
            steps = np.asarray(z["steps"], dtype=np.int64)
            peak = int(_scalar(z, "peak_step"))
            peak_index = steps.tolist().index(peak)
            values = np.asarray(z["wall_wss"][peak_index], dtype=np.float64)
        sample_idx = np.linspace(0, len(values) - 1, len(item["mm"]), dtype=np.int64)
        item["peak_wss_pa"] = values[sample_idx]


def _ilo_peak_wss_multiples(
    items: list[dict], path: Path, xy_lim: float, z_lim: float, *, cols: int = 7,
) -> None:
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.55), squeeze=False)
    vmax = float(np.log1p(WSS_DISPLAY_MAX_PA))
    artist = None
    for ax, item in zip(axes.ravel(), items):
        cloud = item["mm"]
        display = np.log1p(np.clip(item["peak_wss_pa"], 0.0, WSS_DISPLAY_MAX_PA))
        artist = ax.scatter(
            cloud[:, 0], cloud[:, 2], c=display, cmap="magma", vmin=0.0, vmax=vmax,
            s=0.75, linewidths=0,
        )
        ax.set_title(_ilo_label(item["unit_id"]), fontsize=6.5)
        ax.axhline(0, color="0.72", lw=0.5)
        ax.axvline(0, color="0.72", lw=0.5)
        ax.set_xlim(-xy_lim, xy_lim)
        ax.set_ylim(-z_lim, z_lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    if artist is not None:
        ticks_pa = np.asarray([0, 1, 5, 10, 30, 100, 300], dtype=np.float64)
        color_axis = fig.add_axes([0.955, 0.18, 0.012, 0.64])
        cbar = fig.colorbar(artist, cax=color_axis, ticks=np.log1p(ticks_pa))
        cbar.ax.set_yticklabels([str(int(value)) for value in ticks_pa])
        cbar.set_label("peak WSS (Pa); log1p display; clipped at 300 Pa")
    fig.suptitle(
        f"ILO-before{len(items)} · peak-step WSS · fixed common frame and fixed display scale",
        fontsize=11, y=0.999,
    )
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.02, top=0.97, wspace=0.08, hspace=0.20)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _three_way_overlay(
    groups: list[tuple[str, list[dict]]], path: Path, xy_lim: float, z_lim: float,
    *, normalized: bool,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    planes = ((0, 2, "X", "Z"), (1, 2, "Y", "Z"), (0, 1, "X", "Y"))
    key = "norm" if normalized else "mm"
    for ax, (a, b, la, lb) in zip(axes, planes):
        for label, items in groups:
            cloud = np.vstack([item[key] for item in items])
            ax.scatter(
                cloud[:, a], cloud[:, b], s=0.28, alpha=0.045,
                color=PALETTE[label], linewidths=0, label=f"{label} n={len(items)}",
            )
        ax.axhline(0, color="#222222", lw=0.65, ls="--")
        ax.axvline(0, color="#222222", lw=0.65, ls="--")
        if normalized:
            ax.set_xlim(-1.02, 1.02)
            ax.set_ylim(-1.02, 1.02)
            unit = "normalized"
        else:
            ax.set_xlim(-(z_lim if a == 2 else xy_lim), z_lim if a == 2 else xy_lim)
            ax.set_ylim(-(z_lim if b == 2 else xy_lim), z_lim if b == 2 else xy_lim)
            unit = "mm"
        ax.set_aspect("equal")
        ax.set_xlabel(f"{la} ({unit})")
        ax.set_ylabel(f"{lb} ({unit})")
    axes[0].legend(markerscale=12, fontsize=8, frameon=True)
    scope = "bundle normalized coordinates" if normalized else "common physical frame (mm)"
    fig.suptitle(
        f"AG{len(groups[0][1])} + AAA{len(groups[1][1])} + "
        f"ILO-before{len(groups[2][1])} · {scope} · +Z proximal / -Z iliac",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _ilo_group_overlay(
    items: list[dict], title: str, path: Path, xy_lim: float, z_lim: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.5))
    planes = ((0, 2, "X", "Z"), (1, 2, "Y", "Z"), (0, 1, "X", "Y"))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, len(items))))
    handles: list[Line2D] = []
    labels: list[str] = []
    for item, color in zip(items, colors):
        cloud = item["mm"]
        for ax, (a, b, _la, _lb) in zip(axes, planes):
            ax.scatter(cloud[:, a], cloud[:, b], s=0.55, alpha=0.18, color=color, linewidths=0)
        suffix = f" [{'|'.join(item['flags'])}]" if item["flags"] else ""
        labels.append(_ilo_label(item["unit_id"]) + suffix)
        handles.append(Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5,
            markerfacecolor=color, markeredgecolor="none",
        ))
    for ax, (a, b, la, lb) in zip(axes, planes):
        ax.axhline(0, color="#222222", lw=0.6, ls="--")
        ax.axvline(0, color="#222222", lw=0.6, ls="--")
        ax.set_xlim(-(z_lim if a == 2 else xy_lim), z_lim if a == 2 else xy_lim)
        ax.set_ylim(-(z_lim if b == 2 else xy_lim), z_lim if b == 2 else xy_lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{la} (mm)")
        ax.set_ylabel(f"{lb} (mm)")
    legend = fig.legend(
        handles, labels, loc="center left", bbox_to_anchor=(0.82, 0.5),
        fontsize=7.5, frameon=True, title="red label = soft flag",
    )
    for text, item in zip(legend.get_texts(), items):
        text.set_color("#C44E52" if item["flags"] else "#222222")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.815, 0.95])
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _grouped_review(
    items: list[dict], out_dir: Path, xy_lim: float, z_lim: float,
) -> list[dict]:
    grouped_dir = out_dir / "grouped20_ILO_before"
    grouped_dir.mkdir(parents=True, exist_ok=True)
    groups = _stratified_random_groups(items, seed=20260717, group_size=20)
    rows: list[dict] = []
    cursor = 0
    for group_index, group in enumerate(groups, 1):
        start, end = cursor + 1, cursor + len(group)
        stem = f"ILO_before_group{group_index:02d}_rank{start:03d}-{end:03d}"
        front = grouped_dir / f"{stem}_front_common_mm_xz.png"
        overlay = grouped_dir / f"{stem}_common_mm_ortho_overlay.png"
        flagged = sum(bool(item["flags"]) for item in group)
        scope = (
            f"ILO-before v4 · group {group_index}/{len(groups)} · rank {start}-{end} · "
            f"soft flags {flagged}/{len(group)}"
        )
        _ilo_small_multiples(group, scope + " · front X-Z", front, xy_lim, z_lim, cols=5)
        _ilo_group_overlay(group, scope + " · common physical frame", overlay, xy_lim, z_lim)
        for position, item in enumerate(group, 1):
            rows.append({
                "unit_id": item["unit_id"],
                "group": group_index,
                "group_position": position,
                "display_order": cursor + position,
                "soft_flags": "|".join(item["flags"]),
                "review_status": "soft_flag_review" if item["flags"] else "final_approved",
                "front_figure": str(front),
                "overlay_figure": str(overlay),
            })
        cursor = end
    return rows


def _load_wss_rows(audit_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    reference = json.loads(REFERENCE_QUALITY_AUDIT.read_text(encoding="utf-8"))["rows"]
    ag = [row for row in reference if row["unit_id"].startswith("AG/") and row.get("quality_decision") == "include"]
    aaa = [row for row in reference if row["unit_id"].startswith("AAA/") and row.get("quality_decision") == "include"]
    ilo_payload = json.loads((audit_dir / "ilo_before_wss_geometry_audit.json").read_text(encoding="utf-8"))
    ilo = [row for row in ilo_payload["rows"] if row["qa_pass"]]
    return ag, aaa, ilo


def _wss_distribution(audit_dir: Path, path: Path) -> None:
    ag, aaa, ilo = _load_wss_rows(audit_dir)
    fields = (
        ("all_wss_p90", "All-time WSS p90 (Pa)"),
        ("peak_wss_p90", "Peak-step WSS p90 (Pa)"),
        ("peak_wss_p100", "Peak-step WSS maximum (Pa)"),
    )
    groups = (("AG", ag), ("AAA", aaa), ("ILO-before", ilo))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    rng = np.random.default_rng(20260717)
    for ax, (field, title) in zip(axes, fields):
        values = [[float(row[field]) for row in rows] for _label, rows in groups]
        boxes = ax.boxplot(
            values,
            tick_labels=[label for label, _rows in groups],
            patch_artist=True,
        )
        for patch, (label, _rows) in zip(boxes["boxes"], groups):
            patch.set_facecolor(PALETTE[label])
            patch.set_alpha(0.28)
            patch.set_edgecolor(PALETTE[label])
        for index, ((label, _rows), vals) in enumerate(zip(groups, values), 1):
            jitter = rng.normal(0, 0.045, size=len(vals))
            ax.scatter(
                index + jitter, vals, s=11, alpha=0.45, color=PALETTE[label],
                edgecolors="none",
            )
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Pa · log scale")
        ax.grid(axis="y", color="0.88", lw=0.6)
    fig.suptitle("WSS distributions by accepted cohort", fontsize=12)
    fig.text(
        0.5, 0.01,
        f"AG76 and AAA57 are current numerical references; ILO-before includes {len(ilo)} review-round-2 candidates.",
        ha="center", fontsize=8, color="0.35",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _wss_peak_scatter(audit_dir: Path, path: Path) -> None:
    ag, aaa, ilo = _load_wss_rows(audit_dir)
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    for label, rows in (("AG", ag), ("AAA", aaa), ("ILO-before", ilo)):
        x = [float(row["peak_wss_p90"]) for row in rows]
        y = [float(row["peak_wss_p100"]) for row in rows]
        ax.scatter(x, y, s=24, alpha=0.6, color=PALETTE[label], label=f"{label} n={len(rows)}")
    for row in sorted(ilo, key=lambda item: float(item["peak_wss_p100"]), reverse=True)[:5]:
        ax.annotate(
            row["unit_id"].split("/")[1],
            (float(row["peak_wss_p90"]), float(row["peak_wss_p100"])),
            xytext=(4, 4), textcoords="offset points", fontsize=6.5, color="#244A7C",
        )
    ax.set_xlabel("Peak-step WSS p90 (Pa)")
    ax.set_ylabel("Peak-step WSS maximum (Pa)")
    ax.set_yscale("log")
    ax.grid(color="0.88", lw=0.6)
    ax.legend(frameon=True)
    ax.set_title("Peak-step WSS bulk level versus maximum")
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def generate(audit_dir: Path, out_dir: Path) -> dict:
    audit_dir = audit_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    final = json.loads(REFERENCE_FINAL_WHITELIST.read_text(encoding="utf-8"))
    ag_ids = list(final["AG"])
    aaa_ids = list(final["AAA"])
    ilo_ids = load_ilo_before_whitelist(audit_dir / "ilo_before_whitelist.json")
    if (len(ag_ids), len(aaa_ids), len(ilo_ids)) != (76, 63, 41):
        raise RuntimeError("最终通过可视化作用域必须为 AG76 + AAA63 + ILO-before41")

    ag = [_bundle_item(C.OUT_ROOT, unit) for unit in ag_ids]
    aaa = [_bundle_item(C.OUT_ROOT, unit) for unit in aaa_ids]
    ilo = [_bundle_item(C.OUT_ROOT, unit) for unit in ilo_ids]
    ilo.sort(key=lambda item: item["unit_id"])
    _attach_peak_wss(ilo)
    qa_payload = json.loads(
        (audit_dir / "ilo_before_wss_geometry_audit.json").read_text(encoding="utf-8")
    )
    qa_rows = qa_payload["rows"]
    qa_by_unit = {row["unit_id"]: row for row in qa_rows}
    for item in ilo:
        if qa_by_unit[item["unit_id"]].get("wss_watch_flags") and "WSS" not in item["flags"]:
            item["flags"].append("WSS")
    xy_lim, z_lim = _limits(ag + aaa + ilo)

    _ilo_small_multiples(
        ilo,
        "ILO-before41 · final approved · STL landmark v4 · common physical frame (mm) · front X-Z",
        out_dir / "01_ILO_before41_common_mm_xz.png",
        xy_lim, z_lim,
    )
    groups = [("AG", ag), ("AAA", aaa), ("ILO-before", ilo)]
    _three_way_overlay(
        groups, out_dir / "02_AG76_AAA63_ILO41_common_mm_ortho_overlay.png",
        xy_lim, z_lim, normalized=False,
    )
    _three_way_overlay(
        groups, out_dir / "03_AG76_AAA63_ILO41_bundle_norm_ortho_overlay.png",
        xy_lim, z_lim, normalized=True,
    )
    _ilo_peak_wss_multiples(
        ilo, out_dir / "04_ILO_before41_peak_WSS_log1p_fixed.png", xy_lim, z_lim,
    )
    _wss_distribution(audit_dir, out_dir / "05_WSS_distribution_reference.png")
    _wss_peak_scatter(audit_dir, out_dir / "06_WSS_peak_scatter.png")

    pd.DataFrame(qa_rows).to_csv(out_dir / "07_frame_wss_metrics.csv", index=False)
    (out_dir / "07_frame_wss_metrics.json").write_text(
        json.dumps(qa_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    grouped_rows = _grouped_review(ilo, out_dir, xy_lim, z_lim)
    for row in grouped_rows:
        qa = qa_by_unit[row["unit_id"]]
        row["qa_pass"] = qa["qa_pass"]
        row["wss_watch_flags"] = qa.get("wss_watch_flags", "")
        row["peak_wss_p90"] = qa.get("peak_wss_p90")
        row["peak_wss_p100"] = qa.get("peak_wss_p100")
    pd.DataFrame(grouped_rows).to_csv(out_dir / "review_index.csv", index=False)

    flagged_dir = out_dir / "flagged_ILO_case_pages"
    flagged_dir.mkdir(exist_ok=True)
    flagged = [item for item in ilo if item["flags"] or qa_by_unit[item["unit_id"]].get("wss_watch_flags")]
    if flagged:
        raise RuntimeError("最终通过清单不应仍含软旗标或 WSS watch")
    for item in flagged:
        _flagged_page(item, flagged_dir / f"{item['unit_id'].replace('/', '__')}.png")

    exclusions = json.loads(
        (audit_dir / "ilo_before_exclusions.json").read_text(encoding="utf-8")
    )
    deletion_path = audit_dir / "ilo_after_derived_deletion_result.json"
    if not deletion_path.is_file():
        deletion_path = BASELINE_REPORT_DIR / "ilo_after_derived_deletion_result.json"
    deletion = json.loads(deletion_path.read_text(encoding="utf-8"))
    summary = {
        "scope": "final approved ILO-before visual record; no training split or model action",
        "counts": {
            "raw_ilo_before": 61,
            "ilo_before_candidates": len(ilo),
            "ilo_before_excluded": exclusions["counts"]["unique_excluded_before"],
            "ag_geometry_reference": len(ag),
            "aaa_geometry_reference": len(aaa),
            "active_ilo_after_bundles": deletion["remaining_after_bundle_report_files"],
            "flagged_ilo_pages": len(flagged),
        },
        "common_limits_mm": {"xy": xy_lim, "z": z_lim},
        "coordinate_convention": "+Z proximal aorta; -Z iliac branches; +X original STL world +X",
        "per_case_visual_rescaling": False,
        "wss_display": "raw Pa preserved; log1p display fixed to 0-300 Pa; clipping is visual only",
        "group_shuffle_seed": 20260717,
        "promotion_authorized": False,
        "manual_review_round": 3,
        "manual_review_status": "final_approved",
        "manual_geometry_review_approved": True,
        "training_split_created": False,
        "outputs": {
            "review_index": str(out_dir / "review_index.csv"),
            "metrics_csv": str(out_dir / "07_frame_wss_metrics.csv"),
        },
    }
    (out_dir / "visualization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(generate(args.audit_dir, args.out_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
