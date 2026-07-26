#!/usr/bin/env python3
"""Visualize every failed STL-to-wall surface-area mapping from an audit report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import surface as S


PROJECTIONS = ((0, 1, "XY"), (2, 0, "ZX"), (2, 1, "ZY"))


def _sample(points: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(points) <= limit:
        return np.arange(len(points))
    return np.random.default_rng(seed).choice(len(points), limit, replace=False)


def _equal_limits(ax, first: np.ndarray, second: np.ndarray, a: int, b: int) -> None:
    xy = np.concatenate((first[:, [a, b]], second[:, [a, b]]), axis=0)
    low, high = xy.min(axis=0), xy.max(axis=0)
    center = (low + high) / 2.0
    half = max(float((high - low).max()) * 0.54, 1e-6)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal", adjustable="box")


def _category(report: dict) -> str:
    if report.get("n_positive_wall_weights", 0) < 5000:
        return "coordinate / geometry mismatch"
    if report.get("wall_crop_applied"):
        return "full-STL tail vs cropped CFD wall"
    return "mapping tolerance failure"


def _case_figure(case: dict, report: dict, out_path: Path) -> dict:
    wall = np.asarray(case["wall_coords_raw"], dtype=np.float64)
    stl, _ = S.load_stl(case["original_stl_path"])
    stl = stl * float(case["original_stl_scale_to_mm"])
    translation = (wall.min(axis=0) + wall.max(axis=0)) / 2.0 - (
        stl.min(axis=0) + stl.max(axis=0)
    ) / 2.0
    centered_stl = stl + translation
    wall_idx = _sample(wall, 18000, S.stable_seed(case["unit_id"], "wall_plot"))
    stl_idx = _sample(stl, 18000, S.stable_seed(case["unit_id"], "stl_plot"))
    wall_plot, stl_plot = wall[wall_idx], stl[stl_idx]
    centered_plot = centered_stl[stl_idx]

    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)
    for column, (a, b, label) in enumerate(PROJECTIONS):
        axis = axes[0, column]
        axis.scatter(wall_plot[:, a], wall_plot[:, b], s=1.0, c="#1769aa", alpha=0.32,
                     linewidths=0, label="CFD wall")
        axis.scatter(stl_plot[:, a], stl_plot[:, b], s=1.0, c="#ff7f0e", alpha=0.32,
                     linewidths=0, label="scaled original STL")
        _equal_limits(axis, wall_plot, stl_plot, a, b)
        axis.set_title(f"Raw coordinates · {label}")
        axis.set_xlabel("xyz"[a] + " (mm)")
        axis.set_ylabel("xyz"[b] + " (mm)")
        if column == 0:
            axis.legend(loc="best", markerscale=5, fontsize=8)

        axis = axes[1, column]
        axis.scatter(wall_plot[:, a], wall_plot[:, b], s=1.0, c="#1769aa", alpha=0.32,
                     linewidths=0)
        axis.scatter(centered_plot[:, a], centered_plot[:, b], s=1.0, c="#ff7f0e",
                     alpha=0.32, linewidths=0)
        _equal_limits(axis, wall_plot, centered_plot, a, b)
        axis.set_title(f"BBox-center translation only · {label}")
        axis.set_xlabel("xyz"[a] + " (mm)")
        axis.set_ylabel("xyz"[b] + " (mm)")

    figure.suptitle(
        f"{case['unit_id']} · {_category(report)}\n"
        f"bidirectional p95/bbox={report['bidirectional_p95_over_bbox']:.3f}, "
        f"max/bbox={report['bidirectional_max_over_bbox']:.3f}, "
        f"positive wall weights={report['n_positive_wall_weights']:,}\n"
        "Bottom row is diagnostic only; no translation was used for area weights.",
        fontsize=12,
    )
    figure.savefig(out_path, dpi=180)
    plt.close(figure)
    return {
        "unit_id": case["unit_id"],
        "category": _category(report),
        "figure": str(out_path),
        "diagnostic_bbox_center_translation_mm": translation.tolist(),
        "wall_bbox_min_mm": wall.min(axis=0).tolist(),
        "wall_bbox_max_mm": wall.max(axis=0).tolist(),
        "stl_bbox_min_mm": stl.min(axis=0).tolist(),
        "stl_bbox_max_mm": stl.max(axis=0).tolist(),
    }


def _pca_projection(wall: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = wall.mean(axis=0)
    _, _, vh = np.linalg.svd(wall - center, full_matrices=False)
    basis = vh[:2].T
    return (wall - center) @ basis, (points - center) @ basis


def _summary(cases: list[dict], reports: dict[str, dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    scatter = None
    for axis, case in zip(axes.flat, cases):
        report = reports[case["unit_id"]]
        wall = np.asarray(case["wall_coords_raw"], dtype=np.float64)
        stl, _ = S.load_stl(case["original_stl_path"])
        stl = stl * float(case["original_stl_scale_to_mm"])
        distance = cKDTree(wall).query(stl, k=1)[0]
        relative = distance / max(float(report["bbox_diag_mm"]), 1e-12)
        wall_idx = _sample(wall, 8000, S.stable_seed(case["unit_id"], "summary_wall"))
        stl_idx = _sample(stl, 16000, S.stable_seed(case["unit_id"], "summary_stl"))
        wall_2d, stl_2d = _pca_projection(wall, stl)
        axis.scatter(wall_2d[wall_idx, 0], wall_2d[wall_idx, 1], s=1.0,
                     c="#5c6b73", alpha=0.18, linewidths=0)
        scatter = axis.scatter(
            stl_2d[stl_idx, 0], stl_2d[stl_idx, 1], s=1.2,
            c=np.clip(relative[stl_idx], 0.0, 0.10), cmap="turbo", vmin=0.0, vmax=0.10,
            alpha=0.70, linewidths=0,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            f"{case['unit_id']}\n{_category(report)} · p95={report['bidirectional_p95_over_bbox']:.3f}",
            fontsize=9,
        )
        axis.set_xticks([]); axis.set_yticks([])
    colorbar = figure.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.78, pad=0.01)
    colorbar.set_label("STL vertex nearest-wall distance / wall bbox diagonal (clipped at 0.10)")
    figure.suptitle(
        "Surface-area mapping failures · PCA projection in raw coordinates\n"
        "Gray: CFD wall · color: scaled original STL vertex mapping distance",
        fontsize=13,
    )
    figure.savefig(output, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    cfg = C.ExpConfig.from_json(args.config)
    stats = D.load_wss_stats(cfg.data.wss_stats_path)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    reports = {row["unit_id"]: row for row in audit["cases"] if row["status"] != "passed"}
    loaded = {}
    for partition in ("train", "test"):
        for case in D.load_partition(
            cfg.data.split_path, partition, stats, target=cfg.data.target,
            target_normalization=cfg.data.target_normalization,
            data_root=cfg.data.data_root,
            required_frame_version=cfg.data.required_frame_version,
        ):
            if case["unit_id"] in reports:
                loaded[case["unit_id"]] = case
    missing = sorted(set(reports) - set(loaded))
    if missing:
        raise RuntimeError(f"failed audit cases not loadable from split: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_records = []
    cases = [loaded[unit_id] for unit_id in audit["failed_cases"]]
    for case in cases:
        safe_name = case["unit_id"].replace("/", "__")
        case_records.append(_case_figure(
            case, reports[case["unit_id"]], args.output_dir / f"{safe_name}.png"
        ))
    summary_path = args.output_dir / "area_mapping_failures_summary.png"
    _summary(cases, reports, summary_path)

    rows = []
    for record in case_records:
        report = reports[record["unit_id"]]
        rows.append({
            "unit_id": record["unit_id"],
            "partition": report["partition"],
            "category": record["category"],
            "p95_over_bbox": report["bidirectional_p95_over_bbox"],
            "max_over_bbox": report["bidirectional_max_over_bbox"],
            "positive_wall_weights": report["n_positive_wall_weights"],
            "wall_crop_applied": report["wall_crop_applied"],
            "failures": ";".join(report["failures"]),
            "figure": record["figure"],
        })
    with (args.output_dir / "area_mapping_failures.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest = {
        "source_audit": str(args.audit),
        "source_config": str(args.config),
        "summary_figure": str(summary_path),
        "important": "BBox-center translation is visualization-only and was not used for mapping.",
        "cases": [{**record, "audit": reports[record["unit_id"]]} for record in case_records],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"n_cases": len(cases), "output_dir": str(args.output_dir),
                      "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
