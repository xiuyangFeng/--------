#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export an exact fixed-support PointNet++ SA evaluation trace for one case.

It supports the two directly comparable Q-series configurations: Q0
(FPS-2000 support) and Q2V (vertex-random5000 support).  Both use the same
fixed-count FPS centres (500 -> 125 -> 32), but their input support sampling
is intentionally kept distinct.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.nn import fps, radius

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import surface as S
from training_wss_min.tools.visualize_pointnetpp_sa import _write_center_mark_csv, _write_edges
from training_wss_min.tools.visualize_sampling import _bbox_diag, _load_stl, _write_vtp


DEFAULT_CONFIG = C.PROJECT_ROOT / (
    "training_wss_min/configs/pointnetpp_v4/"
    "ag_aaa_v4_stratified_sa3_random5000_fpscenter_sep.json"
)
DEFAULT_CASE = "fast/RAN_QING_BO"
DEFAULT_OUTPUT = C.PROJECT_ROOT / (
    "例子/06_PointNet++_SA三层采样与分组/"
    "Q2V-10477_vertex-random5000_FPS-center_fast_RAN_QING_BO_ParaView"
)


def _digest(*arrays: np.ndarray) -> str:
    hasher = hashlib.sha256()
    for array in arrays:
        hasher.update(np.ascontiguousarray(array).view(np.uint8))
    return hasher.hexdigest()


def _fixed_fps_and_group(pos: torch.Tensor, batch: torch.Tensor, *, count: int,
                         radius_value: float, nsample: int) -> tuple[torch.Tensor, ...]:
    """Exactly mirror PointNetSetAbstraction._fixed_indices for eval FPS."""
    if int(batch.max().item()) != 0:
        raise ValueError("this audit exporter accepts one case at a time")
    n_points = int(pos.size(0))
    actual_count = min(int(count), n_points)
    idx = fps(pos, batch, ratio=actual_count / n_points, random_start=False)[:actual_count]
    if idx.numel() != actual_count:
        raise RuntimeError(f"FPS produced {idx.numel()} centres, expected {actual_count}")
    pos_q, batch_q = pos[idx], batch[idx]
    assignment = radius(pos, pos_q, radius_value, batch, batch_q,
                        max_num_neighbors=int(nsample))
    return idx, pos_q, batch_q, assignment[0], assignment[1]


def _write_stage_files(stages: list[dict], output_dir: Path) -> None:
    for stage in stages:
        prefix = f"sa{stage['stage']}"
        _write_vtp(
            stage["source_xyz_mm"], output_dir / f"{prefix}_source_points.vtp",
            scalars={
                "full_wall_index": stage["source_lineage"],
                "group_membership_count": stage["membership_count"],
                "is_grouped": (stage["membership_count"] > 0).astype(np.int8),
            },
        )
        _write_vtp(
            stage["center_xyz_mm"], output_dir / f"{prefix}_centers.vtp",
            scalars={
                "center_id": np.arange(len(stage["center_xyz_mm"]), dtype=np.int64),
                "full_wall_index": stage["center_lineage"],
                "group_size": stage["group_size"],
            },
        )
        _write_edges(
            stage["center_xyz_mm"][stage["row"]],
            stage["source_xyz_mm"][stage["col"]],
            output_dir / f"{prefix}_group_edges.vtp",
            stage["row"], stage["source_lineage"][stage["col"]], stage["distance_norm"],
        )
        with (output_dir / f"{prefix}_assignments.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "stage", "center_id", "center_full_wall_index", "source_index",
                "source_full_wall_index", "distance_norm", "group_size",
            ])
            for center_id, source_id, distance in zip(
                stage["row"], stage["col"], stage["distance_norm"]
            ):
                writer.writerow([
                    stage["stage"], int(center_id),
                    int(stage["center_lineage"][center_id]), int(source_id),
                    int(stage["source_lineage"][source_id]), float(distance),
                    int(stage["group_size"][center_id]),
                ])


def _stage_summary(stage: dict) -> dict[str, int | float]:
    membership = stage["membership_count"]
    grouped = membership > 0
    return {
        "stage": int(stage["stage"]),
        "source_points": int(len(membership)),
        "centers": int(len(stage["center_xyz_mm"])),
        "ball_query_radius_norm": float(stage["radius_norm"]),
        "radius_mm_for_this_case": float(stage["radius_mm"]),
        "nsample_maximum": int(stage["nsample"]),
        "assignments": int(membership.sum()),
        "unique_group_points": int(grouped.sum()),
        "group_coverage_rate": float(grouped.mean()),
        "multi_group_points": int((membership > 1).sum()),
        "multi_group_rate_among_source": float((membership > 1).mean()),
        "multi_group_rate_among_grouped": float((membership[ grouped] > 1).mean()) if grouped.any() else 0.0,
        "mean_groups_per_source_point": float(membership.mean()),
        "max_groups_per_source_point": int(membership.max()),
        "center_in_own_group": int(np.sum(membership[stage["idx"]] > 0)),
        "trace_sha256": stage["trace_sha256"],
    }


def export_fixed_support_sa_trace(
    config_path: Path, case_label: str, output_dir: Path, partition: str = "auto",
    experiment: str = "Q2V-10477", support_n_points: int | None = None,
    support_sampling: str | None = None, radii_override: tuple[float, float, float] | None = None,
) -> Path:
    cfg = C.ExpConfig.from_json(config_path)
    if cfg.model.name != "pointnetpp":
        raise ValueError("SA trace requires model.name='pointnetpp'")
    selected_support_n = int(support_n_points or cfg.data.support_n_points)
    selected_support_sampling = support_sampling or cfg.data.support_sampling
    selected_radii = radii_override or tuple(float(value) for value in cfg.model.sa_radius)
    if selected_support_sampling not in {"random", "fps"}:
        raise ValueError("only random or FPS fixed support is supported")
    if selected_support_n <= 0 or len(selected_radii) != 3:
        raise ValueError("support count must be positive and exactly three SA radii are required")
    if tuple(cfg.model.sa_center_counts) != (500, 125, 32):
        raise ValueError("expected Q2V fixed FPS centre counts 500, 125, 32")

    stats = D.load_wss_stats(cfg.data.wss_stats_path)
    partitions = ("train", "test") if partition == "auto" else (partition,)
    matches: list[tuple[str, int, dict]] = []
    for partition_name in partitions:
        cases = D.load_partition(
            cfg.data.split_path, partition_name, stats, strict=True,
            target=cfg.data.target, target_normalization=cfg.data.target_normalization,
        )
        matches.extend(
            (partition_name, index, case) for index, case in enumerate(cases)
            if f"{case['cohort'].removeprefix('AG/')}/{case['case']}" == case_label
        )
    if len(matches) != 1:
        raise ValueError(f"expected one case matching {case_label!r}, found {len(matches)}")
    partition_name, _, case = matches[0]

    # This is the literal support-point path in evaluate.predict_case_norm().
    support_seed = S.stable_seed(cfg.eval.support_seed, case["unit_id"], "eval_support")
    support_idx = D.sample_indices(
        case, cfg.data, support_seed, n_points=selected_support_n,
        sampling=selected_support_sampling,
    )
    pos = torch.from_numpy(np.ascontiguousarray(case["pos"][support_idx]))
    batch = torch.zeros(len(pos), dtype=torch.long)
    lineage = np.asarray(support_idx, dtype=np.int64)
    scale = float(case["coord_scale_scalar"])

    stages: list[dict] = []
    for stage_number, (count, radius_value, nsample) in enumerate(zip(
        cfg.model.sa_center_counts, selected_radii, cfg.model.sa_nsample
    ), start=1):
        idx_t, pos_q_t, batch_q_t, row_t, col_t = _fixed_fps_and_group(
            pos, batch, count=int(count), radius_value=float(radius_value), nsample=int(nsample),
        )
        source_norm = pos.numpy()
        center_norm = pos_q_t.numpy()
        idx, row, col = (value.numpy().astype(np.int64) for value in (idx_t, row_t, col_t))
        membership = np.bincount(col, minlength=len(source_norm)).astype(np.int64)
        stage = {
            "stage": stage_number, "radius_norm": float(radius_value),
            "radius_mm": float(radius_value) * scale, "nsample": int(nsample),
            "source_xyz_norm": source_norm, "center_xyz_norm": center_norm,
            "source_xyz_mm": source_norm * scale, "center_xyz_mm": center_norm * scale,
            "idx": idx, "row": row, "col": col,
            "distance_norm": np.linalg.norm(source_norm[col] - center_norm[row], axis=1),
            "group_size": np.bincount(row, minlength=len(center_norm)).astype(np.int64),
            "membership_count": membership, "source_lineage": lineage.copy(),
            "center_lineage": lineage[idx], "trace_sha256": _digest(idx, row, col),
        }
        stages.append(stage)
        pos, batch, lineage = pos_q_t, batch_q_t, lineage[idx]

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = Path(case["bundle_path"])
    with np.load(bundle, allow_pickle=True) as bundle_data:
        wall_raw_xyz = bundle_data["wall_coords_raw"].astype(np.float64)
        centroid = bundle_data["transform_centroid"].astype(np.float64)
        rotation = bundle_data["transform_rotation"].astype(np.float64)
    stl_path = C.PROJECT_ROOT / "data_new" / case["cohort"] / case["case"] / f"{case['case']}.stl"
    stl_raw_xyz, triangles = _load_stl(stl_path)
    stl_scale = _bbox_diag(wall_raw_xyz) / max(_bbox_diag(stl_raw_xyz), 1e-12)
    _write_vtp((stl_raw_xyz * stl_scale - centroid) @ rotation,
               output_dir / "vessel_surface.vtp", triangles=triangles)
    _write_stage_files(stages, output_dir)
    center_mark_counts = _write_center_mark_csv(stages, output_dir)

    summaries = [_stage_summary(stage) for stage in stages]
    report_path = output_dir / f"{experiment}_group_overlap_summary.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    manifest = {
        "purpose": "Exact fixed-support PointNet++ SA evaluation trace; no training or model inference.",
        "experiment": experiment, "config": str(config_path), "case": case_label,
        "partition": partition_name, "source_bundle": str(bundle), "source_stl": str(stl_path),
        "input_sampling": f"{selected_support_sampling}-{len(support_idx)} fixed support",
        "support_seed": int(support_seed), "eval_support_seed_base": int(cfg.eval.support_seed),
        "sa_center_sampling": "evaluation deterministic FPS (random_start=False)",
        "sa_center_counts": [500, 125, 32],
        "counterfactual_overrides": {
            "support_n_points": support_n_points,
            "support_sampling": support_sampling,
            "sa_radii": list(radii_override) if radii_override else None,
        },
        "grouping_coordinate_system": "per-case normalized model coordinates",
        "render_coordinate_system": "registered rigid frame in millimetres",
        "stages": summaries, "paraview_center_mark_csv": {"file": "sa_centers_marked.csv", "counts": center_mark_counts},
    }
    (output_dir / "manifest_sa3.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "README_ParaView.md").write_text(
        f"# {experiment}：固定 support + FPS centres\n\n"
        f"- 病例：`{case_label}`（`{partition_name}`）；固定 `{selected_support_sampling}-{len(support_idx)}` 支持点。\n"
        "- SA centre：评估模式 FPS，固定 `500 → 125 → 32`；每层 ball-query 最多 16 个成员。\n"
        "- 重点图：`07_SA相邻group重叠示例.png`（灰=完整 random5000 壁面点，紫=两个相邻 group 的共同成员）。\n"
        f"- 数值：`{experiment}_group_overlap_summary.csv` 给出每层全部 group 的复用/覆盖率；"
        "`adjacent_group_overlap_report.csv` 给出图中所选相邻 pair 的 Jaccard 重叠率。\n"
        "- ParaView 可直接打开 `SA3_全部中心与分组点.vtm` 后点击 Apply、按 R；或打开 CSV 后用 Table To Points，X/Y/Z 分别选择 x_mm/y_mm/z_mm。\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export exact fixed-support PointNet++ SA trace")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--partition", choices=("auto", "train", "test"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment", default="Q2V-10477",
                        help="label stored in the trace manifest and README")
    parser.add_argument("--support-n-points", type=int,
                        help="counterfactual support-point count; does not train a model")
    parser.add_argument("--support-sampling", choices=("fps", "random"),
                        help="counterfactual support sampling; does not train a model")
    parser.add_argument("--radii", type=float, nargs=3, metavar=("R1", "R2", "R3"),
                        help="counterfactual SA ball-query radii; does not train a model")
    args = parser.parse_args()
    print(export_fixed_support_sa_trace(
        args.config, args.case, args.output_dir, args.partition, args.experiment,
        args.support_n_points, args.support_sampling,
        tuple(args.radii) if args.radii else None,
    ))


if __name__ == "__main__":
    main()
