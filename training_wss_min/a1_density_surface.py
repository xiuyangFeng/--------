#!/usr/bin/env python3
"""Round-5 A1 density, neighbourhood-cap, STL mapping and truth-oracle audit.

This is deliberately val-only for model probes and never accepts a test partition.
It performs no training and never changes a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from training_wss_min import config as C
from training_wss_min import dataset as D
from training_wss_min import metrics as M


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "training_wss_min/runs/_round5/a1_density_surface"
DEFAULT_SPLIT = ROOT / "training/splits/split_AG_wss_min_v2_dev1.json"
RUN_NAMES = {
    1234: "r4_dev1_b1_tgtw_fixedq_s1234",
    7: "r4_dev1_b1_tgtw_fixedq_s7",
    2025: "r4_dev1_b1_tgtw_fixedq_s2025",
}
ORACLE_THRESHOLDS = {
    "r2_field_raw_min": 0.90,
    "r2_casemean_min": 0.85,
    "top10_ratio_min": 0.85,
    "top10_ratio_max": 1.15,
    "top10_iou_min": 0.60,
}
MAPPING_THRESHOLDS_MM = {
    "wall_to_triangle_p95_max": 0.50,
    "wall_to_triangle_max_max": 2.00,
    "surface_to_wall_p95_max": 1.00,
    "surface_coverage_1mm_min": 0.99,
}


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text())


def _norm_key(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


def _clean_stem(path: Path) -> str:
    return path.stem.replace("-all", "").replace("_all", "").replace("(1)", "").strip()


def stl_index() -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for path in sorted((ROOT / "stl_data").glob("*/*.stl")):
        out.setdefault(_norm_key(_clean_stem(path)), []).append(path)
    return out


def find_stl(cohort: str, case: str, index: Dict[str, List[Path]]) -> Path | None:
    subset = cohort.split("/")[-1]
    scored = []
    for path in index.get(_norm_key(case), []):
        score = 3 * (path.parent.name == subset) + 2 * (path.parent.name == "name_data")
        scored.append((score, str(path), path))
    return sorted(scored, key=lambda item: (-item[0], item[1]))[0][2] if scored else None


def load_stl(path: Path):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()
    triangle = vtk.vtkTriangleFilter()
    triangle.SetInputData(reader.GetOutput())
    triangle.Update()
    poly = triangle.GetOutput()
    pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    cells = vtk_to_numpy(poly.GetPolys().GetData())
    faces, i = [], 0
    while i < len(cells):
        n = int(cells[i])
        if n == 3:
            faces.append(cells[i + 1:i + 4])
        i += n + 1
    return pts, np.asarray(faces, dtype=np.int64), poly, numpy_to_vtk


def bbox_diag(points: np.ndarray) -> float:
    return float(np.linalg.norm(points.max(0) - points.min(0)))


def infer_stl_scale(stl_raw: np.ndarray, wall_mm: np.ndarray, unit_factor: float,
                    extent_mismatch: bool) -> Tuple[float, str, Dict[str, float]]:
    """Choose among explicit unit hypotheses without fitting a rigid transform."""
    stl_diag, wall_diag = bbox_diag(stl_raw), bbox_diag(wall_mm)
    ratio = wall_diag / max(stl_diag, 1e-12)
    candidates = {"bbox_ratio": ratio, "stl_mm": 1.0, "stl_native": unit_factor}
    # Even for extent mismatch, record ambiguity.  The closest wall extent is used because
    # the quantitative target is the post-crop CFD wall stored in the bundle.
    scores = {name: abs(math.log(max(stl_diag * scale, 1e-12) / max(wall_diag, 1e-12)))
              for name, scale in candidates.items()}
    kind = min(scores, key=scores.get)
    return float(candidates[kind]), kind, scores


def transformed_stl_poly(stl_raw: np.ndarray, faces: np.ndarray, bundle, scale: float):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    centroid = bundle["transform_centroid"].astype(np.float64)
    rotation = bundle["transform_rotation"].astype(np.float64)
    coord_scale = float(bundle["coord_scale"])
    points_mm = stl_raw * scale
    points_norm = ((points_mm - centroid) @ rotation) / coord_scale
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points_mm, deep=True))
    cells = vtk.vtkCellArray()
    for face in faces:
        tri = vtk.vtkTriangle()
        for j in range(3):
            tri.GetPointIds().SetId(j, int(face[j]))
        cells.InsertNextCell(tri)
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetPolys(cells)
    return points_mm, points_norm, poly


def closest_triangle_distances(points: np.ndarray, poly) -> Tuple[np.ndarray, np.ndarray]:
    import vtk
    from vtkmodules.vtkCommonCore import reference

    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()
    distances = np.empty(len(points), dtype=np.float64)
    cell_ids = np.empty(len(points), dtype=np.int64)
    closest = [0.0, 0.0, 0.0]
    for i, point in enumerate(points):
        cell_id, sub_id, dist2 = reference(0), reference(0), reference(0.0)
        locator.FindClosestPoint(point, closest, cell_id, sub_id, dist2)
        distances[i] = math.sqrt(max(float(dist2), 0.0))
        cell_ids[i] = int(cell_id)
    return distances, cell_ids


def sample_triangle_surface(points: np.ndarray, faces: np.ndarray, n: int, seed: int) -> np.ndarray:
    tri = points[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    valid = np.isfinite(areas) & (areas > 0)
    tri, areas = tri[valid], areas[valid]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(tri), size=n, replace=True, p=areas / areas.sum())
    r1, r2 = rng.random(n), rng.random(n)
    flip = r1 + r2 > 1.0
    r1[flip], r2[flip] = 1.0 - r1[flip], 1.0 - r2[flip]
    t = tri[chosen]
    return t[:, 0] + r1[:, None] * (t[:, 1] - t[:, 0]) + r2[:, None] * (t[:, 2] - t[:, 0])


def reuse_stats(target_ids: np.ndarray, n_targets: int, prefix: str) -> Dict:
    counts = np.bincount(target_ids, minlength=n_targets)
    positive = counts[counts > 0]
    return {
        f"{prefix}_unique_targets": int(len(positive)),
        f"{prefix}_target_coverage": float(len(positive) / max(n_targets, 1)),
        f"{prefix}_duplicate_source_fraction": float(1.0 - len(positive) / max(len(target_ids), 1)),
        f"{prefix}_reuse_p50": float(np.percentile(positive, 50)) if len(positive) else 0.0,
        f"{prefix}_reuse_p95": float(np.percentile(positive, 95)) if len(positive) else 0.0,
        f"{prefix}_reuse_max": int(positive.max()) if len(positive) else 0,
    }


def mapping_audit(cases: Sequence[Dict], out: Path, surface_samples: int) -> Tuple[List[Dict], Dict]:
    index, rows, missing = stl_index(), [], []
    for ci, case in enumerate(cases):
        label = f"{case['cohort']}/{case['case']}"
        stl_path = find_stl(case["cohort"], case["case"], index)
        if stl_path is None:
            missing.append(label)
            rows.append({"case": label, "status": "missing_stl", "mapping_pass": False})
            continue
        bundle_path = Path(case["bundle_path"])
        with np.load(bundle_path, allow_pickle=True) as bundle:
            stl_raw, faces, _, _ = load_stl(stl_path)
            wall_mm = bundle["wall_coords_raw"].astype(np.float64)
            scale, scale_kind, scale_scores = infer_stl_scale(
                stl_raw, wall_mm, float(bundle["unit_factor"]), bool(bundle["unit_extent_mismatch"]))
            stl_mm, _, poly = transformed_stl_poly(stl_raw, faces, bundle, scale)
            wall_to_tri, triangle_ids = closest_triangle_distances(wall_mm, poly)
            surface = sample_triangle_surface(stl_mm, faces, surface_samples, 71011 + ci)
            surface_dist, surface_wall_ids = cKDTree(wall_mm).query(surface, k=1)
            vertex_dist, vertex_wall_ids = cKDTree(wall_mm).query(stl_mm, k=1)
            p95_wt = float(np.percentile(wall_to_tri, 95))
            max_wt = float(wall_to_tri.max())
            p95_sw = float(np.percentile(surface_dist, 95))
            coverage = float(np.mean(surface_dist <= 1.0))
            passed = (p95_wt <= MAPPING_THRESHOLDS_MM["wall_to_triangle_p95_max"] and
                      max_wt <= MAPPING_THRESHOLDS_MM["wall_to_triangle_max_max"] and
                      p95_sw <= MAPPING_THRESHOLDS_MM["surface_to_wall_p95_max"] and
                      coverage >= MAPPING_THRESHOLDS_MM["surface_coverage_1mm_min"])
            row = {
                "case": label, "status": "ok", "stl_path": str(stl_path),
                "n_wall": len(wall_mm), "n_stl_vertices": len(stl_mm), "n_stl_triangles": len(faces),
                "unit_factor_mesh_to_mm": float(bundle["unit_factor"]),
                "unit_extent_mismatch": bool(bundle["unit_extent_mismatch"]),
                "wall_crop_applied": bool(bundle["wall_crop_applied"]),
                "stl_scale_to_mm": scale, "stl_scale_kind": scale_kind,
                "stl_scale_score_bbox_ratio": scale_scores["bbox_ratio"],
                "stl_scale_score_mm": scale_scores["stl_mm"],
                "stl_scale_score_native": scale_scores["stl_native"],
                "rotation_det": float(np.linalg.det(bundle["transform_rotation"])),
                "wall_to_triangle_p50_mm": float(np.percentile(wall_to_tri, 50)),
                "wall_to_triangle_p95_mm": p95_wt,
                "wall_to_triangle_p99_mm": float(np.percentile(wall_to_tri, 99)),
                "wall_to_triangle_max_mm": max_wt,
                "surface_to_wall_p50_mm": float(np.percentile(surface_dist, 50)),
                "surface_to_wall_p95_mm": p95_sw,
                "surface_to_wall_p99_mm": float(np.percentile(surface_dist, 99)),
                "surface_coverage_within_1mm": coverage,
                "vertex_to_wall_p95_mm": float(np.percentile(vertex_dist, 95)),
                "mapping_pass": bool(passed),
            }
            row.update(reuse_stats(triangle_ids, len(faces), "wall_to_triangle"))
            row.update(reuse_stats(vertex_wall_ids, len(wall_mm), "vertex_to_wall"))
            row.update(reuse_stats(surface_wall_ids, len(wall_mm), "surface_to_wall"))
            rows.append(row)
        print(f"[mapping {ci+1}/{len(cases)}] {label}: {rows[-1]['status']}", flush=True)
    write_csv(out / "mapping_per_case.csv", rows)
    ok = [r for r in rows if r["status"] == "ok"]
    summary = {
        "thresholds_mm": MAPPING_THRESHOLDS_MM,
        "n_cases": len(rows), "n_with_stl": len(ok), "missing_stl": missing,
        "n_pass": sum(bool(r["mapping_pass"]) for r in ok),
        "all_available_val_cases_pass": bool(ok) and all(bool(r["mapping_pass"]) for r in ok),
    }
    (out / "mapping_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return rows, summary


def oracle_metrics(cases: Sequence[Dict], out: Path, k: int = 3) -> Tuple[List[Dict], Dict]:
    rows, truths, preds, positions = [], [], [], []
    oracle_arrays = {}
    for ci, case in enumerate(cases):
        idx = D.farthest_point_sample(case["pos"], min(2000, len(case["pos"])), 1234 + 7919 * ci)
        dist, nei = cKDTree(case["pos"][idx]).query(case["pos"], k=min(k, len(idx)))
        if dist.ndim == 1:
            dist, nei = dist[:, None], nei[:, None]
        exact = dist[:, 0] <= 1e-12
        weights = 1.0 / np.maximum(dist, 1e-12)
        pred = np.sum(weights * case["y_raw"][idx][nei], axis=1) / np.sum(weights, axis=1)
        pred[exact] = case["y_raw"][idx][nei[exact, 0]]
        true = case["y_raw"].astype(np.float64)
        basic = M.basic_metrics(true, pred)
        cal = M.calibration_metrics(true, pred)
        hot = M.hotspot_localization_metrics(true, pred, case["pos"])
        label = f"{case['cohort']}/{case['case']}"
        rows.append({
            "case": label, "n_full": len(true), "n_sparse": len(idx), "method": "idw3",
            "r2": basic["r2"], "mae": basic["mae"], "rmse": basic["rmse"],
            "top10_pred_true_ratio": cal["top10_pred_true_ratio"],
            "top10_iou": hot["top10_iou"], "peak_point_dist_norm": hot["peak_point_dist"],
        })
        truths.append(true); preds.append(pred); positions.append(case["pos"])
        oracle_arrays[label] = (true, pred, case["pos"])
    pt, pp = np.concatenate(truths), np.concatenate(preds)
    field, cal = M.basic_metrics(pt, pp), M.calibration_metrics(pt, pp)
    hot = M.hotspot_localization_metrics(pt, pp, np.concatenate(positions))
    r2s = np.asarray([r["r2"] for r in rows], dtype=float)
    summary = {
        "method": "deterministic_fps2000_to_full_idw3", "thresholds": ORACLE_THRESHOLDS,
        "n_cases": len(rows), "r2_field_raw": field["r2"],
        "r2_casemean": float(np.nanmean(r2s)),
        "r2_casemedian": float(np.nanmedian(r2s)), "r2_case_p10": float(np.nanpercentile(r2s, 10)),
        "negative_r2_cases": int(np.sum(r2s < 0)), "mae_field": field["mae"],
        "top10_pred_true_ratio": cal["top10_pred_true_ratio"], "top10_iou": hot["top10_iou"],
    }
    summary["oracle_pass"] = bool(
        summary["r2_field_raw"] >= ORACLE_THRESHOLDS["r2_field_raw_min"] and
        summary["r2_casemean"] >= ORACLE_THRESHOLDS["r2_casemean_min"] and
        ORACLE_THRESHOLDS["top10_ratio_min"] <= summary["top10_pred_true_ratio"] <= ORACLE_THRESHOLDS["top10_ratio_max"] and
        summary["top10_iou"] >= ORACLE_THRESHOLDS["top10_iou_min"])
    write_csv(out / "oracle_per_case.csv", rows)
    (out / "oracle_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    plot_oracle_failures(rows, oracle_arrays, out / "failure_figures")
    return rows, summary


def plot_oracle_failures(rows: List[Dict], arrays: Dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    for row in sorted(rows, key=lambda r: r["r2"])[:6]:
        true, pred, pos = arrays[row["case"]]
        err = np.abs(true - pred)
        vmax, emax = np.percentile(true, 99), max(float(np.percentile(err, 99)), 1e-9)
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
        for ax, values, title, limit, cmap in zip(
                axes, (true, pred, err), ("CFD truth", "FPS2000 IDW3 oracle", "absolute error"),
                (vmax, vmax, emax), ("viridis", "viridis", "magma")):
            sc = ax.scatter(pos[:, 2], pos[:, 0], c=np.clip(values, 0, limit), s=1, cmap=cmap,
                            vmin=0, vmax=limit)
            ax.set_aspect("equal"); ax.set_title(title); fig.colorbar(sc, ax=ax, shrink=.7)
        fig.suptitle(f"{row['case']}  R2={row['r2']:.4f}  IoU={row['top10_iou']:.4f}")
        fig.savefig(out / f"oracle__{row['case'].replace('/', '__')}.png", dpi=120)
        plt.close(fig)


def predict_probe(out: Path, device: str) -> Tuple[List[Dict], Dict]:
    import torch
    from training_wss_min.evaluate import load_model_from_run, load_wss_stats_for_run, predict_case_norm

    rows, seed_summary = [], {}
    for seed, run_name in RUN_NAMES.items():
        run = ROOT / "training_wss_min/runs" / run_name
        cfg, feat_stats, model, _ = load_model_from_run(run, device)
        model.eval()
        if Path(cfg.data.split_path).resolve() != DEFAULT_SPLIT.resolve():
            raise RuntimeError(f"unexpected split in {run_name}: {cfg.data.split_path}")
        wss_stats = load_wss_stats_for_run(run)
        cases = D.load_partition(str(DEFAULT_SPLIT), "val", wss_stats, strict=True)
        can_true, can_pred, full_same_pred, can_pos = [], [], [], []
        for ci, case in enumerate(cases):
            idx = D.farthest_point_sample(case["pos"], min(2000, len(case["pos"])), seed + 7919 * ci)
            canonical = dict(case)
            for key in ("pos", "y_raw", "y_norm", "abscissa_norm", "local_radius", "curvature", "coord_scale", "radius_gradient"):
                canonical[key] = case[key][idx]
            pc_norm = predict_case_norm(model, canonical, cfg.data.input_features, feat_stats, device)
            pf_norm = predict_case_norm(model, case, cfg.data.input_features, feat_stats, device)[idx]
            # Force 1-D before metrics.  A checkpoint head may expose (N, 1);
            # leaving it unsqueezed would broadcast against (N,) into an NxN matrix.
            pc = np.clip(D.denormalize_wss(pc_norm, wss_stats), 0, None).reshape(-1)
            pf = np.clip(D.denormalize_wss(pf_norm, wss_stats), 0, None).reshape(-1)
            true = case["y_raw"][idx].astype(np.float64).reshape(-1)
            mc, mf, consistency = M.basic_metrics(true, pc), M.basic_metrics(true, pf), M.basic_metrics(pc, pf)
            cal_c, cal_f = M.calibration_metrics(true, pc), M.calibration_metrics(true, pf)
            hot_c = M.hotspot_localization_metrics(true, pc, case["pos"][idx])
            hot_f = M.hotspot_localization_metrics(true, pf, case["pos"][idx])
            rows.append({
                "seed": seed, "run": run_name, "partition": "val", "case": f"{case['cohort']}/{case['case']}",
                "n_full": len(case["pos"]), "n_canonical": len(idx),
                "canonical_r2_vs_truth": mc["r2"], "full_sameindex_r2_vs_truth": mf["r2"],
                "full_minus_canonical_r2": mf["r2"] - mc["r2"],
                "canonical_full_sameindex_r2": consistency["r2"],
                "canonical_full_sameindex_mae": consistency["mae"],
                "canonical_top10_ratio": cal_c["top10_pred_true_ratio"],
                "full_sameindex_top10_ratio": cal_f["top10_pred_true_ratio"],
                "canonical_top10_iou": hot_c["top10_iou"], "full_sameindex_top10_iou": hot_f["top10_iou"],
            })
            can_true.append(true); can_pred.append(pc); full_same_pred.append(pf)
            can_pos.append(case["pos"][idx])
            print(f"[probe s{seed} {ci+1}/{len(cases)}] {case['case']}", flush=True)
        pt, pc, pf = np.concatenate(can_true), np.concatenate(can_pred), np.concatenate(full_same_pred)
        pos = np.concatenate(can_pos)
        seed_rows = [r for r in rows if r["seed"] == seed]
        cal_c, cal_f = M.calibration_metrics(pt, pc), M.calibration_metrics(pt, pf)
        hot_c = M.hotspot_localization_metrics(pt, pc, pos)
        hot_f = M.hotspot_localization_metrics(pt, pf, pos)
        seed_summary[str(seed)] = {
            "canonical_r2_field": M.r2_score(pt, pc), "full_sameindex_r2_field": M.r2_score(pt, pf),
            "canonical_r2_casemean": float(np.mean([r["canonical_r2_vs_truth"] for r in seed_rows])),
            "full_sameindex_r2_casemean": float(np.mean([r["full_sameindex_r2_vs_truth"] for r in seed_rows])),
            "canonical_full_sameindex_r2": M.r2_score(pc, pf),
            "canonical_full_sameindex_mae": M.mae(pc, pf),
            "canonical_top10_ratio": cal_c["top10_pred_true_ratio"],
            "full_sameindex_top10_ratio": cal_f["top10_pred_true_ratio"],
            "canonical_top10_iou": hot_c["top10_iou"],
            "full_sameindex_top10_iou": hot_f["top10_iou"],
        }
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    write_csv(out / "density_probe_per_case.csv", rows)
    summary = {"partition": "val", "seeds": seed_summary, "n_rows": len(rows)}
    (out / "density_probe_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return rows, summary


def repeatability_audit(out: Path, device: str, n_repeats: int = 3) -> Tuple[List[Dict], Dict]:
    """Quantify CUDA radius/cap inference non-determinism on seed1234 val."""
    from training_wss_min.evaluate import load_model_from_run, load_wss_stats_for_run, predict_case_norm

    run = ROOT / "training_wss_min/runs" / RUN_NAMES[1234]
    cfg, feat_stats, model, _ = load_model_from_run(run, device)
    model.eval()
    stats = load_wss_stats_for_run(run)
    cases = D.load_partition(str(DEFAULT_SPLIT), "val", stats, strict=True)
    rows, pair_rows, summary = [], [], {"seed": 1234, "partition": "val", "n_repeats": n_repeats}
    for density in ("canonical2000", "full"):
        pooled_true = []
        pooled_pred = [[] for _ in range(n_repeats)]
        for ci, case in enumerate(cases):
            idx = D.farthest_point_sample(case["pos"], min(2000, len(case["pos"])), 1234 + 7919 * ci)
            if density == "canonical2000":
                target = dict(case)
                for key in ("pos", "y_raw", "y_norm", "abscissa_norm", "local_radius",
                            "curvature", "coord_scale", "radius_gradient"):
                    target[key] = case[key][idx]
                true = case["y_raw"][idx].astype(np.float64).reshape(-1)
            else:
                target = case
                true = case["y_raw"].astype(np.float64).reshape(-1)
            predictions = []
            for repeat in range(n_repeats):
                norm = predict_case_norm(model, target, cfg.data.input_features, feat_stats, device)
                pred = np.clip(D.denormalize_wss(norm, stats), 0, None).reshape(-1)
                predictions.append(pred)
                pooled_pred[repeat].append(pred)
                rows.append({"density": density, "case": f"{case['cohort']}/{case['case']}",
                             "repeat": repeat, "r2_vs_truth": M.r2_score(true, pred),
                             "mae_vs_truth": M.mae(true, pred)})
            for a in range(n_repeats):
                for b in range(a + 1, n_repeats):
                    pair_rows.append({"density": density, "case": f"{case['cohort']}/{case['case']}",
                                      "repeat_a": a, "repeat_b": b,
                                      "prediction_r2": M.r2_score(predictions[a], predictions[b]),
                                      "prediction_mae": M.mae(predictions[a], predictions[b])})
            pooled_true.append(true)
        yt = np.concatenate(pooled_true)
        yp = [np.concatenate(items) for items in pooled_pred]
        summary[density] = {
            "repeat_r2_field_vs_truth": [M.r2_score(yt, pred) for pred in yp],
            "repeat_mae_vs_truth": [M.mae(yt, pred) for pred in yp],
            "pairwise_prediction_r2": [M.r2_score(yp[a], yp[b]) for a in range(n_repeats)
                                       for b in range(a + 1, n_repeats)],
            "pairwise_prediction_mae": [M.mae(yp[a], yp[b]) for a in range(n_repeats)
                                         for b in range(a + 1, n_repeats)],
        }
    write_csv(out / "inference_repeatability_per_case.csv", rows)
    write_csv(out / "inference_repeatability_pairs.csv", pair_rows)
    (out / "inference_repeatability_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return rows, summary


def cap_audit(cases: Sequence[Dict], out: Path, device: str) -> Tuple[List[Dict], Dict]:
    cfg = C.ExpConfig.from_json(ROOT / "training_wss_min/runs/r4_dev1_b1_tgtw_fixedq_s1234/config.json")
    rows = []
    for ci, case in enumerate(cases):
        idx = D.farthest_point_sample(case["pos"], min(2000, len(case["pos"])), 1234 + 7919 * ci)
        for density, initial in (("canonical2000", case["pos"][idx]), ("full", case["pos"])):
            source = np.ascontiguousarray(initial)
            for layer, (ratio, radius, cap, radius_scale) in enumerate(zip(
                    cfg.model.sa_ratios, cfg.model.sa_radius, cfg.model.sa_nsample,
                    [cfg.model.invres_radius_scale] * len(cfg.model.sa_ratios)), 1):
                # PyG evaluation uses deterministic FPS (random_start=False).  The audit uses
                # the repository's deterministic NumPy FPS so it is reproducible on CPU nodes
                # without depending on torch-cluster's device-specific FPS kernel.
                n_query = max(1, int(math.ceil(len(source) * float(ratio))))
                qidx = D.farthest_point_sample(source, n_query, seed=0)
                query = source[qidx]
                for block, src, qry, rad in (("sa", source, query, radius),
                                              ("invres", query, query, radius * radius_scale)):
                    src_np, qry_np = src, qry
                    counts = cKDTree(src_np).query_ball_point(qry_np, rad, return_length=True)
                    effective = np.minimum(counts, cap)
                    raw_edges, kept_edges = int(counts.sum()), int(effective.sum())
                    rows.append({
                        "case": f"{case['cohort']}/{case['case']}", "density": density,
                        "layer": layer, "block": block, "radius_norm": rad, "cap": cap,
                        "n_source": len(src_np), "n_query": len(qry_np),
                        "neighbors_raw_mean": float(counts.mean()), "neighbors_raw_p50": float(np.percentile(counts, 50)),
                        "neighbors_raw_p95": float(np.percentile(counts, 95)), "neighbors_raw_max": int(counts.max()),
                        "neighbors_effective_mean": float(effective.mean()),
                        "query_truncation_rate": float(np.mean(counts > cap)),
                        "edge_truncation_fraction": float((raw_edges - kept_edges) / max(raw_edges, 1)),
                    })
                source = query
        print(f"[cap {ci+1}/{len(cases)}] {case['case']}", flush=True)
    write_csv(out / "neighborhood_cap_per_case_layer.csv", rows)
    grouped = {}
    for density in ("canonical2000", "full"):
        for layer in range(1, len(cfg.model.sa_ratios) + 1):
            for block in ("sa", "invres"):
                rr = [r for r in rows if r["density"] == density and r["layer"] == layer and r["block"] == block]
                grouped[f"{density}/L{layer}/{block}"] = {
                    "mean_effective_neighbors": float(np.mean([r["neighbors_effective_mean"] for r in rr])),
                    "mean_query_truncation_rate": float(np.mean([r["query_truncation_rate"] for r in rr])),
                    "mean_edge_truncation_fraction": float(np.mean([r["edge_truncation_fraction"] for r in rr])),
                }
    summary = {"n_cases": len(cases), "center_sampling": "deterministic_numpy_fps_proxy_for_pyg_eval_fps",
               "groups": grouped}
    (out / "neighborhood_cap_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--sections", nargs="+",
                    choices=("probe", "cap", "mapping", "oracle", "repeatability", "all", "finalize"),
                    default=["all"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--surface-samples", type=int, default=10000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sections = {"probe", "cap", "mapping", "oracle", "repeatability"} if "all" in args.sections else set(args.sections)
    sections.discard("finalize")
    # Fixed val-only scope; this script has intentionally no partition CLI.
    stats = D.load_wss_stats(ROOT / "data_wss_min/fold_stats/wss_stats_v2_dev1.json")
    val_cases = D.load_partition(str(DEFAULT_SPLIT), "val", stats, strict=True)
    run_meta = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "sections": sorted(sections),
        "split": str(DEFAULT_SPLIT), "partition": "val", "test_access": False,
        "oracle_thresholds": ORACLE_THRESHOLDS, "mapping_thresholds_mm": MAPPING_THRESHOLDS_MM,
        "script_sha256": sha256(Path(__file__)),
        "run_config_sha256": {str(s): sha256(ROOT / "training_wss_min/runs" / name / "config.json")
                              for s, name in RUN_NAMES.items()},
    }
    (args.out / "run_manifest.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))
    summaries = {}
    if "oracle" in sections:
        _, summaries["oracle"] = oracle_metrics(val_cases, args.out)
    if "mapping" in sections:
        _, summaries["mapping"] = mapping_audit(val_cases, args.out, args.surface_samples)
    if "cap" in sections:
        _, summaries["cap"] = cap_audit(val_cases, args.out, args.device)
    if "probe" in sections:
        _, summaries["probe"] = predict_probe(args.out, args.device)
    if "repeatability" in sections:
        _, summaries["repeatability"] = repeatability_audit(args.out, args.device)
    # A1 sections may be run independently on CPU/GPU nodes.  Finalization merges
    # already materialized section summaries without recomputing them.
    for name, filename in (("oracle", "oracle_summary.json"),
                           ("mapping", "mapping_summary.json"),
                           ("cap", "neighborhood_cap_summary.json"),
                           ("probe", "density_probe_summary.json"),
                           ("repeatability", "inference_repeatability_summary.json")):
        path = args.out / filename
        if path.is_file() and name not in summaries:
            summaries[name] = _json(path)
    run_meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_meta["summaries"] = summaries
    (args.out / "a1_summary.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
