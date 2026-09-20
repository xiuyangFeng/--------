#!/usr/bin/env python3
"""Same-point volume diagnostics; reads saved predictions, never reruns a model.

Profiles are branch-restricted, volume-weighted 5 mm bins of relative pressure
and axial velocity. Gaussian smoothing is a diagnostic, not an orthogonal
spectral decomposition or an upper bound on attainable accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import h5py
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wss_v5 import contract as WC

SIGMAS_MM = (5, 10, 20, 40)
BIN_WIDTH_MM = 5.0


def finite_json(value):
    """Existing reports use NaN for empty regions; preserve these as JSON null."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def scalar_moments(true, pred):
    true, pred = np.asarray(true, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if true.shape != pred.shape or true.ndim != 1 or not len(true):
        raise ValueError("scalar metrics require matching, nonempty one-dimensional arrays")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("non-finite diagnostic prediction or label")
    return {"n": len(true), "mean": float(true.mean()), "variance": float(true.var()),
            "mse": float(np.mean((true - pred) ** 2)), "mae": float(np.mean(np.abs(true - pred)))}


def summarize_scalar(moments):
    if not moments:
        return {"n_cases": 0, "n_points": 0, "r2_casebalanced": None,
                "r2_pooled": None, "rmse": None, "rmse_casebalanced": None}
    count = np.array([m["n"] for m in moments], dtype=np.float64)
    mean = np.array([m["mean"] for m in moments])
    variance = np.array([m["variance"] for m in moments])
    mse = np.array([m["mse"] for m in moments])
    pooled_mean = np.average(mean, weights=count)
    cb_variance = np.mean(variance + (mean - mean.mean()) ** 2)
    pooled_variance = np.average(variance + (mean - pooled_mean) ** 2, weights=count)
    pooled_mse = np.average(mse, weights=count)
    return {"n_cases": len(moments), "n_points": int(count.sum()),
            "r2_casebalanced": float(1 - mse.mean() / cb_variance) if cb_variance > 0 else None,
            "r2_pooled": float(1 - pooled_mse / pooled_variance) if pooled_variance > 0 else None,
            "rmse": float(np.sqrt(pooled_mse)), "rmse_casebalanced": float(np.sqrt(mse.mean()))}


def branch_profiles(segment, s_mm, volume, true, pred):
    """Aggregate cells, smooth within each segment, and compute case-weighted errors."""
    segment = np.asarray(segment)
    s_mm, volume, true, pred = [np.asarray(a, dtype=np.float64) for a in (s_mm, volume, true, pred)]
    if not all(a.ndim == 1 and len(a) == len(segment) for a in (s_mm, volume, true, pred)):
        raise ValueError("profile arrays must be aligned one-dimensional cell arrays")
    if not len(segment) or not all(np.isfinite(a).all() for a in (s_mm, volume, true, pred)) or np.any(volume <= 0):
        raise ValueError("profiles require finite values and positive cell volumes")
    bins = np.floor(s_mm / BIN_WIDTH_MM).astype(np.int64)
    bias = float(np.average(pred - true, weights=volume))
    rows = []
    for sid in np.unique(segment):
        mask = segment == sid
        bin_ids, inverse = np.unique(bins[mask], return_inverse=True)
        weight = np.bincount(inverse, weights=volume[mask])
        yt = np.bincount(inverse, weights=true[mask] * volume[mask]) / weight
        yp = np.bincount(inverse, weights=pred[mask] * volume[mask]) / weight
        centers = (bin_ids + 0.5) * BIN_WIDTH_MM
        residual = yp - yt
        smooth = {}
        for sigma in SIGMAS_MM:
            kernel = np.exp(-0.5 * ((centers[:, None] - centers[None, :]) / sigma) ** 2)
            kernel /= kernel.sum(axis=1, keepdims=True)
            smooth[sigma] = (kernel @ residual, kernel @ (residual - bias))
        for index, bid in enumerate(bin_ids):
            row = {"segment_id": int(sid), "bin_index": int(bid), "s_center_mm": float(centers[index]),
                   "volume_m3": float(weight[index]), "true": float(yt[index]), "pred": float(yp[index]),
                   "residual": float(residual[index])}
            for sigma in SIGMAS_MM:
                row[f"residual_sigma{sigma}_mm"] = float(smooth[sigma][0][index])
                row[f"debiased_residual_sigma{sigma}_mm"] = float(smooth[sigma][1][index])
            rows.append(row)
    weights = np.array([row["volume_m3"] for row in rows])
    result = {"n_cells": len(segment), "n_bins": len(rows), "n_segments": len(np.unique(segment)),
              "case_bias": bias, "case_bias_squared": bias ** 2,
              "full_residual_mse": float(np.average((pred - true) ** 2, weights=volume)),
              "binned_residual_mse": float(np.average([row["residual"] ** 2 for row in rows], weights=weights)),
              "sigmas_mm": {}}
    for sigma in SIGMAS_MM:
        result["sigmas_mm"][str(sigma)] = {
            "residual_mse": float(np.average([row[f"residual_sigma{sigma}_mm"] ** 2 for row in rows], weights=weights)),
            "debiased_residual_mse": float(np.average([row[f"debiased_residual_sigma{sigma}_mm"] ** 2 for row in rows], weights=weights))}
    return rows, result


def summarize_longwave(per_case):
    cases = list(per_case.values())
    if not cases:
        raise ValueError("no cases available for long-wave diagnostics")
    return {"n_cases": len(cases),
            "case_bias_mse_casemean": float(np.mean([c["case_bias_squared"] for c in cases])),
            "case_bias_abs_casemean": float(np.mean([abs(c["case_bias"]) for c in cases])),
            "full_residual_mse_casemean": float(np.mean([c["full_residual_mse"] for c in cases])),
            "binned_residual_mse_casemean": float(np.mean([c["binned_residual_mse"] for c in cases])),
            "sigmas_mm": {str(sigma): {
                "residual_mse_casemean": float(np.mean([c["sigmas_mm"][str(sigma)]["residual_mse"] for c in cases])),
                "debiased_residual_mse_casemean": float(np.mean([c["sigmas_mm"][str(sigma)]["debiased_residual_mse"] for c in cases]))}
                for sigma in SIGMAS_MM}}


def local_geometry(unit_id, bundle_path, snapshot_root):
    volume_path = bundle_path.parent / "volume.npz"
    with np.load(volume_path) as vol:
        segment = vol["vol_segment_id"].copy()
        volume = vol["vol_volume_m3"].astype(np.float64)
        radial = vol["vol_radial_aligned"].astype(np.float64)
    with np.load(bundle_path, allow_pickle=True) as bundle:
        rotation = bundle["transform_rotation"].astype(np.float64)
    with h5py.File(WC.case_dir(unit_id, snapshot_root) / "case.h5", "r") as h5:
        columns = json.loads(h5["geometry"].attrs["atlas_columns"])
        table = h5["geometry/atlas_table"][()]
        atlas_row = h5["volume_static/atlas_row"][()].astype(np.int64)
        s_mm = h5["volume_static/s_from_root_mm"][()].astype(np.float64)
        snapshot_segment = h5["volume_static/segment_id"][()]
    if not np.array_equal(segment, snapshot_segment) or len(atlas_row) != len(volume):
        raise ValueError(f"view/snapshot cell ordering mismatch: {unit_id}")
    tangent = table[:, [columns.index(f"tangent_{axis}") for axis in "xyz"]][atlas_row].astype(np.float64)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError(f"zero centerline tangent: {unit_id}")
    axial = (tangent / norm) @ rotation.T
    radial -= np.sum(radial * axial, axis=1, keepdims=True) * axial
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    valid = radial_norm[:, 0] > 1e-6
    radial = np.where(valid[:, None], radial / np.maximum(radial_norm, 1e-12), 0.0)
    return {"segment": segment, "s_mm": s_mm, "volume": volume,
            "axial": axial, "radial": radial, "circ": np.cross(axial, radial), "valid": valid}


def write_profile(rows, output, task, unit_id, make_plots):
    stem = unit_id.replace("/", "__") + "_" + task
    profile_path = output / "profiles" / f"{stem}.csv"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if make_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
        for index, sid in enumerate(sorted({row["segment_id"] for row in rows})):
            selected = [row for row in rows if row["segment_id"] == sid]
            x = [row["s_center_mm"] for row in selected]
            color = f"C{index % 10}"
            axes[0].plot(x, [row["true"] for row in selected], color=color, label=f"segment {sid} true")
            axes[0].plot(x, [row["pred"] for row in selected], color=color, linestyle="--")
            axes[1].plot(x, [row["residual_sigma20_mm"] for row in selected], color=color)
        units = "Pa" if task == "pressure" else "m/s"
        axes[0].set(ylabel=f"{task} ({units})", title=f"{unit_id}; dashed = predicted")
        axes[0].legend(ncol=4, fontsize=7)
        axes[1].set(xlabel="s from root (mm)", ylabel=f"20 mm smoothed residual ({units})")
        axes[1].axhline(0, color="black", linewidth=.5)
        plot_path = output / "plots" / f"{stem}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=130)
        plt.close(fig)
    return str(profile_path.relative_to(output))


def diagnose(run_dir, ckpt="best", partition="test", snapshot_root=WC.SNAPSHOT_ROOT, make_plots=True):
    run_dir, snapshot_root = Path(run_dir), Path(snapshot_root)
    eval_dir = run_dir / "eval" / f"ckpt_{ckpt}"
    prediction_root = eval_dir / "predictions"
    manifest = json.loads((prediction_root / "manifest.json").read_text())
    target = manifest["target"]
    if target not in ("pressure_mixed", "velocity", "velocity_pressure"):
        raise ValueError(f"unsupported diagnostic target: {target}")
    entries = manifest["partitions"][partition]
    if not entries or len({entry["unit_id"] for entry in entries}) != len(entries):
        raise ValueError("prediction manifest has empty or duplicate case identities")
    output = eval_dir / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    tasks = {}
    if target in ("pressure_mixed", "velocity_pressure"):
        tasks["pressure"] = {"longwave": {"per_case": {}}}
    if target in ("velocity", "velocity_pressure"):
        tasks["velocity"] = {"longwave": {"per_case": {}}, "per_case_components": {}}
    components = {name: [] for name in ("u", "v", "w", "axial", "radial", "circ", "speed")}
    vector_cases = []
    for entry in entries:
        unit_id = entry["unit_id"]
        file_path = prediction_root / entry["file"]
        with np.load(file_path) as saved:
            if str(saved["unit_id"].item()) != unit_id or str(saved["target"].item()) != target:
                raise ValueError(f"prediction identity mismatch: {file_path}")
            query = saved["query_idx"].astype(np.int64)
            kind = saved["point_kind"]
            pred, true = saved["pred_raw"].astype(np.float64), saved["true_raw"].astype(np.float64)
            n_wall = int(saved["n_wall"])
        if pred.shape != true.shape or len(pred) != len(query) or len(np.unique(query)) != len(query):
            raise ValueError(f"invalid saved prediction rows: {unit_id}")
        geometry = local_geometry(unit_id, Path(entry["bundle_path"]), snapshot_root)
        n_volume = len(geometry["volume"])
        expected = np.arange(n_wall if target == "velocity" else 0, n_wall + n_volume)
        if not np.array_equal(query, expected) or not np.array_equal(kind, (query >= n_wall).astype(kind.dtype)):
            raise ValueError(f"diagnostics require full query coverage in original case order: {unit_id}")
        interior = kind == 1
        cell_idx = query[interior] - n_wall
        # This explicit association keeps joint wall rows out of every velocity statistic.
        if not np.array_equal(cell_idx, np.arange(n_volume)):
            raise ValueError(f"interior query/cell mapping mismatch: {unit_id}")
        if "pressure" in tasks:
            yp = pred[interior, 3] if target == "velocity_pressure" else pred[interior]
            yt = true[interior, 3] if target == "velocity_pressure" else true[interior]
            rows, result = branch_profiles(geometry["segment"], geometry["s_mm"], geometry["volume"], yt, yp)
            result["profile_csv"] = write_profile(rows, output, "pressure", unit_id, make_plots)
            tasks["pressure"]["longwave"]["per_case"][unit_id] = result
        if "velocity" in tasks:
            yp, yt = pred[interior, :3], true[interior, :3]
            case_components = {}
            for index, name in enumerate(("u", "v", "w")):
                moment = scalar_moments(yt[:, index], yp[:, index])
                components[name].append(moment)
                case_components[name] = summarize_scalar([moment])
            valid = geometry["valid"]
            for name in ("axial", "radial", "circ"):
                if np.any(valid):
                    moment = scalar_moments(np.sum(yt[valid] * geometry[name][valid], axis=1),
                                            np.sum(yp[valid] * geometry[name][valid], axis=1))
                    components[name].append(moment)
                    case_components[name] = summarize_scalar([moment])
                else:
                    case_components[name] = summarize_scalar([])
            moment = scalar_moments(np.linalg.norm(yt, axis=1), np.linalg.norm(yp, axis=1))
            components["speed"].append(moment)
            case_components["speed"] = summarize_scalar([moment])
            vector_mse = float(np.mean(np.sum((yt - yp) ** 2, axis=1)))
            vector_cases.append((len(yt), vector_mse))
            case_components["vector_rmse_m_s"] = float(np.sqrt(vector_mse))
            case_components["local_frame_excluded_points"] = int((~valid).sum())
            tasks["velocity"]["per_case_components"][unit_id] = case_components
            axial_true = np.sum(yt * geometry["axial"], axis=1)
            axial_pred = np.sum(yp * geometry["axial"], axis=1)
            rows, result = branch_profiles(geometry["segment"], geometry["s_mm"], geometry["volume"], axial_true, axial_pred)
            result["profile_csv"] = write_profile(rows, output, "axial_velocity", unit_id, make_plots)
            tasks["velocity"]["longwave"]["per_case"][unit_id] = result
        print(f"[diagnostics] {ckpt}/{partition} {unit_id}", flush=True)
    for task, result in tasks.items():
        result["longwave"]["summary"] = summarize_longwave(result["longwave"]["per_case"])
        metrics_path = eval_dir / task / "metrics.json" if target == "velocity_pressure" else eval_dir / "metrics.json"
        result["evaluation_metrics_path"] = str(metrics_path)
        metrics = json.loads(metrics_path.read_text())[partition]
        result["evaluation"] = {key: metrics[key] for key in ("field_casebalanced", "group_casebalanced", "query_groups") if key in metrics}
    if "velocity" in tasks:
        tasks["velocity"]["components"] = {name: summarize_scalar(values) for name, values in components.items()}
        tasks["velocity"]["vector_rmse_m_s"] = float(np.sqrt(np.average([v[1] for v in vector_cases], weights=[v[0] for v in vector_cases])))
        tasks["velocity"]["vector_rmse_casebalanced_m_s"] = float(np.sqrt(np.mean([v[1] for v in vector_cases])))
    summary = {"schema_version": 1, "target": target, "run_dir": str(run_dir.resolve()), "checkpoint": ckpt,
               "partition": partition, "case_ids": [entry["unit_id"] for entry in entries], "tasks": tasks,
               "method": {"prediction_manifest": str(prediction_root / "manifest.json"),
                          "snapshot_root": str(snapshot_root), "bin_width_mm": BIN_WIDTH_MM,
                          "bin_key": "(segment_id, floor(s_from_root_mm / 5))", "sigmas_mm": list(SIGMAS_MM),
                          "within_bin_weights": "cell volume_m3", "within_case_metric_weights": "bin volume sum",
                          "smoothing": "Gaussian across occupied 5 mm bins within each segment; row-normalized at boundaries/gaps",
                          "cross_case_weights": "equal", "pressure_scope": "interior only for longwave; wall/interior full metrics unchanged",
                          "velocity_local_metrics": "exclude degenerate radial frame from axial/radial/circ metrics, matching historical decomposition",
                          "velocity_longwave": "axial component at all interior cells with valid atlas tangents",
                          "debiased": "subtract case volume-weighted raw residual mean before smoothing",
                          "interpretation": "diagnostic smoothing; not an orthogonal spectral decomposition or theoretical accuracy bound"}}
    summary = finite_json(summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--ckpt", choices=("best", "last"), default="best")
    parser.add_argument("--partition", default="test", choices=("train", "val", "test"))
    parser.add_argument("--snapshot-root", type=Path, default=WC.SNAPSHOT_ROOT)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    diagnose(args.run_dir, args.ckpt, args.partition, args.snapshot_root, not args.no_plots)


if __name__ == "__main__":
    main()
