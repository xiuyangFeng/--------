#!/usr/bin/env python3
"""Evaluate a completed checkpoint when the surface-area QA gate blocks test.

This is deliberately a diagnostic-only path.  It reports the historical
vertex metrics on every case and surface-area metrics only on the subset whose
STL-to-wall mapping passes the frozen QA gate.  It never replaces the normal
``training_wss_min.evaluate`` output or weakens its strict default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from training_wss_min import dataset as D
from training_wss_min import evaluate as E
from training_wss_min import metrics as M
from training_wss_min import surface as S


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _drop_uniform_area_fields(result: Dict) -> None:
    """Remove area-labelled values produced with diagnostic uniform weights."""
    for key in (
        "primary_physical_or_normalized_casebalanced_area",
        "area_hotspot",
        "group_casebalanced_area_r2",
    ):
        result.pop(key, None)
    for region in result.get("per_case", {}).values():
        region.pop("area_overall", None)
        region.pop("area_hotspot", None)
    if "normalized" in result:
        _drop_uniform_area_fields(result["normalized"])


def _group_casebalanced_vertex_r2(
    cases: Sequence[Dict], true_by_case: Sequence[np.ndarray], pred_by_case: Sequence[np.ndarray]
) -> Dict[str, float]:
    groups = {"AG": [], "AAA": [], "AAA_ruputer": [], "AAA_unruputer": []}
    for index, case in enumerate(cases):
        unit = case["unit_id"]
        groups["AG" if unit.startswith("AG/") else "AAA"].append(index)
        if unit.startswith("AAA/ruputer/"):
            groups["AAA_ruputer"].append(index)
        elif unit.startswith("AAA/unruputer/"):
            groups["AAA_unruputer"].append(index)
    output = {}
    for name, indices in groups.items():
        if indices:
            output[name] = M.casebalanced_field_metrics(
                [true_by_case[i] for i in indices], [pred_by_case[i] for i in indices]
            )["r2"]
    if "AG" in output and "AAA" in output:
        output["AG_AAA_domain_macro"] = (output["AG"] + output["AAA"]) / 2.0
    return output


def _evaluate_predictions(
    cases: List[Dict], pred_norm: List[np.ndarray], wss_stats: Dict
) -> tuple[Dict, List[np.ndarray]]:
    true_norm = [np.asarray(case["y_norm"], dtype=np.float64) for case in cases]
    normalized = E._evaluate_space(
        cases, true_norm, pred_norm, save_dir=None, make_plots=False,
        plot_space="normalized target",
    )
    pred_raw = []
    for prediction in pred_norm:
        raw = D.denormalize_wss(prediction, wss_stats)
        if wss_stats.get("method") == "log_z":
            raw = np.clip(raw, 0, None)
        pred_raw.append(np.asarray(raw, dtype=np.float64))
    true_raw = [np.asarray(case["y_raw"], dtype=np.float64) for case in cases]
    physical = E._evaluate_space(
        cases, true_raw, pred_raw, save_dir=None, make_plots=False,
        plot_space="physical target",
    )
    physical["metric_space"] = "physical_target"
    physical["normalized"] = normalized
    physical["group_casebalanced_vertex_r2"] = _group_casebalanced_vertex_r2(
        cases, true_raw, pred_raw
    )
    normalized["group_casebalanced_vertex_r2"] = _group_casebalanced_vertex_r2(
        cases, true_norm, pred_norm
    )
    return physical, pred_raw


def _write_per_case(result: Dict, output: Path) -> None:
    rows = []
    normalized = result["normalized"]
    for case_id in sorted(result["per_case"]):
        physical_case = result["per_case"][case_id]
        normalized_case = normalized["per_case"][case_id]
        row = {"case": case_id}
        for prefix, case_result in (
            ("physical", physical_case), ("normalized", normalized_case)
        ):
            overall = case_result["overall"]
            row.update({f"{prefix}_{key}": overall[key] for key in (
                "r2", "mae", "rmse", "nrmse_range"
            )})
            for key, value in case_result["hotspot"].items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    row[f"{prefix}_legacy_vertex_{key}"] = value
        rows.append(row)
    keys = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", choices=("best", "last"), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plots", action="store_true", help="write physical WSS heatmaps")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, feature_stats, model, checkpoint = E.load_model_from_run(
        args.run_dir, device, args.checkpoint
    )
    wss_stats = E.load_wss_stats_for_run(args.run_dir)
    cases = D.load_partition(
        cfg.data.split_path, "test", wss_stats, target=cfg.data.target,
        target_normalization=cfg.data.target_normalization,
        data_root=cfg.data.data_root,
        required_frame_version=cfg.data.required_frame_version,
    )

    mapping_reports = []
    valid_indices = []
    for index, case in enumerate(cases):
        weights, report = S.area_weights_for_case(case, strict=False)
        mapping_reports.append(report)
        if report["status"] == "passed":
            case["surface_area_weights"] = weights
            valid_indices.append(index)
        else:
            # Uniform weights are used only to execute the legacy metric code;
            # every area-labelled result from this all-case pass is removed.
            case["surface_area_weights"] = np.full(len(case["pos"]), 1.0 / len(case["pos"]))

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    pred_norm = [
        np.asarray(E.predict_case_norm(
            model, case, cfg.data.input_features, feature_stats, device, cfg=cfg
        ), dtype=np.float64)
        for case in cases
    ]
    elapsed = time.perf_counter() - started

    all_case_result, pred_raw = _evaluate_predictions(cases, pred_norm, wss_stats)
    _drop_uniform_area_fields(all_case_result)
    all_case_result["efficiency"] = {
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "elapsed_seconds": elapsed,
        "seconds_per_case": elapsed / len(cases),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
        ),
        "device": device,
    }

    valid_cases = [cases[i] for i in valid_indices]
    valid_pred_norm = [pred_norm[i] for i in valid_indices]
    area_result, _ = _evaluate_predictions(valid_cases, valid_pred_norm, wss_stats)
    failed_reports = [report for report in mapping_reports if report["status"] != "passed"]
    area_status = {
        "status": "incomplete_test27" if failed_reports else "complete_test27",
        "n_test_cases": len(cases),
        "n_valid_cases": len(valid_cases),
        "excluded_cases": [report["unit_id"] for report in failed_reports],
        "reason": "surface-area mapping QA gate failed",
        "metrics_scope": "valid-mapping subset only",
    }

    payload = {
        "diagnostic_only": True,
        "normal_evaluate_gate_weakened": False,
        "checkpoint": {
            "name": args.checkpoint,
            "epoch": int(checkpoint["epoch"]),
            "path": str(args.run_dir / E.checkpoint_filename(args.checkpoint)),
            "sha256": sha256(args.run_dir / E.checkpoint_filename(args.checkpoint)),
        },
        "protocol": {
            "split_path": cfg.data.split_path,
            "split_sha256": sha256(Path(cfg.data.split_path)),
            "stats_path": cfg.data.wss_stats_path,
            "stats_sha256": sha256(Path(cfg.data.wss_stats_path)),
            "runtime_config_path": str(args.run_dir / "config.json"),
            "runtime_config_sha256": sha256(args.run_dir / "config.json"),
        },
        "test_all27_vertex_metrics": all_case_result,
        "test_surface_area_valid_subset": {
            "scope": area_status,
            "metrics": area_result,
        },
        "surface_area_mapping_reports": mapping_reports,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_per_case(all_case_result, args.output_dir / "per_case_metrics.csv")
    if args.plots:
        for case, prediction in zip(cases, pred_raw):
            E._plot_case(
                case, np.asarray(case["y_raw"], dtype=np.float64), prediction,
                args.output_dir / "heatmaps", "physical WSS",
            )
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "epoch": int(checkpoint["epoch"]),
        "test_cases": len(cases),
        "surface_area_valid_cases": len(valid_cases),
        "surface_area_failed_cases": area_status["excluded_cases"],
        "output": str(args.output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
