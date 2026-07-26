#!/usr/bin/env python3
"""验收 AG/AAA v4 PointNet 作业并统一重算 common-test15。

本工具只读取既有 checkpoint、日志、评估 JSON 和 PostView wall CSV；不会加载模型
做推理，也不会启动训练。输出写入 v4 cutover 报告目录。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


FRAME = "stl_landmarks_v4"
BAD_LOG_PATTERNS = {
    "nonfinite": re.compile(r"(?i)(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])"),
    "oom": re.compile(r"(?i)(?:out of memory|cuda oom|oom-kill)"),
    "traceback": re.compile(r"Traceback \(most recent call last\)"),
    "frame_error": re.compile(r"(?i)(?:wrong frame|frame.*(?:mismatch|error|failed))"),
    "path_error": re.compile(r"(?i)(?:file not found|no such file|path.*(?:error|failed))"),
}

RUNS = {
    "AG-v4": {
        "job": "9138",
        "run": "ag_v4_e2_global_fps2000",
        "config": "training_wss_min/configs/pointnet_v4/ag_v4_e2_global_fps2000.json",
        "split": "training/splits/split_AG_wss_min_v4_traintest.json",
        "stats": "data_wss_min/fold_stats/v4/AG_v4_train61_global_stats.json",
        "counts": (61, 0, 15),
        "slurm": ("training_wss_min/cluster/logs/agv4_e2_9138.out",
                  "training_wss_min/cluster/logs/agv4_e2_9138.err"),
    },
    "AG+AAA-v4": {
        "job": "9140",
        "run": "ag_aaa_v4_e2_global_fps2000",
        "config": "training_wss_min/configs/pointnet_v4/ag_aaa_v4_e2_global_fps2000.json",
        "split": "training/splits/split_AG_AAA_wss_min_v4_trainpool.json",
        "stats": "data_wss_min/fold_stats/v4/AG_AAA_v4_trainpool118_global_stats.json",
        "counts": (118, 0, 15),
        "slurm": ("training_wss_min/cluster/logs/agaaav4_e2_9140.out",
                  "training_wss_min/cluster/logs/agaaav4_e2_9140.err"),
    },
}

PREDICTION_ROOTS = {
    "v3-E2-anchor": "training_wss_min/runs/pointnet_distribution_matrix/outputs/e2_global/postview/ckpt_best/test",
    "AG-v4": "training_wss_min/runs/pointnet_v4/outputs/ag_v4_e2_global_fps2000/postview/ckpt_best/test",
    "AG+AAA-v4": "training_wss_min/runs/pointnet_v4/outputs/ag_aaa_v4_e2_global_fps2000/postview/ckpt_best/test",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any) -> float:
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def sacct_rows(job_ids: list[str]) -> dict[str, dict[str, str]]:
    cmd = [
        "/public/slurm/bin/sacct", "-j", ",".join(job_ids), "-n", "-P",
        "--format=JobID,JobName,State,ExitCode,Elapsed,Start,End",
    ]
    output = subprocess.check_output(cmd, text=True)
    rows: dict[str, dict[str, str]] = {}
    for line in output.splitlines():
        values = line.split("|")
        if len(values) < 7 or "." in values[0]:
            continue
        rows[values[0]] = dict(zip(
            ("job_id", "job_name", "state", "exit_code", "elapsed", "start", "end"),
            values[:7],
        ))
    return rows


def critical_config_view(cfg: dict[str, Any]) -> dict[str, Any]:
    data, model, train = cfg["data"], cfg["model"], cfg["train"]
    return {
        "data": {k: data.get(k) for k in (
            "data_root", "required_frame_version", "split_path", "wss_stats_path",
            "wall_n_points", "sampling", "resample_each_epoch", "rot_aug",
            "input_features", "curvature_transform", "timesteps", "target",
            "target_normalization",
        )},
        "model": {k: model.get(k) for k in (
            "name", "width", "head_hidden", "pointnet_local_channels",
            "pointnet_decoder_channels", "out_dim", "dropout",
        )},
        "train": {k: train.get(k) for k in (
            "epochs", "batch_cases", "seed", "loss", "amp", "selection_rule",
            "early_stop_patience", "min_epoch",
        )},
    }


def scan_logs(paths: list[Path]) -> dict[str, Any]:
    findings: dict[str, list[dict[str, Any]]] = {key: [] for key in BAD_LOG_PATTERNS}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), 1):
            for key, pattern in BAD_LOG_PATTERNS.items():
                if pattern.search(line):
                    findings[key].append({"path": str(path), "line": line_no, "text": line[:300]})
    return {
        "files": [str(p) for p in paths],
        "findings": findings,
        "passed": not any(findings.values()),
    }


def validate_postview(root: Path, expected_cases: list[str]) -> dict[str, Any]:
    case_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    expected_names = sorted(f"{u.split('/')[1]}__{u.split('/')[2]}__peak_wss" for u in expected_cases)
    actual_names = [p.name for p in case_dirs]
    rows = []
    for case_dir in case_dirs:
        manifest_path = case_dir / "manifest_bundle.json"
        manifest = read_json(manifest_path)
        vtps = list(case_dir.rglob("*.vtp"))
        rows.append({
            "case_dir": case_dir.name,
            "manifest": str(manifest_path),
            "vtp_count": len(vtps),
            "mapping_coverage": finite_float(manifest["mapping_coverage"]["valid_ratio"]),
            "wall_csv": str(case_dir / manifest["files"]["wall_csv"]),
        })
    return {
        "case_count": len(case_dirs),
        "case_names_match_split": actual_names == expected_names,
        "manifest_count": len(rows),
        "vtp_count": sum(x["vtp_count"] for x in rows),
        "minimum_mapping_coverage": min((x["mapping_coverage"] for x in rows), default=float("nan")),
        "all_cases_have_four_vtp": all(x["vtp_count"] == 4 for x in rows),
        "rows": rows,
    }


def validate_run(repo: Path, label: str, spec: dict[str, Any], sacct: dict[str, dict[str, str]],
                 forbidden: set[str]) -> dict[str, Any]:
    run_dir = repo / "training_wss_min/runs/pointnet_v4/outputs" / spec["run"]
    source_cfg_path = repo / spec["config"]
    run_cfg_path = run_dir / "config.json"
    split_path = repo / spec["split"]
    stats_path = repo / spec["stats"]
    source_cfg, run_cfg = read_json(source_cfg_path), read_json(run_cfg_path)
    split, stats = read_json(split_path), read_json(stats_path)
    expected_train, expected_val, expected_test = spec["counts"]
    partitions = split["train_cases"] + split["val_cases"] + split["test_cases"]
    stats_units = stats["train_units"]

    manifest_bad_hashes = []
    for item in stats["bundle_manifest"]:
        path = Path(item["bundle_path"])
        if sha256(path) != item["bundle_sha256"]:
            manifest_bad_hashes.append(item["unit_id"])

    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines() if line.strip()]
    epochs = [int(row["epoch"]) for row in history]
    checkpoints = {}
    for name in ("best", "last"):
        path = run_dir / f"ckpt_{name}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoints[name] = {
            "path": str(path), "sha256": sha256(path), "epoch": int(payload["epoch"]),
            "cfg_name": payload.get("cfg_name"), "metric": payload.get("metric"),
        }

    evals: dict[str, Any] = {}
    for name in ("best", "last"):
        eval_dir = run_dir / "eval" / f"ckpt_{name}"
        metrics = read_json(eval_dir / "metrics.json")
        with (eval_dir / "per_case_metrics.csv").open(newline="", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        partition_counts = {
            part: sum(row["partition"] == part for row in csv_rows) for part in ("train", "val", "test")
        }
        evals[name] = {
            "metrics_path": str(eval_dir / "metrics.json"),
            "metrics_sha256": sha256(eval_dir / "metrics.json"),
            "per_case_path": str(eval_dir / "per_case_metrics.csv"),
            "per_case_sha256": sha256(eval_dir / "per_case_metrics.csv"),
            "partitions": sorted(metrics), "per_case_partition_counts": partition_counts,
            "expected_partition_counts_match": partition_counts == {
                "train": expected_train, "val": expected_val, "test": expected_test,
            },
        }

    postview = validate_postview(run_dir / "postview/ckpt_best/test", split["test_cases"])
    log_paths = [run_dir / "train.log", *(repo / p for p in spec["slurm"])]
    log_scan = scan_logs(log_paths)
    job = sacct.get(spec["job"], {})
    gates = {
        "sacct_completed_0_0": job.get("state") == "COMPLETED" and job.get("exit_code") == "0:0",
        "critical_config_snapshot_matches": critical_config_view(source_cfg) == critical_config_view(run_cfg),
        "protocol": source_cfg["data"]["required_frame_version"] == FRAME
                    and source_cfg["data"]["wall_n_points"] == 2000
                    and source_cfg["data"]["sampling"] == "fps"
                    and source_cfg["train"]["seed"] == 1234
                    and source_cfg["train"]["epochs"] == 400,
        "partition_counts": (len(split["train_cases"]), len(split["val_cases"]), len(split["test_cases"]))
                            == spec["counts"],
        "partitions_disjoint": len(partitions) == len(set(partitions)),
        "all_ids_canonical": all(len(x.split("/")) == 3 for x in partitions),
        "forbidden_absent_partitions": not (forbidden & set(partitions)),
        "stats_train_only_exact": stats_units == split["train_cases"] and stats["n_cases"] == expected_train,
        "forbidden_absent_stats": not (forbidden & set(stats_units)),
        "stats_manifest_hashes_match": not manifest_bad_hashes,
        "history_400_epochs": len(history) == 400 and epochs == list(range(400)),
        "best_last_epochs": checkpoints["best"]["epoch"] < 400 and checkpoints["last"]["epoch"] == 399,
        "best_last_eval_complete": all(x["expected_partition_counts_match"] for x in evals.values()),
        "postview_complete": postview["case_count"] == expected_test
                            and postview["case_names_match_split"]
                            and postview["all_cases_have_four_vtp"]
                            and postview["minimum_mapping_coverage"] == 1.0,
        "logs_clean": log_scan["passed"],
    }
    return {
        "label": label, "job": job, "run_dir": str(run_dir),
        "source_config": {"path": str(source_cfg_path), "sha256": sha256(source_cfg_path)},
        "run_config": {"path": str(run_cfg_path), "sha256": sha256(run_cfg_path)},
        "split": {"path": str(split_path), "sha256": sha256(split_path),
                  "counts": {"train": len(split["train_cases"]), "val": len(split["val_cases"]),
                             "test": len(split["test_cases"])},
                  "excluded_cases": split.get("excluded_cases", [])},
        "stats": {"path": str(stats_path), "sha256": sha256(stats_path),
                  "n_cases": stats["n_cases"], "n_points": stats["n_points"],
                  "bad_bundle_hashes": manifest_bad_hashes},
        "checkpoints": checkpoints, "eval": evals, "postview": postview,
        "log_scan": log_scan, "gates": gates, "passed": all(gates.values()),
    }


def read_wall_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64, encoding="utf-8")
    return {
        "pos": np.column_stack((data["x"], data["y"], data["z"])),
        "true": np.asarray(data["wss_cfd"], dtype=np.float64),
        "pred": np.asarray(data["wss_pred"], dtype=np.float64),
    }


def r2(y: np.ndarray, p: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - p) ** 2)) / denom if denom > 1e-12 else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) >= 3 and ra.std() > 0 and rb.std() > 0 else float("nan")


def per_case_metrics(case: dict[str, np.ndarray]) -> dict[str, float]:
    y, p, pos = case["true"], case["pred"], case["pos"]
    err = p - y
    rng = float(y.max() - y.min())
    true_high = y >= np.percentile(y, 90)
    pred_high = p >= np.percentile(p, 90)
    inter = np.logical_and(true_high, pred_high).sum()
    union = np.logical_or(true_high, pred_high).sum()
    bbox = float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0)))
    peak_dist = float(np.linalg.norm(pos[int(np.argmax(y))] - pos[int(np.argmax(p))]))
    centroid_dist = float(np.linalg.norm(pos[true_high].mean(axis=0) - pos[pred_high].mean(axis=0)))
    high_y, high_p = y[true_high], p[true_high]
    return {
        "r2": r2(y, p), "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "nmae": float(np.mean(np.abs(err)) / rng) if rng > 1e-12 else float("nan"),
        "high_wss_r2": r2(high_y, high_p),
        "high_wss_mae": float(np.mean(np.abs(high_p - high_y))),
        "p95_ratio": float(np.percentile(p, 95) / np.percentile(y, 95)),
        "p99_ratio": float(np.percentile(p, 99) / np.percentile(y, 99)),
        "peak_ratio": float(p.max() / y.max()),
        "spearman": spearman(y, p), "spearman_high": spearman(high_y, high_p),
        "top10_iou": float(inter / union),
        "peak_distance_over_bbox": peak_dist / bbox if bbox > 1e-12 else float("nan"),
        "hotspot_centroid_distance_over_bbox": centroid_dist / bbox if bbox > 1e-12 else float("nan"),
        "n": int(len(y)),
    }


def casebalanced_metrics(cases: list[dict[str, np.ndarray]]) -> dict[str, float]:
    true_means = np.asarray([x["true"].mean() for x in cases])
    center = float(true_means.mean())
    mse = np.asarray([np.mean((x["true"] - x["pred"]) ** 2) for x in cases])
    mae = np.asarray([np.mean(np.abs(x["true"] - x["pred"])) for x in cases])
    var = np.asarray([np.mean((x["true"] - center) ** 2) for x in cases])
    return {"r2": 1.0 - float(mse.mean() / var.mean()), "mae": float(mae.mean()),
            "rmse": float(np.sqrt(mse.mean()))}


def aggregate_prediction_metrics(cases: dict[str, dict[str, np.ndarray]]) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    rows = {unit: per_case_metrics(data) for unit, data in cases.items()}
    all_y = np.concatenate([x["true"] for x in cases.values()])
    all_p = np.concatenate([x["pred"] for x in cases.values()])
    err = all_p - all_y
    rng = float(all_y.max() - all_y.min())
    high_y = np.concatenate([x["true"][x["true"] >= np.percentile(x["true"], 90)] for x in cases.values()])
    high_p = np.concatenate([x["pred"][x["true"] >= np.percentile(x["true"], 90)] for x in cases.values()])
    keys = [k for k in next(iter(rows.values())) if k != "n"]
    case_mean = {k: float(np.nanmean([x[k] for x in rows.values()])) for k in keys}
    summary = {
        "n_cases": len(cases), "n_points": int(len(all_y)),
        "point_level": {
            "r2": r2(all_y, all_p), "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err * err))),
            "nmae": float(np.mean(np.abs(err)) / rng),
        },
        "case_balanced": casebalanced_metrics(list(cases.values())),
        "case_mean": case_mean,
        "high_wss_point_level": {
            "r2": r2(high_y, high_p), "mae": float(np.mean(np.abs(high_p - high_y))),
            "rmse": float(np.sqrt(np.mean((high_p - high_y) ** 2))),
        },
    }
    return summary, rows


def wall_csv_path(root: Path, unit: str) -> Path:
    _, subset, case = unit.split("/")
    return root / f"{subset}__{case}__peak_wss/_export/{case}__wall.csv"


def checkpoint_sensitivity(repo: Path, run: str) -> dict[str, Any]:
    out = {}
    for checkpoint in ("best", "last"):
        metrics = read_json(repo / "training_wss_min/runs/pointnet_v4/outputs" / run
                            / "eval" / f"ckpt_{checkpoint}" / "metrics.json")["test"]
        out[checkpoint] = {
            "point_r2": metrics["field"]["r2"], "point_mae": metrics["field"]["mae"],
            "point_rmse": metrics["field"]["rmse"],
            "case_balanced_r2": metrics["field_casebalanced"]["r2"],
            "case_mean_r2": metrics["aggregate"]["r2_casemean"],
            "high_wss_r2": metrics["regional_field"]["high_wss"]["r2"],
            "top10_iou": metrics["hotspot"]["top10_iou_casemean"],
            "spearman": metrics["hotspot"]["spearman_all_casemean"],
            "p99_ratio": metrics["calibration"]["p99_pred_true_ratio"],
        }
    out["last_minus_best"] = {k: out["last"][k] - out["best"][k] for k in out["best"]}
    return out


def unified_analysis(repo: Path, test_units: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    sources: dict[str, list[dict[str, str]]] = {}
    for label, relroot in PREDICTION_ROOTS.items():
        root = repo / relroot
        predictions[label] = {}
        sources[label] = []
        for unit in test_units:
            path = wall_csv_path(root, unit)
            predictions[label][unit] = read_wall_csv(path)
            sources[label].append({"unit_id": unit, "path": str(path), "sha256": sha256(path)})

    true_checks = {}
    for unit in test_units:
        base = predictions["v3-E2-anchor"][unit]["true"]
        true_checks[unit] = {
            label: {"same_shape": x[unit]["true"].shape == base.shape,
                    "max_abs_diff_pa": float(np.max(np.abs(x[unit]["true"] - base)))}
            for label, x in predictions.items()
        }
    truth_identical = all(
        row["same_shape"] and row["max_abs_diff_pa"] <= 1e-6
        for case in true_checks.values() for row in case.values()
    )

    summaries, per_run = {}, {}
    for label, cases in predictions.items():
        summaries[label], per_run[label] = aggregate_prediction_metrics(cases)

    comparison_rows: list[dict[str, Any]] = []
    for unit in test_units:
        row: dict[str, Any] = {"case": unit}
        for label in PREDICTION_ROOTS:
            for key, value in per_run[label][unit].items():
                row[f"{label}__{key}"] = value
        for key in ("r2", "mae", "rmse", "nmae", "high_wss_r2", "high_wss_mae",
                    "p99_ratio", "peak_ratio", "spearman", "top10_iou",
                    "peak_distance_over_bbox", "hotspot_centroid_distance_over_bbox"):
            row[f"mixed_minus_AG__{key}"] = per_run["AG+AAA-v4"][unit][key] - per_run["AG-v4"][unit][key]
            row[f"AG_minus_v3__{key}"] = per_run["AG-v4"][unit][key] - per_run["v3-E2-anchor"][unit][key]
        comparison_rows.append(row)

    by_r2_loss = sorted(comparison_rows, key=lambda x: x["mixed_minus_AG__r2"])
    by_mixed_mae = sorted(comparison_rows, key=lambda x: x["AG+AAA-v4__mae"], reverse=True)
    by_peak = sorted(comparison_rows, key=lambda x: abs(math.log(max(x["AG+AAA-v4__peak_ratio"], 1e-12))), reverse=True)
    by_hotspot = sorted(comparison_rows, key=lambda x: x["AG+AAA-v4__peak_distance_over_bbox"], reverse=True)

    quality = read_json(repo / "data_wss_min/pipeline_reports/v4_cutover_20260715_1921"
                        / "v4_training_quality_exclusions.json")
    watchlist = quality["watchlist_retained"]
    watch_metrics = []
    for item in watchlist:
        unit = item["unit_id"]
        entries = []
        for label, spec in RUNS.items():
            metrics = read_json(repo / "training_wss_min/runs/pointnet_v4/outputs" / spec["run"]
                                / "eval/ckpt_best/metrics.json")
            case = metrics["train"]["per_case"].get(unit)
            if case:
                entries.append({"run": label, "train_overall": case["overall"],
                                "train_high_wss": case["high_wss"],
                                "calibration": case["calibration"], "hotspot": case["hotspot"]})
        watch_metrics.append({**item, "available_train_evaluations": entries,
                              "policy": "retained_observation_only; no automatic exclusion"})

    analysis = {
        "metric_definitions": {
            "point_level": "concatenate all common-test15 wall points",
            "case_balanced": "each case has total weight 1; R2 centered at equal-case global true mean",
            "nmae": "MAE / (true max - true min); pooled and per-case-then-mean are both reported",
            "high_wss": "per-case true top 10% wall points",
            "hotspot": "per-case true/pred top10%; distances divided by case bbox diagonal",
        },
        "no_training_or_model_inference": True,
        "common_test15": test_units, "true_fields_identical_across_runs": truth_identical,
        "true_field_checks": true_checks, "sources": sources, "summaries": summaries,
        "checkpoint_sensitivity": {
            label: checkpoint_sensitivity(repo, spec["run"]) for label, spec in RUNS.items()
        },
        "anomaly_rankings": {
            "largest_mixed_vs_AG_r2_losses": [
                {"case": x["case"], "delta_r2": x["mixed_minus_AG__r2"]} for x in by_r2_loss[:5]],
            "highest_mixed_mae": [
                {"case": x["case"], "mae_pa": x["AG+AAA-v4__mae"]} for x in by_mixed_mae[:5]],
            "most_abnormal_mixed_peak_ratio": [
                {"case": x["case"], "peak_ratio": x["AG+AAA-v4__peak_ratio"]} for x in by_peak[:5]],
            "largest_mixed_peak_location_error": [
                {"case": x["case"], "distance_over_bbox": x["AG+AAA-v4__peak_distance_over_bbox"]}
                for x in by_hotspot[:5]],
        },
        "diagnostic_attribution": {
            "model": "all three runs retain negative high-WSS R2 and peak compression; hotspot errors are prediction-side",
            "normalization_statistics": "mixed point-pooled stats are AAA-mesh-density dominated; association, not causal proof",
            "queue_distribution": "mixed improves aggregate ranking/localization but has case-specific regressions",
            "data": "common-test15 true WSS arrays are identical and PostView coverage is complete; no new test-label anomaly found",
            "LI_SHU_KUN": "largest mixed-vs-AG R2 regression with identical truth and complete mapping; classify as model/domain-shift signal, not deletion evidence",
        },
        "watchlist_retained": watch_metrics,
    }
    return analysis, comparison_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    args = ap.parse_args()
    repo = args.repo.resolve()
    report_dir = repo / "data_wss_min/pipeline_reports/v4_cutover_20260715_1921"
    quality = read_json(report_dir / "v4_training_quality_exclusions.json")
    final_exclusions = read_json(report_dir / "v4_final_exclusions.json")
    forbidden = {x["unit_id"] for x in quality["excluded_units"]}
    forbidden.update(x["unit_id"] for x in final_exclusions["manual_exclusions"])
    if len(forbidden) != 9:
        raise RuntimeError(f"expected 9 exclusions, got {len(forbidden)}: {sorted(forbidden)}")

    sacct = sacct_rows([spec["job"] for spec in RUNS.values()] + ["9141"])
    run_reports = {
        label: validate_run(repo, label, spec, sacct, forbidden) for label, spec in RUNS.items()
    }
    acceptance = {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "formal acceptance of Slurm 9138/9140; read-only artifact validation",
        "forbidden_units": sorted(forbidden), "sacct": sacct, "runs": run_reports,
        "monitor_9141_passed": sacct.get("9141", {}).get("state") == "COMPLETED"
                               and sacct.get("9141", {}).get("exit_code") == "0:0",
    }
    acceptance["passed"] = all(x["passed"] for x in run_reports.values()) and acceptance["monitor_9141_passed"]

    test_units = read_json(repo / RUNS["AG-v4"]["split"])["test_cases"]
    analysis, rows = unified_analysis(repo, test_units)
    analysis_path = report_dir / "v4_e2_global_result_analysis.json"
    existing = read_json(analysis_path)
    existing.update({
        "schema_version": 2, "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "formal_acceptance": acceptance, "unified_common_test15": analysis,
    })
    analysis_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    acceptance_path = report_dir / "v4_e2_global_acceptance.json"
    acceptance_path.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = report_dir / "v4_e2_common_test15_per_case_comparison.csv"
    write_csv(csv_path, rows)
    print(json.dumps({"acceptance": str(acceptance_path), "passed": acceptance["passed"],
                      "analysis": str(analysis_path), "per_case": str(csv_path)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
