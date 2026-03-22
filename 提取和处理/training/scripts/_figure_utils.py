from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def ensure_dir(path: Path | str) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 结构非法: {path}")
    return data


def save_json(path: Path, data: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_manifest(manifest_path: Path) -> Dict[str, object]:
    return load_json(manifest_path)


def lazy_import_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "读取预测结果失败：当前环境缺少 torch。请先安装训练依赖，例如 `pip install -r training/requirements.txt`。"
        ) from exc
    return torch


def load_prediction_payload(prediction_path: Path) -> Dict[str, object]:
    torch = lazy_import_torch()
    payload = torch.load(prediction_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"预测文件结构非法: {prediction_path}")
    return payload


def load_prediction_arrays(prediction_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    payload = load_prediction_payload(prediction_path)
    y_true = payload["y_true"].detach().cpu().numpy()
    y_pred = payload["y_pred"].detach().cpu().numpy()
    x = payload["x"].detach().cpu().numpy()
    case_name = str(payload.get("case_name", prediction_path.stem))
    return y_true, y_pred, x, case_name


def compute_case_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(np.square(err))))
    rmse_p = float(np.sqrt(np.mean(np.square(err[:, 3]))))
    vel_true = np.linalg.norm(y_true[:, :3], axis=1)
    vel_pred = np.linalg.norm(y_pred[:, :3], axis=1)
    rmse_vel_mag = float(np.sqrt(np.mean(np.square(vel_pred - vel_true))))
    mae = float(np.mean(np.abs(err)))
    return {
        "rmse": rmse,
        "rmse_p": rmse_p,
        "rmse_vel_mag": rmse_vel_mag,
        "mae": mae,
        "num_nodes": int(y_true.shape[0]),
    }


def maybe_subsample(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(y_true) <= max_points:
        return y_true, y_pred
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(y_true), size=max_points, replace=False)
    return y_true[indices], y_pred[indices]


def aggregate_predictions(
    manifest_items: Sequence[Dict[str, object]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    per_case_metrics: Dict[str, Dict[str, float]] = {}

    for item in manifest_items:
        prediction_path = Path(str(item["prediction_path"])).resolve()
        y_true, y_pred, _x, fallback_case_name = load_prediction_arrays(prediction_path)
        case_name = str(item.get("case_name", fallback_case_name))
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        per_case_metrics[case_name] = compute_case_metrics(y_true, y_pred)

    if not y_true_all:
        raise ValueError("manifest 中没有可用预测项")

    return np.concatenate(y_true_all, axis=0), np.concatenate(y_pred_all, axis=0), per_case_metrics


def resolve_run_dirs(runs_root: Path, patterns: Sequence[str], run_dirs: Sequence[str]) -> List[Path]:
    resolved: List[Path] = []
    seen = set()

    for run_dir in run_dirs:
        path = Path(run_dir).resolve()
        if path.exists() and path not in seen:
            resolved.append(path)
            seen.add(path)

    for pattern in patterns:
        for candidate in sorted(runs_root.glob(pattern)):
            path = candidate.parent if candidate.name.endswith(".json") else candidate
            path = path.resolve()
            if path.exists() and path not in seen:
                resolved.append(path)
                seen.add(path)
    return resolved
