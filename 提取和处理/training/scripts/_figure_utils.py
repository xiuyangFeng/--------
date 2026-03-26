from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

VALID_REGIONS = ("all", "interior", "wall")
_IS_WALL_FEATURE_IDX = 9  # pipeline.config.NODE_FEATURE_NAMES.index("is_wall")


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


# ---------------------------------------------------------------------------
# Region filtering utilities
# ---------------------------------------------------------------------------

def _resolve_wall_mask_from_payload(payload: Dict[str, object]) -> np.ndarray:
    """Return boolean array (True = wall node) from a prediction payload.

    Prefers ``graph_path`` for accurate ``is_wall`` even when the training
    ``feature_mask`` zeroed out that column (e.g. MLP baseline).
    """
    torch = lazy_import_torch()
    y_true = payload["y_true"]
    n = int(y_true.shape[0])

    graph_path = payload.get("graph_path")
    if graph_path:
        path = Path(str(graph_path))
        if path.is_file():
            try:
                from pipeline.dataset import load_graph_data
                data = load_graph_data(path)
                x_full = data.x.detach().cpu().numpy()
                if x_full.shape[0] == n and x_full.shape[1] > _IS_WALL_FEATURE_IDX:
                    return x_full[:, _IS_WALL_FEATURE_IDX] > 0.5
            except Exception:
                pass

    x = payload["x"]
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if x.shape[1] <= _IS_WALL_FEATURE_IDX:
        warnings.warn(
            "payload['x'] 列数不足，无法提取 is_wall；默认全部标记为 interior",
            stacklevel=2,
        )
        return np.zeros(n, dtype=bool)
    wall = x[:, _IS_WALL_FEATURE_IDX] > 0.5
    if not wall.any():
        warnings.warn(
            "is_wall 全为 0（可能被 feature_mask 屏蔽），区域过滤结果可能不可靠",
            stacklevel=2,
        )
    return wall


def build_region_mask(wall_mask: np.ndarray, region: str) -> np.ndarray:
    """Return boolean selection mask for the given region."""
    if region == "all":
        return np.ones(len(wall_mask), dtype=bool)
    if region == "interior":
        return ~wall_mask
    if region == "wall":
        return wall_mask
    raise ValueError(f"Unsupported region: {region!r}; choose from {VALID_REGIONS}")


def read_regional_metric(
    run_dir: Path, region: str, metric_key: str,
) -> Optional[float]:
    """Read a single metric value from the regional eval JSON of a run."""
    regional_json = run_dir / "predictions_test" / "regional_eval" / "fig_A5_regional_metrics.json"
    if not regional_json.exists():
        return None
    data = load_json(regional_json)
    region_data = data.get(region)
    if not isinstance(region_data, dict):
        return None
    val = region_data.get(metric_key)
    return float(val) if val is not None else None


def read_regional_metrics_dict(
    run_dir: Path, region: str,
) -> Optional[Dict[str, float]]:
    """Read all metrics for a region from the regional eval JSON."""
    regional_json = run_dir / "predictions_test" / "regional_eval" / "fig_A5_regional_metrics.json"
    if not regional_json.exists():
        return None
    data = load_json(regional_json)
    region_data = data.get(region)
    if not isinstance(region_data, dict):
        return None
    return {k: float(v) for k, v in region_data.items() if isinstance(v, (int, float))}


# ---------------------------------------------------------------------------
# Metrics & aggregation
# ---------------------------------------------------------------------------

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
    region: str = "all",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    """Aggregate predictions across cases, optionally filtering by region.

    Parameters
    ----------
    region : str
        ``"all"`` (default / legacy), ``"interior"``, or ``"wall"``.
    """
    if region not in VALID_REGIONS:
        raise ValueError(f"region={region!r} 非法；可选: {VALID_REGIONS}")

    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    per_case_metrics: Dict[str, Dict[str, float]] = {}

    for item in manifest_items:
        prediction_path = Path(str(item["prediction_path"])).resolve()

        if region == "all":
            y_true, y_pred, _x, fallback_case_name = load_prediction_arrays(prediction_path)
        else:
            payload = load_prediction_payload(prediction_path)
            y_true = payload["y_true"].detach().cpu().numpy()
            y_pred = payload["y_pred"].detach().cpu().numpy()
            fallback_case_name = str(payload.get("case_name", prediction_path.stem))
            wall_mask = _resolve_wall_mask_from_payload(payload)
            node_mask = build_region_mask(wall_mask, region)
            y_true = y_true[node_mask]
            y_pred = y_pred[node_mask]

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
