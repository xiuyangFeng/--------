"""Pa 量纲 WSS 临床指标（纯后处理，无 GPU）。

读取 predict_field 导出的 predictions_*.pt + normalization_params_global.json，
inverse z-score 还原 Pa，产出病例级 Pearson/Spearman 与汇总 JSON/CSV。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..analysis.regional_eval import build_region_masks, load_node_features_for_region_masks
from ..core.utils import ensure_dir
from ._figure_utils import load_manifest, load_prediction_payload, save_json


def _inverse_zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std < 1e-12:
        return np.full_like(values, mean, dtype=np.float64)
    return values.astype(np.float64) * std + mean


def _load_norm_params(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _stats_for_key(norm: Dict, key: str) -> Tuple[float, float]:
    stats = norm.get("statistics", {})
    if key in stats:
        return float(stats[key]["mean"]), float(stats[key]["std"])
    wss_local = norm.get("wss_local", {})
    comp = key.replace("wss_", "")
    if comp in wss_local:
        return float(wss_local[comp]["mean"]), float(wss_local[comp]["std"])
    raise KeyError(f"归一化参数缺少 {key}")


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and sorted_vals[j] == sorted_vals[i]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _corr(y_true: np.ndarray, y_pred: np.ndarray, *, rank: bool = False) -> float:
    if y_true.size < 2:
        return float("nan")
    if rank:
        y_true = _rankdata(y_true)
        y_pred = _rankdata(y_pred)
    yt = y_true - np.mean(y_true)
    yp = y_pred - np.mean(y_pred)
    denom = float(np.sqrt(np.sum(yt ** 2) * np.sum(yp ** 2)))
    if denom <= 1e-20:
        return float("nan")
    return float(np.sum(yt * yp) / denom)


def _denorm_wss_payload(
    payload: Dict,
    norm: Dict,
    frame: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """返回 (y_true_pa, y_pred_pa, target_names) 壁面有效行。"""
    target_names = list(payload.get("wss_target_names") or ["wss", "wss_x", "wss_y", "wss_z"])
    y_pred = payload["y_wss_pred"].detach().cpu().numpy()
    y_true = payload["y_wss_true"].detach().cpu().numpy()

    pred_pa = np.zeros_like(y_pred, dtype=np.float64)
    true_pa = np.zeros_like(y_true, dtype=np.float64)
    for i, name in enumerate(target_names):
        mean, std = _stats_for_key(norm, name)
        pred_pa[:, i] = _inverse_zscore(y_pred[:, i], mean, std)
        true_pa[:, i] = _inverse_zscore(y_true[:, i], mean, std)

    features = load_node_features_for_region_masks(payload)
    wall = build_region_masks(features)["wall"].detach().cpu().numpy().astype(bool)
    finite = np.isfinite(true_pa).all(axis=1) & np.isfinite(pred_pa).all(axis=1)
    valid = wall & finite
    return true_pa[valid], pred_pa[valid], target_names


def _case_aggregate(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _plot_case_scatter(rows: Sequence[Dict[str, object]], key: str, save_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("缺少 matplotlib") from exc

    xt = np.asarray([float(r[f"{key}_true_pa"]) for r in rows], dtype=np.float64)
    yp = np.asarray([float(r[f"{key}_pred_pa"]) for r in rows], dtype=np.float64)
    lo = float(min(np.min(xt), np.min(yp)))
    hi = float(max(np.max(xt), np.max(yp)))
    pad = (hi - lo) * 0.04 if hi > lo else 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xt, yp, s=28, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", linewidth=1.2)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_xlabel(f"CFD case {key} (Pa)")
    ax.set_ylabel(f"Predicted case {key} (Pa)")
    ax.set_title(f"Case-level WSS {key} (Pa)")
    ax.text(
        0.05, 0.95,
        f"Pearson={_corr(xt, yp):.3f}\nSpearman={_corr(xt, yp, rank=True):.3f}",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_clinical_eval(
    manifest_path: Path,
    norm_params_path: Path,
    output_dir: Path,
    frame_tag: str = "global",
) -> Dict:
    manifest = load_manifest(manifest_path)
    norm = _load_norm_params(norm_params_path)
    items = manifest.get("items", [])
    output_dir = ensure_dir(output_dir)

    by_case_true: Dict[str, List[np.ndarray]] = defaultdict(list)
    by_case_pred: Dict[str, List[np.ndarray]] = defaultdict(list)
    mag_idx = 0

    for item in items:
        pred_path = Path(str(item["prediction_path"]))
        if not pred_path.is_file():
            pred_path = manifest_path.parent / pred_path.name
        payload = load_prediction_payload(pred_path)
        if "y_wss_pred" not in payload or "y_wss_true" not in payload:
            continue
        frame = str(payload.get("wss_target_frame", frame_tag))
        y_true_pa, y_pred_pa, names = _denorm_wss_payload(payload, norm, frame)
        if y_true_pa.size == 0:
            continue
        if "wss" in names:
            mag_idx = names.index("wss")
        case_name = str(item.get("case_name", payload.get("case_name", "unknown")))
        by_case_true[case_name].append(y_true_pa[:, mag_idx])
        by_case_pred[case_name].append(y_pred_pa[:, mag_idx])

    case_rows: List[Dict[str, object]] = []
    for case_name in sorted(by_case_true):
        yt = np.concatenate(by_case_true[case_name])
        yp = np.concatenate(by_case_pred[case_name])
        t_stats = _case_aggregate(yt)
        p_stats = _case_aggregate(yp)
        case_rows.append({
            "case_name": case_name,
            "n_wall": int(yt.size),
            "mean_true_pa": t_stats["mean"],
            "mean_pred_pa": p_stats["mean"],
            "p95_true_pa": t_stats["p95"],
            "p95_pred_pa": p_stats["p95"],
            "max_true_pa": t_stats["max"],
            "max_pred_pa": p_stats["max"],
        })

    summary_rows: List[Dict[str, object]] = []
    for metric_key, true_key, pred_key in (
        ("mean", "mean_true_pa", "mean_pred_pa"),
        ("p95", "p95_true_pa", "p95_pred_pa"),
        ("max", "max_true_pa", "max_pred_pa"),
    ):
        xt = np.asarray([float(r[true_key]) for r in case_rows], dtype=np.float64)
        yp = np.asarray([float(r[pred_key]) for r in case_rows], dtype=np.float64)
        summary_rows.append({
            "metric": metric_key,
            "n_cases": len(case_rows),
            "pearson": _corr(xt, yp),
            "spearman": _corr(xt, yp, rank=True),
        })

    _write_csv(output_dir / "wss_pa_case_metrics.csv", case_rows)
    if case_rows:
        _plot_case_scatter(case_rows, "mean", output_dir / "fig_case_level_wss_mean_pa.png")
        _plot_case_scatter(case_rows, "p95", output_dir / "fig_case_level_wss_p95_pa.png")

    summary = {
        "frame_tag": frame_tag,
        "unit": "Pa",
        "norm_params": str(norm_params_path),
        "manifest": str(manifest_path),
        "n_cases": len(case_rows),
        "case_level": summary_rows,
        "note": "inverse z-score: x_pa = x_norm * std + mean（wss 段为 Pa 量纲纯 z-score）",
    }
    save_json(output_dir / "wss_pa_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Pa 量纲 WSS 临床指标（纯后处理）")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--norm-params", required=True, type=Path, help="normalization_params_global.json")
    parser.add_argument("--output-dir", default="", type=Path)
    parser.add_argument("--frame-tag", default="global", choices=["global", "local_v1"])
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else manifest_path.parent / "wss_clinical_pa"
    )
    summary = run_clinical_eval(
        manifest_path,
        args.norm_params.resolve(),
        output_dir,
        frame_tag=args.frame_tag,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
