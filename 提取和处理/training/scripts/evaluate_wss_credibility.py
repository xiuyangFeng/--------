"""Direct WSS-head credibility evaluation for one prediction manifest.

This script evaluates ``y_wss_pred`` against ``y_wss_true`` exported by
``predict_field``. It is intentionally separate from ``export_hemo`` because
V3 predicts WSS with a dedicated head; deriving WSS from the predicted velocity
field answers a different question.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from ..analysis.regional_eval import build_region_masks, load_node_features_for_region_masks
from ..core.utils import ensure_dir
from ._figure_utils import load_manifest, load_prediction_payload, maybe_subsample, save_json


def _safe_float(value: float) -> float | None:
    if np.isfinite(value):
        return float(value)
    return None


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-20:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank implementation with deterministic tie handling."""
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
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
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


def _basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float | None]:
    diff = y_pred.astype(np.float64) - y_true.astype(np.float64)
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}n": int(y_true.size),
        f"{p}rmse": _safe_float(float(np.sqrt(np.mean(diff ** 2)))) if y_true.size else None,
        f"{p}mae": _safe_float(float(np.mean(np.abs(diff)))) if y_true.size else None,
        f"{p}r2": _safe_float(_r2_score(y_true, y_pred)) if y_true.size else None,
        f"{p}pearson": _safe_float(_corr(y_true, y_pred)) if y_true.size else None,
        f"{p}spearman": _safe_float(_corr(y_true, y_pred, rank=True)) if y_true.size else None,
    }


def _topk_overlap(y_true: np.ndarray, y_pred: np.ndarray, frac: float) -> Dict[str, float | None]:
    n = int(y_true.size)
    if n == 0:
        return {
            "k": 0,
            "overlap": None,
            "precision": None,
            "recall": None,
            "dice": None,
            "jaccard": None,
        }
    k = max(1, int(round(n * frac)))
    true_idx = np.argpartition(y_true, -k)[-k:]
    pred_idx = np.argpartition(y_pred, -k)[-k:]
    true_set = set(int(i) for i in true_idx)
    pred_set = set(int(i) for i in pred_idx)
    inter = len(true_set & pred_set)
    union = len(true_set | pred_set)
    precision = inter / len(pred_set) if pred_set else float("nan")
    recall = inter / len(true_set) if true_set else float("nan")
    dice = 2.0 * inter / (len(true_set) + len(pred_set)) if true_set or pred_set else float("nan")
    jaccard = inter / union if union else float("nan")
    return {
        "k": k,
        "overlap": inter,
        "precision": _safe_float(precision),
        "recall": _safe_float(recall),
        "dice": _safe_float(dice),
        "jaccard": _safe_float(jaccard),
    }


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


def _plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("生成图像失败：当前环境缺少 matplotlib。") from exc

    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    pad = (hi - lo) * 0.04 if hi > lo else 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(y_true, y_pred, gridsize=70, cmap="Blues", mincnt=1)
    fig.colorbar(hb, ax=ax, label="count")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", linewidth=1.2)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("CFD WSS magnitude")
    ax.set_ylabel("Predicted WSS magnitude")
    ax.set_title(title)
    metrics = _basic_metrics(y_true, y_pred, prefix="wss_mag")
    ax.text(
        0.05,
        0.95,
        f"R2={metrics['wss_mag_r2']:.3f}\nRMSE={metrics['wss_mag_rmse']:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_case_scatter(rows: Sequence[Dict[str, object]], key: str, save_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("生成图像失败：当前环境缺少 matplotlib。") from exc

    xt = np.asarray([float(r[f"{key}_true"]) for r in rows], dtype=np.float64)
    yp = np.asarray([float(r[f"{key}_pred"]) for r in rows], dtype=np.float64)
    lo = float(min(np.min(xt), np.min(yp)))
    hi = float(max(np.max(xt), np.max(yp)))
    pad = (hi - lo) * 0.04 if hi > lo else 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xt, yp, s=28, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "r--", linewidth=1.2)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_xlabel(f"CFD case {key}")
    ax.set_ylabel(f"Predicted case {key}")
    ax.set_title(f"Case-level WSS {key}")
    ax.text(
        0.05,
        0.95,
        f"Pearson={_corr(xt, yp):.3f}\nSpearman={_corr(xt, yp, rank=True):.3f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_overlap_bars(rows: Sequence[Dict[str, object]], save_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("生成图像失败：当前环境缺少 matplotlib。") from exc

    labels = [str(r["top_fraction"]) for r in rows]
    dice = [float(r["mean_dice"]) for r in rows]
    jaccard = [float(r["mean_jaccard"]) for r in rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, dice, width, label="Dice", color="#3182bd")
    ax.bar(x + width / 2, jaccard, width, label="Jaccard", color="#31a354")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_xlabel("True/predicted high-WSS fraction")
    ax.set_ylabel("Overlap score")
    ax.set_title("High-WSS region overlap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _summarize_overlap(rows: Sequence[Dict[str, object]], top_fracs: Sequence[float]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for frac in top_fracs:
        prefix = f"top{int(round(frac * 100))}"
        dice = np.asarray([float(r[f"{prefix}_dice"]) for r in rows], dtype=np.float64)
        jaccard = np.asarray([float(r[f"{prefix}_jaccard"]) for r in rows], dtype=np.float64)
        precision = np.asarray([float(r[f"{prefix}_precision"]) for r in rows], dtype=np.float64)
        recall = np.asarray([float(r[f"{prefix}_recall"]) for r in rows], dtype=np.float64)
        out.append(
            {
                "top_fraction": frac,
                "n_samples": len(rows),
                "mean_dice": _safe_float(float(np.nanmean(dice))),
                "mean_jaccard": _safe_float(float(np.nanmean(jaccard))),
                "mean_precision": _safe_float(float(np.nanmean(precision))),
                "mean_recall": _safe_float(float(np.nanmean(recall))),
            }
        )
    return out


def _mean_metric(rows: Sequence[Dict[str, object]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
    if not vals:
        return None
    return _safe_float(float(np.mean(vals)))


def _find_overlap_metric(rows: Sequence[Dict[str, object]], frac: float, key: str) -> float | None:
    for row in rows:
        if abs(float(row.get("top_fraction", -1.0)) - frac) < 1e-9:
            value = row.get(key)
            return float(value) if value is not None else None
    return None


def _resolve_prediction_path(raw_path: object, manifest_dir: Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_file():
        return path.resolve()
    by_name = manifest_dir / path.name
    if by_name.is_file():
        return by_name.resolve()
    return path.resolve()


def _iter_prediction_payloads(items: Iterable[Dict[str, object]], manifest_dir: Path):
    for item in items:
        prediction_path = _resolve_prediction_path(item["prediction_path"], manifest_dir)
        payload = load_prediction_payload(prediction_path)
        yield item, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 WSS head 直接输出的可信性指标")
    parser.add_argument("--manifest", required=True, help="predict_field.py 导出的 manifest.json")
    parser.add_argument("--output-dir", default="", help="输出目录，默认 manifest 同级 wss_direct")
    parser.add_argument("--max-scatter-points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--top-frac",
        action="append",
        type=float,
        default=[],
        help="高 WSS 区域比例，可重复传入；默认 0.05 和 0.10",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    items = manifest.get("items", [])
    if not isinstance(items, list):
        raise SystemExit("manifest.json 缺少合法的 items 列表")

    output_dir = ensure_dir(Path(args.output_dir).resolve()) if args.output_dir else ensure_dir(manifest_path.parent / "wss_direct")
    top_fracs = args.top_frac or [0.05, 0.10]

    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    by_case_true: Dict[str, List[np.ndarray]] = defaultdict(list)
    by_case_pred: Dict[str, List[np.ndarray]] = defaultdict(list)
    by_region_true: Dict[str, List[np.ndarray]] = defaultdict(list)
    by_region_pred: Dict[str, List[np.ndarray]] = defaultdict(list)
    sample_rows: List[Dict[str, object]] = []

    for item, payload in _iter_prediction_payloads(items, manifest_path.parent):
        if "y_wss_true" not in payload or "y_wss_pred" not in payload:
            raise SystemExit("预测文件缺少 y_wss_true/y_wss_pred；请确认该 run 使用 WSS head 并重新 predict_field。")
        y_true = payload["y_wss_true"].detach().cpu().numpy()
        y_pred = payload["y_wss_pred"].detach().cpu().numpy()
        target_names = list(payload.get("wss_target_names") or ["wss", "wss_x", "wss_y", "wss_z"])
        mag_idx = target_names.index("wss") if "wss" in target_names else 0
        y_true = y_true[:, mag_idx].astype(np.float64)
        y_pred = y_pred[:, mag_idx].astype(np.float64)

        features = load_node_features_for_region_masks(payload)
        masks = build_region_masks(features)
        wall_mask = masks["wall"].detach().cpu().numpy().astype(bool)
        if not wall_mask.any():
            continue
        yt_wall = y_true[wall_mask]
        yp_wall = y_pred[wall_mask]
        case_name = str(item.get("case_name", payload.get("case_name", "unknown")))
        sample_id = str(item.get("sample_id", payload.get("sample_id", "")))

        all_true.append(yt_wall)
        all_pred.append(yp_wall)
        by_case_true[case_name].append(yt_wall)
        by_case_pred[case_name].append(yp_wall)

        row: Dict[str, object] = {
            "case_name": case_name,
            "sample_id": sample_id,
            "n_wall": int(yt_wall.size),
            "mean_true": _safe_float(float(np.mean(yt_wall))),
            "mean_pred": _safe_float(float(np.mean(yp_wall))),
            "p95_true": _safe_float(float(np.quantile(yt_wall, 0.95))),
            "p95_pred": _safe_float(float(np.quantile(yp_wall, 0.95))),
            "max_true": _safe_float(float(np.max(yt_wall))),
            "max_pred": _safe_float(float(np.max(yp_wall))),
        }
        row.update(_basic_metrics(yt_wall, yp_wall, prefix="wss_mag"))
        for frac in top_fracs:
            prefix = f"top{int(round(frac * 100))}"
            overlap = _topk_overlap(yt_wall, yp_wall, frac)
            for key, value in overlap.items():
                row[f"{prefix}_{key}"] = value
        sample_rows.append(row)

        for region_name, mask_t in masks.items():
            region_mask = mask_t.detach().cpu().numpy().astype(bool) & wall_mask
            if region_mask.any():
                by_region_true[region_name].append(y_true[region_mask])
                by_region_pred[region_name].append(y_pred[region_mask])

    if not all_true:
        raise SystemExit("未收集到任何壁面 WSS 样本。")

    y_true_all = np.concatenate(all_true, axis=0)
    y_pred_all = np.concatenate(all_pred, axis=0)
    point_metrics = _basic_metrics(y_true_all, y_pred_all, prefix="wss_mag")

    case_rows: List[Dict[str, object]] = []
    for case_name in sorted(by_case_true):
        yt = np.concatenate(by_case_true[case_name], axis=0)
        yp = np.concatenate(by_case_pred[case_name], axis=0)
        row = {
            "case_name": case_name,
            "n_wall_samples": int(yt.size),
            "mean_true": _safe_float(float(np.mean(yt))),
            "mean_pred": _safe_float(float(np.mean(yp))),
            "p95_true": _safe_float(float(np.quantile(yt, 0.95))),
            "p95_pred": _safe_float(float(np.quantile(yp, 0.95))),
            "max_true": _safe_float(float(np.max(yt))),
            "max_pred": _safe_float(float(np.max(yp))),
        }
        row.update(_basic_metrics(yt, yp, prefix="wss_mag"))
        case_rows.append(row)

    case_summary: Dict[str, object] = {"n_cases": len(case_rows)}
    for key in ("mean", "p95", "max"):
        true_vals = np.asarray([float(r[f"{key}_true"]) for r in case_rows], dtype=np.float64)
        pred_vals = np.asarray([float(r[f"{key}_pred"]) for r in case_rows], dtype=np.float64)
        case_summary[f"{key}_rmse"] = _safe_float(float(np.sqrt(np.mean((pred_vals - true_vals) ** 2))))
        case_summary[f"{key}_mae"] = _safe_float(float(np.mean(np.abs(pred_vals - true_vals))))
        case_summary[f"{key}_pearson"] = _safe_float(_corr(true_vals, pred_vals))
        case_summary[f"{key}_spearman"] = _safe_float(_corr(true_vals, pred_vals, rank=True))

    region_rows: List[Dict[str, object]] = []
    for region_name in sorted(by_region_true):
        yt = np.concatenate(by_region_true[region_name], axis=0)
        yp = np.concatenate(by_region_pred[region_name], axis=0)
        row = {"region_name": region_name}
        row.update(_basic_metrics(yt, yp, prefix="wss_mag"))
        region_rows.append(row)

    overlap_summary = _summarize_overlap(sample_rows, top_fracs)

    _write_csv(output_dir / "wss_sample_metrics.csv", sample_rows)
    _write_csv(output_dir / "wss_case_metrics.csv", case_rows)
    _write_csv(output_dir / "wss_region_metrics.csv", region_rows)
    _write_csv(output_dir / "high_wss_overlap_by_sample.csv", sample_rows)
    _write_csv(output_dir / "high_wss_overlap_summary.csv", overlap_summary)
    save_json(output_dir / "wss_point_metrics.json", point_metrics)
    save_json(output_dir / "wss_case_correlation.json", case_summary)
    save_json(output_dir / "high_wss_overlap.json", {"top_fractions": top_fracs, "summary": overlap_summary})

    scatter_true, scatter_pred = maybe_subsample(y_true_all, y_pred_all, args.max_scatter_points, args.seed)
    _plot_scatter(scatter_true, scatter_pred, output_dir / "fig_wss_mag_scatter.png", "WSS magnitude: predicted vs CFD")
    if len(case_rows) >= 2:
        _plot_case_scatter(case_rows, "p95", output_dir / "fig_case_p95_scatter.png")
        _plot_case_scatter(case_rows, "mean", output_dir / "fig_case_mean_scatter.png")
    _plot_overlap_bars(overlap_summary, output_dir / "fig_high_wss_overlap_bar.png")

    summary = {
        "manifest_path": str(manifest_path),
        "num_samples": len(sample_rows),
        "num_cases": len(case_rows),
        "num_wall_points_total": int(y_true_all.size),
        "value_space": "stored WSS target values from y_wss_true/y_wss_pred; use the matching normalization metadata when converting to Pa",
        "point_metrics": point_metrics,
        "case_summary": case_summary,
        "high_wss_overlap": overlap_summary,
        "quick_view": {
            "point_r2": point_metrics.get("wss_mag_r2"),
            "point_rmse": point_metrics.get("wss_mag_rmse"),
            "case_p95_spearman": case_summary.get("p95_spearman"),
            "case_mean_spearman": case_summary.get("mean_spearman"),
            "top10_mean_dice": _find_overlap_metric(overlap_summary, 0.10, "mean_dice"),
        },
        "files": {
            "point_metrics": str(output_dir / "wss_point_metrics.json"),
            "case_metrics": str(output_dir / "wss_case_metrics.csv"),
            "case_correlation": str(output_dir / "wss_case_correlation.json"),
            "region_metrics": str(output_dir / "wss_region_metrics.csv"),
            "high_wss_overlap": str(output_dir / "high_wss_overlap.json"),
            "wss_scatter": str(output_dir / "fig_wss_mag_scatter.png"),
            "case_p95_scatter": str(output_dir / "fig_case_p95_scatter.png"),
            "high_wss_overlap_bar": str(output_dir / "fig_high_wss_overlap_bar.png"),
        },
    }
    save_json(output_dir / "wss_credibility_summary.json", summary)
    print(output_dir / "wss_credibility_summary.json")


if __name__ == "__main__":
    main()
