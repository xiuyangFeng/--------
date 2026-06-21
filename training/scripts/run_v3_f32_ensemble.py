"""TODO-32: 3-seed best_wss 预测平均 ensemble（0 重训，需 predictions_test_best_wss/*.pt）。"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..core.utils import ensure_dir
from ._figure_utils import load_json, load_manifest, load_prediction_payload, save_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = [
    "outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed1_20260522_124946",
    "outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed2_20260523_124511",
    "outputs/field/field_v3_pointnext_localpool_main01_geom_pw_asymw_a_wall13000_near2000_split_AG_v1_seed3_20260523_124511",
]
BASELINE_SEED1_R2 = 0.39896678924560547
BASELINE_SEED1_TOP10_DICE = 0.22817533536824156


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-20:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _topk_dice(y_true: np.ndarray, y_pred: np.ndarray, frac: float) -> float:
    n = y_true.size
    if n == 0:
        return float("nan")
    k = max(1, int(round(n * frac)))
    t_idx = set(int(i) for i in np.argpartition(y_true, -k)[-k:])
    p_idx = set(int(i) for i in np.argpartition(y_pred, -k)[-k:])
    inter = len(t_idx & p_idx)
    return 2.0 * inter / (len(t_idx) + len(p_idx))


def _sample_key(item: Dict) -> str:
    case_name = str(item.get("case_name", ""))
    sample_id = str(item.get("sample_id", ""))
    if not case_name or not sample_id:
        raise ValueError(f"manifest item 缺少 case_name/sample_id: {item}")
    return f"{case_name}::{sample_id}"


def _wall_magnitude(payload: Dict) -> Tuple[np.ndarray, np.ndarray]:
    from ..analysis.regional_eval import build_region_masks, load_node_features_for_region_masks

    y_true = payload["y_wss_true"].detach().cpu().numpy()[:, 0].astype(np.float64)
    y_pred = payload["y_wss_pred"].detach().cpu().numpy()[:, 0].astype(np.float64)
    features = load_node_features_for_region_masks(payload)
    wall = build_region_masks(features)["wall"].detach().cpu().numpy().astype(bool)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    valid = wall & finite
    return y_true[valid], y_pred[valid]


def _load_run_manifests(run_dirs: Sequence[Path]) -> List[Tuple[int, Path, List[Dict]]]:
    out = []
    for run_dir in run_dirs:
        summary = load_json(run_dir / "summary.json")
        seed = int(summary["seed"])
        manifest_path = run_dir / "predictions_test_best_wss" / "manifest.json"
        manifest = load_manifest(manifest_path)
        items = manifest.get("items") or []
        out.append((seed, manifest_path, items))
    return out


def _aggregate_predictions(
    paths: Sequence[Path],
) -> Tuple[np.ndarray, np.ndarray]:
    payloads = [load_prediction_payload(p) for p in paths]
    y_true, _ = _wall_magnitude(payloads[0])
    preds = np.stack([_wall_magnitude(p)[1] for p in payloads], axis=0)
    if preds.shape[1:] != y_true.shape:
        raise ValueError("ensemble 样本壁面点数不一致")
    y_pred = preds.mean(axis=0)
    return y_true, y_pred


def _single_seed_metrics(run_dir: Path, top_fracs: Sequence[float]) -> Dict[str, object]:
    summary = load_json(run_dir / "summary.json")
    seed = int(summary["seed"])
    test_best = summary.get("test_metrics_best_wss") or {}
    manifest = load_manifest(run_dir / "predictions_test_best_wss" / "manifest.json")
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    dice_acc: Dict[str, List[float]] = {f"top{int(round(f * 100))}": [] for f in top_fracs}
    for item in manifest.get("items") or []:
        pred_path = Path(str(item["prediction_path"]))
        if not pred_path.is_file():
            pred_path = run_dir / "predictions_test_best_wss" / pred_path.name
        payload = load_prediction_payload(pred_path)
        yt, yp = _wall_magnitude(payload)
        all_true.append(yt)
        all_pred.append(yp)
        for frac in top_fracs:
            dice_acc[f"top{int(round(frac * 100))}"].append(_topk_dice(yt, yp, frac))
    yt_all = np.concatenate(all_true)
    yp_all = np.concatenate(all_pred)
    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "summary_wss_r2_wss": test_best.get("wss_r2_wss"),
        "predict_wall_wss_r2_wss": _r2(yt_all, yp_all),
        "topk_dice_mean": {k: float(np.nanmean(v)) for k, v in dice_acc.items()},
        "n_samples": len(all_true),
        "n_wall_points": int(yt_all.size),
    }


def run_ensemble(
    run_dirs: Sequence[Path],
    output_dir: Path,
    top_fracs: Sequence[float] = (0.05, 0.10),
) -> Dict[str, object]:
    loaded = _load_run_manifests(run_dirs)
    n_seeds = len(loaded)
    seed_ids = [seed for seed, _, _ in loaded]

    by_key: Dict[str, Dict[int, Path]] = defaultdict(dict)
    for seed, _manifest_path, items in loaded:
        for item in items:
            key = _sample_key(item)
            pred_path = Path(str(item["prediction_path"]))
            if not pred_path.is_file():
                raise FileNotFoundError(f"seed{seed} 缺少预测: {pred_path}")
            by_key[key][seed] = pred_path

    common_keys = sorted(k for k, mapping in by_key.items() if len(mapping) == n_seeds)
    if not common_keys:
        raise SystemExit("三 seed 无共同 case_name::sample_id；请先完成 predict_field。")

    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    dice_rows: List[Dict[str, object]] = []

    for key in common_keys:
        paths = [by_key[key][seed] for seed in seed_ids]
        y_true, y_pred = _aggregate_predictions(paths)
        all_true.append(y_true)
        all_pred.append(y_pred)
        case_name, sample_id = key.split("::", 1)
        row = {
            "case_name": case_name,
            "sample_id": sample_id,
            "sample_key": key,
            "n_seeds": len(paths),
            "n_wall": int(y_true.size),
        }
        for frac in top_fracs:
            row[f"top{int(round(frac * 100))}_dice"] = _topk_dice(y_true, y_pred, frac)
        dice_rows.append(row)

    yt = np.concatenate(all_true)
    yp = np.concatenate(all_pred)
    topk_mean = {
        f"top{int(round(frac * 100))}": float(np.nanmean([r[f"top{int(round(frac * 100))}_dice"] for r in dice_rows]))
        for frac in top_fracs
    }
    ensemble_r2 = _r2(yt, yp)
    per_seed = [_single_seed_metrics(run_dir, top_fracs) for run_dir in run_dirs]
    seed_r2_mean = float(np.mean([s["predict_wall_wss_r2_wss"] for s in per_seed]))

    summary = {
        "label": "V3P-F32-AsymW-a-ensemble",
        "align_key": "case_name::sample_id",
        "n_samples": len(common_keys),
        "n_wall_points": int(yt.size),
        "seeds": seed_ids,
        "run_dirs": [str(r) for r in run_dirs],
        "wss_r2_wss": ensemble_r2,
        "topk_dice_mean": topk_mean,
        "per_seed": per_seed,
        "comparison_vs_seed1_4957": {
            "baseline_wss_r2_wss": BASELINE_SEED1_R2,
            "baseline_top10_dice": BASELINE_SEED1_TOP10_DICE,
            "ensemble_delta_r2": ensemble_r2 - BASELINE_SEED1_R2,
            "ensemble_delta_top10_dice": topk_mean.get("top10", float("nan")) - BASELINE_SEED1_TOP10_DICE,
            "per_seed_mean_r2": seed_r2_mean,
            "per_seed_mean_delta_r2": seed_r2_mean - BASELINE_SEED1_R2,
        },
        "go_no_go": {
            "ensemble_r2_go_gt_042": ensemble_r2 > 0.42,
            "ensemble_r2_noise_vs_4957": abs(ensemble_r2 - BASELINE_SEED1_R2) <= 0.005,
            "ensemble_top10_dice_improved": topk_mean.get("top10", 0.0) > BASELINE_SEED1_TOP10_DICE,
            "verdict": (
                "marginal_gain"
                if topk_mean.get("top10", 0.0) > BASELINE_SEED1_TOP10_DICE
                and abs(ensemble_r2 - BASELINE_SEED1_R2) <= 0.01
                else "no_go"
            ),
        },
        "per_sample_dice": dice_rows,
        "note": "壁面 |WSS| 点级 R² + 每样本 top-k Dice；对齐键必须为 case_name::sample_id",
    }
    ensure_dir(output_dir)
    save_json(output_dir / "f32_ensemble_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="AsymW-a 三 seed 预测 ensemble（TODO-32）")
    parser.add_argument("--run-dir", action="append", default=[], help="run 目录，可重复；默认 AsymW 三 seed")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "field" / "f0_decision" / "f32_ensemble")
    args = parser.parse_args()

    run_dirs = [REPO_ROOT / p for p in (args.run_dir or DEFAULT_RUNS)]
    summary = run_ensemble(run_dirs, args.output_dir.resolve())
    print(json.dumps({
        "wss_r2_wss": summary["wss_r2_wss"],
        "topk_dice_mean": summary["topk_dice_mean"],
        "n_samples": summary["n_samples"],
        "comparison": summary["comparison_vs_seed1_4957"],
        "verdict": summary["go_no_go"]["verdict"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
