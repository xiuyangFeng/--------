#!/usr/bin/env python3
"""V3P M-E oracle 补跑：O9 全 81 帧 · O3 input→POD 系数 · O11 val 校准。

前置：I6-diag run 须已有
  - predictions_test_best_wss/manifest.json
  - predictions_val_best_wss/manifest.json（GPU predict_field --subset val）

产物：``outputs/field/f0_decision/v3p_me_oracle_followup_<date>.json``

用法::

    python -m training.scripts.run_v3_me_oracle_followup
    python -m training.scripts.run_v3_me_oracle_followup --skip-o11  # val 预测未完成时
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.splits import SplitSpec
from ..core.utils import ensure_dir
from ._figure_utils import load_manifest, load_prediction_payload, save_json
from .run_v3_f0_decision import (
    DEFAULT_NORM_PARAMS,
    REPO_ROOT,
    _denorm_wss_mag,
    _load_graph,
    _load_norm_stats,
    _r2_score,
)
from .run_v3_i_oracle_bundle import _pod_coeff_predictability
from .run_v3_me_oracle_wave import (
    DEFAULT_REFERENCE_RUN,
    _graph_global_cond,
    _mean_finite,
    _safe_json,
    _verdict,
)

NODE_IDX = {
    "is_wall": 9,
    "Abscissa": 3,
    "NormRadius": 4,
}


def _wall_wss_phys(data, stats: Mapping[str, Dict[str, float]]) -> np.ndarray:
    x = data.x.numpy()
    wall = x[:, NODE_IDX["is_wall"]] > 0.5
    y_wss = data.y_wss.numpy()
    return _denorm_wss_mag(y_wss[wall, 0], stats)


def _frame_index(sample_id: str) -> int:
    m = re.search(r"merged-(\d+)", sample_id)
    return int(m.group(1)) if m else 0


def oracle_o9_full_frames_temporal(
    split: SplitSpec,
    data_root: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    graphs_subdir: str,
    max_frames_per_case: int,
    n_harmonics: int,
    holdout_frac: float,
    min_frames: int,
    max_cases: Optional[int],
) -> Dict[str, Any]:
    """E-O · 全帧时间序列 + holdout 谐波重构（避免 in-sample R²=1 伪信号）。"""
    holdout_r2s: List[float] = []
    insample_r2s: List[float] = []
    frame_counts: List[int] = []
    case_names: List[str] = []

    cases = list(split.test_cases)
    if max_cases is not None:
        cases = cases[:max_cases]

    for case_rel in cases:
        case_dir = data_root / case_rel / "processed" / graphs_subdir
        if not case_dir.is_dir():
            continue
        series: List[Tuple[int, float]] = []
        graphs = sorted(case_dir.glob("*.pt"), key=lambda p: _frame_index(p.stem))
        if max_frames_per_case > 0:
            graphs = graphs[:max_frames_per_case]
        for gp in graphs:
            data = _load_graph(gp)
            wss = _wall_wss_phys(data, stats)
            series.append((_frame_index(gp.stem), float(np.mean(wss))))
        if len(series) < min_frames:
            continue
        series.sort(key=lambda t: t[0])
        y = np.array([v for _, v in series], dtype=np.float64)
        n = y.size
        n_train = max(int(math.floor(n * (1.0 - holdout_frac))), min_frames // 2)
        n_train = min(n_train, n - max(4, n // 5))
        if n_train < 4 or n - n_train < 4:
            continue

        def _design(length: int) -> np.ndarray:
            t = np.arange(length, dtype=np.float64)
            cols = [np.ones(length)]
            for k in range(1, n_harmonics + 1):
                w = 2 * math.pi * k / max(length, 1)
                cols.append(np.sin(w * t))
                cols.append(np.cos(w * t))
            return np.column_stack(cols)

        X_tr = _design(n_train)
        y_tr = y[:n_train]
        coef, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        insample_r2s.append(_r2_score(y_tr, X_tr @ coef))

        X_ho = _design(n - n_train)
        y_ho = y[n_train:]
        # 用 train 系数在外推时间轴上预测 holdout（相位按全局 frame index 续接）
        t_ho = np.arange(n_train, n, dtype=np.float64)
        cols = [np.ones(n - n_train)]
        for k in range(1, n_harmonics + 1):
            w = 2 * math.pi * k / max(n, 1)
            cols.append(np.sin(w * t_ho))
            cols.append(np.cos(w * t_ho))
        X_ho_phase = np.column_stack(cols)
        pred_ho = X_ho_phase @ coef
        holdout_r2s.append(_r2_score(y_ho, pred_ho))
        frame_counts.append(n)
        case_names.append(case_rel)

    mean_holdout = _mean_finite(holdout_r2s)
    mean_insample = _mean_finite(insample_r2s)
    go = mean_holdout is not None and mean_holdout >= 0.90
    return {
        "direction": "E-O",
        "mode": "full_frames_holdout",
        "max_frames_per_case": max_frames_per_case,
        "n_harmonics": n_harmonics,
        "holdout_frac": holdout_frac,
        "min_frames": min_frames,
        "n_cases": len(holdout_r2s),
        "mean_frames_per_case": _safe_json(float(np.mean(frame_counts)) if frame_counts else None),
        "mean_insample_r2": _safe_json(mean_insample),
        "mean_holdout_r2": _safe_json(mean_holdout),
        "per_case_holdout_r2": [_safe_json(v) for v in holdout_r2s],
        "per_case_frames": frame_counts,
        "cases": case_names,
        "wave5815_reference": {"mean_temporal_recon_r2": 1.0, "note": "9 frames in-sample overfit"},
        "go_thresholds": {"holdout_r2_min": 0.90},
        "verdict": _verdict(bool(go)),
        "recommendation": (
            "Go → 时间系数/Fourier 建模 worth 1 GPU 探针；"
            "No-Go → 逐帧独立可接受 / 时间低秩不成立"
        ),
    }


def oracle_o3_pod_input_coeff(
    split_path: Path,
    stats: Mapping[str, Dict[str, float]],
    *,
    grid_s: int,
    n_sectors: int,
    n_modes: int,
    max_graphs_per_case: int,
) -> Dict[str, Any]:
    """E-I/E-C · 几何/BC → POD 系数可预测性（补 wave5815 缺失项）。"""
    rep = _pod_coeff_predictability(
        split_path, stats,
        grid_s=grid_s, n_sectors=n_sectors, n_modes=n_modes,
        max_graphs_per_case=max_graphs_per_case,
    )
    if "error" in rep:
        return {
            "direction": "E-I/E-C",
            **rep,
            "verdict": "no_go",
            "recommendation": "数据不足，无法 probe",
        }
    coef_mean = float(rep.get("coef_r2_mean") or 0.0)
    recon_mean = float(rep.get("recon_field_r2_mean") or 0.0)
    go = coef_mean >= 0.15 and recon_mean >= 0.20
    weak = coef_mean >= 0.05 and recon_mean >= 0.15
    return {
        "direction": "E-I/E-C",
        "probe": rep,
        "coef_r2_mean": rep.get("coef_r2_mean"),
        "recon_field_r2_mean": rep.get("recon_field_r2_mean"),
        "gate_coef_predictable": rep.get("gate_coef_predictable"),
        "gate_recon_from_input": rep.get("gate_recon_from_input"),
        "go_thresholds": {"coef_r2_mean": 0.15, "recon_field_r2_mean": 0.20},
        "verdict": _verdict(go, weak=weak and not go),
        "recommendation": (
            "Go → POD/低秩系数回归 GPU 探针；"
            "weak → 与 I2-PC 同级观察；No-Go → 表示换轨仍靠 E-A"
        ),
    }


def _collect_manifest_wss(
    manifest_path: Path,
    case_filter: Optional[set],
    limit: int,
) -> Tuple[np.ndarray, np.ndarray]:
    manifest = load_manifest(manifest_path)
    yt_list: List[np.ndarray] = []
    yp_list: List[np.ndarray] = []
    for item in manifest.get("items") or []:
        if len(yt_list) >= limit:
            break
        case_name = str(item.get("case_name", ""))
        if case_filter is not None and case_name not in case_filter:
            continue
        pred_path = REPO_ROOT / item["prediction_path"]
        if not pred_path.is_file():
            continue
        payload = load_prediction_payload(pred_path)
        t = payload["y_wss_true"].detach().cpu().numpy()[:, 0].astype(np.float64)
        p = payload["y_wss_pred"].detach().cpu().numpy()[:, 0].astype(np.float64)
        m = np.isfinite(t) & np.isfinite(p)
        if int(np.sum(m)) < 10:
            continue
        yt_list.append(t[m])
        yp_list.append(p[m])
    if not yt_list:
        return np.empty(0), np.empty(0)
    return np.concatenate(yt_list), np.concatenate(yp_list)


def oracle_o11_calibration_val_test(
    val_manifest: Path,
    test_manifest: Path,
    split: SplitSpec,
    *,
    max_val_items: int,
    max_test_items: int,
) -> Dict[str, Any]:
    """E-Q · val 拟合 isotonic · test 评一次。"""
    from sklearn.isotonic import IsotonicRegression

    if not val_manifest.is_file():
        return {"direction": "E-Q", "error": "missing_val_manifest", "path": str(val_manifest), "verdict": "no_go"}
    if not test_manifest.is_file():
        return {"direction": "E-Q", "error": "missing_test_manifest", "path": str(test_manifest), "verdict": "no_go"}

    val_cases = set(split.val_cases)
    test_cases = set(split.test_cases)
    yv_t, yv_p = _collect_manifest_wss(val_manifest, val_cases, max_val_items)
    yt_t, yt_p = _collect_manifest_wss(test_manifest, test_cases, max_test_items)

    if yv_t.size < 500:
        return {
            "direction": "E-Q",
            "error": "insufficient_val",
            "n_val_points": int(yv_t.size),
            "verdict": "no_go",
        }
    if yt_t.size < 500:
        return {
            "direction": "E-Q",
            "error": "insufficient_test",
            "n_test_points": int(yt_t.size),
            "verdict": "no_go",
        }

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(yv_p, yv_t)
    cal = iso.predict(yt_p)
    r2_raw = _r2_score(yt_t, yt_p)
    r2_cal = _r2_score(yt_t, cal)
    rmse_raw = float(np.sqrt(np.mean((yt_t - yt_p) ** 2)))
    rmse_cal = float(np.sqrt(np.mean((yt_t - cal) ** 2)))
    rmse_drop = (rmse_raw - rmse_cal) / max(rmse_raw, 1e-20)
    delta_r2 = float(r2_cal) - float(r2_raw)
    spearman_raw = float(np.corrcoef(yt_t, yt_p)[0, 1]) if np.std(yt_p) > 1e-12 else float("nan")

    go = rmse_drop >= 0.10 or delta_r2 >= 0.05
    return {
        "direction": "E-Q",
        "val_manifest": str(val_manifest),
        "test_manifest": str(test_manifest),
        "n_val_points": int(yv_t.size),
        "n_test_points": int(yt_t.size),
        "test_r2_raw": _safe_json(r2_raw),
        "test_r2_calibrated": _safe_json(r2_cal),
        "test_rmse_raw": _safe_json(rmse_raw),
        "test_rmse_calibrated": _safe_json(rmse_cal),
        "rmse_relative_drop": _safe_json(rmse_drop),
        "delta_r2": _safe_json(delta_r2),
        "test_spearman_raw": _safe_json(spearman_raw),
        "go_thresholds": {"rmse_drop_min": 0.10, "or_delta_r2": 0.05},
        "verdict": _verdict(bool(go)),
        "recommendation": (
            "Go → 幅值标定/E-G  worth 探针；No-Go → 空间模式瓶颈（对照 O10）"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V3P M-E oracle followup O9/O3/O11")
    ap.add_argument("--split", type=Path, default=REPO_ROOT / "training" / "splits" / "split_AG_v1.json")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data_new" / "AG")
    ap.add_argument("--graphs-subdir", default="graphs")
    ap.add_argument("--norm-params", type=Path, default=DEFAULT_NORM_PARAMS)
    ap.add_argument("--reference-run-dir", type=Path, default=REPO_ROOT / DEFAULT_REFERENCE_RUN)
    ap.add_argument("--output-name", default="")
    ap.add_argument("--max-frames-per-case", type=int, default=81)
    ap.add_argument("--max-cases-temporal", type=int, default=16)
    ap.add_argument("--n-harmonics", type=int, default=4)
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    ap.add_argument("--min-frames", type=int, default=40)
    ap.add_argument("--grid-s", type=int, default=64)
    ap.add_argument("--n-sectors", type=int, default=8)
    ap.add_argument("--n-modes", type=int, default=20)
    ap.add_argument("--max-graphs-per-case", type=int, default=3)
    ap.add_argument("--max-pred-items", type=int, default=2000)
    ap.add_argument("--skip-o9", action="store_true")
    ap.add_argument("--skip-o3", action="store_true")
    ap.add_argument("--skip-o11", action="store_true")
    args = ap.parse_args()

    split_path = args.split.resolve()
    split = SplitSpec.from_json(split_path)
    stats = _load_norm_stats(args.norm_params.resolve())
    ref_run = args.reference_run_dir.resolve()

    oracles: Dict[str, Any] = {}
    if not args.skip_o9:
        oracles["O9_full"] = oracle_o9_full_frames_temporal(
            split, args.data_root.resolve(), stats,
            graphs_subdir=args.graphs_subdir,
            max_frames_per_case=args.max_frames_per_case,
            n_harmonics=args.n_harmonics,
            holdout_frac=args.holdout_frac,
            min_frames=args.min_frames,
            max_cases=args.max_cases_temporal,
        )
    if not args.skip_o3:
        oracles["O3_input_coeff"] = oracle_o3_pod_input_coeff(
            split_path, stats,
            grid_s=args.grid_s, n_sectors=args.n_sectors, n_modes=args.n_modes,
            max_graphs_per_case=args.max_graphs_per_case,
        )
    if not args.skip_o11:
        val_m = ref_run / "predictions_val_best_wss" / "manifest.json"
        test_m = ref_run / "predictions_test_best_wss" / "manifest.json"
        oracles["O11_calib"] = oracle_o11_calibration_val_test(
            val_m, test_m, split,
            max_val_items=args.max_pred_items,
            max_test_items=args.max_pred_items,
        )

    go_list = [k for k, v in oracles.items() if v.get("verdict") == "go"]
    weak_list = [k for k, v in oracles.items() if v.get("verdict") == "weak_no_go"]

    report = {
        "label": "V3P-M-E-oracle-followup",
        "date": date.today().isoformat(),
        "motivation": "补 wave5815：O9 全81帧holdout · O3 input→POD系数 · O11 val/test校准",
        "reference_run_dir": str(ref_run),
        "prior_oracle": "outputs/field/f0_decision/v3p_me_oracle_wave_20260626.json",
        "oracles": oracles,
        "summary": {
            "go": go_list,
            "weak_no_go": weak_list,
            "no_go": [k for k in oracles if k not in go_list and k not in weak_list],
            "next_hop": (
                "任一条 Go → 对应方向单命题 GPU config；"
                "全 No-Go → G5 叙事收口"
            ),
        },
    }

    out_name = args.output_name or f"v3p_me_oracle_followup_{date.today().strftime('%Y%m%d')}.json"
    out_path = REPO_ROOT / "outputs" / "field" / "f0_decision" / out_name
    ensure_dir(out_path.parent)
    save_json(out_path, report)
    print(f"已写入: {out_path}")
    print(f"Go: {go_list} · weak: {weak_list}")


if __name__ == "__main__":
    main()
