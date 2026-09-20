#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""完整点云评估（A 路核心）：稀疏训练的模型 -> 推理整条血管所有壁面点。

用法：
  python -m training_wss_min.evaluate --run-dir training_wss_min/runs/<name>
  python -m training_wss_min.evaluate --config training_wss_min/configs/<name>.json

产出（写入 run_dir/eval/）：
- metrics.json           整体 + 分区 + 逐病例指标（val & test）
- per_case_metrics.csv   逐病例一行，便于扫表/画点数-精度曲线
- heatmaps/<case>.png    每个 test 病例：真值 / 预测 / 误差 三联图（峰值收缩期）
`evaluate_partition` 同时被 train.py 用作 val 监控（make_plots=False）。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from . import config as C
from . import dataset as D
from . import metrics as M
from . import surface as S
from .models import build_model


@torch.no_grad()
def predict_case_norm(model, case: Dict, input_features, feat_stats, device: str,
                      cfg: C.ExpConfig | None = None, return_all_channels: bool = False,
                      frame_index: int | None = None) -> np.ndarray:
    """在完整点云上预测（标准化空间）。单例过大 OOM 时回退 CPU。

    return_all_channels=True 时保留多通道输出（如 gaussian_nll 的 [mu, logvar]），默认只返回通道 0。"""
    if cfg is None or not cfg.eval.fixed_support:
        if "support_pool" in case:
            raise ValueError("volume targets require eval.fixed_support=True (support=wall, query=volume rows)")
    if cfg is not None and cfg.eval.fixed_support:
        support_n = int(cfg.data.support_n_points or cfg.data.wall_n_points)
        support_sampling = cfg.data.support_sampling or cfg.data.sampling
        seed = S.stable_seed(cfg.eval.support_seed, case["unit_id"], "eval_support")
        support_idx = D.sample_support_indices(case, cfg.data, seed, n_points=support_n,
                                               sampling=support_sampling, stream="eval_support")
        rows = D.query_rows(case)
        support_pos = torch.from_numpy(np.ascontiguousarray(case["pos"][support_idx])).to(device)
        support_x = torch.from_numpy(D.build_features(
            case, support_idx, input_features, feat_stats, frame_index=frame_index
        )).to(device)
        support_batch = torch.zeros(len(support_idx), dtype=torch.long, device=device)
        geometry = D.case_geometry(case) if cfg.data.local_geometry else None
        encode_extra = ({"geometry": torch.from_numpy(np.ascontiguousarray(geometry[support_idx])).to(device)}
                        if geometry is not None else {})
        encoded = model.encode_support(
            support_pos, support_x, support_batch, unit_ids=[case["unit_id"]],
            epoch=0, global_seed=cfg.eval.support_seed, evaluation=True,
            **encode_extra,
        )
        chunk = int(cfg.eval.query_chunk_size) or len(rows)
        outputs = []
        for start in range(0, len(rows), chunk):
            idx = rows[start:start + chunk]
            pos = torch.from_numpy(np.ascontiguousarray(case["pos"][idx])).to(device)
            x = torch.from_numpy(D.build_features(case, idx, input_features, feat_stats, frame_index=frame_index)).to(device)
            batch = torch.zeros(len(idx), dtype=torch.long, device=device)
            decode_extra = {}
            if geometry is not None:
                decode_extra["geometry"] = torch.from_numpy(np.ascontiguousarray(geometry[idx])).to(device)
            if cfg.data.query_patch_nsample:
                patch = D.build_query_patch(case, idx, input_features, feat_stats,
                                            nsample=cfg.data.query_patch_nsample, frame_index=frame_index)
                decode_extra["patch"] = {k: torch.as_tensor(v, device=device) for k, v in patch.items()}
            outputs.append(model.decode_query(encoded, pos, x, batch, **decode_extra).detach().float().cpu())
        output = torch.cat(outputs)
        if output.ndim == 2 and not return_all_channels:
            output = output[:, 0]
        return output.numpy()
    batch = D.case_to_batch(case, input_features, feat_stats, device=device)
    try:
        out = model(batch["pos"], batch["x"], batch["batch"])
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        model_cpu = model.to("cpu")
        batch = D.case_to_batch(case, input_features, feat_stats, device="cpu")
        out = model_cpu(batch["pos"], batch["x"], batch["batch"])
        model.to(device)
    out = out.detach().float().cpu()
    if out.ndim == 2 and not return_all_channels:
        out = out[:, 0]
    return out.numpy()


def evaluate_partition(model, cases: List[Dict], cfg: C.ExpConfig, feat_stats: Dict,
                       wss_stats: Dict, device: str,
                       save_dir: Path | None = None, make_plots: bool = False,
                       predictions_dir: Path | None = None) -> Dict:
    """入口：timesteps='peak'（历史行为）直接单帧评估；'random_frame' 按 eval.eval_frames 逐帧评估，
    顶层结果恒为峰值帧（与单帧 run 同口径），另加 frames / pooled_frames / tawss。"""
    if getattr(cfg.data, "timesteps", "peak") != "random_frame":
        return _evaluate_partition_frame(model, cases, cfg, feat_stats, wss_stats, device,
                                         save_dir=save_dir, make_plots=make_plots,
                                         predictions_dir=predictions_dir)
    if predictions_dir is not None:
        raise ValueError("saved same-point predictions currently require peak-frame volume targets")
    return _evaluate_partition_frames(model, cases, cfg, feat_stats, wss_stats, device,
                                      save_dir=save_dir, make_plots=make_plots)


def _parse_eval_frames(spec: str, steps: np.ndarray, peak_index: int) -> List[int]:
    spec = (spec or "peak").strip().lower()
    if spec == "peak":
        return [int(peak_index)]
    if spec == "all":
        return list(range(len(steps)))
    wanted = [int(x) for x in spec.split(",") if x.strip()]
    lookup = {int(st): i for i, st in enumerate(steps)}
    missing = [w for w in wanted if w not in lookup]
    if missing:
        raise ValueError(f"eval_frames steps not in bundle: {missing}")
    return [lookup[w] for w in wanted]


def _compact(res: Dict) -> Dict:
    keep = ("field", "field_casebalanced", "aggregate", "calibration", "regional_field", "group_casebalanced")
    out = {k: res[k] for k in keep if k in res}
    out["hotspot"] = {k: v for k, v in res.get("hotspot", {}).items() if k.endswith("casemean") or k in ("top10_iou",)}
    return out


def _evaluate_partition_frames(model, cases, cfg, feat_stats, wss_stats, device, *, save_dir, make_plots) -> Dict:
    steps = np.asarray(cases[0]["steps"])
    peak_index = int(cases[0]["peak_index"])
    frames = _parse_eval_frames(cfg.eval.eval_frames, steps, peak_index)
    if peak_index not in frames:
        frames = [peak_index] + frames
    per_frame: Dict[str, Dict] = {}
    pred_norm_store: Dict[int, List[np.ndarray]] = {}
    top = None
    for k in frames:
        for case in cases:
            D.select_frame(case, k)
        res = _evaluate_partition_frame(model, cases, cfg, feat_stats, wss_stats, device,
                                        save_dir=save_dir if k == peak_index else None,
                                        make_plots=make_plots and k == peak_index,
                                        frame_index=k, return_predictions=True)
        pred_norm_store[k] = res.pop("_pred_norm_by_case")
        entry = {"step": int(steps[k]), "frame_index": int(k), "physical": _compact(res), "normalized": _compact(res["normalized"])}
        per_frame[str(int(steps[k]))] = entry
        if k == peak_index:
            top = res
    for case in cases:
        D.select_frame(case, peak_index)
    assert top is not None
    # pooled over evaluated frames (case-balanced: each case's frames concatenated) —— 含帧间方差，口径偏高
    frames_sorted = sorted(pred_norm_store)
    true_n = [np.concatenate([np.asarray(c["y_norm_frames"][k], dtype=np.float64) for k in frames_sorted]) for c in cases]
    pred_n = [np.concatenate([pred_norm_store[k][i] for k in frames_sorted]) for i in range(len(cases))]
    true_p = [np.concatenate([np.asarray(c["y_raw_frames"][k], dtype=np.float64) for k in frames_sorted]) for c in cases]
    pred_p = [np.clip(D.denormalize_wss(p, wss_stats), 0, None) for p in pred_n]
    pooled = {
        "n_frames": len(frames_sorted), "steps": [int(steps[k]) for k in frames_sorted],
        "note": "pooled over frames; includes between-frame (waveform-driven) variance, reads higher than per-frame R²",
        "physical": {"field_casebalanced": M.casebalanced_field_metrics(true_p, pred_p), "field": M.basic_metrics(np.concatenate(true_p), np.concatenate(pred_p)),
                     "calibration": M.calibration_metrics(np.concatenate(true_p), np.concatenate(pred_p))},
        "normalized": {"field_casebalanced": M.casebalanced_field_metrics(true_n, pred_n), "field": M.basic_metrics(np.concatenate(true_n), np.concatenate(pred_n))},
    }
    # TAWSS over the evaluated frames (mean |WSS| per node), physical Pa, full scalar suite
    ta_true = [np.mean(np.stack([np.asarray(c["y_raw_frames"][k], dtype=np.float64) for k in frames_sorted]), axis=0) for c in cases]
    ta_pred = [np.mean(np.stack([np.clip(D.denormalize_wss(pred_norm_store[k][i], wss_stats), 0, None) for k in frames_sorted]), axis=0) for i in range(len(cases))]
    tawss = _evaluate_space(cases, ta_true, ta_pred, save_dir=None, make_plots=False, plot_space="TAWSS", include_area=False)
    tawss.pop("per_case", None)
    tawss["n_frames"] = len(frames_sorted)
    tawss["definition"] = "time-average of |WSS| over evaluated frames (all 81 = full cycle); prediction averaged in Pa"
    # per-frame R² curve for quick reading
    curve = [{"step": v["step"], "r2_casebalanced_physical": v["physical"]["field_casebalanced"]["r2"], "r2_casebalanced_normalized": v["normalized"]["field_casebalanced"]["r2"],
              "r2_casemean_physical": v["physical"]["aggregate"]["r2_casemean"], "mae_pa": v["physical"]["field"]["mae"]} for v in sorted(per_frame.values(), key=lambda e: e["step"])]
    top["time_evaluation"] = {"timesteps": "random_frame", "eval_frames": cfg.eval.eval_frames, "top_level_frame_step": int(steps[peak_index]),
                              "n_frames_evaluated": len(frames_sorted)}
    top["frames"] = per_frame
    top["frame_curve"] = curve
    top["pooled_frames"] = pooled
    top["tawss"] = tawss
    return top


def _evaluate_partition_frame(model, cases: List[Dict], cfg: C.ExpConfig, feat_stats: Dict,
                              wss_stats: Dict, device: str,
                              save_dir: Path | None = None, make_plots: bool = False,
                              frame_index: int | None = None, return_predictions: bool = False,
                              predictions_dir: Path | None = None) -> Dict:
    model.eval()
    include_area = cfg.eval.surface_metric_mode == "both_strict"
    for case in cases:
        case.setdefault("unit_id", f"{case.get('cohort', 'AG/unknown')}/{case.get('case', 'unknown')}")
        if include_area and "surface_area_weights" not in case:
            required = {"wall_coords_raw", "original_stl_path", "original_stl_scale_to_mm"}
            missing = sorted(required - set(case))
            if missing:
                raise RuntimeError(
                    f"strict surface-area evaluation requires {missing}: {case['unit_id']}"
                )
            case["surface_area_weights"], case["surface_area_report"] = \
                S.area_weights_for_case(case, strict=True)
    is_joint = cfg.data.target == "velocity_pressure"
    is_vector = cfg.data.target == "velocity"
    pointwise_jensen = bool(getattr(cfg.eval, "pointwise_jensen", False))
    pred_norm_by_case = []
    pred_logvar_by_case = []
    for case in cases:
        y_pred_norm = predict_case_norm(
            model, case, cfg.data.input_features, feat_stats, device, cfg=cfg,
            return_all_channels=is_vector or is_joint or pointwise_jensen, frame_index=frame_index,
        )
        y_pred_norm = np.asarray(y_pred_norm, dtype=np.float64)
        if pointwise_jensen:
            pred_logvar_by_case.append(y_pred_norm[:, 1])
            y_pred_norm = y_pred_norm[:, 0]
        pred_norm_by_case.append(y_pred_norm)

    if predictions_dir is not None:
        if cfg.data.target not in (*C.VOLUME_TARGETS, "wss"):
            raise ValueError("--save-predictions supports WSS and volume targets")
        if cfg.data.target == "wss" and (pointwise_jensen or cfg.data.target_normalization != "global_stats"):
            raise ValueError("saved WSS predictions require global_stats without Jensen correction")
        predictions_dir.mkdir(parents=True, exist_ok=True)
        save_prediction = _save_wss_prediction if cfg.data.target == "wss" else _save_volume_prediction
        entries = [save_prediction(c, p, cfg.data.target, wss_stats, predictions_dir)
                   for c, p in zip(cases, pred_norm_by_case)]
        (predictions_dir / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "target": cfg.data.target, "cases": entries,
        }, indent=2, ensure_ascii=False))
    if cfg.data.target in C.VOLUME_TARGETS:
        # 体场目标：真值/几何都切到 query 行（内部单元，压力混合时含壁面行）
        rows_by_case = [D.query_rows(c) for c in cases]
        cases = [D.subset_case(c, rows) for c, rows in zip(cases, rows_by_case)]
    true_norm_by_case = [np.asarray(c["y_norm"], dtype=np.float64) for c in cases]
    if is_joint:
        return _evaluate_joint(cases, pred_norm_by_case, cfg, wss_stats,
                               save_dir=save_dir, make_plots=make_plots)
    if is_vector:
        return _evaluate_velocity(cases, true_norm_by_case, pred_norm_by_case, cfg, wss_stats,
                                  save_dir=save_dir, make_plots=make_plots)
    normalized = _evaluate_space(
        cases, true_norm_by_case, pred_norm_by_case,
        save_dir=save_dir, make_plots=make_plots, plot_space="normalized target",
        include_area=include_area,
    )
    normalized["normalization"] = {
        "mode": cfg.data.target_normalization,
        "case_metadata": {
            f"{c['cohort']}/{c['case']}": c.get("target_normalization_meta", {})
            for c in cases
        },
        "predictions_clipped_for_metrics": False,
    }
    normalized["surface_metric_mode"] = cfg.eval.surface_metric_mode
    normalized["surface_area_metrics_status"] = (
        "complete_strict" if include_area else "not_requested"
    )

    if cfg.data.target_normalization == "case_max":
        result = dict(normalized)
        result["metric_space"] = "normalized_target"
        result["normalized"] = normalized
        return result

    pred_raw_by_case = []
    for index, pred_norm in enumerate(pred_norm_by_case):
        if pointwise_jensen:
            pred_raw = D.denormalize_wss_lognormal_mean(
                pred_norm, pred_logvar_by_case[index], wss_stats,
                logvar_min=cfg.train.nll_logvar_min, logvar_max=cfg.train.nll_logvar_max,
            )
        else:
            pred_raw = D.denormalize_wss(pred_norm, wss_stats)
        if wss_stats.get("method") == "log_z":
            pred_raw = np.clip(pred_raw, 0, None)
        pred_raw_by_case.append(np.asarray(pred_raw, dtype=np.float64))
    true_raw_by_case = [np.asarray(c["y_raw"], dtype=np.float64) for c in cases]
    if cfg.data.target == "pressure_mixed":
        physical = _evaluate_space(cases, true_raw_by_case, pred_raw_by_case, save_dir=None, make_plots=False,
                                   plot_space="physical target", include_area=False)
        physical["metric_space"] = "physical_target"
        physical["target"] = "pressure_mixed"
        physical["target_units"] = "Pa relative to case volume-mean pressure at peak step"
        physical["surface_metric_mode"] = "volume_query_rows"
        physical["surface_area_metrics_status"] = "not_applicable"
        physical["normalized"] = normalized
        physical["query_groups"] = _query_group_breakdown(cases, true_raw_by_case, pred_raw_by_case,
                                                          true_norm_by_case, pred_norm_by_case)
        return physical
    physical = _evaluate_space(
        cases, true_raw_by_case, pred_raw_by_case,
        save_dir=None, make_plots=False, plot_space="physical target",
        include_area=include_area,
    )
    physical["metric_space"] = "physical_target"
    physical["back_transform"] = (
        "pointwise_lognormal_mean_exp(mu+sigma^2/2)" if pointwise_jensen else "exp(mu)"
    )
    physical["surface_metric_mode"] = cfg.eval.surface_metric_mode
    physical["surface_area_metrics_status"] = (
        "complete_strict" if include_area else "not_requested"
    )
    physical["normalized"] = normalized
    if return_predictions:
        physical["_pred_norm_by_case"] = pred_norm_by_case
    return physical


def _save_wss_prediction(case, pred_norm, target, stats, predictions_dir):
    relative = Path(case["unit_id"]) / "predictions.npz"
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe case unit_id")
    output = predictions_dir / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = D.denormalize_wss(np.asarray(pred_norm, dtype=np.float64), stats)
    clipped = np.clip(raw, 0, None) if stats.get("method") == "log_z" else raw
    np.savez_compressed(output, pred_norm=pred_norm, true_norm=case["y_norm"],
                        pred_pa_unclipped=raw, pred_pa=clipped, true_pa=case["y_raw"],
                        row_index=np.arange(len(pred_norm), dtype=np.int64))
    import hashlib
    return {"unit_id": case["unit_id"], "file": str(relative), "n_points": len(pred_norm),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "metric_clipping": "nonnegative after inverse log_z" if stats.get("method") == "log_z" else "none"}


def denormalize_volume_prediction(pred_norm: np.ndarray, target: str, stats: Dict) -> np.ndarray:
    """Invert the frozen task-wise linear transforms without losing joint channels."""
    pred_norm = np.asarray(pred_norm, dtype=np.float64)
    if target == "velocity":
        return pred_norm * np.asarray(stats["velocity"]["std"]) + np.asarray(stats["velocity"]["mean"])
    if target == "velocity_pressure":
        if pred_norm.ndim != 2 or pred_norm.shape[1] != 4:
            raise ValueError("joint predictions must have four channels [ux,uy,uz,p]")
        velocity = denormalize_volume_prediction(pred_norm[:, :3], "velocity", stats)
        pressure = pred_norm[:, 3] * float(stats["linear"]["std"]) + float(stats["linear"]["mean"])
        return np.column_stack([velocity, pressure])
    if target != "pressure_mixed":
        raise ValueError(f"not a volume target: {target}")
    return pred_norm * float(stats["linear"]["std"]) + float(stats["linear"]["mean"])


def _save_volume_prediction(case, pred_norm, target, stats, predictions_dir):
    rows = np.asarray(D.query_rows(case), dtype=np.int64)
    true_raw = np.asarray(case["y_raw"])[rows]
    pred_raw = denormalize_volume_prediction(pred_norm, target, stats)
    if pred_raw.shape != true_raw.shape or len(rows) != len(pred_norm):
        raise ValueError(f"prediction/label/query shape mismatch: {case['unit_id']}")
    relative = Path(case["unit_id"]) / "predictions.npz"
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unit_id must be a relative canonical case path")
    output = predictions_dir / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, query_idx=rows,
                        point_kind=np.asarray(case["point_kind"])[rows],
                        pred_norm=np.asarray(pred_norm, dtype=np.float32),
                        pred_raw=pred_raw.astype(np.float32), true_raw=true_raw.astype(np.float32),
                        unit_id=np.asarray(case["unit_id"]), target=np.asarray(target),
                        n_wall=np.int64(case["n_wall"]))
    return {"unit_id": case["unit_id"], "file": str(relative), "n_query": len(rows),
            "n_wall": int(case["n_wall"]), "shape": list(pred_raw.shape),
            "bundle_path": str(Path(case["bundle_path"]).resolve())}


def _evaluate_joint(cases, pred_norm_by_case, cfg, stats, *, save_dir, make_plots):
    """One shared inference over the union, followed by task-specific rows/channels."""
    pressure_cases, velocity_cases, pressure_pred, velocity_pred = [], [], [], []
    for case, pred in zip(cases, pred_norm_by_case):
        if pred.ndim != 2 or pred.shape[1] != 4:
            raise ValueError("joint evaluation requires all four prediction channels")
        pressure_case = dict(case)
        pressure_case["y_raw"] = np.asarray(case["y_raw"])[:, 3]
        pressure_case["y_norm"] = np.asarray(case["y_norm"])[:, 3]
        pressure_cases.append(pressure_case)
        pressure_pred.append(pred[:, 3])
        interior = np.flatnonzero(np.asarray(case["point_kind"]) == 1)
        if not len(interior):
            raise ValueError(f"joint case has no interior rows: {case['unit_id']}")
        velocity_case = D.subset_case(case, interior)
        velocity_case["y_raw"] = np.asarray(velocity_case["y_raw"])[:, :3]
        velocity_case["y_norm"] = np.asarray(velocity_case["y_norm"])[:, :3]
        velocity_cases.append(velocity_case)
        velocity_pred.append(pred[interior, :3])
    pn = [np.asarray(c["y_norm"], dtype=np.float64) for c in pressure_cases]
    pt = [np.asarray(c["y_raw"], dtype=np.float64) for c in pressure_cases]
    pp = [denormalize_volume_prediction(p, "pressure_mixed", stats) for p in pressure_pred]
    normalized = _evaluate_space(pressure_cases, pn, pressure_pred,
                                 save_dir=save_dir / "pressure" if save_dir else None,
                                 make_plots=make_plots, plot_space="normalized target", include_area=False)
    normalized["normalization"] = {"mode": cfg.data.target_normalization,
        "case_metadata": {f"{c['cohort']}/{c['case']}": c.get("target_normalization_meta", {}) for c in cases},
        "predictions_clipped_for_metrics": False}
    normalized["surface_metric_mode"] = cfg.eval.surface_metric_mode
    normalized["surface_area_metrics_status"] = "not_requested"
    pressure = _evaluate_space(pressure_cases, pt, pp, save_dir=None, make_plots=False,
                               plot_space="physical target", include_area=False)
    pressure.update(metric_space="physical_target", target="pressure_mixed",
                    target_units="Pa relative to case volume-mean pressure at peak step",
                    surface_metric_mode="volume_query_rows", surface_area_metrics_status="not_applicable",
                    normalized=normalized,
                    query_groups=_query_group_breakdown(pressure_cases, pt, pp, pn, pressure_pred))
    velocity = _evaluate_velocity(velocity_cases, [c["y_norm"] for c in velocity_cases], velocity_pred,
                                  cfg, stats, save_dir=save_dir / "velocity" if save_dir else None,
                                  make_plots=make_plots)
    return {"target": "velocity_pressure", "channel_order": ["ux", "uy", "uz", "p_rel"],
            "shared_inference": "one support encoding per case; full wall/interior query union in chunks",
            "tasks": {"pressure": pressure, "velocity": velocity}}



def _strip_per_case(res: Dict) -> Dict:
    return {k: v for k, v in res.items() if k != "per_case"}


def _query_group_breakdown(cases, true_raw, pred_raw, true_norm, pred_norm) -> Dict[str, Dict]:
    """压力混合目标：按 point_kind（0 壁面 / 1 内部）分别评一遍物理与归一化空间。"""
    out = {}
    for name, kind in (("wall", 0), ("interior", 1)):
        masks = [np.asarray(c["point_kind"]) == kind for c in cases]
        sub_cases = [D.subset_case(c, np.flatnonzero(m)) for c, m in zip(cases, masks)]
        tr = [t[m] for t, m in zip(true_raw, masks)]
        pr = [p[m] for p, m in zip(pred_raw, masks)]
        tn = [t[m] for t, m in zip(true_norm, masks)]
        pn = [p[m] for p, m in zip(pred_norm, masks)]
        phys = _strip_per_case(_evaluate_space(sub_cases, tr, pr, save_dir=None, make_plots=False,
                                               plot_space="physical target", include_area=False))
        norm = _strip_per_case(_evaluate_space(sub_cases, tn, pn, save_dir=None, make_plots=False,
                                               plot_space="normalized target", include_area=False))
        phys["normalized"] = norm
        phys["n_points"] = int(sum(int(m.sum()) for m in masks))
        out[name] = phys
    return out


def _vector_metrics(true_vec, pred_vec, true_norm, pred_norm, speed_floor: float = 0.05) -> Dict:
    """速度向量补充指标：逐分量 R²（病例等权 / pooled）、方向余弦、分量 z 空间 pooled R²。"""
    comp = {}
    for i, name in enumerate(("u", "v", "w")):
        tb = [t[:, i] for t in true_vec]
        pb = [p[:, i] for p in pred_vec]
        comp[name] = {
            "r2_casebalanced": M.casebalanced_field_metrics(tb, pb)["r2"],
            "r2_pooled": M.r2_score(np.concatenate(tb), np.concatenate(pb)),
            "rmse": float(np.sqrt(np.mean((np.concatenate(tb) - np.concatenate(pb)) ** 2))),
        }
    cos_case, cos_w_case = [], []
    for t, p in zip(true_vec, pred_vec):
        st = np.linalg.norm(t, axis=1)
        sp = np.linalg.norm(p, axis=1)
        m = st > speed_floor
        if m.sum() < 10:
            continue
        cos = np.sum(t[m] * p[m], axis=1) / np.clip(st[m] * sp[m], 1e-12, None)
        cos_case.append(float(np.mean(cos)))
        cos_w_case.append(float(np.sum(cos * st[m]) / np.sum(st[m])))
    tn = np.concatenate([t.ravel() for t in true_norm])
    pn = np.concatenate([p.ravel() for p in pred_norm])
    return {
        "components": comp,
        "direction_cosine_casemean": float(np.mean(cos_case)) if cos_case else float("nan"),
        "direction_cosine_speedweighted_casemean": float(np.mean(cos_w_case)) if cos_w_case else float("nan"),
        "speed_floor_m_s": speed_floor,
        "component_normalized_r2_pooled": M.r2_score(tn, pn),
        "vector_rmse_m_s": float(np.sqrt(np.mean(np.sum((np.concatenate(true_vec) - np.concatenate(pred_vec)) ** 2, axis=1)))),
    }


def _evaluate_velocity(cases, true_norm_by_case, pred_norm_by_case, cfg, wss_stats, *, save_dir, make_plots) -> Dict:
    """速度目标：模型出三分量，评估按用户口径只做标量（速度幅值 |u|）评估，向量指标作补充。"""
    mean = np.asarray(wss_stats["velocity"]["mean"], dtype=np.float64)
    std = np.asarray(wss_stats["velocity"]["std"], dtype=np.float64)
    pred_vec = [p * std + mean for p in pred_norm_by_case]
    true_vec = [np.asarray(c["y_raw"], dtype=np.float64) for c in cases]
    speed_true = [np.linalg.norm(t, axis=1) for t in true_vec]
    speed_pred = [np.linalg.norm(p, axis=1) for p in pred_vec]
    sp_mean, sp_std = float(wss_stats["speed"]["mean"]), float(wss_stats["speed"]["std"])
    normalized = _evaluate_space(cases, [(t - sp_mean) / sp_std for t in speed_true],
                                 [(p - sp_mean) / sp_std for p in speed_pred], save_dir=save_dir,
                                 make_plots=make_plots, plot_space="normalized speed", include_area=False)
    normalized["normalization"] = {"mode": "speed_linear_z(train138)", "mean": sp_mean, "std": sp_std,
                                   "note": "affine in speed: R² identical to physical, error units differ"}
    normalized["surface_metric_mode"] = "volume_query_rows"
    normalized["surface_area_metrics_status"] = "not_applicable"
    physical = _evaluate_space(cases, speed_true, speed_pred, save_dir=None, make_plots=False,
                               plot_space="physical speed", include_area=False)
    physical["metric_space"] = "physical_target"
    physical["target"] = "velocity"
    physical["scalar_evaluated"] = "speed |u| (m/s) from predicted components"
    physical["surface_metric_mode"] = "volume_query_rows"
    physical["surface_area_metrics_status"] = "not_applicable"
    physical["normalized"] = normalized
    physical["vector"] = _vector_metrics(true_vec, pred_vec, true_norm_by_case, pred_norm_by_case)
    return physical


def _evaluate_space(cases: List[Dict], true_by_case: List[np.ndarray],
                    pred_by_case: List[np.ndarray], *, save_dir: Path | None,
                    make_plots: bool, plot_space: str,
                    include_area: bool = True) -> Dict:
    per_case: Dict[str, Dict] = {}
    for case, y_true, y_pred in zip(cases, true_by_case, pred_by_case):
        reg = M.regional_metrics(case["pos"], case["local_radius"], y_true, y_pred)
        reg["calibration"] = M.calibration_metrics(y_true, y_pred)
        reg["distribution"] = M.distribution_metrics(y_true, y_pred)
        reg["hotspot"] = M.hotspot_localization_metrics(y_true, y_pred, case["pos"])
        reg["legacy_vertex_hotspot"] = dict(reg["hotspot"])
        if include_area:
            reg["area_overall"] = M.weighted_basic_metrics(
                y_true, y_pred, case["surface_area_weights"]
            )
            reg["area_hotspot"] = M.area_hotspot_metrics(
                y_true, y_pred, case["pos"], case["surface_area_weights"]
            )
        per_case[f"{case['cohort']}/{case['case']}"] = reg
        if make_plots and save_dir is not None:
            _plot_case(case, y_true, y_pred, save_dir / "heatmaps", plot_space)

    if not true_by_case:
        raise ValueError("evaluate_partition received no cases")
    pt = np.concatenate(true_by_case)
    pp = np.concatenate(pred_by_case)
    field = M.basic_metrics(pt, pp)
    field.update(M.linear_fit_metrics(pt, pp))
    field_casebalanced = M.casebalanced_field_metrics(true_by_case, pred_by_case)
    field_casebalanced.update(
        M.casebalanced_linear_fit_metrics(true_by_case, pred_by_case)
    )
    result = {
        "aggregate": M.aggregate_case_metrics(per_case),
        "field": field,
        "field_casebalanced": field_casebalanced,
        "calibration": M.calibration_metrics(pt, pp),
        "distribution": M.distribution_metrics(pt, pp),
        "hotspot": M.aggregate_hotspot_metrics(per_case),
        "regional_field": _regional_field(cases, true_by_case, pred_by_case),
        "group_casebalanced": _group_casebalanced(cases, true_by_case, pred_by_case),
        "per_case": per_case,
    }
    if include_area:
        area_hotspot = {}
        for key in ("area_top10_iou", "area_top10_precision", "area_top10_recall",
                    "area_hotspot_centroid_dist", "high_wss_area_mae",
                    "high_wss_area_rmse", "high_wss_area_r2"):
            values = [v["area_hotspot"].get(key, np.nan) for v in per_case.values()]
            values = np.asarray(values, dtype=np.float64)
            area_hotspot[f"{key}_casemean"] = float(np.nanmean(values))
        result["primary_physical_or_normalized_casebalanced_area"] = \
            M.casebalanced_area_metrics(
                true_by_case, pred_by_case, [c["surface_area_weights"] for c in cases]
            )
        result["area_hotspot"] = area_hotspot
        result["group_casebalanced_area_r2"] = _group_casebalanced_area_r2(
            cases, true_by_case, pred_by_case
        )
    return result


def _group_casebalanced(cases, true_by_case, pred_by_case) -> Dict[str, Dict]:
    """病例等权分域指标；ILO 后缀只作预注册分层，不解释临床语义。"""
    groups = {
        "AG": [], "AAA": [], "AAA_ruputer": [], "AAA_unruputer": [],
        "ILO": [], "ILO_0": [], "ILO_1": [],
    }
    for i, case in enumerate(cases):
        unit = case["unit_id"]
        if unit.startswith("AG/"):
            groups["AG"].append(i)
        elif unit.startswith("AAA/ruputer/"):
            groups["AAA"].append(i); groups["AAA_ruputer"].append(i)
        elif unit.startswith("AAA/unruputer/"):
            groups["AAA"].append(i); groups["AAA_unruputer"].append(i)
        elif unit.startswith("ILO/"):
            groups["ILO"].append(i)
            patient = unit.split("/", 2)[1]
            groups[f"ILO_{patient.rsplit('-', 1)[-1]}"].append(i)
    return {
        name: {
            **M.casebalanced_field_metrics(
                [true_by_case[i] for i in ids], [pred_by_case[i] for i in ids]
            ),
            "n_cases": len(ids),
        }
        for name, ids in groups.items() if ids
    }


def _group_casebalanced_area_r2(cases, true_by_case, pred_by_case) -> Dict[str, float]:
    groups = {"AG": [], "AAA": [], "AAA_ruputer": [], "AAA_unruputer": [],
              "ILO": [], "ILO_0": [], "ILO_1": []}
    for i, case in enumerate(cases):
        unit = case["unit_id"]
        if unit.startswith("AG/"):
            groups["AG"].append(i)
        elif unit.startswith("AAA/ruputer/"):
            groups["AAA"].append(i)
            groups["AAA_ruputer"].append(i)
        elif unit.startswith("AAA/unruputer/"):
            groups["AAA"].append(i)
            groups["AAA_unruputer"].append(i)
        elif unit.startswith("ILO/"):
            groups["ILO"].append(i)
            suffix = unit.split("/", 2)[1].rsplit("-", 1)[-1]
            groups[f"ILO_{suffix}"].append(i)
    out = {}
    for name, ids in groups.items():
        if ids:
            out[name] = M.casebalanced_area_metrics(
                [true_by_case[i] for i in ids], [pred_by_case[i] for i in ids],
                [cases[i]["surface_area_weights"] for i in ids],
            )["r2"]
    if "AG" in out and "AAA" in out:
        out["AG_AAA_domain_macro"] = float((out["AG"] + out["AAA"]) / 2.0)
    return out


def _regional_field(cases, pooled_true, pooled_pred) -> Dict:
    """把所有病例的点按分区 pool 起来算 field 级分区指标。"""
    acc = {"bifurcation": ([], []), "stenosis": ([], []), "high_wss": ([], [])}
    for case, yt, yp in zip(cases, pooled_true, pooled_pred):
        masks = M.region_masks(case["pos"], case["local_radius"], yt)
        for name, m in masks.items():
            if m.sum() > 0:
                acc[name][0].append(yt[m]); acc[name][1].append(yp[m])
    out = {}
    for name, (ts, ps) in acc.items():
        if ts:
            out[name] = M.basic_metrics(np.concatenate(ts), np.concatenate(ps))
    return out


def _plot_case(case: Dict, y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path,
               plot_space: str = "target"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    pos = case["pos"]
    # 投影到主轴(z) - x 平面看整条血管；点大小随点数自适应
    px, pz = pos[:, 0], pos[:, 2]
    err = np.abs(y_true - y_pred)
    vmin = float(np.percentile(y_true, 1))
    vmax = float(np.percentile(y_true, 99))
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0
    emax = float(np.percentile(err, 99)) or 1.0
    s = max(1.0, 4000.0 / len(pos))

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    for ax, val, title, lo, hi, cmap in [
        (axes[0], y_true, f"true ({plot_space})", vmin, vmax, "viridis"),
        (axes[1], y_pred, f"pred ({plot_space})", vmin, vmax, "viridis"),
        (axes[2], err, "|err|", 0.0, emax, "magma"),
    ]:
        sc = ax.scatter(pz, px, c=val, s=s, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title); ax.set_aspect("equal"); ax.set_xlabel("z"); ax.set_ylabel("x")
        fig.colorbar(sc, ax=ax, shrink=0.7)
    r2 = M.r2_score(y_true, y_pred)
    fig.suptitle(f"{case['cohort']}/{case['case']}  peak step {case['peak_step']}  "
                 f"field R2={r2:.3f}  n={len(pos)}")
    fig.savefig(out_dir / f"{case['cohort'].replace('/', '_')}__{case['case']}.png", dpi=90)
    plt.close(fig)


def write_reports(result_by_part: Dict[str, Dict], eval_dir: Path):
    eval_dir.mkdir(parents=True, exist_ok=True)
    if result_by_part and all(res.get("target") == "velocity_pressure" for res in result_by_part.values()):
        for task in ("pressure", "velocity"):
            write_reports({part: res["tasks"][task] for part, res in result_by_part.items()}, eval_dir / task)
        metadata = {part: {**{k: v for k, v in res.items() if k != "tasks"},
                           "tasks": {task: {"metrics_path": f"{task}/metrics.json"}
                                     for task in ("pressure", "velocity")}}
                    for part, res in result_by_part.items()}
        (eval_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        return
    (eval_dir / "metrics.json").write_text(json.dumps(result_by_part, indent=2, ensure_ascii=False))
    rows = []
    for part, res in result_by_part.items():
        norm_res = res.get("normalized", res)
        for case, reg in norm_res["per_case"].items():
            row = {"partition": part, "case": case}
            for rname, rm in reg.items():
                if rname == "calibration":
                    for k in ("top10_pred_true_ratio", "p95_pred_true_ratio",
                              "p99_pred_true_ratio", "max_pred_true_ratio",
                              "calibration_slope"):
                        row[f"cal_{k}"] = round(rm.get(k, float("nan")), 4)
                    continue
                if rname == "hotspot":
                    for k in ("high_wss_mae", "top10_iou", "top10_precision",
                              "top10_recall", "spearman_all", "spearman_high_wss",
                              "peak_point_dist", "peak_point_dist_over_bbox",
                              "hotspot_centroid_dist", "hotspot_centroid_dist_over_bbox"):
                        row[f"hot_{k}"] = round(rm.get(k, float("nan")), 4)
                    continue
                if rname in {"legacy_vertex_hotspot", "area_hotspot"}:
                    prefix = "legacy_vertex" if rname.startswith("legacy") else "area"
                    for k, v in rm.items():
                        row[f"{prefix}_{k}"] = round(v, 6) if isinstance(v, float) else v
                    continue
                if rname == "distribution":
                    for k, v in rm.items():
                        row[f"dist_{k}"] = round(v, 6) if isinstance(v, float) else v
                    continue
                row[f"{rname}_r2"] = round(rm.get("r2", float("nan")), 4)
                row[f"{rname}_mae"] = round(rm.get("mae", float("nan")), 6)
                row[f"{rname}_rmse"] = round(rm.get("rmse", float("nan")), 6)
                row[f"{rname}_nrmse"] = round(rm.get("nrmse_range", float("nan")), 4) \
                    if "nrmse_range" in rm else float("nan")
                row[f"{rname}_nmae"] = round(rm.get("nmae_range", float("nan")), 4) \
                    if "nmae_range" in rm else float("nan")
                row[f"{rname}_n"] = rm.get("n", 0)
            # Preserve every historical (normalized, unprefixed) column above,
            # while making the new physical/normalized spaces explicit for v4.
            phys_reg = res.get("per_case", {}).get(case, reg)
            for prefix, space_reg in (("normalized", reg), ("physical", phys_reg)):
                for section in ("overall", "area_overall"):
                    values = space_reg.get(section, {})
                    for key in ("r2", "mae", "rmse", "nrmse_range", "nmae_range"):
                        if key in values:
                            row[f"{prefix}_{section}_{key}"] = values[key]
                for section in ("hotspot", "area_hotspot", "calibration", "distribution"):
                    for key, value in space_reg.get(section, {}).items():
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            row[f"{prefix}_{section}_{key}"] = value
            rows.append(row)
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(eval_dir / "per_case_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    fit_rows = []
    for part, res in result_by_part.items():
        if res.get("metric_space") == "normalized_target":
            spaces = [("normalized", res)]
        else:
            spaces = [("physical", res)]
            normalized = res.get("normalized")
            if normalized is not None and normalized is not res:
                spaces.append(("normalized", normalized))
        for space_name, space_result in spaces:
            for aggregation, section in (
                ("pooled", space_result["field"]),
                ("casebalanced", space_result["field_casebalanced"]),
            ):
                fit_rows.append({
                    "partition": part,
                    "space": space_name,
                    "aggregation": aggregation,
                    "r2_linear_fit": section.get("r2_linear_fit", float("nan")),
                    "linear_fit_slope": section.get("linear_fit_slope", float("nan")),
                    "linear_fit_intercept": section.get(
                        "linear_fit_intercept", float("nan")
                    ),
                    "n": section.get("n", 0),
                    "n_cases": section.get("n_cases", res["aggregate"]["n_cases"]),
                })
    if fit_rows:
        with open(eval_dir / "linear_fit_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fit_rows[0]))
            w.writeheader()
            w.writerows(fit_rows)


def checkpoint_filename(checkpoint: str) -> str:
    value = checkpoint.removesuffix(".pt")
    if value in {"best", "ckpt_best"}:
        return "ckpt_best.pt"
    if value in {"last", "ckpt_last"}:
        return "ckpt_last.pt"
    raise ValueError("checkpoint must be 'best' or 'last'")


def load_model_from_run(run_dir: Path, device: str, checkpoint: str = "best"):
    cfg = C.ExpConfig.from_json(run_dir / "config.json")
    feat_stats = json.loads((run_dir / "feature_stats.json").read_text())
    model = build_model(cfg.model, C.input_dim(cfg)).to(device)
    ckpt = torch.load(
        run_dir / checkpoint_filename(checkpoint), map_location=device, weights_only=False
    )
    model.load_state_dict(ckpt["model"])
    return cfg, feat_stats, model, ckpt


def load_wss_stats_for_run(run_dir: Path) -> Dict:
    stats_path = run_dir / "wss_global_stats.json"
    return D.load_wss_stats(stats_path if stats_path.is_file() else C.GLOBAL_STATS)


def parse_and_guard_partitions(value: str, allow_test: bool = False) -> List[str]:
    """Parse CLI partitions and require an explicit unlock before reading test."""
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("at least one partition is required")
    unknown = sorted(set(parts) - {"train", "val", "test"})
    if unknown:
        raise ValueError(f"unknown partitions: {', '.join(unknown)}")
    if len(parts) != len(set(parts)):
        raise ValueError("duplicate partitions are not allowed")
    if "test" in parts and not allow_test:
        raise PermissionError(
            "test access is locked during development; pass --allow-test only for the "
            "pre-authorized legacy test16 evaluation"
        )
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--partitions", type=str, default="val",
                    help="开发默认 val-only；test 还需 --allow-test 显式解锁")
    ap.add_argument("--allow-test", action="store_true",
                    help="显式解锁配置中已获批且锁定的 test partition")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--save-predictions", action="store_true",
                    help="保存体场逐例同点预测及原case query行号，供无需重新推理的尺度诊断")
    ap.add_argument("--checkpoint", choices=("best", "last"), default="best")
    ap.add_argument("--split-path", type=str, default=None,
                    help="只读评估覆盖 split；不修改 run config/checkpoint")
    ap.add_argument("--back-transform", choices=("config", "exp_mu", "jensen"), default="config",
                    help="只读评估覆盖 log->Pa 回变换口径（config=按 run 配置；exp_mu=exp(mu)；"
                         "jensen=逐点 exp(mu+sigma^2/2)，需 gaussian_nll 双通道头）")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="默认 run_dir/eval/ckpt_<best|last>，用于隔离 checkpoint 结果")
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.config:
        run_dir = C.ExpConfig.from_json(args.config).run_dir
    else:
        raise SystemExit("需要 --run-dir 或 --config")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, feat_stats, model, ckpt = load_model_from_run(run_dir, device, args.checkpoint)
    wss_stats = load_wss_stats_for_run(run_dir)
    eval_dir = Path(args.output_dir) if args.output_dir else run_dir / "eval" / f"ckpt_{args.checkpoint}"
    if args.back_transform != "config":
        want_jensen = args.back_transform == "jensen"
        if want_jensen and (cfg.train.loss != "gaussian_nll" or int(cfg.model.out_dim) != 2):
            ap.error("--back-transform jensen requires a run trained with loss='gaussian_nll' and out_dim=2")
        cfg.eval.pointwise_jensen = want_jensen  # read-only reporting override, the run config on disk is untouched

    try:
        partitions = parse_and_guard_partitions(args.partitions, allow_test=args.allow_test)
    except (ValueError, PermissionError) as exc:
        ap.error(str(exc))

    eval_split_path = args.split_path or cfg.data.split_path
    result_by_part = {}
    for part in partitions:
        cases = D.load_partition(
            eval_split_path, part, wss_stats, target=cfg.data.target,
            target_normalization=cfg.data.target_normalization,
            data_root=cfg.data.data_root,
            required_frame_version=cfg.data.required_frame_version,
            case_features_path=cfg.data.case_features_path,
            timesteps=getattr(cfg.data, "timesteps", "peak"), waveform_path=getattr(cfg.data, "waveform_path", None),
            extra_point_features=C.v6_point_features(cfg),
            point_features_root=getattr(cfg.data, "point_features_root", None),
        )
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        res = evaluate_partition(model, cases, cfg, feat_stats, wss_stats, device,
                                 save_dir=eval_dir, make_plots=(not args.no_plots and part == "test"),
                                 predictions_dir=eval_dir / "predictions" / part if args.save_predictions else None)
        elapsed = time.perf_counter() - t0
        res["efficiency"] = {
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "elapsed_seconds": elapsed,
            "seconds_per_case": elapsed / max(len(cases), 1),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
            "fixed_support": cfg.eval.fixed_support,
            "query_chunk_size": cfg.eval.query_chunk_size,
        }
        res["evaluation_split_path"] = str(Path(eval_split_path).resolve())
        result_by_part[part] = res
        if "tasks" in res:
            for task, task_result in res["tasks"].items():
                task_result["efficiency"] = {**res["efficiency"], "shared_joint_inference": True}
                task_result["evaluation_split_path"] = res["evaluation_split_path"]
                cb = task_result["field_casebalanced"]
                print(f"[{part}/{task}] R2_field_casebalanced={cb['r2']:.4f} MAE={cb['mae']:.4f}")
            continue
        report = res.get("normalized", res)
        agg, fld = report["aggregate"], report["field"]
        cb = report["field_casebalanced"]
        print(f"[{part}] cases={agg['n_cases']}  R2_casemean={agg['r2_casemean']:.4f}  "
              f"R2_field_raw={fld['r2']:.4f}  R2_field_casebalanced={cb['r2']:.4f}  "
              f"R2_fit_raw={fld['r2_linear_fit']:.4f}  "
              f"R2_fit_casebalanced={cb['r2_linear_fit']:.4f}  "
              f"NRMSE_field={fld['nrmse_range']:.4f}  "
              f"MAE={fld['mae']:.4f}")
        for rname, rm in report["regional_field"].items():
            print(f"    {rname:12s} R2={rm['r2']:.4f} NRMSE={rm['nrmse_range']:.4f} n={rm['n']}")

    write_reports(result_by_part, eval_dir)
    if args.save_predictions:
        import hashlib
        provenance_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
            "checkpoint_sha256": run_dir / f"ckpt_{args.checkpoint}.pt",
            "config_sha256": run_dir / "config.json",
            "wss_stats_sha256": run_dir / "wss_global_stats.json",
        }.items()}
        manifest = {"schema_version": 1, "target": cfg.data.target,
                    "checkpoint": args.checkpoint, "evaluation_split_path": str(Path(eval_split_path).resolve()),
                    "partitions": {}, **provenance_hashes}
        for part in partitions:
            saved = json.loads((eval_dir / "predictions" / part / "manifest.json").read_text())
            manifest["partitions"][part] = [{**entry, "file": str(Path(part) / entry["file"])}
                                               for entry in saved["cases"]]
        (eval_dir / "predictions" / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("eval 写入:", eval_dir)


if __name__ == "__main__":
    main()
