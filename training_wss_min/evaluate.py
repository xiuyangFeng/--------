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
from typing import Dict, List, Tuple

import numpy as np
import torch

from . import config as C
from . import dataset as D
from . import metrics as M
from . import surface as S
from . import time_basis as TB
from . import volume_time as VT
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
        section = D.case_section(case) if getattr(cfg.data, "section_tokens", False) else None
        if section is not None:
            encode_extra["section"] = torch.from_numpy(np.ascontiguousarray(section[support_idx])).to(device)
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
            if section is not None:
                decode_extra["section"] = torch.from_numpy(np.ascontiguousarray(section[idx])).to(device)
            if cfg.data.query_patch_nsample:
                patch = D.build_query_patch(case, idx, input_features, feat_stats,
                                            nsample=cfg.data.query_patch_nsample, frame_index=frame_index,
                                            **D._patch_scale_kwargs(cfg.data))
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
    if cfg.data.target in C.VOLUME_TARGETS and getattr(cfg.data, "timesteps", "peak") != "peak":
        return _evaluate_partition_volume_time(model, cases, cfg, feat_stats, wss_stats, device,
                                               save_dir=save_dir, make_plots=make_plots,
                                               predictions_dir=predictions_dir)
    if getattr(cfg.data, "timesteps", "peak") == "time_basis":
        return _evaluate_partition_time_basis(model, cases, cfg, feat_stats, wss_stats, device,
                                              save_dir=save_dir, make_plots=make_plots,
                                              predictions_dir=predictions_dir)
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
    # frame_stats：每帧用自己的 log 均值/标准差（stats 视图），global_stats 保持历史行为
    frame_stats_mode = cfg.data.target_normalization == "frame_stats"
    stats_for = (lambda k: D.frame_stats_view(wss_stats, k)) if frame_stats_mode else (lambda k: wss_stats)
    for k in frames:
        for case in cases:
            D.select_frame(case, k)
        res = _evaluate_partition_frame(model, cases, cfg, feat_stats, stats_for(k), device,
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
    pred_p = [np.concatenate([np.clip(D.denormalize_wss(pred_norm_store[k][i], stats_for(k)), 0, None) for k in frames_sorted])
              for i in range(len(cases))]
    pooled = {
        "n_frames": len(frames_sorted), "steps": [int(steps[k]) for k in frames_sorted],
        "note": "pooled over frames; includes between-frame (waveform-driven) variance, reads higher than per-frame R²",
        "physical": {"field_casebalanced": M.casebalanced_field_metrics(true_p, pred_p), "field": M.basic_metrics(np.concatenate(true_p), np.concatenate(pred_p)),
                     "calibration": M.calibration_metrics(np.concatenate(true_p), np.concatenate(pred_p))},
        "normalized": {"field_casebalanced": M.casebalanced_field_metrics(true_n, pred_n), "field": M.basic_metrics(np.concatenate(true_n), np.concatenate(pred_n))},
    }
    # TAWSS over the evaluated frames (mean |WSS| per node), physical Pa, full scalar suite
    ta_true = [np.mean(np.stack([np.asarray(c["y_raw_frames"][k], dtype=np.float64) for k in frames_sorted]), axis=0) for c in cases]
    ta_pred = [np.mean(np.stack([np.clip(D.denormalize_wss(pred_norm_store[k][i], stats_for(k)), 0, None) for k in frames_sorted]), axis=0) for i in range(len(cases))]
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
    if getattr(cfg.eval, "time_metrics", False):
        true_pa = {k: [np.asarray(c["y_raw_frames"][k], dtype=np.float64) for c in cases] for k in frames_sorted}
        pred_pa = {k: [np.clip(D.denormalize_wss(pred_norm_store[k][i], stats_for(k)), 0, None) for i in range(len(cases))]
                   for k in frames_sorted}
        top["cycle"] = _cycle_metrics(cases, frames_sorted, steps, peak_index, true_pa, pred_pa, wss_stats)
    return top


def _cycle_metrics(cases, frames_sorted, steps, peak_index, true_pa, pred_pa, wss_stats) -> Dict:
    """周期口径（与 experiments/wss_time_ecc_20260918/offline/common.py 的 TimeMetrics 同义）：
    逐帧 R²_cb（Pa 与 ln）均值、峰值帧、谷底四分位（协议波形 Q 最低 20 帧 ∩ 评估帧）、TAWSS 由调用方另报，
    逐点峰时误差（帧，周期环绕）、逐点 ln 时序相关、峰谷幅值（ln）误差。帧权均匀。"""
    floor = float(wss_stats.get("floor", 0.0)); eps = float(wss_stats["eps"])
    ln = lambda a: np.log(np.clip(np.asarray(a, dtype=np.float64), floor, None) + eps)
    r2_pa = {k: M.casebalanced_field_metrics(true_pa[k], pred_pa[k])["r2"] for k in frames_sorted}
    r2_ln = {k: M.casebalanced_field_metrics([ln(t) for t in true_pa[k]], [ln(p) for p in pred_pa[k]])["r2"] for k in frames_sorted}
    wf = cases[0].get("waveform")
    trough = []
    if wf is not None:
        q = np.asarray(wf["_q_norm"], dtype=np.float64)
        low = set(int(i) for i in np.argsort(q)[:20])
        trough = [k for k in frames_sorted if k in low]
    peak_med, peak_p90, ts_corr, amp_err = [], [], [], []
    full_cycle = len(frames_sorted) == len(steps)
    for i in range(len(cases)):
        lt = np.stack([ln(true_pa[k][i]) for k in frames_sorted]); lp = np.stack([ln(pred_pa[k][i]) for k in frames_sorted])
        d = np.abs(lt.argmax(axis=0) - lp.argmax(axis=0))
        if full_cycle:
            d = np.minimum(d, len(frames_sorted) - 1 - d)
        peak_med.append(float(np.median(d))); peak_p90.append(float(np.quantile(d, .9)))
        a = lt - lt.mean(axis=0); b = lp - lp.mean(axis=0)
        den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0)); ok = den > 1e-12
        ts_corr.append(float(np.mean((a * b).sum(axis=0)[ok] / den[ok])) if ok.any() else float("nan"))
        amp_err.append(float(np.median(np.abs((lp.max(0) - lp.min(0)) - (lt.max(0) - lt.min(0))))))
    return {
        "definition": "uniform frame weights; R2 case-balanced per frame; trough = protocol-waveform 20 lowest-Q frames among evaluated frames; "
                      "peak-time error in frames (cyclic when all frames evaluated); ts-corr/amp-err in ln(max(WSS,floor)+eps)",
        "n_frames": len(frames_sorted), "full_cycle": bool(full_cycle),
        "cycle_r2cb_pa": float(np.mean([r2_pa[k] for k in frames_sorted])),
        "cycle_r2cb_ln": float(np.mean([r2_ln[k] for k in frames_sorted])),
        "peak_r2cb_pa": float(r2_pa[peak_index]) if peak_index in r2_pa else float("nan"),
        "trough_frames_steps": [int(steps[k]) for k in trough],
        "trough_r2cb_pa": float(np.mean([r2_pa[k] for k in trough])) if trough else float("nan"),
        "trough_r2cb_ln": float(np.mean([r2_ln[k] for k in trough])) if trough else float("nan"),
        "min_frame_r2cb_pa": float(min(r2_pa.values())), "argmin_frame_step": int(steps[min(r2_pa, key=r2_pa.get)]),
        "peak_time_err_med_frames": float(np.mean(peak_med)), "peak_time_err_p90_frames": float(np.mean(peak_p90)),
        "ts_corr_ln": float(np.nanmean(ts_corr)), "amp_err_ln_med": float(np.mean(amp_err)),
        "frame_r2cb_pa": {int(steps[k]): float(r2_pa[k]) for k in frames_sorted},
        "frame_r2cb_ln": {int(steps[k]): float(r2_ln[k]) for k in frames_sorted},
    }


def _assemble_scalar_wss_result(cases, true_norm_by_case, pred_norm_by_case, cfg, stats, *, save_dir, make_plots,
                                true_raw_by_case) -> Dict:
    """标量 WSS 单帧结果（归一化 + 物理），与 _evaluate_partition_frame 的 wss/global_stats 路径同构；仅时间基头路径使用。"""
    normalized = _evaluate_space(cases, true_norm_by_case, pred_norm_by_case, save_dir=save_dir, make_plots=make_plots,
                                 plot_space="normalized target", include_area=False)
    normalized["normalization"] = {"mode": cfg.data.target_normalization,
                                   "case_metadata": {f"{c['cohort']}/{c['case']}": c.get("target_normalization_meta", {}) for c in cases},
                                   "predictions_clipped_for_metrics": False}
    normalized["surface_metric_mode"] = cfg.eval.surface_metric_mode
    normalized["surface_area_metrics_status"] = "not_requested"
    pred_raw_by_case = [np.clip(np.asarray(D.denormalize_wss(p, stats), dtype=np.float64), 0, None) for p in pred_norm_by_case]
    physical = _evaluate_space(cases, true_raw_by_case, pred_raw_by_case, save_dir=None, make_plots=False,
                               plot_space="physical target", include_area=False)
    physical["metric_space"] = "physical_target"
    physical["back_transform"] = "exp(mu)"
    physical["surface_metric_mode"] = cfg.eval.surface_metric_mode
    physical["surface_area_metrics_status"] = "not_requested"
    physical["normalized"] = normalized
    physical["_pred_pa_by_case"] = pred_raw_by_case
    return physical


def _evaluate_partition_time_basis(model, cases, cfg, feat_stats, wss_stats, device, *, save_dir, make_plots,
                                   predictions_dir) -> Dict:
    """时间基头：一次推理得到每点 [b0, a_1..a_K]，重建全部帧后按 eval.eval_frames 逐帧评估。
    顶层结果恒为峰值帧（与单帧 run 同口径），另加 frames / frame_curve / pooled_frames / tawss / coefficients（/ cycle）。"""
    if cfg.eval.surface_metric_mode == "both_strict":
        raise ValueError("time_basis evaluation supports surface_metric_mode='legacy_vertex' only")
    model.eval()
    basis = cases[0]["time_basis"]; K = int(basis["k"])
    steps = np.asarray(cases[0]["steps"]); peak_index = int(cases[0]["peak_index"])
    frames = _parse_eval_frames(cfg.eval.eval_frames, steps, peak_index)
    if peak_index not in frames:
        frames = [peak_index] + frames
    coef_pred, coef_true, z_pred = [], [], []
    for case in cases:
        case.setdefault("unit_id", f"{case.get('cohort', 'AG/unknown')}/{case.get('case', 'unknown')}")
        c = np.asarray(predict_case_norm(model, case, cfg.data.input_features, feat_stats, device, cfg=cfg,
                                         return_all_channels=True), dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != K + 1:
            raise ValueError(f"time_basis head must output (N, K+1) with K={K}, got {c.shape}")
        coef_pred.append(c); coef_true.append(np.asarray(case["y_norm"], dtype=np.float64))
        z_pred.append(TB.reconstruct(c, basis))  # (T, N)
    stats_for = lambda k: D.frame_stats_view(wss_stats, k)
    per_frame: Dict[str, Dict] = {}; top = None; pred_pa_store: Dict[int, List[np.ndarray]] = {}
    for k in frames:
        true_n = [np.asarray(c["y_norm_frames"][k], dtype=np.float64) for c in cases]
        pred_n = [zp[k] for zp in z_pred]
        true_raw = [np.asarray(c["y_raw_frames"][k], dtype=np.float64) for c in cases]
        res = _assemble_scalar_wss_result(cases, true_n, pred_n, cfg, stats_for(k), save_dir=save_dir if k == peak_index else None,
                                          make_plots=make_plots and k == peak_index, true_raw_by_case=true_raw)
        pred_pa_store[k] = res.pop("_pred_pa_by_case")
        per_frame[str(int(steps[k]))] = {"step": int(steps[k]), "frame_index": int(k), "physical": _compact(res), "normalized": _compact(res["normalized"])}
        if k == peak_index:
            top = res
    assert top is not None
    frames_sorted = sorted(pred_pa_store)
    true_p = [np.concatenate([np.asarray(c["y_raw_frames"][k], dtype=np.float64) for k in frames_sorted]) for c in cases]
    pred_p = [np.concatenate([pred_pa_store[k][i] for k in frames_sorted]) for i in range(len(cases))]
    true_n = [np.concatenate([np.asarray(c["y_norm_frames"][k], dtype=np.float64) for k in frames_sorted]) for c in cases]
    pred_n = [np.concatenate([z_pred[i][k] for k in frames_sorted]) for i in range(len(cases))]
    pooled = {"n_frames": len(frames_sorted), "steps": [int(steps[k]) for k in frames_sorted],
              "note": "pooled over frames; includes between-frame (waveform-driven) variance, reads higher than per-frame R²",
              "physical": {"field_casebalanced": M.casebalanced_field_metrics(true_p, pred_p), "field": M.basic_metrics(np.concatenate(true_p), np.concatenate(pred_p)),
                           "calibration": M.calibration_metrics(np.concatenate(true_p), np.concatenate(pred_p))},
              "normalized": {"field_casebalanced": M.casebalanced_field_metrics(true_n, pred_n), "field": M.basic_metrics(np.concatenate(true_n), np.concatenate(pred_n))}}
    ta_true = [np.mean(np.stack([np.asarray(c["y_raw_frames"][k], dtype=np.float64) for k in frames_sorted]), axis=0) for c in cases]
    ta_pred = [np.mean(np.stack([pred_pa_store[k][i] for k in frames_sorted]), axis=0) for i in range(len(cases))]
    tawss = _evaluate_space(cases, ta_true, ta_pred, save_dir=None, make_plots=False, plot_space="TAWSS", include_area=False)
    tawss.pop("per_case", None); tawss["n_frames"] = len(frames_sorted)
    tawss["definition"] = "time-average of |WSS| over evaluated frames (all 81 = full cycle); prediction reconstructed from the time basis and averaged in Pa"
    curve = [{"step": v["step"], "r2_casebalanced_physical": v["physical"]["field_casebalanced"]["r2"], "r2_casebalanced_normalized": v["normalized"]["field_casebalanced"]["r2"],
              "r2_casemean_physical": v["physical"]["aggregate"]["r2_casemean"], "mae_pa": v["physical"]["field"]["mae"]} for v in sorted(per_frame.values(), key=lambda e: e["step"])]
    coef_metrics = {f"c{j}": M.casebalanced_field_metrics([t[:, j] for t in coef_true], [p[:, j] for p in coef_pred]) for j in range(K + 1)}
    coef_metrics["definition"] = "case-balanced R2/MAE per coefficient channel: c0 = b0 (time-mean of z), c1..cK = basis coefficients"
    top["time_evaluation"] = {"timesteps": "time_basis", "eval_frames": cfg.eval.eval_frames, "top_level_frame_step": int(steps[peak_index]),
                              "n_frames_evaluated": len(frames_sorted), "time_basis_k": K, "time_basis_path": basis["path"], "time_basis_sha256": basis["sha256"]}
    top["frames"] = per_frame; top["frame_curve"] = curve; top["pooled_frames"] = pooled; top["tawss"] = tawss; top["coefficients"] = coef_metrics
    if getattr(cfg.eval, "time_metrics", False):
        true_pa = {k: [np.asarray(c["y_raw_frames"][k], dtype=np.float64) for c in cases] for k in frames_sorted}
        top["cycle"] = _cycle_metrics(cases, frames_sorted, steps, peak_index, true_pa, pred_pa_store, wss_stats)
    if predictions_dir is not None:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for case, cp, ct, zp in zip(cases, coef_pred, coef_true, z_pred):
            relative = Path(case["unit_id"]) / "predictions.npz"
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe case unit_id")
            output = predictions_dir / relative; output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output, pred_coef=cp.astype(np.float32), true_coef=ct.astype(np.float32),
                                pred_pa_peak=pred_pa_store[peak_index][cases.index(case)], true_pa_peak=np.asarray(case["y_raw_frames"][peak_index]),
                                pred_pa_frames=np.stack([np.clip(D.denormalize_wss(zp[k], stats_for(k)), 0, None) for k in range(len(steps))]).astype(np.float32),
                                row_index=np.arange(len(cp), dtype=np.int64), steps=steps)
            import hashlib
            entries.append({"unit_id": case["unit_id"], "file": str(relative), "n_points": len(cp),
                            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
        (predictions_dir / "manifest.json").write_text(json.dumps({"schema_version": 1, "target": "wss_time_basis", "time_basis_k": K, "cases": entries},
                                                                  indent=2, ensure_ascii=False))
    return top


def _evaluate_partition_frame(model, cases: List[Dict], cfg: C.ExpConfig, feat_stats: Dict,
                              wss_stats: Dict, device: str,
                              save_dir: Path | None = None, make_plots: bool = False,
                              frame_index: int | None = None, return_predictions: bool = False,
                              predictions_dir: Path | None = None,
                              pred_norm_by_case: List[np.ndarray] | None = None) -> Dict:
    """pred_norm_by_case 给定时跳过前向，直接用外部预测装配指标（体场时间基头的峰值帧由系数重建）。"""
    model.eval()
    if cfg.data.target == C.MULTI_TARGET and pred_norm_by_case is None:
        return _evaluate_partition_multi(model, cases, cfg, feat_stats, wss_stats, device, save_dir=save_dir,
                                         make_plots=make_plots, frame_index=frame_index, return_predictions=return_predictions,
                                         predictions_dir=predictions_dir)
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
    supplied = pred_norm_by_case
    pred_norm_by_case = [] if supplied is None else [np.asarray(p, dtype=np.float64) for p in supplied]
    pred_logvar_by_case = []
    for case in ([] if supplied is not None else cases):
        y_pred_norm = predict_case_norm(
            model, case, cfg.data.input_features, feat_stats, device, cfg=cfg,
            return_all_channels=is_vector or is_joint or pointwise_jensen, frame_index=frame_index,
        )
        y_pred_norm = np.asarray(y_pred_norm, dtype=np.float64)
        if pointwise_jensen:
            pred_logvar_by_case.append(y_pred_norm[:, 1])
            y_pred_norm = y_pred_norm[:, 0]
        if "offset" in wss_stats and cfg.data.target == "wss":
            # 残差目标：模型输出残差 z，换回标准 log_z 后再算全部指标（口径与其它臂一致）
            rows = D.query_rows(case)
            offset_rows = np.asarray(case["y_offset"])[rows]
            gate_quantile = getattr(cfg.eval, "residual_gate_quantile", None)
            if gate_quantile is not None:
                y_pred_norm = D.gate_residual_prediction(y_pred_norm, offset_rows, wss_stats, float(gate_quantile))
            y_pred_norm = D.residual_to_standard_logz(y_pred_norm, offset_rows, wss_stats)
            case["y_norm_standard"] = D.standard_logz(np.asarray(case["y_raw"], dtype=np.float64), wss_stats).astype(np.float32)
        pred_norm_by_case.append(y_pred_norm)

    if predictions_dir is not None:
        scalar_wall = cfg.data.target in ("wss", *C.CYCLE_TARGETS)
        if not scalar_wall and cfg.data.target not in C.VOLUME_TARGETS:
            raise ValueError("--save-predictions supports WSS / cycle (tawss, osi) and volume targets")
        if scalar_wall and (pointwise_jensen or cfg.data.target_normalization != "global_stats"):
            raise ValueError("saved wall predictions require global_stats without Jensen correction")
        predictions_dir.mkdir(parents=True, exist_ok=True)
        save_prediction = _save_wss_prediction if scalar_wall else _save_volume_prediction
        entries = [save_prediction(c, p, cfg.data.target, wss_stats, predictions_dir)
                   for c, p in zip(cases, pred_norm_by_case)]
        (predictions_dir / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "target": cfg.data.target, "cases": entries,
        }, indent=2, ensure_ascii=False))
    if cfg.data.target in C.VOLUME_TARGETS:
        # 体场目标：真值/几何都切到 query 行（内部单元，压力混合时含壁面行）
        rows_by_case = [D.query_rows(c) for c in cases]
        cases = [D.subset_case(c, rows) for c, rows in zip(cases, rows_by_case)]
    true_norm_by_case = [np.asarray(c.get("y_norm_standard", c["y_norm"]), dtype=np.float64) for c in cases]
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
        if cfg.data.target == "osi":
            pred_raw = np.clip(pred_raw, 0.0, C.OSI_MAX)   # 有界目标：线性 z 反变换可能越界；logit_z 已在界内
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
    masks_above = tuple(float(v) for v in (getattr(cfg.eval, "threshold_masks_above", ()) or ()))
    masks_below = tuple(float(v) for v in (getattr(cfg.eval, "threshold_masks_below", ()) or ()))
    physical = _evaluate_space(
        cases, true_raw_by_case, pred_raw_by_case,
        save_dir=None, make_plots=False, plot_space="physical target",
        include_area=include_area,
        threshold_masks=(masks_above, masks_below) if (masks_above or masks_below) else None,
    )
    physical["metric_space"] = "physical_target"
    physical["back_transform"] = (
        "pointwise_lognormal_mean_exp(mu+sigma^2/2)" if pointwise_jensen
        else {"log_z": "exp(mu)", "logit_z": "scale*sigmoid(x)"}.get(wss_stats.get("method"), "linear")
    )
    if cfg.data.target in C.CYCLE_TARGETS:
        physical["target"] = cfg.data.target
        physical["target_units"] = "Pa (cycle-averaged |WSS|, frames 0-79)" if cfg.data.target == "tawss" else "OSI in [0, 0.5]"
    physical["surface_metric_mode"] = cfg.eval.surface_metric_mode
    physical["surface_area_metrics_status"] = (
        "complete_strict" if include_area else "not_requested"
    )
    physical["normalized"] = normalized
    if cfg.data.target in C.CYCLE_TARGETS and bool(getattr(cfg.eval, "cycle_agreement", True)):
        physical["cycle_agreement"] = _cycle_agreement_block(cases, true_raw_by_case, pred_raw_by_case, cfg, wss_stats)
    if return_predictions:
        physical["_pred_norm_by_case"] = pred_norm_by_case
    return physical


def _evaluate_partition_multi(model, cases, cfg, feat_stats, wss_stats, device, *, save_dir, make_plots, frame_index,
                              return_predictions, predictions_dir) -> Dict:
    """M1 三头（2026-09-21）：一次前向得到 (N, 3) = [峰值帧 WSS, TAWSS, OSI]，每个通道用各自的统计量与单目标评估路径；
    顶层结果 = 峰值帧通道（与 X5D 同口径，含 normalized / physical 全套），其余两头在 result["heads"]。"""
    import copy as _copy
    channels = list(wss_stats["channels"])
    for case in cases:
        case.setdefault("unit_id", f"{case.get('cohort', 'AG/unknown')}/{case.get('case', 'unknown')}")
    preds = [np.asarray(predict_case_norm(model, case, cfg.data.input_features, feat_stats, device, cfg=cfg,
                                          return_all_channels=True, frame_index=frame_index), dtype=np.float64) for case in cases]
    for p in preds:
        if p.ndim != 2 or p.shape[1] != len(channels):
            raise ValueError(f"wss_cycle_multi head must output (N, {len(channels)}), got {p.shape}")
    base = [c for c in channels if c in C.MULTI_CHANNELS]
    aux = [c for c in channels if c not in C.MULTI_CHANNELS]
    if tuple(base) != tuple(C.MULTI_CHANNELS) or channels[:len(base)] != base:
        raise ValueError(f"multi channels must start with {C.MULTI_CHANNELS}, got {channels}")
    results: Dict[str, Dict] = {}
    for j, name in enumerate(base):
        sub = _copy.deepcopy(cfg)
        sub.data.target = name
        sub.eval.threshold_masks_above = tuple(cfg.eval.threshold_masks_above) if name == "osi" else ()
        sub.eval.threshold_masks_below = tuple(cfg.eval.threshold_masks_below) if name == "tawss" else ()
        sub_cases = [dict(case, y_raw=np.asarray(case["y_raw"])[:, j], y_norm=np.asarray(case["y_norm"])[:, j]) for case in cases]
        results[name] = _evaluate_partition_frame(
            model, sub_cases, sub, feat_stats, wss_stats[name], device,
            save_dir=save_dir if j == 0 else None, make_plots=make_plots and j == 0, frame_index=frame_index,
            return_predictions=return_predictions,
            predictions_dir=(Path(predictions_dir) / name) if predictions_dir is not None else None,
            pred_norm_by_case=[p[:, j] for p in preds])
    top = results[channels[0]]
    top["heads"] = {name: results[name] for name in base[1:]}
    top["multi_target"] = {"target": C.MULTI_TARGET, "channels": channels, "top_level_channel": channels[0],
                           "definition": "one forward pass, out_dim=3; channel 0 = peak-frame WSS (log_z, comparable with X5D), "
                                         "heads = TAWSS (log_z) and OSI (stats method of the multi file); equal-weight channel MSE"}
    manifest_channels = list(base)
    if aux:
        aux_heads = _evaluate_multi_aux(model, cases, preds, channels, aux, cfg, feat_stats, wss_stats, device,
                                        frame_index=frame_index, predictions_dir=predictions_dir)
        top["heads"].update(aux_heads)
        top["multi_target"].update(aux_channels=aux, definition=(
            f"one forward pass, out_dim={len(channels)}; channel 0 = peak-frame WSS (log_z), heads = TAWSS (log_z), OSI and the auxiliary "
            f"channels {aux} (linear z; loss only, never a deployment output); 'osi_derived' = 0.5(1 - min(1, |(r_axial, r_circ)|)) "
            f"from the mean-vector channels when both are present; equal-weight channel MSE"))
        if "osi_derived" in aux_heads:
            manifest_channels.append("osi_derived")
    if predictions_dir is not None:
        # 每个通道各自写了 <predictions_dir>/<channel>/manifest.json；CLI 回读的是分区顶层 manifest，这里汇总成同一合同
        # （cases 条目的 file 相对 predictions_dir，即 <channel>/<unit_id>/predictions.npz），并附逐通道清单路径。
        # 辅助通道另存在 <predictions_dir>/aux_<name>/ 下（不进顶层清单）；派生 OSI 与三通道同合同。
        root = Path(predictions_dir)
        per_channel = {name: json.loads((root / name / "manifest.json").read_text()) for name in manifest_channels}
        cases_entries = [{**entry, "file": str(Path(name) / entry["file"]), "channel": name}
                         for name in manifest_channels for entry in per_channel[name]["cases"]]
        (root / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "target": C.MULTI_TARGET, "channels": manifest_channels if aux else channels,
            "channel_manifests": {name: str(Path(name) / "manifest.json") for name in manifest_channels},
            "cases": cases_entries,
        }, indent=2, ensure_ascii=False))
    if return_predictions:
        top["_pred_norm_by_case"] = preds
    return top


def _evaluate_multi_aux(model, cases, preds, channels, aux, cfg, feat_stats, wss_stats, device, *, frame_index, predictions_dir) -> Dict:
    """M1 辅助通道（2026-09-25）：逐通道 R²_cb（归一化 / 原始空间）+ 可选保存；mean_axial & mean_circ 同在时，
    派生 OSI 走 OSI 单目标评估全套（阈值掩膜、cycle_agreement、同合同保存），另报标签自洽（真值分量派生 OSI 对真值 OSI）。"""
    import copy as _copy
    out: Dict[str, Dict] = {}
    for name in aux:
        j = channels.index(name)
        st = wss_stats[name]
        t_norm = [np.asarray(c["y_norm"], dtype=np.float64)[:, j] for c in cases]
        p_norm = [np.asarray(p, dtype=np.float64)[:, j] for p in preds]
        t_raw = [np.asarray(c["y_raw"], dtype=np.float64)[:, j] for c in cases]
        p_raw = [np.asarray(D.denormalize_wss(p, st), dtype=np.float64) for p in p_norm]
        out[f"aux_{name}"] = {
            "normalized": {"field_casebalanced": M.casebalanced_field_metrics(t_norm, p_norm)},
            "field_casebalanced": M.casebalanced_field_metrics(t_raw, p_raw),
            "per_case_r2": {c["unit_id"]: float(M.r2_score(t, p)) for c, t, p in zip(cases, t_raw, p_raw)},
        }
        if predictions_dir is not None:
            root = Path(predictions_dir) / f"aux_{name}"
            for case, tn, pn, tr, pr in zip(cases, t_norm, p_norm, t_raw, p_raw):
                target = root / case["unit_id"]
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target / "predictions.npz", pred_norm=pn, true_norm=tn.astype(np.float32), pred_raw=pr,
                                    true_raw=tr.astype(np.float32), row_index=np.arange(len(tr), dtype=np.int64), channel=np.array(name))
    if {"mean_axial", "mean_circ"} <= set(aux):
        ja, jc, jo = channels.index("mean_axial"), channels.index("mean_circ"), channels.index("osi")
        d_pred = [D.derived_osi(D.denormalize_wss(np.asarray(p, dtype=np.float64)[:, ja], wss_stats["mean_axial"]),
                                D.denormalize_wss(np.asarray(p, dtype=np.float64)[:, jc], wss_stats["mean_circ"])) for p in preds]
        d_true = [D.derived_osi(np.asarray(c["y_raw"], dtype=np.float64)[:, ja], np.asarray(c["y_raw"], dtype=np.float64)[:, jc]) for c in cases]
        osi_true = [np.asarray(c["y_raw"], dtype=np.float64)[:, jo] for c in cases]
        sub = _copy.deepcopy(cfg)
        sub.data.target = "osi"
        sub.eval.threshold_masks_above = tuple(cfg.eval.threshold_masks_above)
        sub.eval.threshold_masks_below = ()
        sub_cases = [dict(case, y_raw=np.asarray(case["y_raw"])[:, jo], y_norm=np.asarray(case["y_norm"])[:, jo]) for case in cases]
        res = _evaluate_partition_frame(
            model, sub_cases, sub, feat_stats, wss_stats["osi"], device, save_dir=None, make_plots=False, frame_index=frame_index,
            return_predictions=False, predictions_dir=(Path(predictions_dir) / "osi_derived") if predictions_dir is not None else None,
            pred_norm_by_case=[D.normalize_wss(x, wss_stats["osi"]) for x in d_pred])
        res["label_consistency"] = {
            "definition": "true OSI vs 0.5(1 - min(1, |(r_axial, r_circ)|)) from the true mean-vector channels (tangent-plane projection loss)",
            "field_casebalanced": M.casebalanced_field_metrics(osi_true, d_true),
            "mean_abs_diff_casemean": float(np.mean([np.mean(np.abs(a - b)) for a, b in zip(osi_true, d_true)])),
        }
        out["osi_derived"] = res
    return out


def _cycle_agreement_block(cases, true_by_case, pred_by_case, cfg, wss_stats) -> Dict:
    """周期积分量落地一致性（矩阵 §7 补充，2026-09-21）：逐例 CCC / 相对误差 / 热点与阈值 Dice / 分段均值误差 + 病例级 Bland–Altman。"""
    log_space = cfg.data.target == "tawss"
    floor = float(wss_stats.get("floor", 0.05)) if log_space else float(getattr(cfg.eval, "cycle_osi_rel_floor", 0.01))
    above = tuple(float(v) for v in (getattr(cfg.eval, "threshold_masks_above", ()) or ()))
    below = tuple(float(v) for v in (getattr(cfg.eval, "threshold_masks_below", ()) or ()))
    per_case: Dict[str, Dict] = {}
    for case, yt, yp in zip(cases, true_by_case, pred_by_case):
        per_case[case["unit_id"]] = M.cycle_agreement_case_metrics(
            yt, yp, log_space=log_space, floor=floor, rel_tolerance=float(getattr(cfg.eval, "cycle_rel_tolerance", 0.3)),
            segment_ids=case.get("_wall_semantic_id"), min_segment_points=int(getattr(cfg.eval, "cycle_min_segment_points", 200)),
            masks_above=above, masks_below=below)
    summary = M.cycle_agreement_aggregate(per_case)
    summary["definition"] = ("per-case Lin CCC (" + ("ln(max(y,floor)) for TAWSS" if log_space else "linear for OSI") + "), median relative error "
                             "|pred-true|/max(true,floor) and share within tolerance, top-10% Dice, threshold-mask Dice with true/pred area shares, "
                             "semantic-segment mean relative error (segments with >= min points); case-level Bland-Altman (bias, 95% LoA) of the case "
                             "mean (relative) and of mask area shares (absolute); segments keyed by wss_v5 SEMANTIC_LABELS id")
    summary["per_case"] = per_case
    return summary


def _evaluate_partition_volume_time(model, cases, cfg, feat_stats, wss_stats, device, *,
                                    save_dir, make_plots, predictions_dir) -> Dict:
    """体场多帧（压力 / 速度 × 81 帧）：顶层恒为峰值帧全点云，与单帧体场 run（PF6 / VF6）同口径；
    另加 `cycle` 块 —— 固定子采样点上的整周期指标（见 volume_time.cycle_eval_rows 的口径说明）。

    两种时间臂共用同一套 cycle 口径：
    - `timesteps='random_frame'`（T0，相位查询）：时间特征随帧变化，必须逐帧前向；
    - `timesteps='time_basis'`（TB）：一次前向出系数，重建全部帧。
    """
    target = cfg.data.target
    kind = "pressure" if target == "pressure_mixed" else "velocity"
    peak_index = int(cases[0]["peak_index"])
    steps = np.asarray(cases[0]["steps"])
    n_frames = int(cases[0]["n_frames"])
    frame_stats_mode = cfg.data.target_normalization == "frame_stats"
    stats_for = (lambda k: D.frame_stats_view(wss_stats, k)) if frame_stats_mode else (lambda k: wss_stats)
    basis = cases[0].get("time_basis")
    coef_true = [np.asarray(c["y_norm"], dtype=np.float64) for c in cases] if basis is not None else None

    # ---- 顶层：峰值帧、全点云 ----
    for case in cases:
        D.select_frame(case, peak_index)        # y_raw/y_norm 换成峰值帧标签（TB 下同时覆盖系数）
    peak_pred, coef_pred = None, None
    if basis is not None:
        # 一次全点云前向拿系数，峰值帧场与系数通道指标都从它导出，不重复推理
        model.eval()
        coef_pred = []
        for case in cases:
            case.setdefault("unit_id", f"{case.get('cohort', 'AG/unknown')}/{case.get('case', 'unknown')}")
            coef_pred.append(np.asarray(predict_case_norm(model, case, cfg.data.input_features, feat_stats,
                                                          device, cfg=cfg, return_all_channels=True),
                                        dtype=np.float64))
        peak_pred = [TB.reconstruct_frame(c, basis, peak_index) for c in coef_pred]
    top = _evaluate_partition_frame(model, cases, cfg, feat_stats, stats_for(peak_index), device,
                                    save_dir=save_dir, make_plots=make_plots, frame_index=peak_index,
                                    predictions_dir=predictions_dir, pred_norm_by_case=peak_pred)

    # ---- 周期：固定子采样点、全部 81 帧 ----
    met = VT.VolumeTimeMetrics(_cycle_q_norm(cases[0], n_frames), kind)
    per_case: Dict[str, Dict] = {}
    n_eval_points = []
    for case in cases:
        rows = VT.cycle_eval_rows(case, target)
        n_eval_points.append(len(rows))
        saved_pool = case.get("query_pool")
        case["query_pool"] = rows
        try:
            src = case["volume_frames"].take_rows(rows)
            y_true = np.stack([src.frame(k) for k in range(n_frames)]).astype(np.float64)
            if basis is None:
                y_pred = np.stack([
                    denormalize_volume_prediction(
                        predict_case_norm(model, case, cfg.data.input_features, feat_stats, device, cfg=cfg,
                                          return_all_channels=(kind == "velocity"), frame_index=k),
                        target, stats_for(k))
                    for k in range(n_frames)])
            else:
                coef = np.asarray(predict_case_norm(model, case, cfg.data.input_features, feat_stats, device,
                                                    cfg=cfg, return_all_channels=True), dtype=np.float64)
                z = TB.reconstruct(coef, basis)                      # (T, n_sub[, 3])
                y_pred = np.stack([denormalize_volume_prediction(z[k], target, stats_for(k))
                                   for k in range(n_frames)])
        finally:
            if saved_pool is None:
                case.pop("query_pool", None)
            else:
                case["query_pool"] = saved_pool
        met.add(y_true, y_pred)
        one = VT.VolumeTimeMetrics(_cycle_q_norm(case, n_frames), kind)
        one.add(y_true, y_pred)
        per_case[case["unit_id"]] = {k: v for k, v in one.summary().items()
                                     if k not in ("frame_r2cb", "frame_nmae_range")}
        print(f"  [cycle] {case['unit_id']} points={len(rows)}", flush=True)

    cycle = met.summary()
    cycle.update(
        n_eval_points_per_case=[int(v) for v in n_eval_points],
        eval_point_protocol=(f"fixed per-case subsample: {VT.SUBSAMPLE_CELLS} interior cells"
                             + ("" if kind == "velocity" else f" + {VT.SUBSAMPLE_WALL} wall nodes")
                             + f" (seed {VT.SUBSAMPLE_SEED}, keyed by unit_id); peak frame is still full-cloud"),
        units="Pa" if kind == "pressure" else "m/s",
        frame_steps=[int(v) for v in steps],
        per_case=per_case,
    )
    top["cycle"] = cycle
    top["time_evaluation"] = {
        "timesteps": cfg.data.timesteps, "target": target,
        "top_level_frame_step": int(steps[peak_index]), "n_frames_evaluated": n_frames,
        "frame_reference": "p_rel(x,t) = p(x,t) - p_volume_mean(t)" if kind == "pressure"
                           else "velocity rotated into the atlas aligned frame",
        "target_normalization": cfg.data.target_normalization,
    }
    if basis is not None:
        width = int(basis["k"]) + 1
        top["time_evaluation"].update(time_basis_k=int(basis["k"]), time_basis_channels=int(basis["channels"]),
                                      time_basis_path=basis["path"], time_basis_sha256=basis["sha256"])
        top["coefficients"] = _volume_coefficient_metrics(cases, coef_true, coef_pred, width,
                                                          int(basis["channels"]))
    return top


def _cycle_q_norm(case: Dict, n_frames: int) -> np.ndarray:
    """谷底四分位需要名义入口流量；缺 waveform 时退化为均匀（trough = 前 20 帧），并在结果里可见。"""
    wf = case.get("waveform")
    if wf is None:
        return np.arange(n_frames, dtype=np.float64)
    return np.asarray(wf["_q_norm"], dtype=np.float64)


def _volume_coefficient_metrics(cases, coef_true, coef_pred, width, channels) -> Dict:
    """时间基系数通道的逐通道 R²_cb（c0 = b0 = z 的时间均值，c1..cK = 基系数）。

    真值系数是整例行（load_case 的投影覆盖 wall∪interior），预测只在评估 query 行上，按行对齐。"""
    rows_by_case = [D.query_rows(c) for c in cases]
    out = {}
    for ch in range(channels):
        for j in range(width):
            col = ch * width + j
            name = f"c{j}" if channels == 1 else f"{'uvw'[ch]}_c{j}"
            out[name] = M.casebalanced_field_metrics([t[rows, col] for t, rows in zip(coef_true, rows_by_case)],
                                                     [p[:, col] for p in coef_pred])
    out["definition"] = ("case-balanced R2/MAE per coefficient channel on the evaluation query rows: "
                         "c0 = b0 (time-mean of z), c1..cK = basis coefficients"
                         + ("; velocity components share one basis" if channels > 1 else ""))
    return out


def _save_wss_prediction(case, pred_norm, target, stats, predictions_dir):
    relative = Path(case["unit_id"]) / "predictions.npz"
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe case unit_id")
    output = predictions_dir / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = D.denormalize_wss(np.asarray(pred_norm, dtype=np.float64), stats)
    if target == "osi":
        clipped, clipping = np.clip(raw, 0.0, C.OSI_MAX), f"clipped to [0, {C.OSI_MAX}] (OSI)"
    elif stats.get("method") == "log_z":
        clipped, clipping = np.clip(raw, 0, None), "nonnegative after inverse log_z"
    else:
        clipped, clipping = raw, "none"
    np.savez_compressed(output, pred_norm=pred_norm, true_norm=case.get("y_norm_standard", case["y_norm"]),
                        pred_pa_unclipped=raw, pred_pa=clipped, true_pa=case["y_raw"],
                        row_index=np.arange(len(pred_norm), dtype=np.int64), target=np.array(str(target)))
    import hashlib
    return {"unit_id": case["unit_id"], "file": str(relative), "n_points": len(pred_norm),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "target": str(target),
            "metric_clipping": clipping}


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


def _threshold_mask_block(true_by_case, pred_by_case, above, below) -> Dict[str, Dict]:
    """物理空间阈值掩膜（周期积分量矩阵 §7）：逐例 IoU/precision/recall/面积份额误差的病例均值与中位。"""
    out: Dict[str, Dict] = {}
    specs = [(f"above_{thr:g}", float(thr), True) for thr in above] + [(f"below_{thr:g}", float(thr), False) for thr in below]
    for name, thr, is_above in specs:
        rows = [M.threshold_mask_metrics(t, p, thr, above=is_above) for t, p in zip(true_by_case, pred_by_case)]
        block: Dict[str, float] = {"threshold": thr, "direction": "above" if is_above else "below", "n_cases": len(rows)}
        for key in ("iou", "precision", "recall", "true_frac", "pred_frac", "frac_abs_err"):
            vals = np.asarray([r[key] for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            block[f"{key}_casemean"] = float(vals.mean()) if vals.size else float("nan")
            block[f"{key}_casemed"] = float(np.median(vals)) if vals.size else float("nan")
        out[name] = block
    return out


def _evaluate_space(cases: List[Dict], true_by_case: List[np.ndarray],
                    pred_by_case: List[np.ndarray], *, save_dir: Path | None,
                    make_plots: bool, plot_space: str,
                    include_area: bool = True,
                    threshold_masks: Tuple[Tuple[float, ...], Tuple[float, ...]] | None = None) -> Dict:
    per_case: Dict[str, Dict] = {}
    for case, y_true, y_pred in zip(cases, true_by_case, pred_by_case):
        reg = M.regional_metrics(case["pos"], case["local_radius"], y_true, y_pred)
        reg["calibration"] = M.calibration_metrics(y_true, y_pred)
        reg["distribution"] = M.distribution_metrics(y_true, y_pred)
        reg["hotspot"] = M.hotspot_localization_metrics(y_true, y_pred, case["pos"])
        reg["legacy_vertex_hotspot"] = dict(reg["hotspot"])
        if threshold_masks and (threshold_masks[0] or threshold_masks[1]):
            reg["threshold_masks"] = {
                **{f"above_{thr:g}": M.threshold_mask_metrics(y_true, y_pred, thr, above=True) for thr in threshold_masks[0]},
                **{f"below_{thr:g}": M.threshold_mask_metrics(y_true, y_pred, thr, above=False) for thr in threshold_masks[1]},
            }
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
    if threshold_masks and (threshold_masks[0] or threshold_masks[1]):
        result["threshold_masks"] = _threshold_mask_block(true_by_case, pred_by_case, threshold_masks[0], threshold_masks[1])
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
    if value in {"ema", "ckpt_ema"}:
        return "ckpt_ema.pt"
    raise ValueError("checkpoint must be 'best', 'last' or 'ema'")


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
    ap.add_argument("--checkpoint", choices=("best", "last", "ema"), default="best")
    ap.add_argument("--split-path", type=str, default=None,
                    help="只读评估覆盖 split；不修改 run config/checkpoint")
    ap.add_argument("--back-transform", choices=("config", "exp_mu", "jensen"), default="config",
                    help="只读评估覆盖 log->Pa 回变换口径（config=按 run 配置；exp_mu=exp(mu)；"
                         "jensen=逐点 exp(mu+sigma^2/2)，需 gaussian_nll 双通道头）")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="默认 run_dir/eval/ckpt_<best|last>，用于隔离 checkpoint 结果")
    ap.add_argument("--data-root", type=str, default=None,
                    help="只读评估覆盖 bundle 根（2026-09-22：评估不在 run 视图根里的新病例）；不改 run config")
    ap.add_argument("--point-features-root", type=str, nargs="*", default=None,
                    help="只读评估覆盖 sidecar 根列表（与 --data-root 配套）")
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
            data_root=args.data_root or cfg.data.data_root,
            required_frame_version=cfg.data.required_frame_version,
            case_features_path=cfg.data.case_features_path,
            timesteps=getattr(cfg.data, "timesteps", "peak"), waveform_path=getattr(cfg.data, "waveform_path", None),
            extra_point_features=C.v6_point_features(cfg),
            point_features_root=args.point_features_root or getattr(cfg.data, "point_features_root", None),
            time_basis_path=getattr(cfg.data, "time_basis_path", None), time_basis_k=int(getattr(cfg.model, "time_basis_k", 0)),
            volume_time_sidecar_root=getattr(cfg.data, "volume_time_sidecar_root", None),
            volume_h5_root=getattr(cfg.data, "volume_h5_root", None),
            cycle_view_root=getattr(cfg.data, "cycle_view_root", None),
            multi_aux_channels=tuple(getattr(cfg.data, "multi_aux_channels", ()) or ()),
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
        cyc = res.get("cycle")
        if cyc:
            print(f"    cycle R2cb={cyc['cycle_r2cb']:.4f} nmae={cyc.get('cycle_nmae_range', float('nan')):.4f} "
                  f"peak_nmae={cyc.get('peak_nmae_range', float('nan')):.4f}", flush=True)
        for rname, rm in report["regional_field"].items():
            print(f"    {rname:12s} R2={rm['r2']:.4f} NRMSE={rm['nrmse_range']:.4f} n={rm['n']}")

    write_reports(result_by_part, eval_dir)
    if args.save_predictions:
        import hashlib
        provenance_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
            "checkpoint_sha256": run_dir / f"ckpt_{args.checkpoint}.pt",
            "config_sha256": run_dir / "config.json",
            "wss_stats_sha256": run_dir / "wss_global_stats.json",
        }.items() if path.is_file()}  # mocked/partial run dirs (tests) carry no files to hash
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
