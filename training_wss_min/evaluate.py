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
                      cfg: C.ExpConfig | None = None) -> np.ndarray:
    """在完整点云上预测（标准化空间）。单例过大 OOM 时回退 CPU。"""
    if cfg is not None and cfg.eval.fixed_support:
        support_n = int(cfg.data.support_n_points or cfg.data.wall_n_points)
        support_sampling = cfg.data.support_sampling or cfg.data.sampling
        seed = S.stable_seed(cfg.eval.support_seed, case["unit_id"], "eval_support")
        support_idx = D.sample_indices(case, cfg.data, seed, n_points=support_n,
                                       sampling=support_sampling, stream="eval_support")
        support_pos = torch.from_numpy(np.ascontiguousarray(case["pos"][support_idx])).to(device)
        support_x = torch.from_numpy(D.build_features(
            case, support_idx, input_features, feat_stats
        )).to(device)
        support_batch = torch.zeros(len(support_idx), dtype=torch.long, device=device)
        encoded = model.encode_support(
            support_pos, support_x, support_batch, unit_ids=[case["unit_id"]],
            epoch=0, global_seed=cfg.eval.support_seed, evaluation=True,
        )
        chunk = int(cfg.eval.query_chunk_size) or len(case["pos"])
        outputs = []
        for start in range(0, len(case["pos"]), chunk):
            idx = np.arange(start, min(start + chunk, len(case["pos"])))
            pos = torch.from_numpy(np.ascontiguousarray(case["pos"][idx])).to(device)
            x = torch.from_numpy(D.build_features(case, idx, input_features, feat_stats)).to(device)
            batch = torch.zeros(len(idx), dtype=torch.long, device=device)
            outputs.append(model.decode_query(encoded, pos, x, batch).detach().float().cpu())
        output = torch.cat(outputs)
        if output.ndim == 2:
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
    if out.ndim == 2:
        out = out[:, 0]
    return out.numpy()


def evaluate_partition(model, cases: List[Dict], cfg: C.ExpConfig, feat_stats: Dict,
                       wss_stats: Dict, device: str,
                       save_dir: Path | None = None, make_plots: bool = False) -> Dict:
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
    pred_norm_by_case = []
    for case in cases:
        y_pred_norm = predict_case_norm(
            model, case, cfg.data.input_features, feat_stats, device, cfg=cfg
        )
        pred_norm_by_case.append(np.asarray(y_pred_norm, dtype=np.float64))

    true_norm_by_case = [np.asarray(c["y_norm"], dtype=np.float64) for c in cases]
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
    for pred_norm in pred_norm_by_case:
        pred_raw = D.denormalize_wss(pred_norm, wss_stats)
        if wss_stats.get("method") == "log_z":
            pred_raw = np.clip(pred_raw, 0, None)
        pred_raw_by_case.append(np.asarray(pred_raw, dtype=np.float64))
    true_raw_by_case = [np.asarray(c["y_raw"], dtype=np.float64) for c in cases]
    physical = _evaluate_space(
        cases, true_raw_by_case, pred_raw_by_case,
        save_dir=None, make_plots=False, plot_space="physical target",
        include_area=include_area,
    )
    physical["metric_space"] = "physical_target"
    physical["surface_metric_mode"] = cfg.eval.surface_metric_mode
    physical["surface_area_metrics_status"] = (
        "complete_strict" if include_area else "not_requested"
    )
    physical["normalized"] = normalized
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
    ap.add_argument("--checkpoint", choices=("best", "last"), default="best")
    ap.add_argument("--split-path", type=str, default=None,
                    help="只读评估覆盖 split；不修改 run config/checkpoint")
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
        )
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        res = evaluate_partition(model, cases, cfg, feat_stats, wss_stats, device,
                                 save_dir=eval_dir, make_plots=(not args.no_plots and part == "test"))
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
    print("eval 写入:", eval_dir)


if __name__ == "__main__":
    main()
