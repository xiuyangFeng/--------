#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3P-Diag-00 诊断脚本（计划 §9.1.1 / §11.1 P0-G）

在仓库根目录、GNN 环境：
  python -m training.scripts.run_v3_diag00 \\
    --config training/configs/field/generated/v3_pointcloud/V3P-Diag-00_seed1.json

产出目录默认：outputs/field/diagnostics/v3p_diag00_seed{seed}/

说明：
  - 数据统计类诊断读原始 .pt 图，不依赖训练 run。
  - loss 校准 / rotation 对比 / WSS 模长一致性需一次模型前向（可用 --skip-forward 跳过）。
  - 不代替完整 `train_field.py` 1 epoch；训练仍请按 V3 配置单独提交。
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from pipeline.config import (
    NODE_FEATURE_NAMES,
    SAMPLING_CONFIG,
    TARGET_NAMES,
    WSS_TARGET_NAMES,
)
from training.core.config import ExperimentConfig
from training.core.data import (
    FieldGraphDataset,
    build_dataloader,
    build_feature_mask,
    build_required_data_keys,
)
from training.core.losses import build_loss_plugin
from training.core.models import build_model
from training.core.splits import SplitSpec
from training.core.utils import resolve_device, set_seed


def _load_graph(path: Path) -> Any:
    return torch.load(path, weights_only=False)


def _resolve_json_path(base: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _iter_graph_paths(data_root: Path, case_names: Sequence[str], graphs_subdir: str) -> List[Path]:
    out: List[Path] = []
    for case in case_names:
        d = data_root / case / graphs_subdir
        if not d.is_dir():
            continue
        out.extend(sorted(d.glob("*.pt")))
    return out


def _quantile_report(arr: np.ndarray, qs: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {f"q{int(q*1000)}": float("nan") for q in qs}
    return {f"q{int(q * 1000)}": float(np.quantile(a, q)) for q in qs}


def _case_key_from_graph_path(data_root: Path, graph_path: Path) -> str:
    rel = graph_path.relative_to(data_root)
    parts = rel.parts
    if "processed" not in parts:
        return "/".join(parts[:-2]) if len(parts) >= 3 else str(rel.parent)
    i = parts.index("processed")
    return "/".join(parts[:i]) if i > 0 else str(rel.parent)


def _collect_wall_velocity_truth(paths: List[Path], data_root: Path) -> Dict[str, Any]:
    is_i = NODE_FEATURE_NAMES.index("is_wall")
    u_i, v_i, w_i = TARGET_NAMES.index("u"), TARGET_NAMES.index("v"), TARGET_NAMES.index("w")
    gmax = {"max_abs_u": 0.0, "max_abs_v": 0.0, "max_abs_w": 0.0}
    per_case: Dict[str, Dict[str, float]] = {}
    n_wall_pts = 0
    for p in paths:
        data = _load_graph(p)
        x = data.x.detach().cpu().numpy()
        y = data.y.detach().cpu().numpy()
        wall = x[:, is_i] > 0.5
        if not np.any(wall):
            continue
        case_name = _case_key_from_graph_path(data_root, p)
        u = np.abs(y[wall, u_i])
        v = np.abs(y[wall, v_i])
        w = np.abs(y[wall, w_i])
        n_wall_pts += int(wall.sum())
        cm = {
            "max_abs_u": float(np.max(u)) if u.size else 0.0,
            "max_abs_v": float(np.max(v)) if v.size else 0.0,
            "max_abs_w": float(np.max(w)) if w.size else 0.0,
        }
        prev = per_case.get(case_name, {"max_abs_u": 0.0, "max_abs_v": 0.0, "max_abs_w": 0.0})
        per_case[case_name] = {
            "max_abs_u": max(prev["max_abs_u"], cm["max_abs_u"]),
            "max_abs_v": max(prev["max_abs_v"], cm["max_abs_v"]),
            "max_abs_w": max(prev["max_abs_w"], cm["max_abs_w"]),
        }
        gmax["max_abs_u"] = max(gmax["max_abs_u"], cm["max_abs_u"])
        gmax["max_abs_v"] = max(gmax["max_abs_v"], cm["max_abs_v"])
        gmax["max_abs_w"] = max(gmax["max_abs_w"], cm["max_abs_w"])
    return {
        "description": "训练集壁面节点 CFD 真值速度（z-score 标签空间）最大绝对值；用于 §4.1.1 noslip 档位。",
        "n_graphs": len(paths),
        "wall_point_count_summed": n_wall_pts,
        "global_max_abs_on_wall": gmax,
        "per_case_max_abs_on_wall": per_case,
    }


def _collect_interior_dist_to_wall(
    paths: List[Path], boundary_mm: float
) -> Dict[str, Any]:
    is_i = NODE_FEATURE_NAMES.index("is_wall")
    dist_i = NODE_FEATURE_NAMES.index("dist_to_wall")
    all_dist: List[np.ndarray] = []
    per_graph_max: List[Dict[str, Any]] = []
    violations = 0
    for p in paths:
        data = _load_graph(p)
        x = data.x.detach().cpu().numpy()
        if x.shape[1] <= dist_i:
            raise ValueError(f"{p}: 节点特征维度过小，缺少 dist_to_wall（索引 {dist_i}）")
        interior = x[:, is_i] <= 0.5
        if not np.any(interior):
            continue
        d = x[interior, dist_i].astype(np.float64)
        all_dist.append(d)
        gm = float(np.max(d))
        per_graph_max.append({"graph": str(p), "interior_max_dist_to_wall_mm": gm})
        if gm >= boundary_mm - 1e-6:
            violations += 1
    if not all_dist:
        raise RuntimeError("未找到任何内部点，无法校验 §2.1.1 dist_to_wall。")
    cat = np.concatenate(all_dist)
    ok = bool(np.max(cat) < boundary_mm - 1e-9)
    return {
        "boundary_threshold_mm": boundary_mm,
        "pipeline_sampling_lock": {
            "boundary_core_ratio": list(SAMPLING_CONFIG["boundary_core_ratio"]),
            "allow_core_fallback": SAMPLING_CONFIG["allow_core_fallback"],
            "boundary_threshold_mm": SAMPLING_CONFIG["boundary_threshold"],
        },
        "interior_dist_stats_mm": {
            "max": float(np.max(cat)),
            "q950": float(np.quantile(cat, 0.95)),
            "q500": float(np.quantile(cat, 0.50)),
            "mean": float(np.mean(cat)),
        },
        "mask_equivalence_ok": ok,
        "n_graphs_with_interior_violation": violations,
        "note": "V3 要求 interior_max_dist_to_wall < boundary_threshold；若不成立请勿启动 Anchor/Base/Main。",
    }


def _stack_wall_wss(paths: List[Path]) -> np.ndarray:
    is_i = NODE_FEATURE_NAMES.index("is_wall")
    chunks: List[np.ndarray] = []
    for p in paths:
        data = _load_graph(p)
        if not hasattr(data, "y_wss") or data.y_wss is None:
            continue
        yw = data.y_wss.detach().cpu().numpy()
        x = data.x.detach().cpu().numpy()
        wall = x[:, is_i] > 0.5
        chunks.append(yw[wall].astype(np.float64))
    if not chunks:
        raise RuntimeError("无法收集壁面 WSS：检查 y_wss 是否存在。")
    return np.vstack(chunks)


def _wss_distribution_payload(X: np.ndarray) -> Dict[str, Any]:
    m, d = X.shape
    dims = list(WSS_TARGET_NAMES)
    per_dim: Dict[str, Any] = {}
    out_3s = []
    out_5s = []
    for j, name in enumerate(dims):
        col = X[:, j]
        mu = float(np.mean(col))
        sig = float(np.std(col, ddof=0)) if m else 0.0
        frac3 = float(np.mean(np.abs(col - mu) > 3 * sig)) if sig > 1e-12 else 0.0
        frac5 = float(np.mean(np.abs(col - mu) > 5 * sig)) if sig > 1e-12 else 0.0
        out_3s.append(frac3)
        out_5s.append(frac5)
        sorted_c = np.sort(col)
        per_dim[name] = {
            "mean": mu,
            "std": sig,
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "frac_gt_3sigma_marginal": frac3,
            "frac_gt_5sigma_marginal": frac5,
            "quantiles": _quantile_report(col, [0.5, 0.95, 0.99]),
        }
    mag = X[:, 0]
    comp_mag = np.sqrt(np.sum(X[:, 1:4] ** 2, axis=1))
    p50_m = float(np.quantile(mag, 0.5)) if m else 0.0
    p99_m = float(np.quantile(mag, 0.99)) if m else 0.0
    tail_ratio = float(p99_m / p50_m) if p50_m > 1e-12 else float("inf")
    return {
        "n_wall_samples": int(m),
        "per_dimension": per_dim,
        "outlier_frac_gt_3sigma_per_dim": {dims[k]: out_3s[k] for k in range(d)},
        "outlier_frac_gt_5sigma_per_dim": {dims[k]: out_5s[k] for k in range(d)},
        "magnitude_tail_p99_over_p50": tail_ratio,
        "huber_probe_hint": "若 ≥5% 点任一倍超出 3σ 或 p99/p50>5×，按计划 §5.3 考虑 V3P-WSS-03。",
    }


def _case_level_wss_mean_p95(paths: List[Path]) -> Dict[str, Any]:
    is_i = NODE_FEATURE_NAMES.index("is_wall")
    mag_means: List[float] = []
    mag_p95: List[float] = []
    for p in paths:
        data = _load_graph(p)
        if not hasattr(data, "y_wss") or data.y_wss is None:
            continue
        yw = data.y_wss.detach().cpu().numpy()
        x = data.x.detach().cpu().numpy()
        wall = x[:, is_i] > 0.5
        mag = yw[wall, 0].astype(np.float64)
        if mag.size == 0:
            continue
        mag_means.append(float(np.mean(mag)))
        mag_p95.append(float(np.quantile(mag, 0.95)))
    arr_m = np.asarray(mag_means)
    arr_p = np.asarray(mag_p95)
    return {
        "n_cases_with_wss": int(arr_m.size),
        "cross_case_mean_wss_mag_mean": float(np.mean(arr_m)) if arr_m.size else float("nan"),
        "cross_case_mean_wss_mag_quantiles": _quantile_report(arr_m, [0.05, 0.5, 0.95]),
        "cross_case_p95_wss_mag_quantiles": _quantile_report(arr_p, [0.05, 0.5, 0.95]),
    }


def _sample_tensors_from_graphs(
    paths: List[Path],
    max_nodes_total: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """从图中抽样节点级 z-score 目标与部分几何特征（用于训测分位对比）。"""
    is_i = NODE_FEATURE_NAMES.index("is_wall")
    nr_i = NODE_FEATURE_NAMES.index("NormRadius")
    cur_i = NODE_FEATURE_NAMES.index("Curvature")
    buckets: Dict[str, List[np.ndarray]] = defaultdict(list)
    budget = max_nodes_total
    order = paths.copy()
    rng.shuffle(order)
    for p in order:
        if budget <= 0:
            break
        data = _load_graph(p)
        x = data.x.detach().cpu().numpy()
        y = data.y.detach().cpu().numpy()
        n = x.shape[0]
        take = min(n, budget)
        if take < n:
            idx = rng.choice(n, size=take, replace=False)
        else:
            idx = np.arange(n)
        buckets["u"].append(y[idx, 0])
        buckets["v"].append(y[idx, 1])
        buckets["w"].append(y[idx, 2])
        buckets["p"].append(y[idx, 3])
        buckets["NormRadius"].append(x[idx, nr_i])
        buckets["Curvature"].append(x[idx, cur_i])
        if hasattr(data, "y_wss") and data.y_wss is not None:
            yw = data.y_wss.detach().cpu().numpy()
            wall = x[:, is_i] > 0.5
            widx = idx[np.isin(idx, np.nonzero(wall)[0])]
            if widx.size:
                buckets["wss"].append(yw[widx, 0])
                buckets["wss_x"].append(yw[widx, 1])
                buckets["wss_y"].append(yw[widx, 2])
                buckets["wss_z"].append(yw[widx, 3])
        budget -= take
    return {k: np.concatenate(v) for k, v in buckets.items() if v}


def _train_test_distribution_diff(
    paths_train: List[Path],
    paths_test: List[Path],
    seed: int,
    max_nodes_per_split: int = 800_000,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + 2026)
    tr = _sample_tensors_from_graphs(paths_train, max_nodes_per_split, rng)
    rng = np.random.default_rng(seed + 4242)
    te = _sample_tensors_from_graphs(paths_test, max_nodes_per_split, rng)
    channels = ["u", "v", "w", "p", "wss", "wss_x", "wss_y", "wss_z", "NormRadius", "Curvature"]
    rows: Dict[str, Any] = {}
    warnings: List[str] = []
    for ch in channels:
        if ch not in tr or ch not in te:
            rows[ch] = {"note": "缺通道（通常 test/train 壁面抽样过少导致无 wss_*）"}
            continue
        qa_tr = np.abs(tr[ch])
        qa_te = np.abs(te[ch])
        q99_tr = float(np.quantile(qa_tr, 0.99))
        q99_te = float(np.quantile(qa_te, 0.99))
        q995_tr = float(np.quantile(qa_tr, 0.995))
        q995_te = float(np.quantile(qa_te, 0.995))
        ratio99 = q99_te / q99_tr if q99_tr > 1e-12 else float("inf")
        flag = ratio99 > 1.5 and q99_te > q99_tr
        rows[ch] = {
            "train_abs_q99": q99_tr,
            "test_abs_q99": q99_te,
            "train_abs_q995": q995_tr,
            "test_abs_q995": q995_te,
            "test_over_train_q99_ratio": ratio99,
            "ood_warning_plan_9_1_1": bool(flag),
        }
        if flag:
            warnings.append(f"{ch}: |·| 的测试集 99 分位 / 训练集 99 分位 = {ratio99:.3f} (>1.5)")
    # NormRadius max
    nr_tr_max = float(np.max(tr["NormRadius"])) if "NormRadius" in tr else float("nan")
    nr_te_max = float(np.max(te["NormRadius"])) if "NormRadius" in te else float("nan")
    if nr_te_max > 1.0 + 1e-6:
        warnings.append(f"测试集 NormRadius max={nr_te_max:.4f} > 1.0（min-max 可能越界）")
    rows["_NormRadius_max_train"] = nr_tr_max
    rows["_NormRadius_max_test"] = nr_te_max
    return {
        "channels": rows,
        "ood_warnings": warnings,
    }


def _collect_bc_vectors(data_root: Path, cases: Sequence[str]) -> np.ndarray:
    rows: List[List[float]] = []
    for case in cases:
        bc_path = data_root / case / "processed" / "normalized" / "bc_metadata_normalized.json"
        if not bc_path.exists():
            bc_path = data_root / case / "processed" / "graphs" / "bc_metadata_normalized.json"
        if not bc_path.exists():
            continue
        with open(bc_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        data_block = meta.get("data", {})
        for vec in data_block.values():
            if isinstance(vec, list) and len(vec) >= 5:
                rows.append([float(vec[i]) for i in range(5)])
    if not rows:
        raise RuntimeError("未读到任何 bc_metadata_normalized.json（检查病例路径）。")
    return np.asarray(rows, dtype=np.float64)


def _bc_train_test_report(data_root: Path, train_cases: Sequence[str], test_cases: Sequence[str]) -> Dict[str, Any]:
    B_tr = _collect_bc_vectors(data_root, train_cases)
    B_te = _collect_bc_vectors(data_root, test_cases)
    names = ["BC_Inlet", "BC_O1", "BC_O2", "BC_O3", "BC_O4"]
    rep: Dict[str, Any] = {}
    warnings: List[str] = []
    for j, nm in enumerate(names):
        tr_j = B_tr[:, j]
        te_j = B_te[:, j]
        rep[nm] = {
            "train_q05_q95": [float(np.quantile(tr_j, 0.05)), float(np.quantile(tr_j, 0.95))],
            "test_q05_q95": [float(np.quantile(te_j, 0.05)), float(np.quantile(te_j, 0.95))],
            "train_max_abs": float(np.max(np.abs(tr_j))),
            "test_max_abs": float(np.max(np.abs(te_j))),
        }
        # 粗粒度 OOD：测试 q05 低于训练 q05 很多或反之
        if np.quantile(te_j, 0.05) < np.quantile(tr_j, 0.05) - 3 * np.std(tr_j):
            warnings.append(f"{nm}: 测试集低端 BC 可能显著 OOD（相对训练）")
    return {"per_bc_channel": rep, "warnings": warnings}


def _loss_calibration_batches(
    config: ExperimentConfig,
    split: SplitSpec,
    device: torch.device,
    n_batches: int,
    rotation_prob_high: float,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """在**同一组随机初始化权重**下，对比高旋转增强概率与配置中的 rotation_prob。"""
    feature_mask = build_feature_mask(
        enabled_node_features=config.data.enabled_node_features,
        enabled_global_features=config.data.enabled_global_features,
    )
    req = build_required_data_keys(config.model.name, wss_dim=config.model.wss_dim)
    base_aug = (config.data.augment_config or {}).copy()
    cfg_rot = float(base_aug.get("rotation_prob", 0.0))

    init_seed = config.system.seed + 9001
    set_seed(init_seed, deterministic=config.system.deterministic)
    template = build_model(
        model_name=config.model.name,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        heads=config.model.heads,
        use_transformer_prenorm=config.model.use_transformer_prenorm,
        wss_dim=config.model.wss_dim,
        head_layout=config.model.head_layout,
        wss_head_dropout=config.model.wss_head_dropout,
    ).to(device)
    init_sd = copy.deepcopy(template.state_dict())
    del template

    def run_with_rotation(
        rot_prob: float, state: Dict[str, torch.Tensor]
    ) -> Tuple[List[Dict[str, float]], List[float]]:
        aug = base_aug.copy()
        aug["rotation_prob"] = float(rot_prob)
        ds = FieldGraphDataset(
            root=config.data.data_root,
            case_names=split.train_cases,
            graphs_subdir=config.data.graphs_subdir,
            augment=True,
            augment_config=aug,
            preload=config.data.preload,
            feature_mask=feature_mask,
            required_keys=req,
        )
        loader = build_dataloader(
            ds,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            seed=config.system.seed,
        )
        model = build_model(
            model_name=config.model.name,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            heads=config.model.heads,
            use_transformer_prenorm=config.model.use_transformer_prenorm,
            wss_dim=config.model.wss_dim,
            head_layout=config.model.head_layout,
            wss_head_dropout=config.model.wss_head_dropout,
        ).to(device)
        model.load_state_dict(state)
        wss_tensor = (
            torch.tensor(config.optim.wss_weights, dtype=torch.float32)
            if config.model.wss_dim > 0
            else None
        )
        plugin = build_loss_plugin(
            config.physics,
            interior_loss_boost=config.optim.interior_loss_boost,
            wss_loss_weight=config.optim.wss_loss_weight,
            wss_weights=wss_tensor,
            wss_loss_type=config.optim.wss_loss_type,
            wss_huber_beta=config.optim.wss_huber_beta,
            domain_loss_config=config.optim.domain_loss,
        )
        rows: List[Dict[str, float]] = []
        ints: List[float] = []
        model.train()
        for bi, batch in enumerate(loader):
            if bi >= n_batches:
                break
            batch = batch.to(device)
            out = model(batch)
            if isinstance(out, tuple):
                pred, wss_pred = out
            else:
                pred, wss_pred = out, None
            br = plugin.build_loss(
                model=model,
                batch=batch,
                pred=pred,
                target=batch.y,
                data_weights=torch.tensor(config.optim.target_weights, dtype=torch.float32, device=device),
                epoch=1,
                train=True,
                wss_pred=wss_pred,
            )
            sd = br.scalar_dict()
            rows.append(sd)
            ints.append(sd["loss_interior_velocity"])
        return rows, ints

    rows_hi, ints_hi = run_with_rotation(rotation_prob_high, copy.deepcopy(init_sd))
    rows_lo, ints_lo = run_with_rotation(cfg_rot, copy.deepcopy(init_sd))

    def summarize(rows: List[Dict[str, float]]) -> Dict[str, Any]:
        keys = [
            "loss",
            "weighted_loss_interior_velocity",
            "weighted_loss_noslip_velocity",
            "weighted_loss_interior_pressure",
            "weighted_loss_wall_pressure",
            "weighted_loss_wall_wss",
            "loss_interior_velocity",
        ]
        arr = {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in keys if rows and k in rows[0]}
        return {
            "mean": {k: float(np.mean(v)) for k, v in arr.items()},
            "median": {k: float(np.median(v)) for k, v in arr.items()},
            "max": {k: float(np.max(v)) for k, v in arr.items()},
        }

    calib = {
        "n_batches": min(n_batches, len(rows_hi)),
        "rotation_prob_probe": rotation_prob_high,
        "rotation_prob_config": cfg_rot,
        "init_seed_for_shared_weights": init_seed,
        "summaries": {
            "at_rotation_probe": summarize(rows_hi),
            "at_config_rotation": summarize(rows_lo),
        },
        "L_interior_velocity_across_batches": {
            "rotation_probe_std": float(np.std(np.asarray(ints_hi, dtype=np.float64))),
            "rotation_probe_mean": float(np.mean(np.asarray(ints_hi, dtype=np.float64))),
            "rotation_config_std": float(np.std(np.asarray(ints_lo, dtype=np.float64))),
            "rotation_config_mean": float(np.mean(np.asarray(ints_lo, dtype=np.float64))),
        },
    }
    mean_lo_int = float(np.mean(np.asarray(ints_lo, dtype=np.float64))) if ints_lo else 0.0
    std_lo_int = float(np.std(np.asarray(ints_lo, dtype=np.float64))) if ints_lo else 0.0
    ratio = std_lo_int / mean_lo_int if mean_lo_int > 1e-12 else float("inf")
    calib["L_interior_velocity_I4_hint"] = {
        "batch_mean_L_interior_velocity_at_config_rotation": mean_lo_int,
        "batch_std_L_interior_velocity_at_config_rotation": std_lo_int,
        "std_over_mean": ratio,
        "suggest_lambda_vel_int_0_5_if_std_over_mean_gt_0_5": bool(ratio > 0.5),
    }
    mean_hi = calib["summaries"]["at_rotation_probe"]["mean"]["loss"]
    mean_lo = calib["summaries"]["at_config_rotation"]["mean"]["loss"]
    augment_delta = {"mean_L_total_probe_minus_config": float(mean_hi - mean_lo)}
    return calib, augment_delta


def _wss_magnitude_consistency_test(
    config: ExperimentConfig,
    split: SplitSpec,
    device: torch.device,
    max_batches: int,
) -> Dict[str, Any]:
    feature_mask = build_feature_mask(
        enabled_node_features=config.data.enabled_node_features,
        enabled_global_features=config.data.enabled_global_features,
    )
    req = build_required_data_keys(config.model.name, wss_dim=config.model.wss_dim)
    ds = FieldGraphDataset(
        root=config.data.data_root,
        case_names=split.test_cases,
        graphs_subdir=config.data.graphs_subdir,
        augment=False,
        preload=config.data.preload,
        feature_mask=feature_mask,
        required_keys=req,
    )
    loader = build_dataloader(
        ds,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    model = build_model(
        model_name=config.model.name,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        heads=config.model.heads,
        use_transformer_prenorm=config.model.use_transformer_prenorm,
        wss_dim=config.model.wss_dim,
        head_layout=config.model.head_layout,
        wss_head_dropout=config.model.wss_head_dropout,
    ).to(device)
    model.eval()
    is_i = NODE_FEATURE_NAMES.index("is_wall")
    errs: List[float] = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            batch = batch.to(device)
            out = model(batch)
            if isinstance(out, tuple):
                _, wss_pred = out
            else:
                wss_pred = None
            if wss_pred is None:
                continue
            wm = batch.x[:, is_i].bool()
            if not wm.any():
                continue
            wp = wss_pred[wm]
            mag = wp[:, 0]
            recomputed = torch.sqrt(torch.clamp((wp[:, 1:] ** 2).sum(dim=1), min=0.0))
            denom = torch.clamp(torch.abs(mag), min=1e-8)
            rel = (torch.abs(mag - recomputed) / denom).detach().cpu().numpy()
            errs.append(rel)
    if not errs:
        return {"note": "未收集到 WSS 预测（检查 wss_dim > 0）。"}
    cat = np.concatenate(errs)
    qs = [0.0, 0.25, 0.5, 0.75, 0.95, 1.0]
    cdf = {f"p{int(q*100)}": float(np.quantile(cat, q)) for q in qs}
    return {
        "description": "测试集壁面节点：|pred_wss_mag - sqrt(pred_wss_x²+...)| / |pred_wss_mag| 相对误差分位数（随机初始化模型，用于 §3.3 冗余诊断基线）。",
        "n_wall_predictions": int(cat.size),
        "relative_error_quantiles": cdf,
        "trigger_v3p_wss04_hint": bool(cdf.get("p50", 0.0) > 0.05),
    }


def _write_text(path: Path, lines: Sequence[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _noslip_lines(truth: Dict[str, Any]) -> List[str]:
    g = truth["global_max_abs_on_wall"]
    mu = max(g["max_abs_u"], g["max_abs_v"], g["max_abs_w"])
    if mu < 1e-6:
        decision = "weak_reg"
        lam = 0.1
        note = "壁面速度量级 <1e-6，noslip 降权为弱正则。"
    elif mu < 1e-3:
        decision = "soft"
        lam = 0.5
        note = "介于 1e-6 与 1e-3，维持软无滑移档位。"
    else:
        decision = "raw_truth"
        lam = 1.0
        note = ">=1e-3：监督目标应使用 CFD 壁面真值而非强行置 0。"
    return [
        f"global_max_abs_velocity_wall = {mu:.6e}",
        f"recommended_lambda_vel_noslip = {lam}",
        f"meta.noslip_decision = {decision}",
        note,
    ]


def _weight_calib_lines(calib: Dict[str, Any]) -> List[str]:
    """§4.2.1：基于「配置 rotation 档」各 batch 的中位数粗略给出调档建议。"""
    summaries = calib.get("summaries", {})
    block = summaries.get("at_config_rotation") or {}
    med = block.get("median") or {}
    wkeys = [
        "weighted_loss_interior_velocity",
        "weighted_loss_noslip_velocity",
        "weighted_loss_interior_pressure",
        "weighted_loss_wall_pressure",
        "weighted_loss_wall_wss",
    ]
    vals = {k: float(med.get(k, 0.0)) for k in wkeys}
    if not any(v > 0 for v in vals.values()):
        return ["（无 weighted_loss 中位数：检查 domain_loss 是否启用及 skip-forward）"]
    lines = [
        "median 取自 summaries.at_config_rotation（与训练 JSON 中 augment.rotation_prob 一致）",
        json.dumps(vals, indent=2, ensure_ascii=False),
    ]
    for k, v in vals.items():
        others = [vals[o] for o in wkeys if o != k and vals[o] > 0]
        if not others:
            continue
        om = float(np.median(others))
        if om > 0 and v > 5 * om:
            lines.append(f"建议：{k} 中位数 {v:.4e} > 5× 其它项中位数 {om:.4e} → 考虑将对应 lambda 减半（§4.2.1）。")
    w_wss = vals.get("weighted_loss_wall_wss", 0.0)
    field_sum_med = sum(vals[k] for k in wkeys if k != "weighted_loss_wall_wss")
    if w_wss > 2 * field_sum_med and field_sum_med > 0:
        lines.append("weighted_wss 的中位数 > 2× 其它加权和：建议 lambda_wss 先降到 0.05 档（§4.2.1 / §5.1）。")
    return lines


def _plot_wss_figs(X: np.ndarray, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    dims = list(WSS_TARGET_NAMES)
    for j, name in enumerate(dims):
        col = X[:, j]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(col, bins=80, density=True, alpha=0.75)
        ax.set_title(f"Train wall WSS / {name} (z-score)")
        ax.set_xlabel(name)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_wss_dist_{name}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="V3P-Diag-00 数据统计与 loss 校准（§9.1.1）")
    parser.add_argument("--config", required=True, help="V3P-Diag-00 JSON 路径")
    parser.add_argument("--output-dir", default="", help="默认 outputs/field/diagnostics/v3p_diag00_seed{S}")
    parser.add_argument("--calibration-batches", type=int, default=50, help="loss / augment 对比的前 N 个 batch")
    parser.add_argument("--rotation-probe", type=float, default=0.5, help="对比用的增强旋转概率（计划 §8.6）")
    parser.add_argument("--wss-consistency-batches", type=int, default=200, help="WSS 模长一致性最多前向 batch 数")
    parser.add_argument("--skip-forward", action="store_true", help="跳过 GPU/模型前向（仅数据统计）")
    args = parser.parse_args()

    cwd = Path.cwd().resolve()
    cfg_path = _resolve_json_path(cwd, args.config)
    config = ExperimentConfig.from_json(str(cfg_path))
    config.validate()
    split = SplitSpec.from_json(str(_resolve_json_path(cwd, config.data.split_file)))

    seed = config.system.seed
    out_dir = Path(args.output_dir) if args.output_dir else cwd / "outputs" / "field" / "diagnostics" / f"v3p_diag00_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = _resolve_json_path(cwd, config.data.data_root)
    train_paths = _iter_graph_paths(data_root, split.train_cases, config.data.graphs_subdir)
    test_paths = _iter_graph_paths(data_root, split.test_cases, config.data.graphs_subdir)
    if not train_paths:
        raise RuntimeError(f"训练集未找到任何图：root={data_root}, graphs_subdir={config.data.graphs_subdir}")

    print(
        f"[diag00] 图枚举完成 train_pt={len(train_paths)} test_pt={len(test_paths)} "
        f"→ 后续多次顺序读盘，耗时可数分钟至十几分钟（非悬挂）。",
        flush=True,
    )

    norm_path = data_root / "normalization_params_global.json"
    norm_txt_lines = [
        f"path={norm_path}",
        f"exists={norm_path.exists()}",
    ]
    if norm_path.exists():
        with open(norm_path, "r", encoding="utf-8") as f:
            norm_js = json.load(f)
        stats = norm_js.get("statistics", {})
        norm_txt_lines.append(f"channels_with_statistics={sorted(stats.keys())}")
        _draft = {ch: stats.get(ch, {}) for ch in ["u", "v", "w", "p", "wss", "wss_x", "wss_y", "wss_z"]}
        with open(out_dir / "stats_reference_from_norm_params.json", "w", encoding="utf-8") as f:
            json.dump(_draft, f, indent=2, ensure_ascii=False)
    else:
        norm_txt_lines.append("ERROR: normalization_params_global.json 缺失；V3P-Main-01 不应开跑。")
    _write_text(out_dir / "norm_params_consistency.txt", norm_txt_lines)

    print("[diag00] wall_velocity_truth …", flush=True)
    wall_truth = _collect_wall_velocity_truth(train_paths, data_root)
    with open(out_dir / "wall_velocity_truth.json", "w", encoding="utf-8") as f:
        json.dump(wall_truth, f, indent=2, ensure_ascii=False)
    _write_text(out_dir / "noslip_decision.txt", _noslip_lines(wall_truth))

    print("[diag00] interior_dist_to_wall …", flush=True)
    dist_payload = _collect_interior_dist_to_wall(
        train_paths + test_paths,
        float(SAMPLING_CONFIG["boundary_threshold"]),
    )
    with open(out_dir / "interior_dist_to_wall_stats.json", "w", encoding="utf-8") as f:
        json.dump(dist_payload, f, indent=2, ensure_ascii=False)

    print("[diag00] WSS 分布（壁面堆叠）…", flush=True)
    Xw = _stack_wall_wss(train_paths)
    wss_payload = _wss_distribution_payload(Xw)
    wss_payload["case_level"] = _case_level_wss_mean_p95(train_paths)
    with open(out_dir / "wss_distribution_train.json", "w", encoding="utf-8") as f:
        json.dump(wss_payload, f, indent=2, ensure_ascii=False)
    _plot_wss_figs(Xw, out_dir)

    print("[diag00] train vs test 分位 + BC …", flush=True)
    ttd = _train_test_distribution_diff(train_paths, test_paths, seed=seed)
    ttd["bc_train_vs_test"] = _bc_train_test_report(data_root, split.train_cases, split.test_cases)
    with open(out_dir / "train_test_distribution_diff.json", "w", encoding="utf-8") as f:
        json.dump(ttd, f, indent=2, ensure_ascii=False)

    augment_extra: Dict[str, float] = {}
    calib_json: Dict[str, Any] = {"note": "跳过模型前向（--skip-forward）"}
    if not args.skip_forward:
        device = resolve_device(config.system.device)
        set_seed(seed, deterministic=config.system.deterministic)
        print(
            f"[diag00] GPU 前向：loss 校准 batches={args.calibration_batches} ×2（rotation 对比）… device={device}",
            flush=True,
        )
        calib_json, augment_extra = _loss_calibration_batches(
            config,
            split,
            device,
            n_batches=max(1, args.calibration_batches),
            rotation_prob_high=args.rotation_probe,
        )
        print(
            f"[diag00] WSS 模长一致性（测试集前向，最多 {args.wss_consistency_batches} batch）…",
            flush=True,
        )
        mag_csv = _wss_magnitude_consistency_test(
            config,
            split,
            device,
            max_batches=max(1, args.wss_consistency_batches),
        )
        with open(out_dir / "wss_magnitude_consistency.json", "w", encoding="utf-8") as f:
            json.dump(mag_csv, f, indent=2, ensure_ascii=False)
    else:
        with open(out_dir / "wss_magnitude_consistency.json", "w", encoding="utf-8") as f:
            json.dump({"skipped": True}, f, indent=2)

    with open(out_dir / "weighted_loss_calibration.json", "w", encoding="utf-8") as f:
        json.dump(calib_json, f, indent=2, ensure_ascii=False)

    # weight_calibration.txt（基于 median 规则 §4.2.1）
    if isinstance(calib_json, dict) and "summaries" in calib_json:
        wl_lines = _weight_calib_lines(calib_json)
    else:
        wl_lines = ["跳过或未生成 summaries。"]
    _write_text(out_dir / "weight_calibration.txt", wl_lines)

    aug_lines = [
        f"rotation_probe_prob={args.rotation_probe}",
        json.dumps(augment_extra, indent=2, ensure_ascii=False) if augment_extra else "{}",
        "若 probe（高 rotation_prob）相对配置 rotation 的 mean L_total 差异显著，按 §8.6 写入 meta.augment_decision；V3 主线默认关闭 rotation。",
    ]
    _write_text(out_dir / "augment_decision.txt", aug_lines)

    manifest = {
        "exp_id": config.meta.exp_id,
        "seed": seed,
        "data_root": str(data_root),
        "split_version": split.split_version,
        "output_dir": str(out_dir),
        "artifacts": [
            "norm_params_consistency.txt",
            "wall_velocity_truth.json",
            "noslip_decision.txt",
            "interior_dist_to_wall_stats.json",
            "wss_distribution_train.json",
            "fig_wss_dist_*.png",
            "train_test_distribution_diff.json",
            "weighted_loss_calibration.json",
            "weight_calibration.txt",
            "wss_magnitude_consistency.json",
            "augment_decision.txt",
        ],
    }
    with open(out_dir / "diag_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"V3P-Diag-00 脚本完成，输出目录: {out_dir}")
    if not dist_payload.get("mask_equivalence_ok"):
        print("[WARN] interior max dist_to_wall 未严格小于 boundary_threshold；请检查采样资产。")
    if norm_path.exists() is False:
        print("[WARN] 缺少 normalization_params_global.json")


if __name__ == "__main__":
    main()
