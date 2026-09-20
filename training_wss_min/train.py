#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练入口（配置驱动）。

  python -m training_wss_min.train --config training_wss_min/configs/<name>.json

产物写入 config.run_dir（training_wss_min/runs/<name>/）：
- config.json / feature_stats.json / weight_quantiles.json
- train.log / history.jsonl / history.png
- ckpt_best.pt / ckpt_top{k}.pt / ckpt_last.pt

默认：训练用稀疏子采样，val 监控恒在**完整点云**上。
若 ``selection_rule='train_loss'``（或 split 无 val）：不加载/不评估 val，
按 epoch 训练损失选 best（score=-train_loss），``early_stop_patience<=0`` 关闭早停。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config as C
from . import dataset as D
from .evaluate import evaluate_partition
from .models import build_model
from .objectives import compute_loss, compute_selection_score
from .paired_initialization import build_paired_model
from .runtime import lr_lambda_factory, seed_all, setup_logger


def plot_history(history, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    tr = [h["train_loss"] for h in history]
    ve = [(h["epoch"], h["val_r2_casemean"]) for h in history if "val_r2_casemean" in h]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ep, tr, "b-", label="train loss (norm MSE)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="b"); ax1.set_yscale("log")
    if ve:
        ax2 = ax1.twinx()
        ax2.plot([e for e, _ in ve], [v for _, v in ve], "r.-", label="val R2 (casemean)")
        ax2.set_ylabel("val R2", color="r"); ax2.set_ylim(-0.2, 1.0)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def _save_topk(run_dir: Path, topk: list, model, epoch: int, score: float, cfg_name: str,
               k: int):
    """维护 score 最大的 top-k checkpoint（heap 存 (-score, epoch, path)）。"""
    path = run_dir / f"ckpt_top{len(topk) + 1}_e{epoch}.pt"
    # 先写入临时名，再整理
    tmp = run_dir / f"ckpt_cand_e{epoch}.pt"
    torch.save({"model": model.state_dict(), "epoch": epoch, "metric": score,
                "cfg_name": cfg_name}, tmp)
    heapq.heappush(topk, (score, epoch, tmp))
    while len(topk) > k:
        _, _, old = heapq.heappop(topk)
        if old.exists():
            old.unlink()
    # 同步 ckpt_best = 当前最高分
    best = max(topk, key=lambda t: t[0])
    best_path = run_dir / "ckpt_best.pt"
    torch.save(torch.load(best[2], map_location="cpu", weights_only=False), best_path)


def _record_nonfinite_event(run_dir: Path, epoch: int, step: int, kind: str) -> None:
    with (run_dir / "nonfinite_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"epoch": epoch, "step": step, "kind": kind}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=None, help="覆盖配置里的 epochs（调试用）")
    args = ap.parse_args()

    cfg = C.ExpConfig.from_json(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    run_dir = cfg.run_dir
    log = setup_logger(run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_all(cfg.train.seed)

    log.info("=== train %s ===", cfg.name)
    log.info("device=%s  notes=%s", device, cfg.notes)
    log.info("data: wall_n=%s sampling=%s features=%s target=%s persistent_workers=%s",
             cfg.data.wall_n_points, cfg.data.sampling, cfg.data.input_features,
             cfg.data.target, cfg.data.persistent_workers)
    log.info("support/query: support_n=%s support_sampling=%s query_mode=%s query_n=%s "
             "query_sampling=%s fixed_support_eval=%s",
             cfg.data.support_n_points or cfg.data.wall_n_points,
             cfg.data.support_sampling or cfg.data.sampling, cfg.data.query_mode,
             cfg.data.query_n_points or cfg.data.support_n_points or cfg.data.wall_n_points,
             cfg.data.query_sampling or cfg.data.support_sampling or cfg.data.sampling,
             cfg.eval.fixed_support)

    # ---- 数据 ----
    stats_path = Path(cfg.data.wss_stats_path)
    if not stats_path.is_file():
        stats_path = C.GLOBAL_STATS
    wss_stats = D.load_wss_stats(stats_path)
    tr_cases = D.load_partition(
        cfg.data.split_path, "train", wss_stats, strict=True, target=cfg.data.target,
        target_normalization=cfg.data.target_normalization,
        data_root=cfg.data.data_root,
        required_frame_version=cfg.data.required_frame_version,
        case_features_path=cfg.data.case_features_path,
        timesteps=getattr(cfg.data, "timesteps", "peak"), waveform_path=getattr(cfg.data, "waveform_path", None),
        extra_point_features=C.v6_point_features(cfg),
        point_features_root=getattr(cfg.data, "point_features_root", None),
        time_basis_path=getattr(cfg.data, "time_basis_path", None), time_basis_k=int(getattr(cfg.model, "time_basis_k", 0)),
        volume_time_sidecar_root=getattr(cfg.data, "volume_time_sidecar_root", None),
        volume_h5_root=getattr(cfg.data, "volume_h5_root", None),
    )
    select_by_train_loss = cfg.train.selection_rule in ("train_loss", "train_loss_ema")
    select_by_ema = cfg.train.selection_rule == "train_loss_ema"
    train_loss_ema = None
    val_labels = D.load_split_cases(cfg.data.split_path, "val")
    if select_by_train_loss or not val_labels:
        va_cases = []
    else:
        va_cases = D.load_partition(
            cfg.data.split_path, "val", wss_stats, strict=True, target=cfg.data.target,
            target_normalization=cfg.data.target_normalization,
            data_root=cfg.data.data_root,
            required_frame_version=cfg.data.required_frame_version,
            case_features_path=cfg.data.case_features_path,
            timesteps=getattr(cfg.data, "timesteps", "peak"), waveform_path=getattr(cfg.data, "waveform_path", None),
            extra_point_features=C.v6_point_features(cfg),
            point_features_root=getattr(cfg.data, "point_features_root", None),
            time_basis_path=getattr(cfg.data, "time_basis_path", None), time_basis_k=int(getattr(cfg.model, "time_basis_k", 0)),
            volume_time_sidecar_root=getattr(cfg.data, "volume_time_sidecar_root", None),
            volume_h5_root=getattr(cfg.data, "volume_h5_root", None),
        )
    joint_target = cfg.data.target == "velocity_pressure"
    if joint_target and not cfg.data.feature_stats_path:
        raise ValueError("joint volume runs require frozen historical volume feature_stats_path")
    feature_stats_source = None
    if cfg.data.feature_stats_path:
        feature_stats_source = Path(cfg.data.feature_stats_path)
        if not feature_stats_source.is_file():
            raise FileNotFoundError(
                f"configured feature_stats_path does not exist: {feature_stats_source}"
            )
        feat_stats = json.loads(feature_stats_source.read_text(encoding="utf-8"))
        required = (set(cfg.data.input_features) - {"x", "y", "z"}
                    - set(D.COHORT_FEATURE_KEYS))
        missing = sorted(required - set(feat_stats))
        if missing:
            raise KeyError(
                f"frozen feature stats missing required features {missing}: "
                f"{feature_stats_source}"
            )
    else:
        feat_stats = D.compute_feature_stats(
            tr_cases, cfg.data.input_features, cfg.data.curvature_transform
        )
    # Joint labels combine m/s and Pa. No flattened raw quantile has a meaning,
    # and the fixed equal-task MSE does not consume any loss-weight quantiles.
    wq = ({"mode": "not_used", "reason": "joint equal-task MSE; no mixed-unit quantiles"}
          if joint_target else D.compute_train_weight_quantiles(tr_cases))
    if cfg.train.loss_weight_fixed_quantiles and not joint_target:
        cfg.train.y_norm_q02 = wq["y_norm_q02"]
        cfg.train.y_norm_q98 = wq["y_norm_q98"]
        cfg.train.curv_q02 = wq["curv_q02"]
        cfg.train.curv_q98 = wq["curv_q98"]
        cfg.train.invr_q02 = wq["invr_q02"]
        cfg.train.invr_q98 = wq["invr_q98"]
    if not joint_target:
        cfg.train._raw_p90 = wq["raw_p90"]  # type: ignore[attr-defined]
    if float(getattr(cfg.train, "loss_raw_mse_lambda", 0.0)) > 0:
        if (cfg.data.target != "wss" or cfg.data.target_normalization != "global_stats"
                or cfg.model.out_dim != 1):
            raise ValueError("Pa-MSE requires scalar WSS with global_stats normalization")
        if not math.isfinite(float(wq["raw_p90"])) or float(wq["raw_p90"]) <= 0:
            raise ValueError("Pa-MSE requires a finite positive train-only p90")
    log.info("train cases=%d val cases=%d  selection=%s  feat_stats=%s  fixed_q=%s",
             len(tr_cases), len(va_cases), cfg.train.selection_rule,
             list(feat_stats.keys()), cfg.train.loss_weight_fixed_quantiles)

    cfg.to_json(run_dir / "config.json")
    (run_dir / "feature_stats.json").write_text(json.dumps(feat_stats, indent=2))
    (run_dir / "feature_stats_source.json").write_text(json.dumps({
        "mode": "frozen" if feature_stats_source is not None else "train_recomputed",
        "source_path": str(feature_stats_source.resolve()) if feature_stats_source else None,
        "source_sha256": (
            hashlib.sha256(feature_stats_source.read_bytes()).hexdigest()
            if feature_stats_source is not None else None
        ),
        "train_split_path": str(Path(cfg.data.split_path).resolve()),
    }, indent=2, ensure_ascii=False))
    (run_dir / "wss_global_stats.json").write_text(
        json.dumps(wss_stats, indent=2, ensure_ascii=False)
    )
    (run_dir / "weight_quantiles.json").write_text(
        json.dumps(wq, indent=2, ensure_ascii=False)
    )
    (run_dir / "target_normalization.json").write_text(json.dumps({
        "mode": cfg.data.target_normalization,
        "physical_recovery_enabled": cfg.data.target_normalization == "global_stats",
        "case_metadata": {
            f"{c['cohort']}/{c['case']}": c.get("target_normalization_meta", {})
            for c in tr_cases
        },
    }, indent=2, ensure_ascii=False))

    train_ds = D.WSSMinDataset(tr_cases, cfg.data, feat_stats, training=True,
                               base_seed=cfg.train.seed)
    g = torch.Generator()
    g.manual_seed(cfg.train.seed)
    pw = bool(cfg.data.persistent_workers) and cfg.data.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_cases, shuffle=True,
        collate_fn=D.collate, num_workers=cfg.data.num_workers,
        drop_last=False, persistent_workers=pw,
        generator=g, worker_init_fn=D.worker_init_fn if cfg.data.num_workers > 0 else None,
    )

    # ---- 模型/优化器 ----
    if getattr(cfg.train, "init_reference_config", None):
        model, initialization = build_paired_model(cfg)
        model = model.to(device)
    else:
        model = build_model(cfg.model, C.input_dim(cfg)).to(device)
        initialization = {"mode": "random_initialization"}
    if cfg.train.init_checkpoint_path:
        init_path = Path(cfg.train.init_checkpoint_path)
        if not init_path.is_file():
            raise FileNotFoundError(f"init checkpoint does not exist: {init_path}")
        payload = torch.load(init_path, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        result = model.load_state_dict(state, strict=cfg.train.init_checkpoint_strict)
        if not cfg.train.init_checkpoint_strict and (result.missing_keys or result.unexpected_keys):
            log.warning("non-strict initialization missing=%s unexpected=%s",
                        result.missing_keys, result.unexpected_keys)
        initialization = {
            "mode": "warm_start_weights_only",
            "checkpoint_path": str(init_path.resolve()),
            "checkpoint_sha256": hashlib.sha256(init_path.read_bytes()).hexdigest(),
            "checkpoint_epoch": payload.get("epoch"),
            "checkpoint_metric": payload.get("metric"),
            "strict": bool(cfg.train.init_checkpoint_strict),
            "optimizer_scheduler_reset": True,
        }
        log.info("initialized model weights from %s (epoch=%s metric=%s; optimizer reset)",
                 init_path, payload.get("epoch"), payload.get("metric"))
    (run_dir / "initialization.json").write_text(
        json.dumps(initialization, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # 波 1：可选权重 EMA（影子模型；best/last 选择与保存不变，另存 ckpt_ema.pt）
    ema_decay = float(getattr(cfg.train, "ema_decay", 0.0) or 0.0)
    ema_model = None
    ema_updates = 0
    if ema_decay > 0:
        ema_model = copy.deepcopy(model).eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
        log.info("EMA enabled: decay=%.5f (warm-up decay min(d, (1+t)/(10+t)))", ema_decay)
    n_par = sum(p.numel() for p in model.parameters())
    log.info("model=%s params=%.2fM selection=%s top_k=%d patience=%d",
             cfg.model.name, n_par / 1e6, cfg.train.selection_rule,
             cfg.train.ckpt_top_k, cfg.train.early_stop_patience)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(cfg.train))
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.train.amp and device == "cuda"))
    log_components = joint_target or bool(getattr(cfg.train, "log_loss_components", False))
    guard_finite = log_components or float(getattr(cfg.train, "loss_raw_mse_lambda", 0.0)) > 0
    if log_components and device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    total_amp_overflows = 0

    history = []
    best_metric = -math.inf
    topk: list = []
    evals_without_improve = 0
    hist_path = run_dir / "history.jsonl"
    hist_path.write_text("")
    t0 = time.time()
    stopped_early = False

    region_feature = getattr(cfg.train, "loss_region_feature", None)
    region_column = list(cfg.data.input_features).index(region_feature) if region_feature else None

    for epoch in range(cfg.train.epochs):
        model.train()
        train_ds.set_epoch(epoch)
        ep_loss, nb = 0.0, 0
        ep_loss_units = 0
        ep_sqerr, ep_abserr, ep_points = 0.0, 0.0, 0
        task_errors = {task: {"sqerr": 0.0, "abserr": 0.0, "elements": 0, "points": 0}
                       for task in ("velocity", "pressure")} if joint_target else {}
        component_sums = {}
        finite_grad_norms = []
        epoch_amp_overflows = 0
        for batch in train_loader:
            pos = batch["pos"].to(device)
            x = batch["x"].to(device)
            if region_column is not None:
                batch["loss_region_value"] = x[:, region_column]  # T3 focus region from the standardised input column
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(cfg.train.amp and device == "cuda")):
                if cfg.data.support_n_points is not None or cfg.model.sa_center_counts:
                    pred = model.forward_support_query(
                        batch["support_pos"].to(device), batch["support_x"].to(device),
                        batch["support_batch"].to(device), pos, x,
                        batch["batch"].to(device), unit_ids=batch["unit_ids"], epoch=epoch,
                        global_seed=cfg.train.seed, evaluation=False,
                        **D.local_model_context(batch, device),
                    )
                else:
                    pred = model(pos, x, batch["batch"].to(device))
                loss_parts = {} if log_components else None
                try:
                    loss = compute_loss(pred, batch, cfg.train, device, wss_stats,
                                        components=loss_parts)
                except FloatingPointError:
                    _record_nonfinite_event(run_dir, epoch, nb, "physical_loss_nonfinite_fatal")
                    raise
            if guard_finite and not bool(torch.isfinite(loss)):
                _record_nonfinite_event(run_dir, epoch, nb, "total_loss_nonfinite_fatal")
                raise FloatingPointError(f"non-finite training loss at epoch={epoch} step={nb}")
            loss_units = len(batch["unit_ids"]) if joint_target else 1
            if loss_parts is not None:
                for key, value in loss_parts.items():
                    component_sums[key] = component_sums.get(key, 0.0) + float(value) * loss_units
            target_metric = batch["y"].to(device)
            if joint_target:
                for task, channels in (("velocity", slice(0, 3)), ("pressure", 3)):
                    mask = batch[f"{task}_mask"].to(device)
                    diff = (pred.detach()[mask, channels].float() - target_metric[mask, channels].float())
                    acc = task_errors[task]
                    acc["sqerr"] += float(diff.square().sum())
                    acc["abserr"] += float(diff.abs().sum())
                    acc["elements"] += int(diff.numel())
                    acc["points"] += int(mask.sum())
                ep_points += len(target_metric)
            elif pred.ndim == 2 and target_metric.ndim == 2 and pred.shape == target_metric.shape:
                pred_metric = pred  # vector target
            else:
                pred_metric = pred[:, 0] if pred.ndim == 2 else pred
            if not joint_target:
                diff = (pred_metric.detach().float() - target_metric.float())
                ep_sqerr += float(diff.square().sum().item())
                ep_abserr += float(diff.abs().sum().item())
                ep_points += int(diff.numel())
            scaler.scale(loss).backward()
            if cfg.train.grad_clip > 0 or guard_finite:
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train.grad_clip if cfg.train.grad_clip > 0 else float("inf"),
                )
                if guard_finite:
                    if bool(torch.isfinite(grad_norm)):
                        if log_components:
                            finite_grad_norms.append(float(grad_norm))
                    elif scaler.is_enabled():
                        epoch_amp_overflows += 1
                        _record_nonfinite_event(run_dir, epoch, nb, "amp_gradient_overflow_skipped")
                    else:
                        _record_nonfinite_event(run_dir, epoch, nb, "gradient_nonfinite_fatal")
                        raise FloatingPointError(f"non-finite gradient at epoch={epoch} step={nb}")
            scaler.step(opt); scaler.update()
            if ema_model is not None:
                ema_updates += 1
                decay = min(ema_decay, (1.0 + ema_updates) / (10.0 + ema_updates))
                with torch.no_grad():
                    for shadow, live in zip(ema_model.parameters(), model.parameters()):
                        shadow.mul_(decay).add_(live.detach(), alpha=1.0 - decay)
                    for shadow, live in zip(ema_model.buffers(), model.buffers()):
                        shadow.copy_(live)
            ep_loss += float(loss.item()) * loss_units
            ep_loss_units += loss_units
            nb += 1
        sched.step()
        ep_loss /= max(1, ep_loss_units)

        if joint_target:
            train_mse_norm = sum(acc["sqerr"] / acc["elements"] for acc in task_errors.values()) / 2
            train_mae_norm = sum(acc["abserr"] / acc["elements"] for acc in task_errors.values()) / 2
        else:
            train_mse_norm = ep_sqerr / max(1, ep_points)
            train_mae_norm = ep_abserr / max(1, ep_points)
        rec = {"epoch": epoch, "train_loss": ep_loss,
               "train_mse_norm": train_mse_norm,
               "train_mae_norm": train_mae_norm,
               "train_rmse_norm": math.sqrt(train_mse_norm),
               "train_sampled_points": ep_points,
               "lr": opt.param_groups[0]["lr"],
               "elapsed_s": round(time.time() - t0, 1)}
        if joint_target:
            rec["train_metric_reduction"] = "equal tasks; fixed per-task queries per case"
            for task, acc in task_errors.items():
                mse = acc["sqerr"] / acc["elements"]
                rec.update({f"train_{task}_mse_norm": mse,
                            f"train_{task}_mae_norm": acc["abserr"] / acc["elements"],
                            f"train_{task}_rmse_norm": math.sqrt(mse),
                            f"train_{task}_sampled_points": acc["points"]})
        total_amp_overflows += epoch_amp_overflows
        if log_components:
            rec.update({
                "loss_components": {k: v / max(1, ep_loss_units) for k, v in component_sums.items()},
                "grad_norm_preclip_mean": (
                    sum(finite_grad_norms) / len(finite_grad_norms) if finite_grad_norms else None
                ),
                "grad_norm_preclip_max": max(finite_grad_norms) if finite_grad_norms else None,
                "amp_overflow_steps": epoch_amp_overflows,
                "amp_scale": float(scaler.get_scale()),
                "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0,
                "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved() / 2**20 if device == "cuda" else 0.0,
            })

        do_eval = ((epoch + 1) % cfg.train.eval_every == 0) or (epoch == cfg.train.epochs - 1)
        if select_by_train_loss:
            # Maximize -train_loss so existing top-k / best machinery stays max-based.
            score = -float(ep_loss)
            if select_by_ema:
                alpha = float(cfg.train.selection_ema_alpha)
                train_loss_ema = float(ep_loss) if train_loss_ema is None else alpha * float(ep_loss) + (1.0 - alpha) * train_loss_ema
                rec["train_loss_ema"] = train_loss_ema
                score = -train_loss_ema
            rec["selection_score"] = score
            improved = np.isfinite(score) and score > best_metric + 1e-12
            if improved:
                best_metric = score
                rec["is_best"] = True
            # Avoid writing a checkpoint every epoch; only when it can enter top-k.
            if (cfg.train.ckpt_top_k > 0 and np.isfinite(score)
                    and (len(topk) < cfg.train.ckpt_top_k or score > topk[0][0])):
                _save_topk(run_dir, topk, model, epoch, score, cfg.name, cfg.train.ckpt_top_k)
            if epoch % 10 == 0 or rec.get("is_best") or epoch + 1 == cfg.train.epochs:
                log.info(
                    "epoch %d/%d loss=%.4f norm_mae=%.4f norm_rmse=%.4f lr=%.2e "
                    "| select=train_loss best_loss=%.4f %s",
                    epoch, cfg.train.epochs, ep_loss, train_mae_norm,
                    math.sqrt(train_mse_norm), opt.param_groups[0]["lr"],
                    -best_metric if np.isfinite(best_metric) else float("nan"),
                    "[best]" if rec.get("is_best") else "",
                )
        elif do_eval and va_cases:
            res = evaluate_partition(model, va_cases, cfg, feat_stats, wss_stats, device,
                                     make_plots=False)
            agg, fld = res["aggregate"], res["field"]
            fld_cb = res["field_casebalanced"]
            cal = res.get("calibration", {})
            hotspot = res.get("hotspot", {})
            score = compute_selection_score(cfg.train, agg, fld, fld_cb, cal, hotspot)
            rec.update({
                "val_r2_casemean": agg["r2_casemean"],
                "val_r2_field": fld["r2"],
                "val_r2_field_casebalanced": fld_cb["r2"],
                "val_nrmse_field": fld["nrmse_range"],
                "val_mae_field": fld["mae"],
                "val_top10_pred_true_ratio": cal.get("top10_pred_true_ratio", float("nan")),
                "val_p99_pred_true_ratio": cal.get("p99_pred_true_ratio", float("nan")),
                "val_top10_iou": hotspot.get("top10_iou", float("nan")),
                "val_selection_score": score,
            })
            if np.isfinite(score):
                _save_topk(run_dir, topk, model, epoch, score, cfg.name, cfg.train.ckpt_top_k)
            improved = np.isfinite(score) and score > best_metric + 1e-12
            if improved:
                best_metric = score
                evals_without_improve = 0
                rec["is_best"] = True
            else:
                evals_without_improve += 1
            log.info(
                "epoch %d/%d loss=%.4f lr=%.2e | val R2(c)=%.4f R2(f)=%.4f "
                "MAE=%.4f top10=%.3f IoU=%.3f score=%.4f %s",
                epoch, cfg.train.epochs, ep_loss, opt.param_groups[0]["lr"],
                agg["r2_casemean"], fld["r2"], fld["mae"],
                cal.get("top10_pred_true_ratio", float("nan")),
                hotspot.get("top10_iou", float("nan")), score,
                "[best]" if rec.get("is_best") else "",
            )
            if (cfg.train.early_stop_patience > 0
                    and (epoch + 1) >= cfg.train.min_epoch
                    and evals_without_improve >= cfg.train.early_stop_patience):
                log.info("early stop at epoch %d (patience=%d evals)",
                         epoch, cfg.train.early_stop_patience)
                stopped_early = True
        else:
            if epoch % 10 == 0:
                log.info("epoch %d/%d loss=%.4f lr=%.2e",
                         epoch, cfg.train.epochs, ep_loss, opt.param_groups[0]["lr"])

        history.append(rec)
        with open(hist_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        torch.save({"model": model.state_dict(), "epoch": epoch, "cfg_name": cfg.name},
                   run_dir / "ckpt_last.pt")
        if ema_model is not None:
            rec["ema_updates"] = ema_updates
            torch.save({"model": ema_model.state_dict(), "epoch": epoch, "cfg_name": cfg.name,
                        "ema_decay": ema_decay, "ema_updates": ema_updates}, run_dir / "ckpt_ema.pt")
        if stopped_early:
            break

    plot_history(history, run_dir / "history.png")
    if log_components:
        (run_dir / "training_diagnostics.json").write_text(json.dumps({
            "completed_epochs": len(history), "parameter_count": n_par,
            "elapsed_seconds": time.time() - t0,
            "amp_overflow_steps": total_amp_overflows,
            "fatal_nonfinite_events": 0,
            "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0,
            "peak_cuda_reserved_mb": torch.cuda.max_memory_reserved() / 2**20 if device == "cuda" else 0.0,
            "loss_components": ("equal-task MSE; epoch means weighted by cases" if joint_target else
                                "unweighted and weighted auxiliary terms; epoch means over batches"),
            "gradient_norm": "unscaled, before clipping; finite steps only",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("训练完成 best score=%.4f  用时 %.1f min  early=%s  -> %s",
             best_metric, (time.time() - t0) / 60, stopped_early, run_dir)
    if not (run_dir / "ckpt_best.pt").exists():
        torch.save({"model": model.state_dict(), "epoch": history[-1]["epoch"],
                    "cfg_name": cfg.name}, run_dir / "ckpt_best.pt")


if __name__ == "__main__":
    main()
