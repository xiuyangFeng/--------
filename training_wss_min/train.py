#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练入口（配置驱动）。

  python -m training_wss_min.train --config training_wss_min/configs/<name>.json

产物写入 config.run_dir（training_wss_min/runs/<name>/）：
- config.json / feature_stats.json / weight_quantiles.json
- train.log / history.jsonl / history.png
- ckpt_best.pt / ckpt_top{k}.pt / ckpt_last.pt
训练用稀疏子采样，val 监控恒在**完整点云**上（evaluate.evaluate_partition）。
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config as C
from . import dataset as D
from . import metrics as M
from .pointnext import build_model
from .evaluate import evaluate_partition


def setup_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"wssmin.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(run_dir / "train.log")
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda_factory(cfg: C.TrainConfig):
    def f(epoch: int) -> float:
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / max(1, cfg.warmup_epochs)
        prog = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
        cos = 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        return (cfg.min_lr + (cfg.lr - cfg.min_lr) * cos) / cfg.lr
    return f


def _fixed01(a: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp((a - lo) / (hi - lo + 1e-9), 0, 1)


def _batch01(a: torch.Tensor) -> torch.Tensor:
    lo, hi = torch.quantile(a, 0.02), torch.quantile(a, 0.98)
    return torch.clamp((a - lo) / (hi - lo + 1e-9), 0, 1)


def compute_loss(pred, batch, tcfg: C.TrainConfig, device, wss_stats: dict | None = None):
    """标量回归损失；支持固定阈值加权与 raw-space Huber 辅助（C1）。"""
    y = batch["y"].to(device)
    # NLL: pred 为 (N,2) = [mu, logvar]
    if tcfg.loss == "gaussian_nll":
        if pred.ndim != 2 or pred.shape[-1] != 2:
            raise ValueError("gaussian_nll requires model out_dim=2")
        mu = pred[:, 0]
        logvar = torch.clamp(pred[:, 1], tcfg.nll_logvar_min, tcfg.nll_logvar_max)
        per = 0.5 * (logvar + (y - mu) ** 2 / torch.exp(logvar))
        per = per + tcfg.nll_logvar_reg * (logvar ** 2)
        pred_for_weight = mu
    else:
        if pred.ndim == 2 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        if tcfg.loss == "huber":
            per = torch.nn.functional.huber_loss(pred, y, delta=tcfg.huber_delta, reduction="none")
        else:
            per = (pred - y) ** 2
        pred_for_weight = pred

    weighted = tcfg.loss_geom_weight or tcfg.loss_weight_target
    if weighted:
        w = torch.ones_like(per)
        use_fixed = bool(tcfg.loss_weight_fixed_quantiles)
        if tcfg.loss_geom_weight:
            curv = batch["curv"].to(device)
            invr = batch["invr"].to(device)
            if use_fixed and tcfg.curv_q02 is not None and tcfg.curv_q98 is not None:
                cq = _fixed01(curv, tcfg.curv_q02, tcfg.curv_q98)
            else:
                cq = _batch01(curv)
            if use_fixed and tcfg.invr_q02 is not None and tcfg.invr_q98 is not None:
                iq = _fixed01(invr, tcfg.invr_q02, tcfg.invr_q98)
            else:
                iq = _batch01(invr)
            w = w + tcfg.loss_weight_curv * cq + tcfg.loss_weight_invradius * iq
        if tcfg.loss_weight_target:
            if use_fixed and tcfg.y_norm_q02 is not None and tcfg.y_norm_q98 is not None:
                yq = _fixed01(y, tcfg.y_norm_q02, tcfg.y_norm_q98)
            else:
                yq = _batch01(y)
            w = w + tcfg.loss_weight_target_alpha * yq
        loss = (per * w).sum() / w.sum()
    else:
        loss = per.mean()

    # C1: raw-space Huber 辅助
    lam = float(tcfg.loss_raw_huber_lambda or 0.0)
    if lam > 0 and wss_stats is not None and tcfg.loss != "gaussian_nll":
        y_raw = batch["y_raw"].to(device).float()
        # FP32 denorm
        mu = torch.as_tensor(wss_stats["log"]["mean"], device=device, dtype=torch.float32)
        sd = torch.as_tensor(wss_stats["log"]["std"], device=device, dtype=torch.float32)
        eps = torch.as_tensor(wss_stats["eps"], device=device, dtype=torch.float32)
        pred_raw = torch.exp(pred_for_weight.float() * sd + mu) - eps
        # 记录裁剪（不静默掩盖）：仅用于监控，不改梯度路径上的值过多
        p90 = float(getattr(tcfg, "_raw_p90", None) or wss_stats.get("raw_percentiles", {}).get("p90", 1.0))
        scale = max(p90, 1e-3)
        h = torch.nn.functional.huber_loss(
            pred_raw / scale, y_raw / scale, delta=tcfg.raw_huber_delta, reduction="mean"
        )
        loss = loss + lam * h
    return loss


def compute_selection_score(tcfg: C.TrainConfig, agg, field, cal, hotspot) -> float:
    if tcfg.selection_rule == "r4_composite_v1":
        return M.selection_score_r4_composite_v1(agg, field, cal, hotspot)
    # 兼容旧 ckpt_metric
    if "casemean" in tcfg.ckpt_metric:
        return float(agg.get("r2_casemean", float("-inf")))
    return float(field.get("r2", float("-inf")))


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

    # ---- 数据 ----
    stats_path = Path(cfg.data.wss_stats_path)
    if not stats_path.is_file():
        stats_path = C.GLOBAL_STATS
    wss_stats = D.load_wss_stats(stats_path)
    tr_cases = D.load_partition(cfg.data.split_path, "train", wss_stats, strict=True)
    va_cases = D.load_partition(cfg.data.split_path, "val", wss_stats, strict=True)
    feat_stats = D.compute_feature_stats(
        tr_cases, cfg.data.input_features, cfg.data.curvature_transform
    )
    wq = D.compute_train_weight_quantiles(tr_cases)
    if cfg.train.loss_weight_fixed_quantiles:
        cfg.train.y_norm_q02 = wq["y_norm_q02"]
        cfg.train.y_norm_q98 = wq["y_norm_q98"]
        cfg.train.curv_q02 = wq["curv_q02"]
        cfg.train.curv_q98 = wq["curv_q98"]
        cfg.train.invr_q02 = wq["invr_q02"]
        cfg.train.invr_q98 = wq["invr_q98"]
    cfg.train._raw_p90 = wq["raw_p90"]  # type: ignore[attr-defined]
    log.info("train cases=%d val cases=%d  feat_stats=%s  fixed_q=%s",
             len(tr_cases), len(va_cases), list(feat_stats.keys()),
             cfg.train.loss_weight_fixed_quantiles)

    cfg.to_json(run_dir / "config.json")
    (run_dir / "feature_stats.json").write_text(json.dumps(feat_stats, indent=2))
    (run_dir / "wss_global_stats.json").write_text(
        json.dumps(wss_stats, indent=2, ensure_ascii=False)
    )
    (run_dir / "weight_quantiles.json").write_text(
        json.dumps(wq, indent=2, ensure_ascii=False)
    )

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
    model = build_model(cfg.model, C.input_dim(cfg)).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    log.info("model=%s params=%.2fM selection=%s top_k=%d patience=%d",
             cfg.model.name, n_par / 1e6, cfg.train.selection_rule,
             cfg.train.ckpt_top_k, cfg.train.early_stop_patience)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(cfg.train))
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.train.amp and device == "cuda"))

    history = []
    best_metric = -math.inf
    topk: list = []
    evals_without_improve = 0
    hist_path = run_dir / "history.jsonl"
    hist_path.write_text("")
    t0 = time.time()
    stopped_early = False

    for epoch in range(cfg.train.epochs):
        model.train()
        train_ds.set_epoch(epoch)
        ep_loss, nb = 0.0, 0
        for batch in train_loader:
            pos = batch["pos"].to(device)
            x = batch["x"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(cfg.train.amp and device == "cuda")):
                pred = model(pos, x, batch["batch"].to(device))
                loss = compute_loss(pred, batch, cfg.train, device, wss_stats)
            scaler.scale(loss).backward()
            if cfg.train.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt); scaler.update()
            ep_loss += float(loss.item()); nb += 1
        sched.step()
        ep_loss /= max(1, nb)

        rec = {"epoch": epoch, "train_loss": ep_loss, "lr": opt.param_groups[0]["lr"],
               "elapsed_s": round(time.time() - t0, 1)}

        do_eval = ((epoch + 1) % cfg.train.eval_every == 0) or (epoch == cfg.train.epochs - 1)
        if do_eval and va_cases:
            res = evaluate_partition(model, va_cases, cfg, feat_stats, wss_stats, device,
                                     make_plots=False)
            agg, fld, cal = res["aggregate"], res["field"], res.get("calibration", {})
            hotspot = res.get("hotspot", {})
            score = compute_selection_score(cfg.train, agg, fld, cal, hotspot)
            rec.update({
                "val_r2_casemean": agg["r2_casemean"],
                "val_r2_field": fld["r2"],
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
            if (epoch + 1) >= cfg.train.min_epoch and \
               evals_without_improve >= cfg.train.early_stop_patience:
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
        if stopped_early:
            break

    plot_history(history, run_dir / "history.png")
    log.info("训练完成 best score=%.4f  用时 %.1f min  early=%s  -> %s",
             best_metric, (time.time() - t0) / 60, stopped_early, run_dir)
    if not (run_dir / "ckpt_best.pt").exists():
        torch.save({"model": model.state_dict(), "epoch": history[-1]["epoch"],
                    "cfg_name": cfg.name}, run_dir / "ckpt_best.pt")


if __name__ == "__main__":
    main()
