#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练入口（配置驱动）。

  python -m training_wss_min.train --config training_wss_min/configs/<name>.json

产物写入 config.run_dir（training_wss_min/runs/<name>/）：
- config.json / feature_stats.json   本次实验的完整配置与特征标准化统计
- train.log                          全程日志（控制台同步）
- history.jsonl                      每 epoch 一行：loss / lr / val 指标
- ckpt_best.pt / ckpt_last.pt        最优(按 ckpt_metric)与最新权重
- history.png                        训练曲线
训练用稀疏子采样，val 监控恒在**完整点云**上（evaluate.evaluate_partition）。
"""

from __future__ import annotations

import argparse
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
from .pointnext import build_model
from .evaluate import evaluate_partition


# ---------------------------------------------------------------------------
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


def _q01(a: torch.Tensor) -> torch.Tensor:
    lo, hi = torch.quantile(a, 0.02), torch.quantile(a, 0.98)
    return torch.clamp((a - lo) / (hi - lo + 1e-9), 0, 1)


def compute_loss(pred, batch, tcfg: C.TrainConfig, device):
    y = batch["y"].to(device)
    if tcfg.loss == "huber":
        per = torch.nn.functional.huber_loss(pred, y, delta=tcfg.huber_delta, reduction="none")
    else:
        per = (pred - y) ** 2
    weighted = tcfg.loss_geom_weight or tcfg.loss_weight_target
    if not weighted:
        return per.mean()
    w = torch.ones_like(per)
    if tcfg.loss_geom_weight:
        w = w + tcfg.loss_weight_curv * _q01(batch["curv"].to(device)) \
              + tcfg.loss_weight_invradius * _q01(batch["invr"].to(device))
    if tcfg.loss_weight_target:
        w = w + tcfg.loss_weight_target_alpha * _q01(y)   # 高 WSS 处更重
    return (per * w).sum() / w.sum()


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
    log.info("data: wall_n=%s sampling=%s features=%s target=%s",
             cfg.data.wall_n_points, cfg.data.sampling, cfg.data.input_features, cfg.data.target)

    # ---- 数据 ----
    wss_stats = D.load_wss_stats()
    tr_cases = D.load_partition(cfg.data.split_path, "train", wss_stats)
    va_cases = D.load_partition(cfg.data.split_path, "val", wss_stats)
    feat_stats = D.compute_feature_stats(
        tr_cases, cfg.data.input_features, cfg.data.curvature_transform
    )
    log.info("train cases=%d val cases=%d  feat_stats=%s",
             len(tr_cases), len(va_cases), list(feat_stats.keys()))

    # 保存配置与特征统计（评估复用）
    cfg.to_json(run_dir / "config.json")
    (run_dir / "feature_stats.json").write_text(json.dumps(feat_stats, indent=2))
    (run_dir / "wss_global_stats.json").write_text(
        json.dumps(wss_stats, indent=2, ensure_ascii=False)
    )

    train_ds = D.WSSMinDataset(tr_cases, cfg.data, feat_stats, training=True,
                               base_seed=cfg.train.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_cases, shuffle=True,
                              collate_fn=D.collate, num_workers=cfg.data.num_workers,
                              drop_last=False, persistent_workers=cfg.data.num_workers > 0)

    # ---- 模型/优化器 ----
    model = build_model(cfg.model, C.input_dim(cfg)).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    log.info("model=%s params=%.2fM", cfg.model.name, n_par / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(cfg.train))
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.train.amp and device == "cuda"))

    history = []
    best_metric = -math.inf
    hist_path = run_dir / "history.jsonl"
    hist_path.write_text("")
    t0 = time.time()

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
                loss = compute_loss(pred, batch, cfg.train, device)
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

        # ---- val（完整点云）----
        do_eval = ((epoch + 1) % cfg.train.eval_every == 0) or (epoch == cfg.train.epochs - 1)
        if do_eval and va_cases:
            res = evaluate_partition(model, va_cases, cfg, feat_stats, wss_stats, device,
                                     make_plots=False)
            agg, fld, cal = res["aggregate"], res["field"], res.get("calibration", {})
            rec.update({
                "val_r2_casemean": agg["r2_casemean"],
                "val_r2_field": fld["r2"],
                "val_nrmse_field": fld["nrmse_range"],
                "val_mae_field": fld["mae"],
                "val_top10_pred_true_ratio": cal.get("top10_pred_true_ratio", float("nan")),
                "val_p99_pred_true_ratio": cal.get("p99_pred_true_ratio", float("nan")),
            })
            metric_key = cfg.train.ckpt_metric.replace("val_r2_casemean", "r2_casemean")
            cur = agg.get("r2_casemean", -math.inf) if "casemean" in cfg.train.ckpt_metric \
                else fld.get("r2", -math.inf)
            if np.isfinite(cur) and cur > best_metric:
                best_metric = cur
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "metric": cur, "cfg_name": cfg.name}, run_dir / "ckpt_best.pt")
                rec["is_best"] = True
            log.info("epoch %d/%d loss=%.4f lr=%.2e | val R2(casemean)=%.4f R2(field)=%.4f "
                     "NRMSE=%.4f MAE=%.4f %s",
                     epoch, cfg.train.epochs, ep_loss, opt.param_groups[0]["lr"],
                     agg["r2_casemean"], fld["r2"], fld["nrmse_range"], fld["mae"],
                     "[best]" if rec.get("is_best") else "")
        else:
            if epoch % 10 == 0:
                log.info("epoch %d/%d loss=%.4f lr=%.2e",
                         epoch, cfg.train.epochs, ep_loss, opt.param_groups[0]["lr"])

        history.append(rec)
        with open(hist_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        torch.save({"model": model.state_dict(), "epoch": epoch, "cfg_name": cfg.name},
                   run_dir / "ckpt_last.pt")

    plot_history(history, run_dir / "history.png")
    log.info("训练完成 best %s=%.4f  用时 %.1f min  -> %s",
             cfg.train.ckpt_metric, best_metric, (time.time() - t0) / 60, run_dir)
    # 若从未保存 best（val 为空），把 last 复制为 best
    if not (run_dir / "ckpt_best.pt").exists():
        torch.save({"model": model.state_dict(), "epoch": cfg.train.epochs - 1,
                    "cfg_name": cfg.name}, run_dir / "ckpt_best.pt")


if __name__ == "__main__":
    main()
