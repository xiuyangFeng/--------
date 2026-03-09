from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Optional

import torch

from .io import load_checkpoint, save_checkpoint
from .losses import build_loss_plugin
from .metrics import RegressionMeter


class FieldTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        loss_weights: torch.Tensor,
        grad_clip_norm: Optional[float] = None,
        physics_config=None,
    ):
        # trainer 只负责“如何跑一轮/多轮”，不直接知道实验属于哪一组。
        # 实验语义全部由 config/meta 和外层入口负责。
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.loss_weights = loss_weights.to(device)
        self.grad_clip_norm = grad_clip_norm
        self.loss_plugin = build_loss_plugin(physics_config)

    def run_epoch(self, loader, train: bool, epoch: int) -> Dict[str, float]:
        meter = RegressionMeter()
        self.model.train(mode=train)
        extra_totals = {
            "data_loss": 0.0,
            "physics_continuity_loss": 0.0,
            "physics_momentum_loss": 0.0,
            "physics_no_slip_loss": 0.0,
        }
        num_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            if train:
                self.optimizer.zero_grad()

            pred = self.model(batch)
            # loss_plugin 统一收口 data-only 和 physics-augmented 两种训练模式。
            breakdown = self.loss_plugin.build_loss(
                model=self.model,
                batch=batch,
                pred=pred,
                target=batch.y,
                data_weights=self.loss_weights,
                epoch=epoch,
                train=train,
            )

            if train:
                breakdown.total_loss.backward()
                if self.grad_clip_norm is not None:
                    # 小样本图数据训练时梯度偶发爆炸更常见，这里默认加裁剪更稳妥。
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            meter.update(pred, batch.y, breakdown.total_loss.item())
            scalar_breakdown = breakdown.scalar_dict()
            for key in extra_totals:
                extra_totals[key] += scalar_breakdown[key]
            num_batches += 1

        metrics = meter.compute()
        # 这里把 loss 分项平均到 epoch 级，方便直接写入 history.csv 做后续分析。
        for key, total in extra_totals.items():
            metrics[key] = total / max(1, num_batches)
        metrics["physics_enabled"] = float(self.loss_plugin.is_enabled(epoch))
        return metrics

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        early_stopping_patience: int,
        run_dir: Path,
        save_every: int = 10,
        save_best_only: bool = True,
    ) -> Dict[str, object]:
        best_val = float("inf")
        best_epoch = 0
        patience = 0
        history = []

        csv_path = run_dir / "history.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "train_rmse",
                    "val_rmse",
                    "train_rmse_p",
                    "val_rmse_p",
                    "train_data_loss",
                    "val_data_loss",
                    "train_continuity_loss",
                    "val_continuity_loss",
                    "train_momentum_loss",
                    "val_momentum_loss",
                    "train_no_slip_loss",
                    "val_no_slip_loss",
                    "lr",
                    "epoch_time_sec",
                    "is_best",
                ]
            )

            for epoch in range(1, epochs + 1):
                t0 = time.time()
                train_metrics = self.run_epoch(train_loader, train=True, epoch=epoch)
                val_metrics = self.run_epoch(val_loader, train=False, epoch=epoch)

                self.scheduler.step(val_metrics["loss"])
                current_lr = self.optimizer.param_groups[0]["lr"]
                epoch_time_sec = time.time() - t0
                is_best = val_metrics["loss"] < best_val

                row = [
                    epoch,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    train_metrics["rmse"],
                    val_metrics["rmse"],
                    train_metrics["rmse_p"],
                    val_metrics["rmse_p"],
                    train_metrics["data_loss"],
                    val_metrics["data_loss"],
                    train_metrics["physics_continuity_loss"],
                    val_metrics["physics_continuity_loss"],
                    train_metrics["physics_momentum_loss"],
                    val_metrics["physics_momentum_loss"],
                    train_metrics["physics_no_slip_loss"],
                    val_metrics["physics_no_slip_loss"],
                    current_lr,
                    epoch_time_sec,
                    int(is_best),
                ]
                writer.writerow(row)
                history.append(
                    {
                        "epoch": epoch,
                        "train": train_metrics,
                        "val": val_metrics,
                        "lr": current_lr,
                        "epoch_time_sec": epoch_time_sec,
                        "is_best": is_best,
                    }
                )

                if is_best:
                    best_val = val_metrics["loss"]
                    best_epoch = epoch
                    patience = 0
                    save_checkpoint(self.model, run_dir / "best_model.pt")
                else:
                    patience += 1

                if not save_best_only and epoch % save_every == 0:
                    save_checkpoint(self.model, run_dir / f"checkpoint_epoch_{epoch}.pt")

                # 早停只看验证损失，避免在不同实验中引入额外选择偏差。
                if patience >= early_stopping_patience:
                    break

        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "history": history,
        }

    def evaluate(self, loader, checkpoint_path: Optional[Path] = None) -> Dict[str, float]:
        if checkpoint_path is not None:
            load_checkpoint(self.model, checkpoint_path, self.device)
        # 评估走极大 epoch 编号，只是为了让 warmup 后的 physics 配置在评估时也能被计入指标。
        return self.run_epoch(loader, train=False, epoch=10**9)
