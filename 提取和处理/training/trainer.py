from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Optional

import torch

from .io import load_checkpoint, save_checkpoint
from .losses import weighted_mse_loss
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
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.loss_weights = loss_weights.to(device)
        self.grad_clip_norm = grad_clip_norm

    def run_epoch(self, loader, train: bool) -> Dict[str, float]:
        meter = RegressionMeter()
        self.model.train(mode=train)

        for batch in loader:
            batch = batch.to(self.device)
            if train:
                self.optimizer.zero_grad()

            pred = self.model(batch)
            loss = weighted_mse_loss(pred, batch.y, self.loss_weights)

            if train:
                loss.backward()
                if self.grad_clip_norm is not None:
                    # 小样本图数据训练时梯度偶发爆炸更常见，这里默认加裁剪更稳妥。
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            meter.update(pred, batch.y, loss.item())

        return meter.compute()

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
                    "lr",
                ]
            )

            for epoch in range(1, epochs + 1):
                t0 = time.time()
                train_metrics = self.run_epoch(train_loader, train=True)
                val_metrics = self.run_epoch(val_loader, train=False)

                self.scheduler.step(val_metrics["loss"])
                current_lr = self.optimizer.param_groups[0]["lr"]

                row = [
                    epoch,
                    train_metrics["loss"],
                    val_metrics["loss"],
                    train_metrics["rmse"],
                    val_metrics["rmse"],
                    train_metrics["rmse_p"],
                    val_metrics["rmse_p"],
                    current_lr,
                ]
                writer.writerow(row)
                history.append(
                    {
                        "epoch": epoch,
                        "train": train_metrics,
                        "val": val_metrics,
                        "lr": current_lr,
                        "epoch_time_sec": time.time() - t0,
                    }
                )

                if val_metrics["loss"] < best_val:
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

    @torch.no_grad()
    def evaluate(self, loader, checkpoint_path: Optional[Path] = None) -> Dict[str, float]:
        if checkpoint_path is not None:
            load_checkpoint(self.model, checkpoint_path, self.device)
        return self.run_epoch(loader, train=False)
