from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from .io import load_checkpoint, save_checkpoint
from .losses import build_loss_plugin
from .metrics import RegressionMeter

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


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
        accumulate_grad_batches: int = 1,
        log_dir: Optional[Path] = None,
        use_amp: bool = False,
        warmup_scheduler: Optional[LRScheduler] = None,
        warmup_epochs: int = 0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_scheduler = warmup_scheduler
        self.warmup_epochs = max(0, warmup_epochs)
        self.device = device
        self.loss_weights = loss_weights.to(device)
        self.grad_clip_norm = grad_clip_norm
        self.loss_plugin = build_loss_plugin(physics_config)
        self.accumulate_grad_batches = max(1, accumulate_grad_batches)

        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp) if self.use_amp else None

        self.writer = None
        if log_dir is not None and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=str(log_dir))

    def _log_scalars(self, metrics: Dict[str, float], prefix: str, step: int) -> None:
        if self.writer is None:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def _step_optimizer(self) -> None:
        if self.scaler is not None:
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
        self.optimizer.zero_grad()

    def _step_scheduler(self, val_loss: float, epoch: int) -> float:
        if self.warmup_scheduler is not None and epoch <= self.warmup_epochs:
            self.warmup_scheduler.step()
        elif isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        elif self.scheduler is not None:
            self.scheduler.step()
        return self.optimizer.param_groups[0]["lr"]

    def run_epoch(self, loader, train: bool, epoch: int) -> Dict[str, float]:
        meter = RegressionMeter()
        self.model.train(mode=train)
        extra_totals: Dict[str, float] = {}
        num_batches = 0

        if train:
            self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            batch = batch.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred = self.model(batch)
            # Physics loss uses autograd.grad which needs float32;
            # build_loss is called outside autocast so gradients stay precise.
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
                scaled_loss = breakdown.total_loss / self.accumulate_grad_batches
                if self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                    self._step_optimizer()

            meter.update(pred, batch.y, breakdown.total_loss.item())
            scalar_breakdown = breakdown.scalar_dict()
            for key, val in scalar_breakdown.items():
                extra_totals[key] = extra_totals.get(key, 0.0) + val
            num_batches += 1

        if train and num_batches % self.accumulate_grad_batches != 0:
            self._step_optimizer()

        metrics = meter.compute()
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
        csv_file = open(csv_path, "w", encoding="utf-8", newline="")
        dict_writer: Optional[csv.DictWriter] = None

        try:
            for epoch in range(1, epochs + 1):
                t0 = time.time()
                train_metrics = self.run_epoch(train_loader, train=True, epoch=epoch)
                val_metrics = self.run_epoch(val_loader, train=False, epoch=epoch)

                current_lr = self._step_scheduler(val_metrics["loss"], epoch)
                epoch_time_sec = time.time() - t0
                is_best = val_metrics["loss"] < best_val

                row: Dict[str, object] = {
                    "epoch": epoch,
                }
                row.update({f"train_{k}": v for k, v in sorted(train_metrics.items())})
                row.update({f"val_{k}": v for k, v in sorted(val_metrics.items())})
                row["lr"] = current_lr
                row["epoch_time_sec"] = round(epoch_time_sec, 2)
                row["is_best"] = int(is_best)

                if dict_writer is None:
                    dict_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                    dict_writer.writeheader()
                dict_writer.writerow(row)
                csv_file.flush()

                self._log_scalars(train_metrics, "train", epoch)
                self._log_scalars(val_metrics, "val", epoch)
                self._log_scalars({"lr": current_lr}, "optim", epoch)

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

                if patience >= early_stopping_patience:
                    break
        finally:
            csv_file.close()
            if self.writer is not None:
                self.writer.close()

        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "history": history,
        }

    def evaluate(self, loader, checkpoint_path: Optional[Path] = None) -> Dict[str, float]:
        if checkpoint_path is not None:
            load_checkpoint(self.model, checkpoint_path, self.device)
        return self.run_epoch(loader, train=False, epoch=10**9)
