from __future__ import annotations

import contextlib
import csv
import gc
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from tqdm.auto import tqdm

from pipeline.config import NODE_FEATURE_NAMES

from .io import load_checkpoint, save_checkpoint
from .losses import build_loss_plugin
from .metrics import RegressionMeter, WSSMeter

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
        interior_loss_boost: float = 1.0,
        accumulate_grad_batches: int = 1,
        log_dir: Optional[Path] = None,
        use_amp: bool = False,
        warmup_scheduler: Optional[LRScheduler] = None,
        warmup_epochs: int = 0,
        wss_loss_weight: float = 0.0,
        wss_weights: Optional[torch.Tensor] = None,
        early_stop_wss_weight: float = 0.0,
        wss_loss_type: str = "mse",
        wss_huber_beta: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_scheduler = warmup_scheduler
        self.warmup_epochs = max(0, warmup_epochs)
        self.device = device
        self.loss_weights = loss_weights.to(device)
        self.early_stop_wss_weight = float(early_stop_wss_weight)
        self.grad_clip_norm = grad_clip_norm
        self.wss_loss_weight = float(wss_loss_weight)
        self.loss_plugin = build_loss_plugin(
            physics_config,
            interior_loss_boost=interior_loss_boost,
            wss_loss_weight=wss_loss_weight,
            wss_weights=wss_weights,
            wss_loss_type=wss_loss_type,
            wss_huber_beta=wss_huber_beta,
        )
        self.accumulate_grad_batches = max(1, accumulate_grad_batches)

        # 只有 CUDA 上才真正启用 AMP。
        self.use_amp = use_amp and device.type == "cuda"
        # 启用 AMP 时用 GradScaler 管理缩放。
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp) if self.use_amp else None

        # 默认不写 TensorBoard。
        self.writer = None
        # 同时满足“给了 log_dir”且“环境里安装了 TensorBoard”时才创建 writer。
        if log_dir is not None and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=str(log_dir))

    def _log_scalars(self, metrics: Dict[str, float], prefix: str, step: int) -> None:
        # 没有 writer 时直接跳过日志记录。
        if self.writer is None:
            return
        # 只把标量指标写入 TensorBoard。
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def _step_optimizer(self) -> None:
        # AMP 路径下先处理缩放梯度。
        if self.scaler is not None:
            # 如果设置了梯度裁剪，需要先 unscale 再裁剪。
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            # 执行优化器更新。
            self.scaler.step(self.optimizer)
            # 更新缩放因子。
            self.scaler.update()
        else:
            # 非 AMP 路径直接做梯度裁剪。
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            # 执行一步参数更新。
            self.optimizer.step()
        # 每次参数更新后把梯度清空。
        self.optimizer.zero_grad(set_to_none=True)

    def _step_scheduler(self, val_loss: float, epoch: int) -> float:
        # 如果还处在 warmup 阶段，优先走 warmup 调度器。
        if self.warmup_scheduler is not None and epoch <= self.warmup_epochs:
            self.warmup_scheduler.step()
        # Plateau 调度器需要喂验证损失。
        elif isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        # 其他调度器按 epoch 直接 step。
        elif self.scheduler is not None:
            self.scheduler.step()
        # 返回当前学习率，方便日志记录。
        return self.optimizer.param_groups[0]["lr"]

    @staticmethod
    def _format_metric(value: float) -> str:
        # 统一把浮点指标格式化成 6 位小数。
        return f"{value:.6f}"

    def run_epoch(self, loader, train: bool, epoch: int) -> Dict[str, float]:
        # meter 负责在线累计回归指标。
        meter = RegressionMeter()
        # WSS 指标在线累计（仅壁面节点）。
        wss_meter = WSSMeter() if self.wss_loss_weight > 0 else None
        # train=True 时启用训练模式；否则启用评估模式。
        self.model.train(mode=train)
        # 额外损失分项的累计器。
        extra_totals: Dict[str, float] = {}
        # batch 计数。
        num_batches = 0
        # 当前阶段名称。
        phase = "train" if train else "val"
        # tqdm 进度条。
        progress = tqdm(
            loader,
            desc=f"Epoch {epoch} [{phase}]",
            leave=False,
            dynamic_ncols=True,
            mininterval=1.0,
        )

        _is_wall_idx = NODE_FEATURE_NAMES.index("is_wall")

        # 训练阶段在 epoch 开头先清零梯度。
        if train:
            self.optimizer.zero_grad(set_to_none=True)

        # Disable gradient computation during validation to avoid storing
        # intermediate activations, which can double GPU memory usage.
        grad_ctx = contextlib.nullcontext() if train else torch.no_grad()
        with grad_ctx:
            for batch_idx, batch in enumerate(progress):
                # 把当前 batch 搬到训练设备。
                batch = batch.to(self.device)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    model_output = self.model(batch)
                if isinstance(model_output, tuple):
                    pred, wss_pred = model_output
                else:
                    pred, wss_pred = model_output, None
                breakdown = self.loss_plugin.build_loss(
                    model=self.model,
                    batch=batch,
                    pred=pred,
                    target=batch.y,
                    data_weights=self.loss_weights,
                    epoch=epoch,
                    train=train,
                    wss_pred=wss_pred,
                )
                loss_value = breakdown.total_loss.item()

                if train:
                    # 梯度累积时，先按累积步数缩小 loss。
                    scaled_loss = breakdown.total_loss / self.accumulate_grad_batches
                    # AMP 路径用 scaler.backward。
                    if self.scaler is not None:
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        # 普通路径直接反向传播。
                        scaled_loss.backward()
                    # 达到累积步数后才真正更新一次参数。
                    if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                        self._step_optimizer()

                # 把当前 batch 的预测、真值和 loss 送入指标累计器。
                meter.update(pred, batch.y, loss_value)

                # WSS 指标累计（壁面子集）。
                if wss_meter is not None and wss_pred is not None:
                    wss_target = getattr(batch, "y_wss", None)
                    if wss_target is not None:
                        is_wall = batch.x[:, _is_wall_idx : _is_wall_idx + 1]
                        wss_meter.update(wss_pred, wss_target, is_wall)

                # 读取各个损失分项的标量版本。
                scalar_breakdown = breakdown.scalar_dict()
                # 逐项累计，后面会除以 batch 数取平均。
                for key, val in scalar_breakdown.items():
                    extra_totals[key] = extra_totals.get(key, 0.0) + val
                num_batches += 1
                # 在进度条后缀显示当前 batch loss。
                progress.set_postfix_str(f"loss={self._format_metric(loss_value)}")
                del scalar_breakdown, breakdown, pred, wss_pred, batch

        # 如果最后一个累积周期不足 accumulate_grad_batches，也要补做一次参数更新。
        if train and num_batches % self.accumulate_grad_batches != 0:
            self._step_optimizer()

        # 手动关闭 tqdm 进度条。
        progress.close()

        # 先从 meter 计算整体指标。
        metrics = meter.compute()
        # 再把各个损失分项换算成按 batch 平均的标量。
        for key, total in extra_totals.items():
            metrics[key] = total / max(1, num_batches)
        # 合并壁面 WSS 指标（wss_rmse_*, wss_r2_*）。
        if wss_meter is not None:
            metrics.update(wss_meter.compute())
        # 标记当前 epoch 是否启用 physics。
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
        # best_val 用来跟踪最佳验证损失。
        best_val = float("inf")
        # 记录最佳 epoch。
        best_epoch = 0
        # 早停计数器。
        patience = 0
        # 保存每个 epoch 的历史记录。
        history = []

        # 训练曲线 CSV 的输出路径。
        csv_path = run_dir / "history.csv"
        # 直接打开 CSV 文件准备持续写入。
        csv_file = open(csv_path, "w", encoding="utf-8", newline="")
        # 第一行字段名要等拿到首个 epoch 的 row 后才能确定。
        dict_writer: Optional[csv.DictWriter] = None

        try:
            for epoch in range(1, epochs + 1):
                # 记录当前 epoch 开始时间。
                t0 = time.time()
                # 跑一轮训练集。
                train_metrics = self.run_epoch(train_loader, train=True, epoch=epoch)
                # 跑一轮验证集。
                val_metrics = self.run_epoch(val_loader, train=False, epoch=epoch)

                # 混合验证指标：当 early_stop_wss_weight > 0 时用
                # data_loss + w * wss_loss 取代默认 total loss 做早停与调度。
                if self.early_stop_wss_weight > 0 and "wss_loss" in val_metrics:
                    val_score = (
                        val_metrics["data_loss"]
                        + self.early_stop_wss_weight * val_metrics["wss_loss"]
                    )
                else:
                    val_score = val_metrics["loss"]

                # 用验证分数推进学习率调度器。
                current_lr = self._step_scheduler(val_score, epoch)
                # 统计本轮耗时。
                epoch_time_sec = time.time() - t0
                # 当前验证分数是否刷新最优。
                is_best = val_score < best_val

                # 准备写入 CSV 的这一行。
                row: Dict[str, object] = {
                    "epoch": epoch,
                }
                # 把训练指标加上 train_ 前缀写进去。
                row.update({f"train_{k}": v for k, v in sorted(train_metrics.items())})
                # 把验证指标加上 val_ 前缀写进去。
                row.update({f"val_{k}": v for k, v in sorted(val_metrics.items())})
                # 当前学习率。
                row["lr"] = current_lr
                # 当前 epoch 耗时，保留两位小数。
                row["epoch_time_sec"] = round(epoch_time_sec, 2)
                # 是否为最佳模型，写成 0/1。
                row["is_best"] = int(is_best)

                # 第一个 epoch 时初始化 CSV 表头。
                if dict_writer is None:
                    dict_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                    dict_writer.writeheader()
                # 追加当前 epoch 记录。
                dict_writer.writerow(row)
                # 立刻 flush，避免中断时丢历史。
                csv_file.flush()

                # 训练、验证、优化器指标同时写 TensorBoard。
                self._log_scalars(train_metrics, "train", epoch)
                self._log_scalars(val_metrics, "val", epoch)
                self._log_scalars({"lr": current_lr}, "optim", epoch)

                # 计算打印时将要显示的 patience 值。
                next_patience = 0 if is_best else patience + 1
                summary = (
                    f"[Epoch {epoch}/{epochs}] "
                    f"train_loss={self._format_metric(train_metrics['loss'])} | "
                    f"val_loss={self._format_metric(val_metrics['loss'])} | "
                    f"val_score={self._format_metric(val_score)} | "
                    f"lr={current_lr:.6e} | "
                    f"best_val={self._format_metric(min(best_val, val_score))} | "
                    f"patience={next_patience}/{early_stopping_patience}"
                )
                print(summary)

                # 把当前 epoch 的详细结果留到 history 列表中。
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
                    # 刷新最佳验证分数。
                    best_val = val_score
                    # 记录最佳 epoch。
                    best_epoch = epoch
                    # 最优刷新时重置早停计数。
                    patience = 0
                    # 保存 best checkpoint。
                    save_checkpoint(self.model, run_dir / "best_model.pt")
                    print("已保存 best_model.pt")
                else:
                    # 没有刷新最优时，patience 加一。
                    patience += 1

                # 无论是否最佳，都更新 last checkpoint。
                save_checkpoint(self.model, run_dir / "last_model.pt")

                # 如果配置允许，同时定期额外保存 epoch checkpoint。
                if not save_best_only and epoch % save_every == 0:
                    save_checkpoint(self.model, run_dir / f"checkpoint_epoch_{epoch}.pt")

                # 达到早停阈值就提前结束训练。
                if patience >= early_stopping_patience:
                    break

                # 主动做一次 Python 垃圾回收。
                gc.collect()
                # CUDA 场景下顺便清理缓存显存。
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            # 不管训练是否异常退出，都要关闭 CSV 文件。
            csv_file.close()
            # 同样关闭 TensorBoard writer。
            if self.writer is not None:
                self.writer.close()

        # 返回训练阶段的核心摘要结果。
        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "history": history,
        }

    def evaluate(self, loader, checkpoint_path: Optional[Path] = None) -> Dict[str, float]:
        # 评估前如果给了 checkpoint，就先加载对应参数。
        if checkpoint_path is not None:
            load_checkpoint(self.model, checkpoint_path, self.device)
        # 复用 run_epoch 的验证逻辑；epoch 传大数只是为了让日志不与真实训练轮次冲突。
        return self.run_epoch(loader, train=False, epoch=10**9)
