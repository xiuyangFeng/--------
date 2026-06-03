from __future__ import annotations

import contextlib
import csv
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from tqdm.auto import tqdm

from pipeline.config import NODE_FEATURE_NAMES, WSS_LOCAL_TARGET_NAMES, WSS_TARGET_NAMES

from .config import DomainLossConfig
from .io import load_checkpoint, save_checkpoint
from .losses import DualDomainLossBreakdown, build_loss_plugin
from .metrics import RegressionMeter, WSSMeter, compute_weighted_wss_val_term

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
        domain_loss_config: Optional[DomainLossConfig] = None,
        norm_params_path: Optional[str] = None,
        early_stop_min_delta: float = 0.0,
        val_score_ema_alpha: float = 0.0,
        val_score_wss_weights: Optional[Sequence[float]] = None,
        wss_target_names: Optional[Sequence[str]] = None,
        wss_target_frame: str = "global",
        wss_output_mode: str = "head",
        wss_metric_dim: int = 0,
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
        self.domain_loss_config = domain_loss_config
        self.loss_plugin = build_loss_plugin(
            physics_config,
            interior_loss_boost=interior_loss_boost,
            wss_loss_weight=wss_loss_weight,
            wss_weights=wss_weights,
            wss_loss_type=wss_loss_type,
            wss_huber_beta=wss_huber_beta,
            domain_loss_config=domain_loss_config,
        )
        self.accumulate_grad_batches = max(1, accumulate_grad_batches)
        self.early_stop_min_delta = float(early_stop_min_delta)
        self.val_score_ema_alpha = float(val_score_ema_alpha)
        if val_score_wss_weights is None:
            self._val_score_wss_weights: List[float] = [1.0, 1.0, 1.0, 1.0]
        else:
            self._val_score_wss_weights = [float(w) for w in val_score_wss_weights]
        self._wss_target_names = list(wss_target_names or WSS_TARGET_NAMES)
        self._wss_target_frame = wss_target_frame
        self._wss_output_mode = wss_output_mode
        self._wss_metric_dim = wss_metric_dim
        self._wss_best_metric_key = "wss_r2_mag" if wss_target_frame == "local" else "wss_r2_wss"

        # P0-B: 读取 normalization_params_global.json 中的 per-channel std。
        self.norm_stds: Dict[str, float] = {}
        self._use_domain_loss = domain_loss_config is not None and domain_loss_config.enabled
        if self._use_domain_loss and domain_loss_config.normalize_by_target_std:
            if domain_loss_config.norm_consts:
                self.norm_stds = dict(domain_loss_config.norm_consts)
            elif norm_params_path and Path(norm_params_path).exists():
                with open(norm_params_path, "r", encoding="utf-8") as f:
                    norm_payload = json.load(f)
                stats = norm_payload.get("statistics", {})
                for ch in ("u", "v", "w", "p", "wss", "wss_x", "wss_y", "wss_z", "wss_axial", "wss_circ", "wss_rad"):
                    ch_stats = stats.get(ch, {})
                    if "std" in ch_stats and ch_stats["std"] > 1e-10:
                        self.norm_stds[ch] = ch_stats["std"]
                wss_local = norm_payload.get("wss_local", {})
                if isinstance(wss_local, dict):
                    for comp in ("axial", "circ", "rad"):
                        comp_stats = wss_local.get(comp, {})
                        if comp_stats.get("std", 0) > 1e-10:
                            self.norm_stds[f"wss_local_{comp}"] = comp_stats["std"]
                            self.norm_stds[f"wss_{comp}"] = comp_stats["std"]

        # P0-A: mask 等价关系首 batch 校验标记。
        self._domain_mask_verified = False
        self._track_wss_metrics = self.wss_loss_weight > 0
        if self._use_domain_loss and domain_loss_config.lambda_wss > 0:
            self._track_wss_metrics = True
        if self._wss_output_mode == "vel_diff" and self._wss_metric_dim > 0:
            self._track_wss_metrics = True
        if (
            self._use_domain_loss
            and domain_loss_config is not None
            and domain_loss_config.lambda_wss_slope > 0
        ):
            self._track_wss_metrics = True

        # 只有 CUDA 上才真正启用 AMP。
        self.use_amp = use_amp and device.type == "cuda"
        # 启用 AMP 时用 GradScaler 管理缩放。
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp) if self.use_amp else None

        # 默认不写 TensorBoard。
        self.writer = None
        # 同时满足“给了 log_dir”且“环境里安装了 TensorBoard”时才创建 writer。
        if log_dir is not None and SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=str(log_dir))

    def _align_wss_target(self, wss_target: torch.Tensor, wss_pred: torch.Tensor) -> torch.Tensor:
        """将图数据全维 y_wss 对齐到当前 head 输出维度（如 magnitude-only wss_dim=1）。"""
        if wss_target.shape[1] == wss_pred.shape[1]:
            return wss_target
        base_names = WSS_LOCAL_TARGET_NAMES if self._wss_target_frame == "local" else WSS_TARGET_NAMES
        indices = [base_names.index(n) for n in self._wss_target_names]
        return wss_target[:, indices]

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
        wss_meter = WSSMeter(target_names=self._wss_target_names) if self._track_wss_metrics else None
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

                # P0-A: 首 batch 校验 wall_mask / interior_mask 非空。
                if self._use_domain_loss and not self._domain_mask_verified:
                    is_wall_col = batch.x[:, _is_wall_idx]
                    n_wall_check = is_wall_col.bool().sum().item()
                    n_int_check = (~is_wall_col.bool()).sum().item()
                    if n_wall_check == 0 or n_int_check == 0:
                        raise RuntimeError(
                            f"双域 mask 校验失败: wall={n_wall_check}, interior={n_int_check}。"
                            "请检查图资产的 is_wall 列。"
                        )
                    self._domain_mask_verified = True

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
                        wss_meter.update(
                            wss_pred,
                            self._align_wss_target(wss_target, wss_pred),
                            is_wall,
                        )

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
        # best_val 用来跟踪用于选模型/早停的验证分数（可能与 val_loss 不同）。
        best_val = float("inf")
        # best WSS R²（独立选优，保存 best_wss_model.pt）
        best_wss_r2: Optional[float] = None
        best_wss_epoch = 0
        # 记录最佳 epoch。
        best_epoch = 0
        # 早停计数器。
        patience = 0
        # 保存每个 epoch 的历史记录。
        history = []
        # val_score EMA；在 fit() 内每轮更新，不得跨多次 fit 复用。
        ema_val_score: Optional[float] = None

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

                # V3 双域 loss 路径：用归一化 RMSE 加权和作为验证分数。
                if self._use_domain_loss and self.norm_stds:
                    vel_std = max(
                        (self.norm_stds.get("u", 1.0) + self.norm_stds.get("v", 1.0) + self.norm_stds.get("w", 1.0)) / 3.0,
                        1e-10,
                    )
                    wss_term = compute_weighted_wss_val_term(
                        val_metrics,
                        self._val_score_wss_weights,
                        self.norm_stds,
                        target_names=self._wss_target_names,
                    )
                    # 压力 RMSE 已在归一化空间；勿再除以物理单位 p_std。
                    wall_p_rmse = (
                        val_metrics.get("loss_wall_pressure", 0.0) ** 0.5
                        if "loss_wall_pressure" in val_metrics
                        else val_metrics.get("rmse_p", 0.0)
                    )
                    _P_SCALE = 3.0
                    _vel_supervised = (
                        self.domain_loss_config is not None
                        and getattr(self.domain_loss_config, "lambda_vel_int", 0.0) > 0.0
                    )
                    if _vel_supervised:
                        vel_rmse = val_metrics.get("rmse_vel_mag", 0.0)
                        vel_term = 0.3 * vel_rmse / vel_std
                    else:
                        vel_term = 0.0
                    val_score = wss_term + _P_SCALE * wall_p_rmse + vel_term
                elif self._use_domain_loss:
                    val_score = val_metrics["loss"]
                # 旧路径：混合验证指标。
                elif self.early_stop_wss_weight > 0 and "wss_loss" in val_metrics:
                    val_score = (
                        val_metrics["data_loss"]
                        + self.early_stop_wss_weight * val_metrics["wss_loss"]
                    )
                else:
                    val_score = val_metrics["loss"]

                # 用原始 val_score 推进学习率调度器（勿用 EMA，避免拖慢 LR 衰减）。
                current_lr = self._step_scheduler(val_score, epoch)
                # 统计本轮耗时。
                epoch_time_sec = time.time() - t0

                in_warmup = self.warmup_epochs > 0 and epoch <= self.warmup_epochs
                if not in_warmup and self.val_score_ema_alpha > 0.0:
                    a = self.val_score_ema_alpha
                    ema_val_score = val_score if ema_val_score is None else (
                        a * val_score + (1.0 - a) * ema_val_score
                    )
                    score_for_es = ema_val_score
                else:
                    score_for_es = val_score

                # Warmup 内不参与早停 / best_model；避免前几轮噪声锁死 checkpoint。
                if in_warmup:
                    is_best = False
                else:
                    is_best = score_for_es < (best_val - self.early_stop_min_delta)

                if self._wss_best_metric_key in val_metrics:
                    val_wss_r2 = float(val_metrics[self._wss_best_metric_key])
                    if best_wss_r2 is None or val_wss_r2 > best_wss_r2:
                        best_wss_r2 = val_wss_r2
                        best_wss_epoch = epoch
                        save_checkpoint(self.model, run_dir / "best_wss_model.pt")
                elif "wss_r2_wss" in val_metrics:
                    val_wss_r2 = float(val_metrics["wss_r2_wss"])
                    if best_wss_r2 is None or val_wss_r2 > best_wss_r2:
                        best_wss_r2 = val_wss_r2
                        best_wss_epoch = epoch
                        save_checkpoint(self.model, run_dir / "best_wss_model.pt")

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
                row["val_score"] = val_score
                row["val_score_ema"] = ema_val_score if ema_val_score is not None else val_score
                if self._use_domain_loss and self.norm_stds:
                    row["val_wss_term"] = wss_term
                row["early_stop_score"] = score_for_es
                # P0-B: 写入归一化 std 元信息列，便于事后审计。
                if self.norm_stds:
                    for ch, std_val in sorted(self.norm_stds.items()):
                        row[f"norm_std_{ch}"] = std_val

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

                # 计算打印时将要显示的 patience 值（warmup 内始终视为 0）。
                if in_warmup:
                    next_patience = 0
                else:
                    next_patience = 0 if is_best else patience + 1
                eff_best = min(best_val, score_for_es) if best_val < float("inf") else score_for_es
                summary = (
                    f"[Epoch {epoch}/{epochs}] "
                    f"train_loss={self._format_metric(train_metrics['loss'])} | "
                    f"val_loss={self._format_metric(val_metrics['loss'])} | "
                    f"val_score={self._format_metric(val_score)} | "
                    f"es_score={self._format_metric(score_for_es)} | "
                    f"lr={current_lr:.6e} | "
                    f"best_es={self._format_metric(eff_best)} | "
                    f"patience={next_patience}/{early_stopping_patience}"
                    + (" | warmup" if in_warmup else "")
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
                        "val_score": val_score,
                        "early_stop_score": score_for_es,
                    }
                )

                if is_best:
                    # 刷新最佳验证分数（与早停判定一致：EMA 或原始）。
                    best_val = score_for_es
                    # 记录最佳 epoch。
                    best_epoch = epoch
                    # 最优刷新时重置早停计数。
                    patience = 0
                    # 保存 best checkpoint。
                    save_checkpoint(self.model, run_dir / "best_model.pt")
                    print("已保存 best_model.pt")
                elif not in_warmup:
                    # 没有刷新最优时，patience 加一。
                    patience += 1

                # 无论是否最佳，都更新 last checkpoint。
                save_checkpoint(self.model, run_dir / "last_model.pt")

                # 如果配置允许，同时定期额外保存 epoch checkpoint。
                if not save_best_only and epoch % save_every == 0:
                    save_checkpoint(self.model, run_dir / f"checkpoint_epoch_{epoch}.pt")

                # 达到早停阈值就提前结束训练（warmup 内永不触发）。
                if not in_warmup and patience >= early_stopping_patience:
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

        if best_epoch == 0 and history:
            last_entry = history[-1]
            best_epoch = int(last_entry["epoch"])
            best_val = float(last_entry["early_stop_score"])
            save_checkpoint(self.model, run_dir / "best_model.pt")
            print("警告: warmup 占满或未产生有效选优，已用最后一轮作为 best_model.pt")

        best_true_val_loss = float("nan")
        if best_epoch > 0 and history:
            best_true_val_loss = float(history[best_epoch - 1]["val"]["loss"])
        elif history:
            best_true_val_loss = float(history[-1]["val"]["loss"])

        # 返回训练阶段的核心摘要结果。
        return {
            "best_epoch": best_epoch,
            "best_val_score": best_val,
            "best_val_loss": best_true_val_loss,
            "best_wss_epoch": best_wss_epoch,
            "best_val_wss_r2": best_wss_r2,
            "history": history,
        }

    def evaluate(self, loader, checkpoint_path: Optional[Path] = None) -> Dict[str, float]:
        # 评估前如果给了 checkpoint，就先加载对应参数。
        if checkpoint_path is not None:
            load_checkpoint(self.model, checkpoint_path, self.device)
        # 复用 run_epoch 的验证逻辑；epoch 传大数只是为了让日志不与真实训练轮次冲突。
        return self.run_epoch(loader, train=False, epoch=10**9)
