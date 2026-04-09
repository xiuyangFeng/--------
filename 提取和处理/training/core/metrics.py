from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from pipeline.config import NODE_FEATURE_NAMES, TARGET_NAMES, WSS_TARGET_NAMES


def _compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Shared metric computation for a set of (pred, target) pairs."""
    # 先计算逐元素误差。
    diff = pred - target
    # 整体 RMSE / MAE。
    metrics: Dict[str, float] = {
        "rmse": torch.sqrt((diff ** 2).mean()).item(),
        "mae": diff.abs().mean().item(),
    }

    # R² 需要先计算目标均值、残差平方和和总平方和。
    target_mean = target.mean(dim=0, keepdim=True)
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - target_mean) ** 2).sum(dim=0).clamp_min(1e-12)
    r2 = 1.0 - ss_res / ss_tot

    # 逐个输出维度补充 RMSE / MAE / R²。
    for idx, name in enumerate(TARGET_NAMES):
        metrics[f"rmse_{name}"] = torch.sqrt((diff[:, idx] ** 2).mean()).item()
        metrics[f"mae_{name}"] = diff[:, idx].abs().mean().item()
        metrics[f"r2_{name}"] = r2[idx].item()

    # 额外统计速度模长误差，方便更直观比较速度场质量。
    velocity_pred = pred[:, :3].norm(dim=1)
    velocity_target = target[:, :3].norm(dim=1)
    metrics["rmse_vel_mag"] = torch.sqrt(((velocity_pred - velocity_target) ** 2).mean()).item()
    metrics["mae_vel_mag"] = (velocity_pred - velocity_target).abs().mean().item()
    ss_res_vm = ((velocity_target - velocity_pred) ** 2).sum()
    ss_tot_vm = ((velocity_target - velocity_target.mean()) ** 2).sum().clamp_min(1e-12)
    metrics["r2_vel_mag"] = (1.0 - ss_res_vm / ss_tot_vm).item()
    return metrics


@dataclass
class RegressionMeter:
    """Incremental metric accumulator that avoids storing all predictions.

    Original approach appended every batch's pred/target tensors into lists and
    concatenated them at epoch end.  For a dataset with thousands of batches and
    tens-of-thousands of nodes per graph this can consume tens of GBs of CPU
    memory per epoch.

    This version keeps only O(n_dims) running scalars using the parallel Welford
    algorithm for online variance (needed for R²), so memory cost is constant
    regardless of dataset size.
    """

    # All fields are internal; callers only use update() / compute().
    _n: int = field(default=0, init=False, repr=False)
    _total_loss_sum: float = field(default=0.0, init=False, repr=False)
    num_batches: int = field(default=0, init=False)

    # Running sums of squared / absolute errors, shape [n_dims]
    _sum_sq_err: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _sum_abs_err: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    # Parallel Welford state for target variance (SS_tot), shape [n_dims]
    _target_mean: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _target_M2: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    # Velocity-magnitude tracking (scalar accumulators)
    _sum_sq_vel_err: float = field(default=0.0, init=False, repr=False)
    _sum_abs_vel_err: float = field(default=0.0, init=False, repr=False)
    _vel_target_mean: float = field(default=0.0, init=False, repr=False)
    _vel_target_M2: float = field(default=0.0, init=False, repr=False)

    def update(self, pred: torch.Tensor, target: torch.Tensor, loss: float) -> None:
        # 把预测和真值都 detach 到 CPU，并转成 float，避免累计时占 GPU 显存。
        pred_c = pred.detach().cpu().float()
        target_c = target.detach().cpu().float()
        # 当前 batch 的节点数。
        n = pred_c.size(0)

        # 逐元素误差。
        diff = pred_c - target_c
        # 逐维平方误差和。
        sq_err = (diff ** 2).sum(dim=0)   # [n_dims]
        # 逐维绝对误差和。
        abs_err = diff.abs().sum(dim=0)   # [n_dims]

        # 速度模长误差单独累计。
        vel_target = target_c[:, :3].norm(dim=1)
        vel_diff = pred_c[:, :3].norm(dim=1) - vel_target
        self._sum_sq_vel_err += (vel_diff ** 2).sum().item()
        self._sum_abs_vel_err += vel_diff.abs().sum().item()
        vel_batch_mean = vel_target.mean().item()
        vel_batch_M2 = ((vel_target - vel_batch_mean) ** 2).sum().item()

        # 第一个 batch 直接初始化累计器。
        if self._sum_sq_err is None:
            self._sum_sq_err = sq_err
            self._sum_abs_err = abs_err
            batch_mean = target_c.mean(dim=0)
            self._target_mean = batch_mean
            self._target_M2 = ((target_c - batch_mean) ** 2).sum(dim=0)
            self._vel_target_mean = vel_batch_mean
            self._vel_target_M2 = vel_batch_M2
            self._n = n
        else:
            # 后续 batch 继续累加平方误差与绝对误差。
            self._sum_sq_err += sq_err
            self._sum_abs_err += abs_err
            # Parallel Welford merge: combine existing state (n_old nodes) with
            # this batch (n new nodes).
            n_old = self._n
            n_new = n_old + n
            batch_mean = target_c.mean(dim=0)
            batch_M2 = ((target_c - batch_mean) ** 2).sum(dim=0)
            delta = batch_mean - self._target_mean
            self._target_mean = self._target_mean + delta * (n / n_new)
            self._target_M2 = (
                self._target_M2 + batch_M2 + delta ** 2 * (n_old * n / n_new)
            )
            vel_delta = vel_batch_mean - self._vel_target_mean
            self._vel_target_mean = self._vel_target_mean + vel_delta * (n / n_new)
            self._vel_target_M2 = (
                self._vel_target_M2 + vel_batch_M2 + vel_delta ** 2 * (n_old * n / n_new)
            )
            self._n = n_new

        # loss 按节点数加权累计，后面再除以总节点数得到全局平均损失。
        self._total_loss_sum += loss * n
        # batch 计数加一。
        self.num_batches += 1

    def compute(self) -> Dict[str, float]:
        # 没有数据时返回空字典。
        if self._n == 0 or self._sum_sq_err is None:
            return {}

        # 总节点数。
        n = self._n
        # 输出维度数。
        n_dims = self._sum_sq_err.size(0)
        metrics: Dict[str, float] = {}

        # Overall RMSE / MAE (mean over all nodes × all dims, matching original)
        metrics["rmse"] = (self._sum_sq_err.sum().item() / (n * n_dims)) ** 0.5
        metrics["mae"] = self._sum_abs_err.sum().item() / (n * n_dims)

        # Per-dim metrics
        ss_tot = self._target_M2.clamp_min(1e-12)  # [n_dims]
        r2 = 1.0 - self._sum_sq_err / ss_tot
        for idx, name in enumerate(TARGET_NAMES):
            metrics[f"rmse_{name}"] = (self._sum_sq_err[idx].item() / n) ** 0.5
            metrics[f"mae_{name}"] = self._sum_abs_err[idx].item() / n
            metrics[f"r2_{name}"] = r2[idx].item()

        metrics["rmse_vel_mag"] = (self._sum_sq_vel_err / n) ** 0.5
        metrics["mae_vel_mag"] = self._sum_abs_vel_err / n
        metrics["r2_vel_mag"] = 1.0 - self._sum_sq_vel_err / max(self._vel_target_M2, 1e-12)
        metrics["loss"] = self._total_loss_sum / n
        return metrics


@dataclass
class WSSMeter:
    """Tracks WSS prediction metrics on wall nodes only."""

    _n: int = field(default=0, init=False, repr=False)
    _sum_sq_err: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _sum_abs_err: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _target_mean: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _target_M2: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    def update(self, wss_pred: torch.Tensor, wss_target: torch.Tensor, is_wall: torch.Tensor) -> None:
        wall_mask = is_wall.squeeze(-1).bool()
        if wall_mask.sum() == 0:
            return
        pred_w = wss_pred[wall_mask].detach().cpu().float()
        target_w = wss_target[wall_mask].detach().cpu().float()
        n = pred_w.size(0)
        diff = pred_w - target_w
        sq_err = (diff ** 2).sum(dim=0)
        abs_err = diff.abs().sum(dim=0)

        if self._sum_sq_err is None:
            self._sum_sq_err = sq_err
            self._sum_abs_err = abs_err
            batch_mean = target_w.mean(dim=0)
            self._target_mean = batch_mean
            self._target_M2 = ((target_w - batch_mean) ** 2).sum(dim=0)
            self._n = n
        else:
            self._sum_sq_err += sq_err
            self._sum_abs_err += abs_err
            n_old = self._n
            n_new = n_old + n
            batch_mean = target_w.mean(dim=0)
            batch_M2 = ((target_w - batch_mean) ** 2).sum(dim=0)
            delta = batch_mean - self._target_mean
            self._target_mean = self._target_mean + delta * (n / n_new)
            self._target_M2 = self._target_M2 + batch_M2 + delta ** 2 * (n_old * n / n_new)
            self._n = n_new

    def compute(self) -> Dict[str, float]:
        if self._n == 0 or self._sum_sq_err is None:
            return {}
        n = self._n
        metrics: Dict[str, float] = {}
        ss_tot = self._target_M2.clamp_min(1e-12)
        r2 = 1.0 - self._sum_sq_err / ss_tot
        for idx, name in enumerate(WSS_TARGET_NAMES):
            metrics[f"wss_rmse_{name}"] = (self._sum_sq_err[idx].item() / n) ** 0.5
            metrics[f"wss_r2_{name}"] = r2[idx].item()
        metrics["wss_rmse"] = (self._sum_sq_err.sum().item() / (n * len(WSS_TARGET_NAMES))) ** 0.5
        return metrics


@dataclass
class PerCaseMeter:
    """Accumulates predictions grouped by case_name for per-patient metrics.

    Call ``update()`` once per batch, passing the per-node ``case_names``
    list (length == number of nodes in the batch).
    """
    _case_preds: Dict[str, List[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))
    _case_targets: Dict[str, List[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        case_names: List[str],
    ) -> None:
        # 每次更新都先拷到 CPU，避免病例级累计长期占用 GPU。
        pred_cpu = pred.detach().cpu()
        target_cpu = target.detach().cpu()
        # 按 case_name 分组，把当前 batch 中同一病例的节点切出来。
        for cname in set(case_names):
            mask = [i for i, c in enumerate(case_names) if c == cname]
            idx = torch.tensor(mask, dtype=torch.long)
            self._case_preds[cname].append(pred_cpu[idx])
            self._case_targets[cname].append(target_cpu[idx])

    def compute(self) -> Dict[str, Dict[str, float]]:
        """Returns ``{case_name: {metric_name: value}}``."""
        # 最终输出格式：每个病例对应一组指标。
        results: Dict[str, Dict[str, float]] = {}
        # 逐病例拼接该病例在各个 batch 中积累的节点预测与真值。
        for cname in sorted(self._case_preds):
            p = torch.cat(self._case_preds[cname], dim=0)
            t = torch.cat(self._case_targets[cname], dim=0)
            results[cname] = _compute_metrics(p, t)
            results[cname]["n_nodes"] = p.size(0)
        return results

    def summarize(self) -> Dict[str, Dict[str, float]]:
        """Returns ``{metric_name: {mean, std, min, max, n_cases}}``."""
        # 先得到每个病例的原始指标。
        per_case = self.compute()
        # 没有病例时返回空字典。
        if not per_case:
            return {}
        # 收集所有病例里出现过的指标名。
        all_keys = set()
        for m in per_case.values():
            all_keys.update(k for k in m if k != "n_nodes")

        summary: Dict[str, Dict[str, float]] = {}
        # 对每个指标做病例间均值、标准差、极值统计。
        for key in sorted(all_keys):
            vals = [m[key] for m in per_case.values() if key in m]
            t = torch.tensor(vals, dtype=torch.float64)
            summary[key] = {
                "mean": t.mean().item(),
                "std": t.std().item() if len(t) > 1 else 0.0,
                "min": t.min().item(),
                "max": t.max().item(),
                "n_cases": len(vals),
            }
        return summary

    def reset(self) -> None:
        # 清空所有病例级累计缓存，便于下次重新统计。
        self._case_preds.clear()
        self._case_targets.clear()
