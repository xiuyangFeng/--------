from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from pipeline.config import TARGET_NAMES


def _compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Shared metric computation for a set of (pred, target) pairs."""
    diff = pred - target
    metrics: Dict[str, float] = {
        "rmse": torch.sqrt((diff ** 2).mean()).item(),
        "mae": diff.abs().mean().item(),
    }

    target_mean = target.mean(dim=0, keepdim=True)
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - target_mean) ** 2).sum(dim=0).clamp_min(1e-12)
    r2 = 1.0 - ss_res / ss_tot

    for idx, name in enumerate(TARGET_NAMES):
        metrics[f"rmse_{name}"] = torch.sqrt((diff[:, idx] ** 2).mean()).item()
        metrics[f"mae_{name}"] = diff[:, idx].abs().mean().item()
        metrics[f"r2_{name}"] = r2[idx].item()

    velocity_pred = pred[:, :3].norm(dim=1)
    velocity_target = target[:, :3].norm(dim=1)
    metrics["rmse_vel_mag"] = torch.sqrt(((velocity_pred - velocity_target) ** 2).mean()).item()
    metrics["mae_vel_mag"] = (velocity_pred - velocity_target).abs().mean().item()
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

    def update(self, pred: torch.Tensor, target: torch.Tensor, loss: float) -> None:
        pred_c = pred.detach().cpu().float()
        target_c = target.detach().cpu().float()
        n = pred_c.size(0)

        diff = pred_c - target_c
        sq_err = (diff ** 2).sum(dim=0)   # [n_dims]
        abs_err = diff.abs().sum(dim=0)   # [n_dims]

        vel_diff = pred_c[:, :3].norm(dim=1) - target_c[:, :3].norm(dim=1)
        self._sum_sq_vel_err += (vel_diff ** 2).sum().item()
        self._sum_abs_vel_err += vel_diff.abs().sum().item()

        if self._sum_sq_err is None:
            self._sum_sq_err = sq_err
            self._sum_abs_err = abs_err
            batch_mean = target_c.mean(dim=0)
            self._target_mean = batch_mean
            self._target_M2 = ((target_c - batch_mean) ** 2).sum(dim=0)
            self._n = n
        else:
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
            self._n = n_new

        self._total_loss_sum += loss * n
        self.num_batches += 1

    def compute(self) -> Dict[str, float]:
        if self._n == 0 or self._sum_sq_err is None:
            return {}

        n = self._n
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
        metrics["loss"] = self._total_loss_sum / n
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
        pred_cpu = pred.detach().cpu()
        target_cpu = target.detach().cpu()
        for cname in set(case_names):
            mask = [i for i, c in enumerate(case_names) if c == cname]
            idx = torch.tensor(mask, dtype=torch.long)
            self._case_preds[cname].append(pred_cpu[idx])
            self._case_targets[cname].append(target_cpu[idx])

    def compute(self) -> Dict[str, Dict[str, float]]:
        """Returns ``{case_name: {metric_name: value}}``."""
        results: Dict[str, Dict[str, float]] = {}
        for cname in sorted(self._case_preds):
            p = torch.cat(self._case_preds[cname], dim=0)
            t = torch.cat(self._case_targets[cname], dim=0)
            results[cname] = _compute_metrics(p, t)
            results[cname]["n_nodes"] = p.size(0)
        return results

    def summarize(self) -> Dict[str, Dict[str, float]]:
        """Returns ``{metric_name: {mean, std, min, max, n_cases}}``."""
        per_case = self.compute()
        if not per_case:
            return {}
        all_keys = set()
        for m in per_case.values():
            all_keys.update(k for k in m if k != "n_nodes")

        summary: Dict[str, Dict[str, float]] = {}
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
        self._case_preds.clear()
        self._case_targets.clear()
