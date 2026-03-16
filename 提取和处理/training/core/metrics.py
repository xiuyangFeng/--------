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

    pred_mean = target.mean(dim=0, keepdim=True)
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - pred_mean) ** 2).sum(dim=0).clamp_min(1e-12)
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
    """Accumulates predictions across batches for epoch-level metrics."""
    preds: List[torch.Tensor] = field(default_factory=list)
    targets: List[torch.Tensor] = field(default_factory=list)
    total_loss: float = 0.0
    total_nodes: int = 0
    num_batches: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, loss: float) -> None:
        n = pred.size(0)
        self.preds.append(pred.detach().cpu())
        self.targets.append(target.detach().cpu())
        self.total_loss += loss * n
        self.total_nodes += n
        self.num_batches += 1

    def compute(self) -> Dict[str, float]:
        pred = torch.cat(self.preds, dim=0)
        target = torch.cat(self.targets, dim=0)

        metrics = _compute_metrics(pred, target)
        metrics["loss"] = self.total_loss / max(1, self.total_nodes)
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
