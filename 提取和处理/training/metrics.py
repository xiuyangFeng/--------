from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from pipeline.config import TARGET_NAMES


@dataclass
class RegressionMeter:
    # 这里按整轮累计全部节点预测，再统一算指标，避免 batch 粒度平均带来偏差。
    preds: List[torch.Tensor] = field(default_factory=list)
    targets: List[torch.Tensor] = field(default_factory=list)
    total_loss: float = 0.0
    num_batches: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, loss: float) -> None:
        self.preds.append(pred.detach().cpu())
        self.targets.append(target.detach().cpu())
        self.total_loss += loss
        self.num_batches += 1

    def compute(self) -> Dict[str, float]:
        pred = torch.cat(self.preds, dim=0)
        target = torch.cat(self.targets, dim=0)
        diff = pred - target

        metrics: Dict[str, float] = {
            "loss": self.total_loss / max(1, self.num_batches),
            "rmse": torch.sqrt((diff ** 2).mean()).item(),
            "mae": diff.abs().mean().item(),
        }

        pred_mean = target.mean(dim=0, keepdim=True)
        # 这里按输出维度分别算 R²，避免速度和压力量纲差异相互掩盖。
        ss_res = ((target - pred) ** 2).sum(dim=0)
        ss_tot = ((target - pred_mean) ** 2).sum(dim=0).clamp_min(1e-12)
        r2 = 1.0 - ss_res / ss_tot

        for idx, name in enumerate(TARGET_NAMES):
            metrics[f"rmse_{name}"] = torch.sqrt((diff[:, idx] ** 2).mean()).item()
            metrics[f"mae_{name}"] = diff[:, idx].abs().mean().item()
            metrics[f"r2_{name}"] = r2[idx].item()

        velocity_pred = pred[:, :3].norm(dim=1)
        velocity_target = target[:, :3].norm(dim=1)
        # 速度模长误差经常比单个分量更贴近血流场重建的实际使用场景。
        metrics["rmse_vel_mag"] = torch.sqrt(((velocity_pred - velocity_target) ** 2).mean()).item()
        metrics["mae_vel_mag"] = (velocity_pred - velocity_target).abs().mean().item()
        return metrics
