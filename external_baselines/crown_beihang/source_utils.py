from __future__ import annotations

import torch


def wall_mask_from_velocity(
    targets: torch.Tensor,
    threshold: float = 0.01,
) -> torch.Tensor:
    """
    与源码 model_train_pinn.py 一致：用 |u|^2+|v|^2+|w|^2 <= threshold 推断壁面。
    targets: (B, 4, N) 物理速度 + 归一化前的 p 通道（速度通道为物理量）
    """
    u, v, w = targets[:, 0], targets[:, 1], targets[:, 2]
    return (u.square() + v.square() + w.square()) <= threshold


def sample_point_indices(n: int, choice: int, device: torch.device) -> torch.Tensor:
    choice = min(choice, n)
    return torch.randperm(n, device=device)[:choice]
