"""Shared runtime helpers for training and diagnostic experiments."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .config import TrainConfig


def setup_logger(run_dir: Path) -> logging.Logger:
    """Create an isolated console/file logger for one run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"wssmin.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    file_handler = logging.FileHandler(run_dir / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def seed_all(seed: int) -> None:
    """Seed NumPy and Torch without changing deterministic-kernel policy."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda_factory(cfg: TrainConfig) -> Callable[[int], float]:
    """Build the historical linear-warmup/cosine-decay LR multiplier."""
    def multiplier(epoch: int) -> float:
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / max(1, cfg.warmup_epochs)
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return (cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine) / cfg.lr

    return multiplier
