from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .data import PointCloudBatch


def load_config(path: str | Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def move_batch(batch: PointCloudBatch, device: torch.device) -> PointCloudBatch:
    return PointCloudBatch(
        node_input=batch.node_input.to(device),
        target=batch.target.to(device),
        batch=batch.batch.to(device),
        global_cond=batch.global_cond.to(device),
        case_names=batch.case_names,
        sample_ids=batch.sample_ids,
    )
