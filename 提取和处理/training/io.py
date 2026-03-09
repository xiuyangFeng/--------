from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from .utils import ensure_dir


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)


def append_experiment_index(
    output_root: str | Path,
    row: Dict[str, object],
    fieldnames: Iterable[str],
) -> None:
    output_root = ensure_dir(output_root)
    csv_path = Path(output_root) / "experiment_index.csv"
    exists = csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def sanitize_batch_metadata(values) -> List[str]:
    if isinstance(values, (list, tuple)):
        return [str(v) for v in values]
    return [str(values)]

