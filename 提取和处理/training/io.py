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
    requested_fields = list(fieldnames)

    if not csv_path.exists():
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=requested_fields)
            writer.writeheader()
            writer.writerow(row)
        return

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        existing_rows = list(reader)

    merged_fields = existing_fields.copy()
    for field in requested_fields:
        if field not in merged_fields:
            merged_fields.append(field)

    existing_rows.append({key: row.get(key, "") for key in merged_fields})
    normalized_rows = []
    for old_row in existing_rows:
        normalized_rows.append({key: old_row.get(key, "") for key in merged_fields})

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields)
        writer.writeheader()
        writer.writerows(normalized_rows)


def sanitize_batch_metadata(values) -> List[str]:
    if isinstance(values, (list, tuple)):
        return [str(v) for v in values]
    return [str(values)]
