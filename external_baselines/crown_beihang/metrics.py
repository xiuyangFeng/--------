from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch


def regression_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    names: Sequence[str],
    p_min: float | None = None,
    p_max: float | None = None,
) -> Dict[str, float]:
    if p_min is not None and p_max is not None and "p" in names:
        span = p_max - p_min + 1e-9
        p_idx = list(names).index("p")
        pred = pred.clone()
        target = target.clone()
        pred[:, p_idx] = pred[:, p_idx] * span + p_min
        target[:, p_idx] = target[:, p_idx] * span + p_min

    diff = pred - target
    metrics: Dict[str, float] = {
        "loss_mse": float(torch.mean(diff.square()).item()),
        "mae": float(torch.mean(diff.abs()).item()),
        "rmse": float(torch.sqrt(torch.mean(diff.square())).item()),
    }
    for idx, name in enumerate(names):
        err = diff[:, idx]
        y = target[:, idx]
        ss_res = torch.sum(err.square())
        ss_tot = torch.sum((y - y.mean()).square()).clamp_min(1e-12)
        metrics[f"{name}_mae"] = float(torch.mean(err.abs()).item())
        metrics[f"{name}_rmse"] = float(torch.sqrt(torch.mean(err.square())).item())
        metrics[f"{name}_r2"] = float((1.0 - ss_res / ss_tot).item())

    if all(n in names for n in ("u", "v", "w")):
        u_idx, v_idx, w_idx = names.index("u"), names.index("v"), names.index("w")
        vel_pred = torch.sqrt(pred[:, u_idx].square() + pred[:, v_idx].square() + pred[:, w_idx].square())
        vel_tgt = torch.sqrt(
            target[:, u_idx].square() + target[:, v_idx].square() + target[:, w_idx].square()
        )
        vel_err = vel_pred - vel_tgt
        metrics["vel_mag_mae"] = float(torch.mean(vel_err.abs()).item())
        metrics["vel_mag_rmse"] = float(torch.sqrt(torch.mean(vel_err.square())).item())
        ss_res = torch.sum(vel_err.square())
        ss_tot = torch.sum((vel_tgt - vel_tgt.mean()).square()).clamp_min(1e-12)
        metrics["vel_mag_r2"] = float((1.0 - ss_res / ss_tot).item())

    return metrics


def grouped_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    groups: Sequence[str],
    names: Sequence[str],
    p_min: float | None = None,
    p_max: float | None = None,
) -> List[Dict[str, object]]:
    by_group: Dict[str, List[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        by_group[group].append(idx)

    rows: List[Dict[str, object]] = []
    for group, indices in sorted(by_group.items()):
        idx_tensor = torch.as_tensor(indices, dtype=torch.long)
        metrics = regression_metrics(
            pred[idx_tensor], target[idx_tensor], names, p_min=p_min, p_max=p_max
        )
        row: Dict[str, object] = {"group": group, "n_points": len(indices)}
        row.update(metrics)
        rows.append(row)
    return rows


def write_metric_rows(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
