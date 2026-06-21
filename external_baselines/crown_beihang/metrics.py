from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch


def _physical_pressure(
    pred: torch.Tensor,
    target: torch.Tensor,
    names: Sequence[str],
    p_min: float | None = None,
    p_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if p_min is None or p_max is None or "p" not in names:
        return pred, target

    span = p_max - p_min + 1e-9
    p_idx = list(names).index("p")
    pred = pred.clone()
    target = target.clone()
    pred[:, p_idx] = pred[:, p_idx] * span + p_min
    target[:, p_idx] = target[:, p_idx] * span + p_min
    return pred, target


def metric_ranges(
    target: torch.Tensor,
    names: Sequence[str],
    p_min: float | None = None,
    p_max: float | None = None,
) -> Dict[str, float]:
    _, target = _physical_pressure(target, target, names, p_min=p_min, p_max=p_max)
    ranges: Dict[str, float] = {}
    for idx, name in enumerate(names):
        y = target[:, idx]
        ranges[name] = float((y.max() - y.min()).clamp_min(1e-12).item())

    if all(n in names for n in ("u", "v", "w")):
        u_idx, v_idx, w_idx = names.index("u"), names.index("v"), names.index("w")
        vel_tgt = torch.sqrt(
            target[:, u_idx].square() + target[:, v_idx].square() + target[:, w_idx].square()
        )
        ranges["vel_mag"] = float((vel_tgt.max() - vel_tgt.min()).clamp_min(1e-12).item())
    return ranges


def regression_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    names: Sequence[str],
    p_min: float | None = None,
    p_max: float | None = None,
    nmae_ranges: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    pred, target = _physical_pressure(pred, target, names, p_min=p_min, p_max=p_max)
    if nmae_ranges is None:
        nmae_ranges = metric_ranges(target, names)

    diff = pred - target
    metrics: Dict[str, float] = {
        "loss_mse": float(torch.mean(diff.square()).item()),
        "mae": float(torch.mean(diff.abs()).item()),
        "rmse": float(torch.sqrt(torch.mean(diff.square())).item()),
    }
    nmae_values: List[float] = []
    for idx, name in enumerate(names):
        err = diff[:, idx]
        y = target[:, idx]
        ss_res = torch.sum(err.square())
        ss_tot = torch.sum((y - y.mean()).square()).clamp_min(1e-12)
        denom = max(float(nmae_ranges.get(name, 0.0)), 1e-12)
        nmae = float((torch.mean(err.abs()) / denom).item())
        metrics[f"{name}_mae"] = float(torch.mean(err.abs()).item())
        metrics[f"{name}_rmse"] = float(torch.sqrt(torch.mean(err.square())).item())
        metrics[f"{name}_nmae"] = nmae
        metrics[f"{name}_r2"] = float((1.0 - ss_res / ss_tot).item())
        nmae_values.append(nmae)

    if nmae_values:
        metrics["nmae"] = float(sum(nmae_values) / len(nmae_values))

    if all(n in names for n in ("u", "v", "w")):
        u_idx, v_idx, w_idx = names.index("u"), names.index("v"), names.index("w")
        vel_pred = torch.sqrt(pred[:, u_idx].square() + pred[:, v_idx].square() + pred[:, w_idx].square())
        vel_tgt = torch.sqrt(
            target[:, u_idx].square() + target[:, v_idx].square() + target[:, w_idx].square()
        )
        vel_err = vel_pred - vel_tgt
        vel_denom = max(float(nmae_ranges.get("vel_mag", 0.0)), 1e-12)
        metrics["vel_mag_mae"] = float(torch.mean(vel_err.abs()).item())
        metrics["vel_mag_rmse"] = float(torch.sqrt(torch.mean(vel_err.square())).item())
        metrics["vel_mag_nmae"] = float((torch.mean(vel_err.abs()) / vel_denom).item())
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
    nmae_ranges: Mapping[str, float] | None = None,
) -> List[Dict[str, object]]:
    by_group: Dict[str, List[int]] = defaultdict(list)
    for idx, group in enumerate(groups):
        by_group[group].append(idx)

    if nmae_ranges is None:
        nmae_ranges = metric_ranges(target, names, p_min=p_min, p_max=p_max)

    rows: List[Dict[str, object]] = []
    for group, indices in sorted(by_group.items()):
        idx_tensor = torch.as_tensor(indices, dtype=torch.long)
        metrics = regression_metrics(
            pred[idx_tensor],
            target[idx_tensor],
            names,
            p_min=p_min,
            p_max=p_max,
            nmae_ranges=nmae_ranges,
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
