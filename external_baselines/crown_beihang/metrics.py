from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

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


def update_global_ranges_from_targets(
    mins: MutableMapping[str, float],
    maxs: MutableMapping[str, float],
    target: torch.Tensor,
    names: Sequence[str],
    p_min: float | None = None,
    p_max: float | None = None,
) -> None:
    """按 test 全集 GT 更新 global min/max（对齐 model_error_analysis.py）。"""
    _, target = _physical_pressure(target, target, names, p_min=p_min, p_max=p_max)
    for idx, name in enumerate(names):
        y = target[:, idx]
        y_min = float(y.min().item())
        y_max = float(y.max().item())
        mins[name] = min(float(mins.get(name, y_min)), y_min)
        maxs[name] = max(float(maxs.get(name, y_max)), y_max)

    if all(n in names for n in ("u", "v", "w")):
        u_idx, v_idx, w_idx = names.index("u"), names.index("v"), names.index("w")
        vel_tgt = torch.sqrt(
            target[:, u_idx].square() + target[:, v_idx].square() + target[:, w_idx].square()
        )
        v_min = float(vel_tgt.min().item())
        v_max = float(vel_tgt.max().item())
        mins["vel_mag"] = min(float(mins.get("vel_mag", v_min)), v_min)
        maxs["vel_mag"] = max(float(maxs.get("vel_mag", v_max)), v_max)


def global_ranges_from_minmax(
    mins: Mapping[str, float], maxs: Mapping[str, float]
) -> Dict[str, float]:
    ranges: Dict[str, float] = {}
    for key, y_min in mins.items():
        y_max = maxs.get(key, y_min)
        ranges[key] = max(float(y_max) - float(y_min), 1e-12)
    return ranges


class PooledMetricAccumulator:
    """流式累计全场点指标，避免 1296×全点云驻内存。"""

    def __init__(
        self,
        names: Sequence[str],
        p_min: float | None,
        p_max: float | None,
        nmae_ranges: Mapping[str, float],
        prefix: str = "",
    ) -> None:
        self.names = list(names)
        self.p_min = p_min
        self.p_max = p_max
        self.nmae_ranges = dict(nmae_ranges)
        self.prefix = prefix
        self.n_points = 0
        self._sum_abs: Dict[str, float] = {n: 0.0 for n in self.names}
        self._sum_sq: Dict[str, float] = {n: 0.0 for n in self.names}
        self._sum_y: Dict[str, float] = {n: 0.0 for n in self.names}
        self._sum_y2: Dict[str, float] = {n: 0.0 for n in self.names}
        self._sum_err2: Dict[str, float] = {n: 0.0 for n in self.names}
        self._vel_sum_abs = 0.0
        self._vel_sum_sq = 0.0
        self._vel_sum_y = 0.0
        self._vel_sum_y2 = 0.0
        self._vel_sum_err2 = 0.0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred, target = _physical_pressure(pred, target, self.names, self.p_min, self.p_max)
        n = int(pred.shape[0])
        if n == 0:
            return
        self.n_points += n
        diff = pred - target
        for idx, name in enumerate(self.names):
            err = diff[:, idx]
            y = target[:, idx]
            self._sum_abs[name] += float(err.abs().sum().item())
            self._sum_sq[name] += float(err.square().sum().item())
            self._sum_y[name] += float(y.sum().item())
            self._sum_y2[name] += float(y.square().sum().item())
            self._sum_err2[name] += float(err.square().sum().item())

        if all(n in self.names for n in ("u", "v", "w")):
            u_idx, v_idx, w_idx = self.names.index("u"), self.names.index("v"), self.names.index("w")
            vel_pred = torch.sqrt(
                pred[:, u_idx].square() + pred[:, v_idx].square() + pred[:, w_idx].square()
            )
            vel_tgt = torch.sqrt(
                target[:, u_idx].square() + target[:, v_idx].square() + target[:, w_idx].square()
            )
            vel_err = vel_pred - vel_tgt
            self._vel_sum_abs += float(vel_err.abs().sum().item())
            self._vel_sum_sq += float(vel_err.square().sum().item())
            self._vel_sum_y += float(vel_tgt.sum().item())
            self._vel_sum_y2 += float(vel_tgt.square().sum().item())
            self._vel_sum_err2 += float(vel_err.square().sum().item())

    def _prefixed(self, metrics: Dict[str, float]) -> Dict[str, float]:
        if not self.prefix:
            return metrics
        return {f"{self.prefix}{k}": v for k, v in metrics.items()}

    def finalize(self) -> Dict[str, float]:
        if self.n_points == 0:
            return self._prefixed({"n_points": 0})

        metrics: Dict[str, float] = {"n_points": float(self.n_points)}
        nmae_values: List[float] = []
        total_abs = 0.0
        total_sq = 0.0
        n = float(self.n_points)
        for name in self.names:
            mae = self._sum_abs[name] / n
            mse = self._sum_sq[name] / n
            ss_res = self._sum_err2[name]
            ss_tot = max(self._sum_y2[name] - self._sum_y[name] * self._sum_y[name] / n, 1e-12)
            denom = max(float(self.nmae_ranges.get(name, 0.0)), 1e-12)
            nmae = mae / denom
            metrics[f"{name}_mae"] = mae
            metrics[f"{name}_rmse"] = float(mse**0.5)
            metrics[f"{name}_nmae"] = nmae
            metrics[f"{name}_r2"] = float(1.0 - ss_res / ss_tot)
            nmae_values.append(nmae)
            total_abs += self._sum_abs[name]
            total_sq += self._sum_sq[name]

        metrics["mae"] = total_abs / (n * len(self.names))
        metrics["loss_mse"] = total_sq / (n * len(self.names))
        metrics["rmse"] = float(metrics["loss_mse"] ** 0.5)
        if nmae_values:
            metrics["nmae"] = float(sum(nmae_values) / len(nmae_values))

        if all(n in self.names for n in ("u", "v", "w")):
            vel_mae = self._vel_sum_abs / n
            vel_mse = self._vel_sum_sq / n
            vel_ss_res = self._vel_sum_err2
            vel_ss_tot = max(self._vel_sum_y2 - self._vel_sum_y * self._vel_sum_y / n, 1e-12)
            vel_denom = max(float(self.nmae_ranges.get("vel_mag", 0.0)), 1e-12)
            metrics["vel_mag_mae"] = vel_mae
            metrics["vel_mag_rmse"] = float(vel_mse**0.5)
            metrics["vel_mag_nmae"] = vel_mae / vel_denom
            metrics["vel_mag_r2"] = float(1.0 - vel_ss_res / vel_ss_tot)

        return self._prefixed(metrics)


def paper_nmae_summary(per_sample_rows: Sequence[Mapping[str, object]]) -> Dict[str, float]:
    """对齐论文摘要：逐样本 NMAE 的 mean±std（velocity= u/v/w 均值，pressure= p）。"""
    if not per_sample_rows:
        return {}

    vel_triplets: List[float] = []
    p_vals: List[float] = []
    for row in per_sample_rows:
        u = float(row.get("u_nmae", float("nan")))
        v = float(row.get("v_nmae", float("nan")))
        w = float(row.get("w_nmae", float("nan")))
        p = float(row.get("p_nmae", float("nan")))
        if all(x == x for x in (u, v, w)):
            vel_triplets.append((u + v + w) / 3.0)
        if p == p:
            p_vals.append(p)

    out: Dict[str, float] = {}
    if vel_triplets:
        out["paper_velocity_nmae_mean"] = float(statistics.mean(vel_triplets))
        out["paper_velocity_nmae_std"] = (
            float(statistics.stdev(vel_triplets)) if len(vel_triplets) > 1 else 0.0
        )
    if p_vals:
        out["paper_pressure_nmae_mean"] = float(statistics.mean(p_vals))
        out["paper_pressure_nmae_std"] = (
            float(statistics.stdev(p_vals)) if len(p_vals) > 1 else 0.0
        )
    return out


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
