"""Add phase-resolved physical metrics without changing the round-one contract.

Fields are N x 80 x C, with volume/area weights within each data unit. The
temporal metrics measure physical-value increments, not derivatives per second.
Endpoint difference is frame79 minus frame0, NOT a 0/80 periodic closure test.
Only nested dictionaries/numbers/None are returned so the existing equal-unit
``mean_records`` reducer continues to work without special cases.
"""
from __future__ import annotations

import math
import numpy as np

from .joint_cycle import case_metrics as original_case_metrics, weighted_fields


def case_metrics(task, y, p, weights, angle_floor):
    """Preserve every original metric and add a fixed peak frame and time errors."""
    y, p, weights = (np.asarray(x, dtype=np.float64) for x in (y, p, weights))
    out = original_case_metrics(task, y, p, weights, angle_floor)
    out["per_frame"] = {
        f"{frame:02d}": weighted_fields(y[:, frame:frame + 1], p[:, frame:frame + 1], weights)
        for frame in range(80)
    }
    out["peak_frame21"] = dict(out["per_frame"]["21"])
    out["temporal_difference"] = weighted_fields(np.diff(y, axis=1), np.diff(p, axis=1), weights)
    out["endpoint_difference"] = weighted_fields(y[:, -1:] - y[:, :1], p[:, -1:] - p[:, :1], weights)
    if task == "velocity":
        # Reuse vectorized norms and directions across all frame diagnostics.
        speed_y, speed_p = np.linalg.norm(y, axis=-1), np.linalg.norm(p, axis=-1)
        weights_n = weights / weights.sum()
        mask = speed_y >= angle_floor
        cosine = np.clip((y * p).sum(-1) / np.maximum(speed_y * speed_p, 1e-20), -1., 1.)
        coverage = np.einsum("n,nf->f", weights_n, mask)
        cosine_sum = np.einsum("n,nf->f", weights_n, cosine * mask)
        out["speed"]["per_frame"] = {}
        for frame in range(80):
            key = f"{frame:02d}"
            row = out["per_frame"][key]
            row.update(component_rmse_m_s=row["rmse"], vector_rmse_m_s=math.sqrt(3) * row["rmse"],
                       direction_cosine=float(cosine_sum[frame] / coverage[frame]) if coverage[frame] else None,
                       direction_truth_speed_floor_m_s=float(angle_floor),
                       direction_weight_coverage=float(coverage[frame]))
            out["speed"]["per_frame"][key] = weighted_fields(
                speed_y[:, frame:frame + 1, None], speed_p[:, frame:frame + 1, None], weights)
        out["peak_frame21"] = dict(out["per_frame"]["21"])
        out["speed"]["peak_frame21"] = dict(out["speed"]["per_frame"]["21"])
        for name in ("temporal_difference", "endpoint_difference"):
            out[name]["component_rmse_m_s"] = out[name]["rmse"]
            out[name]["vector_rmse_m_s"] = math.sqrt(3) * out[name]["rmse"]
    if task == "wss":
        from scipy.stats import spearmanr
        a, b = y[:, 21, 0], p[:, 21, 0]
        out["peak_frame21"]["spatial_spearman_unweighted"] = (
            float(spearmanr(a, b).statistic) if np.ptp(a) > 0 and np.ptp(b) > 0 else None)
    return out
