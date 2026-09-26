"""Internal ensemble quality diagnostics and the small public quality summary.

The ensemble (five- or three-model) spread is useful for triage, but exposing a per-point standard
deviation in the normal report makes the result harder to read and encourages
false precision.  This module therefore keeps the numeric diagnostics in an
audit record while exposing only a stable quality level and review reasons.
"""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np


def model_count_phrase(count: int | None = None) -> str:
    """「五模型」 for the five-seed releases (and when the count is unknown, the historical wording), else 「N 个模型」."""
    try:
        n = int(count) if count is not None else 5
    except (TypeError, ValueError):
        n = 5
    return "五模型" if n == 5 or n <= 0 else f"{n} 个模型"


def ensemble_quality(wss_pa: Any, seed_sd_pa: Any, *, seed_count: int | None = None) -> dict[str, Any]:
    """Summarise seed-ensemble consistency without serialising pointwise SD.

    Thresholds are deliberately conservative triage thresholds, not calibrated
    probabilities.  ``quality`` is safe for the main UI; ``audit`` is intended
    for ``quality_audit.json`` and contains aggregate diagnostics only.  The
    reason text names the actual model count (five-seed wording unchanged).
    """
    values = np.asarray(wss_pa, dtype=float).reshape(-1)
    spread = np.asarray(seed_sd_pa, dtype=float).reshape(-1)
    if values.size == 0 or spread.size != values.size:
        raise ValueError("wss_pa and seed_sd_pa must be non-empty arrays of equal length")
    finite = np.isfinite(values) & np.isfinite(spread)
    values, spread = values[finite], np.maximum(spread[finite], 0.0)
    if values.size == 0:
        raise ValueError("ensemble arrays contain no finite values")
    # Relative spread is stabilised near zero, which avoids giant ratios for
    # insignificant near-zero wall values.
    denominator = np.maximum(np.abs(values), 0.25)
    relative = spread / denominator
    abs_p50, abs_p90, abs_max = np.percentile(spread, [50, 90, 99.5]).tolist()
    rel_p50, rel_p90, rel_max = np.percentile(relative, [50, 90, 99.5]).tolist()
    if rel_p90 > 0.35 or abs_p90 > 3.0:
        level, label = "poor", "不稳定，建议复核"
        reasons = [f"{model_count_phrase(seed_count)}离散度较高，结果需要人工复核"]
    elif rel_p90 > 0.20 or abs_p90 > 1.5:
        level, label = "review", "存在不确定性，建议复核"
        reasons = [f"{model_count_phrase(seed_count)}离散度超过常规范围，建议复核热点和分支"]
    else:
        level, label = "good", "模型集成稳定"
        reasons = []
    count = int(seed_count) if seed_count is not None else None
    audit = {
        "schema_version": "wss-deploy.quality-audit/v1",
        "seed_count": count,
        "n_points": int(values.size),
        "spread_pa": {"p50": float(abs_p50), "p90": float(abs_p90), "p99_5": float(abs_max)},
        "relative_spread": {"p50": float(rel_p50), "p90": float(rel_p90), "p99_5": float(rel_max)},
        "thresholds": {"review_relative_p90": 0.20, "poor_relative_p90": 0.35,
                       "review_abs_p90_pa": 1.5, "poor_abs_p90_pa": 3.0},
    }
    return {"quality": {"level": level, "label": label, "reasons": reasons}, "audit": audit}


__all__ = ["ensemble_quality", "model_count_phrase"]
