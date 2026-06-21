"""Statistical testing utilities for multi-seed experiment comparison.

Provides paired hypothesis tests, effect sizes, and confidence intervals
needed for rigorous ablation reporting in the paper.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sp_stats


def paired_ttest(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Two-sided paired t-test.  Returns t-statistic, p-value, and Cohen's d."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert len(a) == len(b) and len(a) >= 2, "Need at least 2 paired observations"
    t_stat, p_value = sp_stats.ttest_rel(a, b)
    diff = a - b
    cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 1e-12 else 0.0
    return {"t_stat": float(t_stat), "p_value": float(p_value), "cohens_d": float(cohens_d)}


def wilcoxon_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Wilcoxon signed-rank test (non-parametric alternative)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    diff = a - b
    if np.allclose(diff, 0):
        return {"statistic": 0.0, "p_value": 1.0}
    stat, p_value = sp_stats.wilcoxon(diff)
    return {"statistic": float(stat), "p_value": float(p_value)}


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for the mean.

    Returns (mean, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    means = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_bootstrap)])
    alpha = (1 - confidence) / 2
    lo = float(np.percentile(means, 100 * alpha))
    hi = float(np.percentile(means, 100 * (1 - alpha)))
    return float(arr.mean()), lo, hi


def summarize_seeds(
    metric_by_seed: Dict[int, float],
) -> Dict[str, float]:
    """Summarize a metric across seeds: mean, std, min, max, 95% CI."""
    vals = list(metric_by_seed.values())
    arr = np.array(vals, dtype=np.float64)
    result = {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_seeds": len(arr),
    }
    if len(arr) >= 2:
        mean, ci_lo, ci_hi = bootstrap_ci(vals)
        result["ci95_lo"] = ci_lo
        result["ci95_hi"] = ci_hi
    return result


def compare_experiments(
    baseline_seeds: Dict[int, float],
    proposed_seeds: Dict[int, float],
    lower_is_better: bool = True,
) -> Dict[str, object]:
    """Compare two experiments across matching seeds.

    Returns summary statistics for both, plus paired test results.
    """
    common_seeds = sorted(set(baseline_seeds) & set(proposed_seeds))
    if len(common_seeds) < 2:
        return {"error": "Need at least 2 common seeds for statistical comparison"}

    a = [baseline_seeds[s] for s in common_seeds]
    b = [proposed_seeds[s] for s in common_seeds]

    base_summary = summarize_seeds(baseline_seeds)
    prop_summary = summarize_seeds(proposed_seeds)

    delta_mean = prop_summary["mean"] - base_summary["mean"]
    direction = "improved" if (lower_is_better and delta_mean < 0) or (not lower_is_better and delta_mean > 0) else "degraded"

    result: Dict[str, object] = {
        "baseline": base_summary,
        "proposed": prop_summary,
        "delta_mean": float(delta_mean),
        "direction": direction,
        "n_common_seeds": len(common_seeds),
    }

    if len(common_seeds) >= 3:
        result["paired_ttest"] = paired_ttest(a, b)
        result["wilcoxon"] = wilcoxon_test(a, b)

    return result
