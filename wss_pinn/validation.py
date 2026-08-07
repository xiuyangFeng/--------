"""Validation aggregation contracts shared by training and focused tests."""

from __future__ import annotations

from typing import Any

import numpy as np


def aggregate_case_validation(
    cases: list[dict[str, Any]], *, legacy_batch_cases: int
) -> dict[str, Any]:
    """Return case-, point- and historical batch-weighted field scores.

    Each input row contains a per-case four-channel standardized MSE in
    ``field_score`` and its query ``points``.  The historical metric recreates
    the old DataLoader batching exactly: point-weight inside a batch, followed
    by an equal mean over batches.
    """
    if not cases:
        raise ValueError("validation cases cannot be empty")
    if int(legacy_batch_cases) <= 0:
        raise ValueError("legacy_batch_cases must be positive")
    scores = np.asarray([row["field_score"] for row in cases], dtype=np.float64)
    points = np.asarray([row["points"] for row in cases], dtype=np.float64)
    if (
        not np.isfinite(scores).all()
        or not np.isfinite(points).all()
        or bool(np.any(points <= 0))
    ):
        raise ValueError("validation score inputs must be finite with positive counts")

    case_balanced = float(np.mean(scores))
    point_balanced = float(np.sum(scores * points) / np.sum(points))
    batches = []
    effective_weights = np.zeros(len(cases), dtype=np.float64)
    batch_count = int(np.ceil(len(cases) / int(legacy_batch_cases)))
    for start in range(0, len(cases), int(legacy_batch_cases)):
        stop = min(start + int(legacy_batch_cases), len(cases))
        batch_points = points[start:stop]
        within = batch_points / np.sum(batch_points)
        score = float(np.sum(scores[start:stop] * within))
        batches.append(
            {
                "case_ids": [row["case_id"] for row in cases[start:stop]],
                "points": int(np.sum(batch_points)),
                "field_score": score,
            }
        )
        effective_weights[start:stop] = within / batch_count
    legacy = float(np.mean([row["field_score"] for row in batches]))
    minimum = float(np.min(effective_weights))
    maximum = float(np.max(effective_weights))
    overweighted = [
        cases[index]["case_id"]
        for index, weight in enumerate(effective_weights)
        if weight > minimum * (1.0 + 1e-12)
    ]
    return {
        "validation_field_score_cb": case_balanced,
        "validation_field_score_point_weighted": point_balanced,
        "validation_field_score_legacy_batch_weighted": legacy,
        "legacy_batches": batches,
        "legacy_effective_case_weights": {
            row["case_id"]: float(effective_weights[index])
            for index, row in enumerate(cases)
        },
        "legacy_weight_ratio_max_to_min": maximum / max(minimum, 1e-30),
        "legacy_overweighted_case_ids": overweighted,
    }

