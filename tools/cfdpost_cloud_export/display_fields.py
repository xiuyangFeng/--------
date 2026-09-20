"""Display-only self-maximum fields with a fixed, auditable source domain.

Fit once on the original colocated case, then apply the same contract to point
clouds, native surfaces and interpolated physical fields. Never fit on a crop or
a smoothed output when it is meant to share the original case's scale.
"""
from __future__ import annotations

import json
import numpy as np


def _denominator(values):
    if not np.isfinite(values).all():
        return {"max": None, "valid": False, "reason": "nonfinite_source", "zero_tolerance": None}
    maximum = float(values.max())
    tolerance = 1e-12 * max(1.0, float(np.abs(values).max()))
    valid = abs(maximum) > tolerance
    return {"max": maximum, "valid": valid, "reason": "valid" if valid else "near_zero_max",
            "zero_tolerance": tolerance}


def fit_selfmax(cfd, pred, prefix, denominator_scope, source_unit):
    """Record two separate maxima; signed scalars use max, not max(abs())."""
    cfd = np.asarray(cfd, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if cfd.ndim != 1 or cfd.shape != pred.shape or not len(cfd):
        raise ValueError("selfmax requires nonempty paired scalar fields; use speed for velocity")
    c, p = _denominator(cfd), _denominator(pred)
    return {
        "version": 1, "kind": "selfmax", "prefix": prefix, "unit": "dimensionless",
        "source_unit": source_unit, "denominator_scope": denominator_scope,
        "source_point_count": len(cfd), "cfd_max": c["max"], "pred_max": p["max"],
        "cfd_denominator": c, "pred_denominator": p,
        "formula": {f"{prefix}_cfd_selfmax": "CFD / max(original CFD)",
                    f"{prefix}_pred_selfmax": "Pred / max(original Pred)",
                    f"{prefix}_selfmax_error_pred_minus_cfd": "Pred/max(original Pred) - CFD/max(original CFD)",
                    f"{prefix}_selfmax_abs_error": "abs(Pred/max(original Pred) - CFD/max(original CFD))"},
        "denominator_basis": "original full colocated case at selected frame, before interpolation",
        "surface_policy": "apply same original denominators after mapping physical CFD/Pred; no refitting",
        "amplitude_difference_removed": True,
        "formal_metrics": "retain original physical colocated metrics; selfmax is a display field",
        "invalid_policy": "near-zero or nonfinite denominator: NaN field and zero validity mask; never substitute 1",
        "signed_scalar_policy": "preserve reference zero and signs in source; divide by signed max, not maxabs or minmax; no clipping",
    }


def selfmax_fields(cfd, pred, contract):
    """Apply fixed denominators to any matching scalar pair, including a surface."""
    cfd = np.asarray(cfd, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if cfd.ndim != 1 or cfd.shape != pred.shape:
        raise ValueError("Expected matching one-dimensional display fields")
    prefix = contract["prefix"]
    arrays = {}
    scaled = []
    masks = []
    for name, values in (("cfd", cfd), ("pred", pred)):
        den = contract[f"{name}_denominator"]
        valid = np.isfinite(values) & bool(den["valid"])
        result = np.full(values.shape, np.nan, dtype=np.float64)
        if den["valid"]:
            result[valid] = values[valid] / den["max"]
        arrays[f"{prefix}_{name}_selfmax"] = result
        arrays[f"{prefix}_{name}_selfmax_valid"] = valid.astype(np.uint8)
        scaled.append(result)
        masks.append(valid)
    valid_error = masks[0] & masks[1]
    error = np.full(cfd.shape, np.nan, dtype=np.float64)
    error[valid_error] = scaled[1][valid_error] - scaled[0][valid_error]
    arrays[f"{prefix}_selfmax_error_pred_minus_cfd"] = error
    arrays[f"{prefix}_selfmax_abs_error"] = np.abs(error)
    arrays[f"{prefix}_selfmax_error_valid"] = valid_error.astype(np.uint8)
    return arrays


def normalization_field_data(contract):
    """Scalar and string metadata to embed in a standalone VTP FieldData block."""
    prefix = contract["prefix"]
    fields = {"display_normalization": json.dumps(contract, ensure_ascii=False, allow_nan=False)}
    for name in ("cfd", "pred"):
        value = contract[f"{name}_max"]
        fields[f"{prefix}_{name}_max"] = float(value) if value is not None else float("nan")
        fields[f"{prefix}_{name}_max_valid"] = int(contract[f"{name}_denominator"]["valid"])
    return fields
