"""Case-level boundary-condition (BC) vector contract for V4 (frozen 2026-09-05).

Raw vector (18 values, order fixed by ``OUTLET_ORDER``)::

    [A_in_mesh_m2, inlet_area_ratio, (A_out_m2, R1, R2, C) x 4 outlets]
    inlet_area_ratio = A_mesh / A_udf = Q_actual_peak / Q_nom_peak

Transformed vector: ``log10`` on every area / R1 / R2 / C field; the inlet
area ratio uses the **fixed physical scale** ``(ratio - 1) / 0.1`` and is never
z-scored.  The ``log10`` fields are z-scored with the train-only *max-aware*
scale ``std = max(population_std, max_abs_deviation / 6)`` (the same policy the
geometry channels use), so every train value lies within ±6 by construction
and no per-case whitelist is needed.

A field that is near-constant on train (population std below
``MIN_TRAIN_STD_FOR_ZSCORE`` = 0.02 dex) must not be z-scored: the builder
raises so the field is either given a fixed scale or dropped.

Why: all 173 UDFs share one inlet waveform, so ``Q_actual_peak`` differs from
the nominal peak only through ``A_mesh / A_udf`` (4 cases, -11.2% .. +0.9%).
Its train coefficient of variation of 1.3% turned those cases into -8σ
"outliers" under the old z-score (2026-09-03 audit §5 / §15.2).  The channel is
kept (1 dimension) because the DATA arm has no BC loss and the labels of the
4 mismatched cases are only consistent with their inputs if the model can see
the actual/nominal flow ratio.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import OUTLET_ORDER

BC_CONTRACT = "v4_bc_vector_2026-09-05"
INLET_AREA_RATIO_INDEX = 1
INLET_AREA_RATIO_SCALE = 0.1
MIN_TRAIN_STD_FOR_ZSCORE = 0.02
MAX_ABS_Z = 6.0
_OUTLET_FIELDS = ("A_out", "R1", "R2", "C")
BC_RAW_NAMES = (
    "A_in_mesh_m2",
    "inlet_area_ratio",
    *[f"{field}_{outlet}" for outlet in OUTLET_ORDER for field in ("A_out_m2", "R1", "R2", "C")],
)
BC_TRANSFORMED_NAMES = (
    "log10_A_in_mesh",
    "inlet_area_ratio_fixed",
    *[f"log10_{field}_{outlet}" for outlet in OUTLET_ORDER for field in _OUTLET_FIELDS],
)
BC_LOG10_INDICES = tuple([0] + [2 + 4 * outlet + field for outlet in range(4) for field in range(4)])
BC_DIM = len(BC_RAW_NAMES)
BC_TRANSFORM_DESCRIPTION = (
    "log10 for areas/R1/R2/C then train138 max-aware z-score "
    "(std = max(population_std, max_abs_deviation/6)); "
    "inlet_area_ratio = (A_mesh/A_udf - 1)/0.1 fixed physical scale, never z-scored"
)


def bc_vector_raw(
    a_in_mesh_m2: float,
    a_in_udf_m2: float,
    outlet_area_m2: Any,
    rcr: Any,
) -> list[float]:
    """Raw 18-vector from solver-time-known quantities (mesh areas, UDF area, RCR parameters)."""

    if len(rcr) != 4 or len(outlet_area_m2) != 4:
        raise ValueError("BC vector requires exactly four outlets")
    vector = [float(a_in_mesh_m2), float(a_in_mesh_m2) / float(a_in_udf_m2)]
    for index, item in enumerate(rcr):
        vector.extend([float(outlet_area_m2[index]), float(item["R1"]), float(item["R2"]), float(item["C"])])
    if len(vector) != BC_DIM:
        raise ValueError(f"BC vector must have {BC_DIM} entries")
    return vector


def bc_vector_from_conditions(conditions: dict[str, Any]) -> list[float]:
    return bc_vector_raw(
        conditions["a_in_mesh_m2"],
        conditions["a_in_udf_m2"],
        conditions["outlet_area_m2"],
        conditions["rcr_mass_flow_basis"],
    )


def transform_bc_raw(raw: np.ndarray) -> np.ndarray:
    """Physical transform (no train statistics): log10 fields and the fixed-scale ratio."""

    value = np.asarray(raw, dtype=np.float64).copy()
    single = value.ndim == 1
    if single:
        value = value[None, :]
    if value.shape[1] != BC_DIM:
        raise ValueError(f"BC raw vector must have {BC_DIM} entries, got {value.shape[1]}")
    log_indices = list(BC_LOG10_INDICES)
    if np.any(value[:, log_indices] <= 0.0):
        raise ValueError("areas and RCR parameters must be positive for log10")
    ratio = value[:, INLET_AREA_RATIO_INDEX]
    if np.any(ratio < 0.5) or np.any(ratio > 2.0):
        raise ValueError(
            "BC raw index 1 is outside [0.5, 2]: not an inlet area ratio "
            "(vector built under the pre-2026-09-05 Q_actual_peak contract?)"
        )
    value[:, log_indices] = np.log10(value[:, log_indices])
    value[:, INLET_AREA_RATIO_INDEX] = (ratio - 1.0) / INLET_AREA_RATIO_SCALE
    return value[0] if single else value


def bc_scale_policy(train_transformed: np.ndarray) -> dict[str, Any]:
    """Train-only normalisation statistics under the frozen policy (raises on a near-constant z-scored field)."""

    value = np.asarray(train_transformed, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != BC_DIM:
        raise ValueError("train BC matrix must be (cases, 18)")
    mean = value.mean(axis=0)
    population_std = value.std(axis=0)
    max_abs_deviation = np.max(np.abs(value - mean), axis=0)
    policy = ["train_zscore_max_aware"] * BC_DIM
    policy[INLET_AREA_RATIO_INDEX] = "fixed_physical"
    std = np.maximum(population_std, max_abs_deviation / MAX_ABS_Z)
    mean[INLET_AREA_RATIO_INDEX] = 0.0
    std[INLET_AREA_RATIO_INDEX] = 1.0
    near_constant = [
        BC_TRANSFORMED_NAMES[index]
        for index in range(BC_DIM)
        if policy[index] != "fixed_physical" and population_std[index] < MIN_TRAIN_STD_FOR_ZSCORE
    ]
    if near_constant:
        raise ValueError(
            f"near-constant BC field(s) {near_constant}: train std < {MIN_TRAIN_STD_FOR_ZSCORE}; "
            "give the field a fixed physical scale or drop it instead of z-scoring it"
        )
    z = (value - mean) / std
    return {
        "contract": BC_CONTRACT,
        "names": list(BC_TRANSFORMED_NAMES),
        "raw_names": list(BC_RAW_NAMES),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "raw_std": population_std.tolist(),
        "max_abs_deviation": max_abs_deviation.tolist(),
        "train_abs_z_max": np.max(np.abs(z), axis=0).tolist(),
        "scale_policy": policy,
        "fixed_scale": {
            "index": INLET_AREA_RATIO_INDEX,
            "name": BC_TRANSFORMED_NAMES[INLET_AREA_RATIO_INDEX],
            "transform": f"(A_mesh/A_udf - 1) / {INLET_AREA_RATIO_SCALE}",
        },
        "min_train_std_for_zscore": MIN_TRAIN_STD_FOR_ZSCORE,
        "max_abs_z": MAX_ABS_Z,
        "transform": BC_TRANSFORM_DESCRIPTION,
    }


def normalize_bc(raw: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """Model input: transformed raw vector normalised with the frozen train statistics."""

    if stats.get("contract") != BC_CONTRACT:
        raise ValueError(
            f"BC stats contract {stats.get('contract')!r} != {BC_CONTRACT!r}: "
            "the bundle stats predate the 2026-09-05 BC contract and must be rebuilt"
        )
    value = transform_bc_raw(raw)
    return ((value - np.asarray(stats["mean"], dtype=np.float64)) / np.asarray(stats["std"], dtype=np.float64)).astype(np.float32)
