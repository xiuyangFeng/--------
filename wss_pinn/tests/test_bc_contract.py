"""Frozen 2026-09-05 BC vector contract: fixed-scale inlet ratio, max-aware z-scale, near-constant guard."""

from __future__ import annotations

import numpy as np
import pytest

from wss_pinn.v4.bc_contract import (
    BC_CONTRACT,
    BC_DIM,
    INLET_AREA_RATIO_INDEX,
    INLET_AREA_RATIO_SCALE,
    bc_scale_policy,
    bc_vector_raw,
    normalize_bc,
    transform_bc_raw,
)

OUT_RE_R2 = 2 + 4 * 3 + 1


def _synthetic_train(count: int = 138, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        a_mesh = 10 ** rng.normal(-3.6, 0.1)
        rcr = [
            {"R1": 10 ** rng.normal(7.5, 0.3), "R2": 10 ** rng.normal(8.5, 0.3), "C": 10 ** rng.normal(-9.5, 0.3)}
            for _ in range(4)
        ]
        rows.append(bc_vector_raw(a_mesh, a_mesh, 10 ** rng.normal(-4.3, 0.15, size=4), rcr))
    raw = np.asarray(rows, dtype=np.float64)
    assert raw.shape == (count, BC_DIM)
    # the four real mismatched cases: -11.2%, -8.9%, -6.6%, +0.9%
    raw[:4, INLET_AREA_RATIO_INDEX] = [0.888, 0.911, 0.934, 1.009]
    return raw


def test_inlet_ratio_uses_fixed_physical_scale_and_train_is_within_six_sigma():
    raw = _synthetic_train()
    transformed = transform_bc_raw(raw)
    assert transformed[0, INLET_AREA_RATIO_INDEX] == pytest.approx((0.888 - 1.0) / INLET_AREA_RATIO_SCALE)
    stats = bc_scale_policy(transformed)
    assert stats["contract"] == BC_CONTRACT
    assert stats["scale_policy"][INLET_AREA_RATIO_INDEX] == "fixed_physical"
    assert stats["mean"][INLET_AREA_RATIO_INDEX] == 0.0 and stats["std"][INLET_AREA_RATIO_INDEX] == 1.0
    assert max(stats["train_abs_z_max"]) <= 6.0 + 1e-9
    z = normalize_bc(raw[0], stats)
    assert z.shape == (BC_DIM,) and z.dtype == np.float32
    assert z[INLET_AREA_RATIO_INDEX] == pytest.approx((0.888 - 1.0) / INLET_AREA_RATIO_SCALE, rel=1e-6)
    # a case with exactly matched areas sits at 0 regardless of the cohort
    assert normalize_bc(raw[10], stats)[INLET_AREA_RATIO_INDEX] == pytest.approx(0.0, abs=1e-6)


def test_near_constant_log_field_is_refused_instead_of_z_scored():
    raw = _synthetic_train()
    raw[:, 0] = 2.5e-4  # constant inlet mesh area
    with pytest.raises(ValueError, match="near-constant"):
        bc_scale_policy(transform_bc_raw(raw))


def test_old_q_actual_peak_vectors_and_old_stats_are_refused():
    raw = _synthetic_train()
    old = raw.copy()
    old[:, INLET_AREA_RATIO_INDEX] = 1.07274e-4  # pre-2026-09-05 contract: Q_actual_peak
    with pytest.raises(ValueError, match="inlet area ratio"):
        transform_bc_raw(old)
    stats = bc_scale_policy(transform_bc_raw(raw))
    stats["contract"] = "pre-2026-09-05"
    with pytest.raises(ValueError, match="contract"):
        normalize_bc(raw[0], stats)


def test_max_aware_scale_bounds_a_single_long_tail_without_whitelist():
    raw = _synthetic_train()
    raw[5, OUT_RE_R2] *= 1.0e3  # a ZOU_LI_SHUN-like out-re R2 three decades above the cohort
    stats = bc_scale_policy(transform_bc_raw(raw))
    z = normalize_bc(raw[5], stats)
    assert 4.0 < abs(float(z[OUT_RE_R2])) <= 6.0 + 1e-6
    assert stats["train_abs_z_max"][OUT_RE_R2] == pytest.approx(6.0, rel=1e-6)
