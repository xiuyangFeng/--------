"""Solver conditions and protocol metadata (audit-only / offline-reference; never model input)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from wss_pinn.utils import sha256_file
from wss_pinn.v4.waveform import parse_udf, q_nom_numpy

from . import contract as C

SIDES = {"left": ("out-le", "out-li"), "right": ("out-re", "out-ri")}


PROTOCOL_R_TOTAL = {"AG": 4.367e5, "AAA_ILO": 3.761e5}  # Pa*s/kg, cohort constants verified 2026-09-06 (172 cases)


def protocol_table(rcr: list[dict[str, Any]], cohort: str | None = None) -> dict[str, Any]:
    by = {row["outlet"]: row for row in rcr}
    g = {o: 1.0 / (by[o]["R1"] + by[o]["R2"]) for o in C.OUTLET_ORDER}
    G = sum(g.values())
    r_total = 1.0 / G
    left = (g["out-le"] + g["out-li"]) / G
    matches = {name: abs(r_total - ref) / ref < 0.005 for name, ref in PROTOCOL_R_TOTAL.items()}
    protocol_id = next((name for name, ok in matches.items() if ok), "exception")
    expected = "AG" if cohort == "AG" else ("AAA_ILO" if cohort in ("AAA", "ILO") else None)
    reasons = []
    if protocol_id == "exception":
        reasons.append(f"r_total {r_total:.4g} matches no cohort constant")
    elif expected and protocol_id != expected:
        reasons.append(f"r_total follows {protocol_id} but cohort {cohort} expects {expected}")
    if abs(left - 0.5) > 0.02:
        reasons.append(f"left conductance share {left:.3f} deviates from 0.5")
    out: dict[str, Any] = {
        "resistance_units": "Pa*s/kg (UDF Windkessel integrates F_FLUX mass flow)",
        "capacitance_units": "kg/Pa",
        "r_total_parallel_dc": r_total,
        "protocol_id": protocol_id,
        "protocol_expected_for_cohort": expected,
        "protocol_exception": bool(reasons),
        "protocol_exception_reasons": reasons,
        "left_conductance_share": left,
        "per_outlet": {},
    }
    for o in C.OUTLET_ORDER:
        r = by[o]
        out["per_outlet"][o] = {
            "R1": r["R1"], "R2": r["R2"], "C": r["C"],
            "R_sum": r["R1"] + r["R2"], "R1_over_R": r["R1"] / (r["R1"] + r["R2"]),
            "R2C_s": r["R2"] * r["C"], "RC_s": (r["R1"] + r["R2"]) * r["C"],
            "conductance_share": g[o] / G,
        }
    for side, (ext, internal) in SIDES.items():
        out[f"{side}_external_conductance_share"] = g[ext] / (g[ext] + g[internal])
    return out


def waveform_samples() -> dict[str, np.ndarray]:
    """Public inlet flow waveform at the 81 export steps (protocol constant, offline reference only)."""
    steps = np.asarray(C.EXPECTED_STEPS, dtype=np.float64)
    time_s = steps * C.STEP_DT_S
    phase_s = time_s - (C.CYCLE_START_STEP * C.STEP_DT_S)
    q = q_nom_numpy(time_s)
    fine = np.linspace(0.0, C.PERIOD_S, 8001)
    qf = q_nom_numpy(fine + C.CYCLE_START_STEP * C.STEP_DT_S)
    dq = np.gradient(qf, fine)
    dq_at = np.interp(phase_s, fine, dq)
    return {"step": steps.astype(np.int32), "time_s": time_s, "phase_s": phase_s, "q_nom_m3s": q, "dq_nom_dt_m3s2": dq_at,
            "q_nom_peak_m3s": np.array(float(qf.max())), "q_nom_mean_m3s": np.array(float(np.trapz(qf, fine) / C.PERIOD_S))}


def read_conditions(udf_path: Path, inlet_bc_face_area_m2: float, interface_areas_m2: dict[str, float], cohort: str | None = None) -> dict[str, Any]:
    udf = parse_udf(udf_path)
    a_udf = float(udf["a_udf_m2"])
    cond = {
        "udf": {"path": str(udf_path), "sha256": sha256_file(udf_path)},
        "period_s": udf["period_s"],
        "fourier": udf["fourier"],
        "rheology": udf["rheology"],
        "rho_kg_m3": C.RHO_KG_M3,
        "A_inlet_udf_m2": a_udf,
        "A_inlet_bc_face_m2": inlet_bc_face_area_m2,
        "inlet_area_ratio": inlet_bc_face_area_m2 / a_udf,
        "A_interface_anatomy_m2": interface_areas_m2,
        "rcr": udf["rcr"],
        "protocol": protocol_table(udf["rcr"], cohort),
    }
    return cond
