"""UDF Fourier waveform and robust RCR/material parsing."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


PERIOD_S = 0.8
OUTLET_PROFILE_ORDER = (
    ("out-le", "outle"),
    ("out-li", "outli"),
    ("out-ri", "outri"),
    ("out-re", "outre"),
)
DEFAULT_FOURIER = {
    "a0": 30.38,
    "a": [3.109, -25.41, -0.8703, 3.069, 5.425, 0.404, -0.4276, -0.8367],
    "b": [30.03, 0.9296, -13.24, -4.541, -0.06318, 2.757, 1.827, 0.8608],
    "w": 6.491,
    "scale": 1.0e-6,
}

FLOAT = r"([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)"


def _mod_period_numpy(time_s: np.ndarray) -> np.ndarray:
    return time_s - np.floor(time_s / PERIOD_S) * PERIOD_S


def q_nom_numpy(time_s: np.ndarray | float, coefficients: dict[str, Any] | None = None) -> np.ndarray:
    coeff = DEFAULT_FOURIER if coefficients is None else coefficients
    time = np.asarray(time_s, dtype=np.float64)
    tt = _mod_period_numpy(time)
    value = np.full_like(tt, float(coeff["a0"]), dtype=np.float64)
    for harmonic, (a_value, b_value) in enumerate(zip(coeff["a"], coeff["b"]), start=1):
        phase = harmonic * tt * float(coeff["w"])
        value += float(a_value) * np.cos(phase) + float(b_value) * np.sin(phase)
    return value * float(coeff.get("scale", 1.0e-6))


def q_nom_torch(time_s: torch.Tensor, coefficients: dict[str, Any] | None = None) -> torch.Tensor:
    """Evaluate the exact piecewise UDF waveform inside the autograd graph."""
    coeff = DEFAULT_FOURIER if coefficients is None else coefficients
    tt = time_s - torch.floor(time_s / PERIOD_S).detach() * PERIOD_S
    value = torch.full_like(tt, float(coeff["a0"]))
    for harmonic, (a_value, b_value) in enumerate(zip(coeff["a"], coeff["b"]), start=1):
        phase = harmonic * tt * float(coeff["w"])
        value = value + float(a_value) * torch.cos(phase) + float(b_value) * torch.sin(phase)
    return value * float(coeff.get("scale", 1.0e-6))


def q_nom_peak(coefficients: dict[str, Any] | None = None) -> tuple[float, float]:
    grid = np.linspace(0.0, PERIOD_S, 200_001, endpoint=False, dtype=np.float64)
    values = q_nom_numpy(grid, coefficients)
    index = int(np.argmax(values))
    return float(grid[index]), float(values[index])


def parse_udf(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")

    def macro(name: str) -> float:
        match = re.search(rf"^\s*#define\s+{re.escape(name)}\s+{FLOAT}\s*$", text, re.MULTILINE)
        if not match:
            raise ValueError(f"missing UDF macro {name}: {path}")
        return float(match.group(1))

    coefficients = {
        "a0": macro("a0"),
        "a": [macro(f"a{index}") for index in range(1, 9)],
        "b": [macro(f"b{index}") for index in range(1, 9)],
        "w": macro("w"),
        "scale": 1.0e-6,
    }
    profile = text.split("DEFINE_PROFILE(my_inlet", 1)[-1].split("DEFINE_PROPERTY", 1)[0]
    denominator_matches = re.findall(rf"/\s*{FLOAT}\s*;", profile, re.MULTILINE)
    if len(denominator_matches) != 1:
        raise ValueError(f"expected one inlet area denominator: {path}")
    a_udf = float(denominator_matches[0])

    rcr = []
    for label, profile_name in OUTLET_PROFILE_ORDER:
        match = re.search(
            rf"DEFINE_PROFILE\(pressure_{profile_name}\b.*?"
            rf"R1\s*=\s*{FLOAT}\s*;.*?R2\s*=\s*{FLOAT}\s*;.*?C\s*=\s*{FLOAT}\s*;",
            text,
            re.DOTALL,
        )
        if not match:
            raise ValueError(f"missing RCR profile {profile_name}: {path}")
        values = [float(match.group(index)) for index in range(1, 4)]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError(f"invalid RCR values for {profile_name}: {values}")
        rcr.append({"outlet": label, "R1": values[0], "R2": values[1], "C": values[2]})

    rheology = {
        "mu_inf_pa_s": macro("A1"),
        "mu_zero_pa_s": macro("B"),
        "lambda_s": macro("D"),
        "a": macro("E"),
        "n": macro("n"),
    }
    if a_udf <= 0 or not math.isfinite(a_udf):
        raise ValueError(f"invalid A_udf={a_udf}: {path}")
    return {
        "period_s": PERIOD_S,
        "fourier": coefficients,
        "a_udf_m2": a_udf,
        "rcr": rcr,
        "rheology": rheology,
    }
