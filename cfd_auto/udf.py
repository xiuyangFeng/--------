"""Library UDF (udf-inlet.c) regenerated for a new mesh.

Only three things in the library UDF depend on the case: the four outlet thread ids (``Lookup_Thread(d, id)`` in
``execute_at_end``), the inlet-area divisor of ``my_inlet`` and the R1/R2/C constants of the four ``pressure_out*``
profiles. Everything else (waveform, Carreau constants, RCR recursion) is copied verbatim from the reference UDF.
The RCR protocol (operator spreadsheet, 2026-09-15 audit) is available for new cases; the regression trial keeps the
reference constants so that only the mesh changes.
"""
from __future__ import annotations

import re
from pathlib import Path

OUTLETS = ("outle", "outli", "outri", "outre")          # UDF naming; zones are out-le, out-li, out-ri, out-re
PROTOCOL_P = 93.33 * 133                                # mean pressure (Pa) used by the spreadsheet
A1_KG_S = {"AG": 0.02842, "AAA": 0.033, "ILO": 0.033}   # total inflow per cohort (AG value from the 2026-09-15 audit)


def read_udf(path: str | Path) -> str:
    """Library UDFs carry GBK comments and CRLF line ends: latin-1 + newline='' round-trips every byte."""
    with open(path, encoding="latin-1", newline="") as fh:
        return fh.read()


def write_udf(path: str | Path, text: str) -> None:
    with open(path, "w", encoding="latin-1", newline="") as fh:
        fh.write(text)


def protocol_rcr(areas_m2: dict[str, float], cohort: str) -> dict[str, dict[str, float]]:
    """Murray r^3 split within each side (le/li, re/ri), half of A1 per side; Rt = P/flow, C = 1.79/Rt,
    R1 = 13.3/(2r)^0.3/s (r in mm, s in m^2), R2 = Rt - R1."""
    a1 = A1_KG_S[cohort]
    out = {}
    for ext, inte in (("outle", "outli"), ("outre", "outri")):
        d_mm = {k: 2 * (areas_m2[k] / 3.141592653589793) ** 0.5 * 1e3 for k in (ext, inte)}
        tot = d_mm[ext] ** 3 + d_mm[inte] ** 3
        for k in (ext, inte):
            flow = a1 * d_mm[k] ** 3 / tot * 0.5
            rt = PROTOCOL_P / flow
            r1 = 13.3 / d_mm[k] ** 0.3 / areas_m2[k]
            out[k] = {"R1": r1, "R2": rt - r1, "C": 1.79 / rt}
    return out


def render(reference_text: str, thread_ids: dict[str, int], inlet_area_m2: float | None = None, rcr: dict[str, dict[str, float]] | None = None) -> str:
    """Reference UDF with new thread ids (and optionally a new inlet area / RCR constants)."""
    text = reference_text
    for key in OUTLETS:
        pat = re.compile(r"(Lookup_Thread\(d,\s*)(\d+)(\);\s*/\*\s*" + key + r"\b)")
        text, n = pat.subn(lambda m: f"{m.group(1)}{int(thread_ids[key])}{m.group(3)}", text)
        if n != 1:
            raise ValueError(f"UDF: expected one Lookup_Thread for {key}, found {n}")
    if inlet_area_m2 is not None:
        block = re.search(r"DEFINE_PROFILE\(my_inlet.*?end_f_loop", text, re.S)
        new_block, n = re.subn(r"(\)\s*/\s*)([-+0-9.eE]+)(\s*;)", lambda m: f"{m.group(1)}{inlet_area_m2:.10g}{m.group(3)}", block.group(0))
        if n != 1:
            raise ValueError("UDF: inlet area divisor not found exactly once")
        text = text[: block.start()] + new_block + text[block.end():]
    if rcr is not None:
        for key in OUTLETS:
            m = re.search(r"DEFINE_PROFILE\(pressure_" + key + r",.*?begin_f_loop", text, re.S)
            blk = m.group(0)
            for name in ("R1", "R2", "C"):
                blk, n = re.subn(r"(\b" + name + r"\s*=\s*)([-+0-9.eE]+)(;)", lambda mm: f"{mm.group(1)}{rcr[key][name]:.4E}{mm.group(3)}", blk, count=1)
                if n != 1:
                    raise ValueError(f"UDF: {name} not found in pressure_{key}")
            text = text[: m.start()] + blk + text[m.end():]
    return text
