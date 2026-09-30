"""CFD protocol definitions (``cfd_auto/protocols/<name>.json``): everything a vessel bed needs besides its geometry —
solver schedule, exported frames, divergence ladder, opening keys, extension rules, mesh parameters and quality gates,
boundary-condition rule, set-up risk thresholds, settings templates for new cases, after-run gates.

The library protocol is ``aortoiliac_rcr4_v1``; its numbers equal the module constants of udf / sanity / rebuild (checked
by the tests). Another bed (e.g. cerebral) = a copy of the file with its own numbers plus that bed's settings template and
UDF template.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

PROTOCOL_DIR = Path(__file__).resolve().parent / "protocols"
DEFAULT = "aortoiliac_rcr4_v1"


def load(name_or_path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Protocol dict by name (``protocols/<name>.json``) or path, with optional nested overrides (batch-level tweaks)."""
    p = Path(name_or_path) if name_or_path and str(name_or_path).endswith(".json") else PROTOCOL_DIR / f"{name_or_path or DEFAULT}.json"
    proto = json.loads(p.read_text())
    return merge(proto, overrides or {})


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def label_steps(proto: dict) -> list[int]:
    """Library step labels of the exported frames (1120..1280 step 2 for the library protocol)."""
    e = proto["exports"]
    return list(range(e["first_step"], e["last_step"] + 1, e["every"]))


def inlet_length_mm(proto: dict, cohort: str) -> float:
    return float(proto["extension"]["inlet_length_mm"][cohort])


def a1_kg_s(proto: dict, cohort: str) -> float:
    return float(proto["bc"]["A1_kg_s"][cohort])
