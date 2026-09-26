"""Safe, scalar comparisons between deployment summaries.

This module intentionally compares the *reported statistics* in two runs.  It
does not load ``field.npz`` and it never attempts a point-to-point alignment:
two summaries may therefore be compared across releases, and across cases,
without pretending that their sampled wall points correspond.

Comparison is fail-closed.  A summary is comparable only when it declares the
WSS field, frame, statistics protocol, thresholds, and the two preprocessing
parameters that affect the reported values.  Legacy summaries are accepted
when those declarations live in their historical ``metadata``/``model_frame``
objects; missing declarations are reported rather than guessed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "wss-deploy.comparison/v1"

# These values are produced from geometry or an individual run and are not
# part of the statistical *contract*.  Comparing them would make otherwise
# compatible summaries from different cases fail the protocol check.
_VOLATILE_PROTOCOL_KEYS = {
    "top5_connectivity_radius_mm",
    "connectivity_radius_mm",
}


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _first_map(summary: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    """Read a key from a current summary, then its legacy metadata envelope."""
    value = summary.get(key)
    if isinstance(value, Mapping):
        return value
    metadata = _mapping(summary.get("metadata"))
    if metadata is not None and isinstance(metadata.get(key), Mapping):
        return metadata[key]
    return None


def _first_value(summary: Mapping[str, Any], key: str) -> Any:
    if key in summary and summary[key] is not None:
        return summary[key]
    metadata = _mapping(summary.get("metadata"))
    if metadata is not None and metadata.get(key) is not None:
        return metadata[key]
    return None


def _canonical(value: Any) -> Any:
    """Make a JSON-safe, deterministic representation for protocol checks."""
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # numpy scalar values are common in hand-built summaries.  Convert only
    # scalar values; arbitrary objects are represented as strings below.
    if hasattr(value, "item"):
        try:
            return _canonical(value.item())
        except Exception:
            pass
    return str(value)


def _number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    if hasattr(value, "item"):
        try:
            return _number(value.item())
        except Exception:
            return None
    return None


def _equal_number(left: Any, right: Any) -> bool:
    a, b = _number(left), _number(right)
    if a is None or b is None:
        return False
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)


def _summary_identity(summary: Mapping[str, Any]) -> dict[str, Any]:
    release = _first_map(summary, "model_release")
    if release is None:
        release = {}
        raw_release = _first_value(summary, "release")
        if raw_release is not None:
            release["release"] = raw_release
    case_id = _first_value(summary, "case_id")
    input_sha = _first_value(summary, "input_sha256")
    if input_sha is None:
        inp = _mapping(_first_value(summary, "input"))
        if inp is not None:
            input_sha = inp.get("sha256")
    run_id = _first_value(summary, "run_id")
    if run_id is None:
        run_id = _first_value(summary, "run_identity")
    metadata = _mapping(_first_value(summary, "case_metadata")) or _mapping(_first_value(summary, "metadata")) or {}
    identity: dict[str, Any] = {
        "case_id": str(case_id) if case_id is not None else None,
        "patient_id": str(metadata.get("patient_id")) if metadata.get("patient_id") else None,
        "scan_label": str(metadata.get("scan_label")) if metadata.get("scan_label") else None,
        "scan_date": str(metadata.get("scan_date")) if metadata.get("scan_date") else None,
        "input_sha256": str(input_sha) if input_sha is not None else None,
        "run_id": str(run_id) if run_id is not None else None,
        "release": _canonical(release),
    }
    return identity


def _time_axis(summary: Mapping[str, Any]) -> tuple[Any | None, str | None]:
    axis = _first_value(summary, "time_axis")
    if axis is None:
        # ``model_frame`` is explicit in the legacy summary format.  It is
        # safe to wrap it, but we do not invent a frame when it is absent.
        frame = _first_map(summary, "model_frame")
        if frame is not None:
            axis = [dict(frame)]
    if not isinstance(axis, Sequence) or isinstance(axis, (str, bytes)) or not axis:
        return None, "missing time_axis/model_frame"
    normalized = []
    for item in axis:
        if not isinstance(item, Mapping):
            return None, "time_axis contains a non-object frame"
        # ``model_frame`` predates the generic axis envelope and uses
        # ``target`` where current records use ``label``.  Compare only the
        # frame identity/provenance keys, while retaining explicit extras on a
        # current multi-frame axis would make old and new equivalent reports
        # needlessly incompatible.
        frame: dict[str, Any] = {}
        if item.get("index") is not None:
            frame["index"] = item.get("index")
        elif len(axis) == 1:
            frame["index"] = 0
        for key in ("step", "time_s"):
            if item.get(key) is not None:
                frame[key] = item.get(key)
        label = item.get("label") if item.get("label") is not None else item.get("target")
        if label is not None:
            frame["label"] = label
        normalized.append(_canonical(frame))
    return normalized, None


def _wss_field(summary: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    fields = _first_map(summary, "fields")
    if fields is None:
        # A generic result envelope may place fields under results.  This is
        # still an explicit declaration, unlike inferring from wss_field_pa.
        results = _mapping(_first_value(summary, "results"))
        fields = _mapping(results.get("fields")) if results else None
    if fields is None:
        return None, "missing fields declaration for WSS"
    descriptor = fields.get("wss")
    if descriptor is None:
        descriptor = fields.get("wss_pa")
    if not isinstance(descriptor, Mapping):
        return None, "missing WSS field descriptor"
    units = str(descriptor.get("units", "")).strip().lower()
    location = str(descriptor.get("location", "")).strip().lower()
    kind = str(descriptor.get("kind", "")).strip().lower()
    components = descriptor.get("components", 1)
    if units != "pa" or location != "wall" or kind != "scalar" or components != 1:
        return None, "WSS field must be a scalar wall field in Pa"
    # Only these properties determine whether the displayed statistic is the
    # same field.  Array keys may differ between old/new storage envelopes.
    return {"units": "Pa", "location": "wall", "kind": "scalar", "components": 1}, None


def _protocol(summary: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    protocol = _first_map(summary, "statistics_protocol")
    if protocol is None:
        results = _mapping(_first_value(summary, "results"))
        stats = _mapping(results.get("statistics")) if results else None
        candidate = _mapping(stats.get("wss")) if stats else None
        # ``results.statistics.wss`` is the value block, not the protocol;
        # only accept it if it explicitly carries a protocol object.
        if candidate is not None and isinstance(candidate.get("protocol"), Mapping):
            protocol = candidate["protocol"]
    if protocol is None:
        return None, "missing statistics_protocol"
    normalized = {str(k): _canonical(v) for k, v in protocol.items()
                  if str(k) not in _VOLATILE_PROTOCOL_KEYS}
    return normalized, None


def _thresholds(summary: Mapping[str, Any]) -> tuple[list[float] | None, str | None]:
    field = _first_map(summary, "wss_field_pa")
    if field is None:
        results = _mapping(_first_value(summary, "results"))
        stats = _mapping(results.get("statistics")) if results else None
        field = _mapping(stats.get("wss")) if stats else None
    protocol = _first_map(summary, "statistics_protocol") or {}
    raw = field.get("thresholds_pa") if field else None
    if raw is None:
        raw = protocol.get("thresholds_pa")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 3:
        return None, "missing WSS thresholds_pa"
    values = [_number(item) for item in raw]
    if any(value is None for value in values):
        return None, "WSS thresholds_pa must be finite numbers"
    values = [float(value) for value in values]
    if not (0 <= values[0] < values[1] < values[2]):
        return None, "WSS thresholds_pa must be strictly increasing"
    return values, None


def _parameters(summary: Mapping[str, Any]) -> tuple[dict[str, float] | None, str | None]:
    params = _first_map(summary, "run_parameters")
    if params is None:
        params = _first_map(summary, "parameters")
    if params is None:
        return None, "missing run_parameters"
    out: dict[str, float] = {}
    for name in ("smooth_mm", "spacing_mm"):
        value = _number(params.get(name))
        if value is None or float(value) <= 0:
            return None, f"missing or invalid run_parameters.{name}"
        out[name] = float(value)
    return out, None


def _mapping_value(summary: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = _first_value(summary, "mapping")
    return value if isinstance(value, Mapping) else None


def _stat_value(summary: Mapping[str, Any], path: Sequence[str]) -> float | None:
    current: Any = summary
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return _number(current)


def _result_statistics(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return generic result statistics while retaining old top-level views."""
    results = _mapping(_first_value(summary, "results"))
    statistics = _mapping(results.get("statistics")) if results else None
    return statistics or {}


def _peak_rows(summary: Mapping[str, Any]) -> list[tuple[str, str, str, float | None]]:
    compatibility = _mapping(_first_value(summary, "compatibility"))
    wss = (_first_map(summary, "wss_field_pa") or
           _mapping(_result_statistics(summary).get("wss")) or
           (_mapping(compatibility.get("wss_field_pa")) if compatibility else None) or {})
    peak = _first_map(summary, "peak") or {}
    if not peak:
        peak = _mapping(compatibility.get("peak")) if compatibility else None
    peak = peak or {}
    p99 = _number(peak.get("p99_pa"))
    if p99 is None:
        p99 = _number(wss.get("p99_pa"))
    if p99 is None:
        p99 = _number(wss.get("p99"))
    max_value = _number(peak.get("max_pa"))
    if max_value is None:
        max_value = _number(wss.get("max_pa"))
    if max_value is None:
        max_value = _number(wss.get("max"))
    mean = _number(wss.get("mean_pa"))
    if mean is None:
        mean = _number(wss.get("mean"))
    return [("peak.p99_pa", "Peak p99", "peak", p99),
            ("peak.max_pa", "Peak maximum", "peak", max_value),
            ("wss_field.mean_pa", "Wall-field mean", "field", mean)]


def _branch_values(summary: Mapping[str, Any]) -> dict[str, float]:
    branches = _first_map(summary, "per_branch") or {}
    if not branches:
        compatibility = _mapping(_first_value(summary, "compatibility"))
        branches = _mapping(compatibility.get("per_branch")) if compatibility else None
    branches = branches or {}
    out: dict[str, float] = {}
    for name, value in branches.items():
        if not isinstance(value, Mapping):
            continue
        number = _number(value.get("wss_p99_pa"))
        if number is None:
            number = _number(value.get("p99_pa"))
        if number is not None:
            out[str(name)] = float(number)
    return out


def _row(metric: str, label: str, scope: str, left: float | None, right: float | None,
         *, branch: str | None = None, units: str = "Pa") -> dict[str, Any]:
    result: dict[str, Any] = {"id": metric, "label": label, "scope": scope, "units": units,
                              "left": left, "right": right,
                              "delta": (right - left if left is not None and right is not None else None)}
    if branch is not None:
        result["branch"] = branch
    return result


# Cycle-integrated scalars of a three-head release (contract §19.2): (id, label, units, path in summary["cycle"]).
_CYCLE_ROWS = (("cycle.tawss.mean_pa", "TAWSS mean", "Pa", ("fields", "tawss", "mean")),
               ("cycle.tawss.p99_pa", "TAWSS p99", "Pa", ("fields", "tawss", "p99")),
               ("cycle.osi.mean", "OSI mean", "1", ("fields", "osi", "mean")),
               # v0.13 derived indices; an older summary without them simply leaves the cells empty.
               ("cycle.rrt.mean_per_pa", "RRT mean", "1/Pa", ("fields", "rrt", "mean")),
               ("cycle.ecap.mean_per_pa", "ECAP mean", "1/Pa", ("fields", "ecap", "mean")),
               ("cycle.stagnation.area_frac", "Stagnation area fraction (TAWSS < 0.4 Pa and OSI > 0.1)", "1",
                ("stagnation", "area_frac")))
_CYCLE_PROTOCOL_KEYS = ("frames", "weight", "period_s", "tawss", "osi")


def _cycle_rows(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    """Cycle rows plus a note; a delta only when both sides carry ``cycle`` with the same definition and criteria."""
    lc, rc = _mapping(_first_value(left, "cycle")), _mapping(_first_value(right, "cycle"))
    if lc is None and rc is None:
        return [], None
    rows = [_row(metric, label, "cycle", _stat_value(lc or {}, path), _stat_value(rc or {}, path), units=units)
            for metric, label, units, path in _CYCLE_ROWS]
    note = None
    if lc is None or rc is None:
        note = "只有{}有周期量（TAWSS / OSI），差值留空".format("左侧" if lc is not None else "右侧")
    else:
        def protocol(cycle):
            definition = _mapping(cycle.get("definition")) or {}
            stagnation = _mapping(cycle.get("stagnation")) or {}
            return _canonical({"definition": {key: definition.get(key) for key in _CYCLE_PROTOCOL_KEYS},
                               "stagnation": stagnation.get("criteria"),
                               "thresholds": {key: (_mapping((_mapping(cycle.get("fields")) or {}).get(key)) or {}).get("thresholds")
                                              for key in ("tawss", "osi")}})
        if protocol(lc) != protocol(rc):
            note = "两侧周期量定义或阈值不同，差值留空"
    if note:
        for item in rows:
            item["delta"] = None
            item["note"] = note
    return rows, note


def compare_summaries(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe scalar comparison envelope for two summaries.

    ``compatible`` means the two reports use the same declared field, frame,
    statistic definitions, thresholds, and preprocessing parameters.  A
    compatible cross-case comparison is useful for aggregate/branch tables;
    it still never claims that points in the two cases align.
    """
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise TypeError("summary inputs must be mappings")

    reasons: list[str] = []
    left_id, right_id = _summary_identity(left), _summary_identity(right)
    left_sha, right_sha = left_id.get("input_sha256"), right_id.get("input_sha256")
    if not left_sha or not right_sha:
        reasons.append("missing input_sha256; cannot establish same-input or cross-case identity")
        kind = "unknown"
    elif left_sha == right_sha:
        kind = "same_input"
        lm, rm = _mapping_value(left), _mapping_value(right)
        if lm is None or rm is None:
            reasons.append("same input requires mapping in both summaries")
        elif _canonical(lm) != _canonical(rm):
            reasons.append("mapping differs for the same input")
    else:
        kind = "cross_case"

    lf, lreason = _wss_field(left); rf, rreason = _wss_field(right)
    if lreason:
        reasons.append(f"left: {lreason}")
    if rreason:
        reasons.append(f"right: {rreason}")
    if lf is not None and rf is not None and lf != rf:
        reasons.append("WSS field declarations differ")

    lt, lreason = _time_axis(left); rt, rreason = _time_axis(right)
    if lreason:
        reasons.append(f"left: {lreason}")
    if rreason:
        reasons.append(f"right: {rreason}")
    if lt is not None and rt is not None and lt != rt:
        reasons.append("time_axis differs")

    lp, lreason = _protocol(left); rp, rreason = _protocol(right)
    if lreason:
        reasons.append(f"left: {lreason}")
    if rreason:
        reasons.append(f"right: {rreason}")
    if lp is not None and rp is not None and lp != rp:
        reasons.append("statistics_protocol differs")

    lth, lreason = _thresholds(left); rth, rreason = _thresholds(right)
    if lreason:
        reasons.append(f"left: {lreason}")
    if rreason:
        reasons.append(f"right: {rreason}")
    if lth is not None and rth is not None and lth != rth:
        reasons.append("WSS thresholds_pa differ")

    lpar, lreason = _parameters(left); rpar, rreason = _parameters(right)
    if lreason:
        reasons.append(f"left: {lreason}")
    if rreason:
        reasons.append(f"right: {rreason}")
    if lpar is not None and rpar is not None and any(not _equal_number(lpar[key], rpar[key]) for key in lpar):
        reasons.append("run_parameters smooth_mm/spacing_mm differ")

    # Duplicate reasons make UI explanations noisy if malformed input repeats
    # a nested issue.  Preserve first-seen ordering for deterministic output.
    reasons = list(dict.fromkeys(reasons))
    compatible = not reasons

    rows: list[dict[str, Any]] = []
    left_peaks, right_peaks = _peak_rows(left), _peak_rows(right)
    for lrow, rrow in zip(left_peaks, right_peaks):
        rows.append(_row(lrow[0], lrow[1], lrow[2], lrow[3], rrow[3]))
    left_branches, right_branches = _branch_values(left), _branch_values(right)
    for branch in sorted(set(left_branches) | set(right_branches)):
        rows.append(_row(f"per_branch.{branch}.p99_pa", f"{branch} p99", "per_branch",
                         left_branches.get(branch), right_branches.get(branch), branch=branch))
    cycle_rows, cycle_note = _cycle_rows(left, right)
    rows.extend(cycle_rows)
    # ``delta`` is intentionally withheld for an incompatible protocol.  The
    # values remain visible for diagnostics, but consumers must not interpret
    # them as a valid comparison.
    if not compatible:
        for item in rows:
            item["delta"] = None
    out = {"schema_version": SCHEMA_VERSION, "compatible": compatible, "kind": kind,
           "reasons": reasons, "left": left_id, "right": right_id,
           "rows": rows}
    if cycle_rows:
        out["cycle"] = {"comparable": cycle_note is None and compatible, "note": cycle_note}
    return out


def compare_summary_files(left: str | Path, right: str | Path) -> dict[str, Any]:
    """Load two UTF-8 summary JSON files and compare their scalar reports."""
    def load(path: str | Path) -> Mapping[str, Any]:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, Mapping):
            raise ValueError(f"summary is not a JSON object: {path}")
        return value
    return compare_summaries(load(left), load(right))


def compare(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Short alias for callers embedding comparison in an API endpoint."""
    return compare_summaries(left, right)


__all__ = ["SCHEMA_VERSION", "compare", "compare_summaries", "compare_summary_files"]
