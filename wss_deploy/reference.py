"""Release-bound geometry ranges and an optional OOF population reference.

Only metadata embedded in ``release.info`` is read.  There is no training-data
lookup or default reference.  A reviewed range check is not an accuracy or
calibrated-probability estimate.

Example geometry declaration (``geometry.*`` checks every reported branch)::

    geometry_reference = {
        "schema_version": "wss-deploy.geometry-reference/v1",
        "id": "geometry-v1", "status": "validated", "release": "release-id",
        "reference_set": "reviewed-anatomy-reference",
        "bounds": [{"path": "cloud.spacing_mm", "units": "mm", "min": 0.4, "max": 0.6},
                   {"path": "geometry.*.radius_min_mm", "units": "mm", "min": 1, "max": 30}]
    }

``population_reference`` uses schema ``wss-deploy.population-reference/v1``
and the same id/status/release/reference_set keys, plus
``source: cv<k>_oof_predictions`` with the same ``fold_count: <k>`` (CV3 for the
v5.1 releases, CV5 from the v5.2d releases on), ``metric: p99_pa``,
``field: {units: Pa, location: wall, kind: scalar, components: 1}``,
``time_axis``, ``statistics_protocol`` and finite ``values_pa``.  The latter
three contracts must match the current result; CFD distributions are rejected.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "wss-deploy.reference-assessment/v1"
_CLOUD_UNITS = {"spacing_mm": "mm", "surface_variation_median": "1", "variation": "1"}
_BRANCH_UNITS = {"radius_min_mm": "mm", "radius_median_mm": "mm",
                 "radius_max_mm": "mm", "length_mm": "mm"}
_NOTE = "仅检查已声明的几何参考范围，不代表预测准确率或校准概率。"


def _map(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _release_id(value: Mapping) -> str | None:
    for key in ("registry_id", "id", "release", "name"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item
    return None


def _profile(meta: Mapping, release_info: Mapping, key: str, schema: str):
    # ``*_profile`` is accepted as an explicit spelling of the same contract;
    # no filesystem or historical default is consulted.
    profile = _map(release_info.get(key) or release_info.get(f"{key}_profile"))
    if not profile:
        return {}, ["当前发布包未声明参考 profile。"]
    reasons = []
    if profile.get("schema_version") != schema or profile.get("status") != "validated":
        reasons.append("参考 profile 未通过声明的版本和验证状态检查。")
    profile_id = profile.get("id") or profile.get("profile_id")
    for name, value in (("id", profile_id), ("reference_set", profile.get("reference_set"))):
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"参考 profile 缺少 {name}。")
    rid = _release_id(release_info)
    result_rid = _release_id(_map(meta.get("model_release"))) or meta.get("release")
    profile_rid = profile.get("release") or profile.get("release_id")
    if not rid or profile_rid != rid or result_rid != rid:
        reasons.append("参考 profile 与当前结果的发布包不一致或缺少绑定。")
    if "id" not in profile and profile_id is not None:
        # Work with a copy so evaluating a record never mutates release.info.
        profile = dict(profile)
        profile["id"] = profile_id
    return profile, reasons


def _selector(path: Any):
    if not isinstance(path, str):
        return None
    if path.startswith("cloud.") and path[6:] in _CLOUD_UNITS:
        return "cloud", None, path[6:], _CLOUD_UNITS[path[6:]]
    if path.startswith("geometry.") and "." in path[9:]:
        branch, field = path[9:].rsplit(".", 1)
        if branch and field in _BRANCH_UNITS:
            return "geometry", branch, field, _BRANCH_UNITS[field]
    return None


def _geometry(meta: Mapping, release_info: Mapping) -> dict:
    profile, reasons = _profile(meta, release_info, "geometry_reference",
                                "wss-deploy.geometry-reference/v1")
    result = {"status": "unknown", "profile_id": profile.get("id"), "checks": [],
              "reasons": reasons, "note": _NOTE}
    if reasons:
        return result
    bounds = profile.get("bounds")
    if isinstance(bounds, Mapping):
        bounds = [dict(_map(spec), path=path) for path, spec in bounds.items()]
    if bounds is None and isinstance(profile.get("ranges"), (list, Mapping)):
        raw = profile["ranges"]
        bounds = ([dict(_map(spec), path=path) for path, spec in raw.items()]
                  if isinstance(raw, Mapping) else raw)
    if not isinstance(bounds, list) or not bounds:
        result["reasons"] = ["参考 profile 缺少数值范围。"]
        return result
    parsed, seen = [], set()
    for bound in bounds:
        bound = _map(bound)
        selector = _selector(bound.get("path"))
        low, high = _number(bound.get("min")), _number(bound.get("max"))
        if (selector is None or bound.get("path") in seen or
                bound.get("units") != selector[3] or low is None or high is None or
                low < 0 or low > high):
            result["reasons"] = ["参考 profile 含不支持的字段、单位或无效数值范围。"]
            return result
        seen.add(bound["path"])
        parsed.append((bound["path"], selector, low, high))
    for path, (section, branch, field, units), low, high in parsed:
        data = _map(meta.get(section))
        if section == "cloud":
            raw_value = data.get(field)
            # The deployment summary currently calls this dimensionless PCA
            # quantity ``surface_variation_median``; ``variation`` is the
            # shorter declarative profile spelling.
            if raw_value is None and field == "variation":
                raw_value = data.get("surface_variation_median")
            values = [(path, raw_value)]
        elif branch == "*":
            values = [(f"geometry.{name}.{field}", _map(row).get(field))
                      for name, row in sorted(data.items(), key=lambda pair: str(pair[0]))]
            values = values or [(path, None)]
        else:
            values = [(path, _map(data.get(branch)).get(field))]
        for measured_path, raw in values:
            value = _number(raw)
            status = "unknown" if value is None else "pass" if low <= value <= high else "review"
            result["checks"].append({"path": measured_path, "units": units, "value": value,
                                      "min": low, "max": high, "status": status})
    states = {check["status"] for check in result["checks"]}
    result["status"] = "review" if "review" in states else "unknown" if "unknown" in states else "pass"
    if "review" in states:
        result["reasons"].append("部分几何测量超出当前发布包声明的参考范围，请复核。")
    if "unknown" in states:
        result["reasons"].append("部分几何测量缺失或不是有限数值，无法完成范围检查。")
    return result


def _field_contract(value: Any) -> dict | None:
    descriptor = _map(value)
    expected = {"units": "Pa", "location": "wall", "kind": "scalar", "components": 1}
    actual = {key: descriptor.get(key, 1 if key == "components" else None) for key in expected}
    return expected if actual == expected else None


def _axis_contract(value: Any) -> list | None:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], Mapping):
        return None
    frame = value[0]
    if (type(frame.get("index")) is not int or type(frame.get("step")) is not int or
            _number(frame.get("time_s")) is None):
        return None
    normalized = {key: frame[key] for key in ("index", "step", "time_s")}
    if frame.get("label") is not None:
        normalized["label"] = frame["label"]
    return [normalized]


# Per-case values that live in the protocol block but do not define the
# statistic (the hotspot connectivity radius follows the case's own point
# spacing).  A population profile must match every other key exactly.
_VOLATILE_PROTOCOL_KEYS = {"top5_connectivity_radius_mm", "connectivity_radius_mm"}


def _canonical_protocol(value: Any) -> dict:
    return {str(k): v for k, v in _map(value).items() if str(k) not in _VOLATILE_PROTOCOL_KEYS}


def _cv_folds(source: str) -> int | None:
    """``k`` of a ``cv<k>_oof_predictions`` (or ``cv<k>_oof`` / ``…distribution``) source, else None."""
    match = re.fullmatch(r"cv([2-9])_oof(?:_(?:pred\w*|distribution\w*))?", source)
    if match is None and "oof" in source and ("pred" in source or "distribution" in source):
        match = re.search(r"cv([2-9])(?!\d)", source)
    return int(match.group(1)) if match else None


def evaluate_population(meta: Mapping, release_info: Mapping) -> dict:
    """Rank p99 against explicitly bound k-fold (CV3 / CV5) out-of-fold predictions only."""
    meta, release_info = _map(meta), _map(release_info)
    profile, reasons = _profile(meta, release_info, "population_reference",
                                "wss-deploy.population-reference/v1")
    result = {"status": "unknown", "profile_id": profile.get("id"), "reasons": reasons,
              "percentile": None, "reference_count": 0,
              "note": "人群位置仅表示相同预测协议下的经验分位，不是风险概率或临床阈值。"}
    if reasons:
        return result
    source = str(profile.get("source", "")).strip().lower().replace("-", "_")
    folds = _cv_folds(source)
    if (folds is None or
            type(profile.get("fold_count")) is not int or profile["fold_count"] != folds or
            profile.get("metric") != "p99_pa"):
        reasons.append("人群参照必须明确来自交叉验证（CV3 / CV5）折外预测的空间 p99，且折数与来源一致。")
    meta_fields = _map(meta.get("fields"))
    field = _field_contract(meta_fields.get("wss") or meta_fields.get("wss_pa"))
    profile_field = profile.get("field")
    if profile_field is None:
        profile_fields = _map(profile.get("fields"))
        profile_field = profile_fields.get("wss") or profile_fields.get("wss_pa")
    if field is None or _field_contract(profile_field) != field:
        reasons.append("人群参照字段、单位或位置与结果不一致。")
    axis = _axis_contract(meta.get("time_axis"))
    if axis is None or _axis_contract(profile.get("time_axis")) != axis:
        reasons.append("人群参照时间帧与结果不一致。")
    protocol = _canonical_protocol(meta.get("statistics_protocol"))
    if (not protocol or protocol.get("support") != "prediction_point_cloud" or
            protocol.get("field") != "fixed_peak_systolic_frame" or
            protocol.get("quantile_method") != "linear" or
            _canonical_protocol(profile.get("statistics_protocol")) != protocol):
        reasons.append("人群参照统计协议与结果不一致或未完整声明。")
    raw_values = profile.get("values_pa")
    if raw_values is None:
        raw_values = profile.get("p99_values_pa")
    if raw_values is None:
        raw_values = profile.get("values") or profile.get("p99_values")
    if raw_values is None:
        raw_values = profile.get("p99_pa")
    if raw_values is None and isinstance(profile.get("distribution"), Mapping):
        distribution = profile["distribution"]
        raw_values = (distribution.get("values_pa") or distribution.get("p99_values_pa")
                      or distribution.get("p99_pa"))
    if isinstance(raw_values, list):
        values = [_number(item) if not isinstance(item, Mapping) else _number(item.get("p99_pa"))
                  for item in raw_values]
    else:
        values = []
    if len(values) < 2 or any(value is None for value in values):
        reasons.append("人群参照至少需要两个有限的 p99 数值。")
    value = _number(_map(meta.get("peak")).get("p99_pa"))
    if value is None:
        reasons.append("当前结果缺少有限的空间 p99。")
    if reasons:
        return result
    less = sum(item < value for item in values)
    equal = sum(item == value for item in values)
    result.update(status="pass", percentile=100.0 * (less + .5 * equal) / len(values),
                  reference_count=len(values), value_pa=value,
                  rank_method="100 * (count_below + 0.5 * count_equal) / reference_count",
                  reference_set=profile["reference_set"], source=profile["source"],
                  reference_median_pa=float(sorted(values)[len(values) // 2]),
                  # 2026-09-30 (U11): the cohort's 90th percentile grades the high-WSS findings; same
                  # linear quantile rule as the statistic itself (statistics_protocol.quantile_method).
                  reference_p90_pa=quantile(values, 0.90), grading_quantile=0.90)
    # Declared limits travel with the result so a report can show them next to the percentile.
    for key in ("reference_support", "caveats", "validation"):
        if profile.get(key) is not None:
            result[key] = profile[key]
    return result


def evaluate(meta: Mapping, release_info: Mapping) -> dict:
    """Evaluate optional release declarations; missing/invalid ones stay unknown.

    The top-level status is the geometry assessment. Population-reference
    eligibility has its own status and cannot promote a geometry result.
    v2 ``population_references`` bound to the result's release are listed (without their values) under
    ``population_references`` so a reader can see which cohort statistics exist; they never change the
    v1 percentile above.
    """
    meta, release_info = _map(meta), _map(release_info)
    out = {"schema_version": SCHEMA_VERSION, **_geometry(meta, release_info),
           "population": evaluate_population(meta, release_info)}
    result_rid = _release_id(_map(meta.get("model_release"))) or meta.get("release")
    listed = [{key: entry.get(key) for key in _V2_LISTED if key in entry}
              for entry in population_references(release_info, include_v1=False)
              if entry.get("model_release") == result_rid]
    if listed:
        out["population_references"] = listed
    return out


# ----------------------------------------------------------------------------- v2 population references (E4)
POPULATION_V2_KEY = "population_references"
SUBGROUPS = ("all", "AAA", "AG", "ILO")
_V2_REQUIRED = ("id", "metric", "field", "statistic", "units", "subgroup", "source", "model_release")
_V2_LISTED = ("id", "metric", "field", "statistic", "units", "n", "subgroup", "source", "model_release",
              "data_version", "built_on", "schema")


def quantile(values, q: float) -> float | None:
    """Linear-interpolation quantile (numpy's default, the protocol's ``quantile_method: linear``) of finite values."""
    clean = sorted(float(v) for v in values if _number(v) is not None)
    if not clean or not (0.0 <= float(q) <= 1.0):
        return None
    position = (len(clean) - 1) * float(q)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(clean) - 1)
    return clean[lower] + (clean[upper] - clean[lower]) * (position - lower)


def _quantile_key(q: float) -> str:
    return f"p{round(100 * float(q)):d}"


def _v2_entry(raw: Any) -> dict | None:
    """A normalised ``population_references`` entry, or None when it is malformed (fails closed)."""
    entry = _map(raw)
    if any(not isinstance(entry.get(key), str) or not entry.get(key).strip() for key in _V2_REQUIRED):
        return None
    if entry["subgroup"] not in SUBGROUPS:
        return None
    out = {key: entry[key] for key in _V2_LISTED if key in entry}
    values = entry.get("values")
    quantiles = entry.get("quantiles")
    if values is not None:
        if not isinstance(values, list) or len(values) < 2 or any(_number(v) is None for v in values):
            return None
        out["values"] = [float(v) for v in values]
        n = entry.get("n")
        if n is not None and (type(n) is not int or n != len(values)):
            return None
        out["n"] = len(values)
    elif isinstance(quantiles, Mapping) and quantiles:
        parsed = {}
        for key, value in quantiles.items():
            if not isinstance(key, str) or not re.fullmatch(r"p(100|[1-9]?[0-9])", key) or _number(value) is None:
                return None
            parsed[key] = float(value)
        out["quantiles"] = parsed
        if entry.get("n") is not None and (type(entry.get("n")) is not int or entry["n"] < 2):
            return None
    else:
        return None
    out["schema"] = "v2"
    return out


def _v1_entry(release_info: Mapping) -> dict | None:
    """The v1 ``population_reference`` (CV3 OOF p99 of peak WSS) in the v2 shape, when it is well formed."""
    profile = _map(release_info.get("population_reference") or release_info.get("population_reference_profile"))
    if not profile or profile.get("schema_version") != "wss-deploy.population-reference/v1" or profile.get("status") != "validated":
        return None
    raw = profile.get("values_pa")
    for key in ("p99_values_pa", "values", "p99_values", "p99_pa"):
        if raw is None:
            raw = profile.get(key)
    values = ([_number(item) if not isinstance(item, Mapping) else _number(item.get("p99_pa")) for item in raw]
              if isinstance(raw, list) else [])
    if len(values) < 2 or any(value is None for value in values):
        return None
    return {"id": profile.get("id") or profile.get("profile_id"), "metric": str(profile.get("metric") or "p99_pa"),
            "field": "wss", "statistic": "p99", "units": "Pa", "n": len(values), "subgroup": "all",
            "source": profile.get("source"), "model_release": profile.get("release") or profile.get("release_id"),
            "built_on": profile.get("built_on"), "values": values, "schema": "v1"}


def population_references(release_info: Mapping | None, *, include_v1: bool = True) -> list[dict]:
    """Every well-formed cohort reference the release declares: v2 ``population_references`` (list of
    ``{id, metric, field, statistic, units, n, subgroup, source, model_release, data_version, built_on,
    values | quantiles}``) plus, with ``include_v1``, the v1 ``population_reference`` in the same shape.
    Malformed v2 entries are skipped (never guessed)."""
    info = _map(release_info)
    out = []
    if include_v1:
        legacy = _v1_entry(info)
        if legacy is not None:
            out.append(legacy)
    raw = info.get(POPULATION_V2_KEY)
    for item in raw if isinstance(raw, list) else []:
        entry = _v2_entry(item)
        if entry is not None:
            out.append(entry)
    return out


def find_population_reference(release_info: Mapping | None, *, metric: str, field: str | None = None,
                              subgroup: str = "all", release_id: str | None = None) -> dict | None:
    """The first reference of ``metric`` (+ ``field``, ``subgroup``, bound release); v2 entries win over v1."""
    matches = [entry for entry in population_references(release_info)
               if entry.get("metric") == metric and entry.get("subgroup") == subgroup
               and (field is None or entry.get("field") == field)
               and (release_id is None or entry.get("model_release") == release_id)]
    matches.sort(key=lambda entry: 0 if entry.get("schema") == "v2" else 1)
    return matches[0] if matches else None


def reference_quantile(entry: Mapping | None, q: float) -> float | None:
    """Quantile ``q`` of a reference entry: interpolated from ``values``, or the exact ``p<NN>`` of ``quantiles``."""
    entry = _map(entry)
    if isinstance(entry.get("values"), list):
        return quantile(entry["values"], q)
    quantiles = _map(entry.get("quantiles"))
    return _number(quantiles.get(_quantile_key(q)))


def population_p90(population: Mapping | None) -> float | None:
    """U11 grading threshold: the 90th percentile of the same-protocol cohort reference of peak-WSS p99.

    ``population`` is ``reference_assessment["population"]`` (the v1 evaluation).  Only a reference that
    passed every same-protocol check (``status == "pass"``) grades; otherwise None (= no reference).
    """
    block = _map(population)
    if block.get("status") != "pass":
        return None
    return _number(block.get("reference_p90_pa"))


def merge_sidecar_v2(release_info: Mapping | None, release_dir, release_id: str | None = None) -> dict:
    """``release_info`` plus the v2 ``population_references`` of ``<release_dir>/reference.json`` (when present).

    ``infer.load_reference_sidecar`` copies only the v1 keys; this adds the v2 list without touching them.
    Entries not bound to ``release_id`` (or the info's own id) are dropped; the input mapping is not mutated.
    """
    from pathlib import Path
    import json
    info = dict(_map(release_info))
    if release_dir is None or POPULATION_V2_KEY in info:
        return info
    path = Path(release_dir) / "reference.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, ValueError):
        doc = None
    raw = _map(doc).get(POPULATION_V2_KEY)
    if not isinstance(raw, list):
        return info
    rid = release_id or _release_id(info)
    info[POPULATION_V2_KEY] = [item for item in raw if _map(item).get("model_release") == rid]
    return info


__all__ = ["POPULATION_V2_KEY", "SCHEMA_VERSION", "SUBGROUPS", "evaluate", "evaluate_population", "find_population_reference",
           "merge_sidecar_v2", "population_p90", "population_references", "quantile", "reference_quantile"]
