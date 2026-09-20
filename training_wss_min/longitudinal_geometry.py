"""Explicit value/mask contract for optional longitudinal geometry inputs.

Reference-surface and point-cloud measurements are separate sources. Missing
values are never imputed as geometry: only the standardised network channel
is set to zero, accompanied by its required, unstandardised validity channel.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "wall_longitudinal_v1"
STATS_SCOPE = "valid_points_in_explicit_train_partition_only"
# public suffix -> candidate value suffix, candidate mask suffix, units
FIELD_SPECS = {
    "log_area": ("area_mm2", "valid", "ln(area_mm2 / 1 mm2)"),
    "area_slope": ("area_slope_per_mm", "area_slope_per_mm_valid", "1/mm"),
    "roundness": ("roundness", "roundness_valid", "dimensionless"),
    "eccentricity": ("eccentricity", "eccentricity_valid", "dimensionless"),
    "upstream_min_area_ratio": ("upstream_min_area_ratio", "upstream_min_area_ratio_valid", "dimensionless"),
    "downstream_min_area_ratio": ("downstream_min_area_ratio", "downstream_min_area_ratio_valid", "dimensionless"),
    "upstream_min_distance": ("upstream_min_distance_mm", "upstream_min_distance_mm_valid", "mm"),
    "downstream_min_distance": ("downstream_min_distance_mm", "downstream_min_distance_mm_valid", "mm"),
}
VALUE_KEYS = tuple(f"geom_{source}_{name}" for source in ("ref", "pc") for name in FIELD_SPECS)
MASK_KEYS = tuple(f"{name}_valid" for name in VALUE_KEYS)
FEATURE_KEYS = tuple(key for name in VALUE_KEYS for key in (name, f"{name}_valid"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_feature_pairs(names) -> tuple[str, ...]:
    selected = set(names) & set(FEATURE_KEYS)
    for name in VALUE_KEYS:
        if (name in selected) != (f"{name}_valid" in selected):
            raise ValueError(f"longitudinal feature {name!r} requires both its value and {name + '_valid'!r} channels")
    return tuple(name for name in VALUE_KEYS if name in selected)


def checked_pair(values, mask, name: str, n: int | None = None):
    values, raw_mask = np.asarray(values), np.asarray(mask)
    if values.ndim != 1 or raw_mask.shape != values.shape or (n is not None and len(values) != n):
        raise ValueError(f"{name}: value/mask must be matching one-dimensional wall arrays")
    if not np.isfinite(raw_mask).all() or not np.isin(raw_mask, [False, True]).all():
        raise ValueError(f"{name}: validity must contain only finite 0/1 values")
    valid = raw_mask.astype(bool)
    if not np.isfinite(values[valid]).all():
        raise ValueError(f"{name}: valid geometry contains nonfinite values")
    return np.asarray(values, dtype=np.float64), valid


def attach_sidecar(case: dict, bundle, names, roots) -> None:
    values = validate_feature_pairs(names)
    if not values:
        return
    paths = [roots] if isinstance(roots, (str, Path)) else list(roots or [])
    if not paths:
        raise ValueError("longitudinal geometry requires data.point_features_root")
    pending = set(values)
    for root in paths:
        path = Path(root) / case["unit_id"] / "features.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            present = [name for name in values if f"wall_{name}" in data or f"wall_{name}_valid" in data]
            if not present:
                continue
            if "longitudinal_schema" not in data or str(data["longitudinal_schema"].item()) != SCHEMA:
                raise ValueError(f"unrecognised longitudinal sidecar schema: {path}")
            if str(data["canonical_id"].item()) != case["unit_id"]:
                raise ValueError(f"longitudinal sidecar case identity differs: {path}")
            if not np.array_equal(data["wall_node_id_cas"], bundle["wall_node_id_cas"]):
                raise ValueError(f"longitudinal sidecar node IDs differ from bundle: {path}")
            if not np.array_equal(data["wall_coords_raw"], bundle["wall_coords_raw"]):
                raise ValueError(f"longitudinal sidecar original coordinates differ from bundle: {path}")
            file_sha = sha256_file(path)
            for name in present:
                if name not in pending:
                    raise ValueError(f"duplicate longitudinal source for {name!r}: {path}")
                if f"wall_{name}" not in data or f"wall_{name}_valid" not in data:
                    raise ValueError(f"longitudinal value and mask must come from the same sidecar: {name}, {path}")
                val, valid = checked_pair(data[f"wall_{name}"], data[f"wall_{name}_valid"], name, len(case["pos"]))
                case[name] = val.astype(np.float32)
                case[f"{name}_valid"] = valid
                case.setdefault("_longitudinal_sources", {})[name] = {"path": str(path), "sha256": file_sha}
                pending.remove(name)
    if pending:
        raise KeyError(f"{case['unit_id']}: longitudinal value/mask pairs missing: {sorted(pending)}")


def compute_stats(cases: list[dict], names) -> dict:
    values = validate_feature_pairs(names)
    if not values:
        return {}
    if not cases:
        raise ValueError("longitudinal statistics require nonempty training cases")
    if any(case.get("_longitudinal_partition") != "train" for case in cases):
        raise ValueError("longitudinal statistics require only explicit train partition cases; use load_partition(..., 'train')")
    units = [case["unit_id"] for case in cases]
    if len(set(units)) != len(units):
        raise ValueError("duplicate training cases in longitudinal feature statistics")
    result = {}
    for name in values:
        chunks = []; count_by_case = {}; sidecar_sha = {}
        for case in cases:
            val, valid = checked_pair(case[name], case[f"{name}_valid"], name)
            chunks.append(val[valid]); count_by_case[case["unit_id"]] = int(valid.sum())
            if name in case.get("_longitudinal_sources", {}):
                sidecar_sha[case["unit_id"]] = case["_longitudinal_sources"][name]["sha256"]
        val = np.concatenate(chunks)
        if not len(val):
            raise ValueError(f"{name}: all training values are invalid; cannot fit normalisation")
        result[name] = {
            "mean": float(val.mean()), "std": float(val.std() + 1e-6), "transform": "none",
            "longitudinal_schema": SCHEMA, "statistics_scope": STATS_SCOPE,
            "mask_feature": f"{name}_valid", "valid_count": len(val),
            "train_units": sorted(units), "valid_count_by_train_unit": count_by_case,
            "sidecar_sha256_by_train_unit": sidecar_sha, "invalid_standardized_value": 0.0,
        }
        result[f"{name}_valid"] = {"mean": 0.0, "std": 1.0, "transform": "none",
                                    "longitudinal_schema": SCHEMA, "normalization": "identity_binary_mask"}
    return result


def validate_frozen_stats(stats: dict, names, train_units=None) -> None:
    for name in validate_feature_pairs(names):
        st = stats.get(name, {})
        if (st.get("longitudinal_schema") != SCHEMA or st.get("statistics_scope") != STATS_SCOPE
                or st.get("mask_feature") != f"{name}_valid" or st.get("valid_count", 0) <= 0
                or st.get("invalid_standardized_value") != 0.0):
            raise ValueError(f"{name}: statistics lack the train-only valid-mask contract")
        if (not np.isfinite(st.get("mean", np.nan)) or not np.isfinite(st.get("std", np.nan))
                or st["std"] <= 0 or st.get("transform") != "none"):
            raise ValueError(f"{name}: invalid longitudinal normalisation statistics")
        if train_units is not None and sorted(st.get("train_units", [])) != sorted(train_units):
            raise ValueError(f"{name}: frozen feature statistics do not match this training split")
        mask_stats = stats.get(f"{name}_valid", {})
        if (mask_stats.get("normalization") != "identity_binary_mask"
                or mask_stats.get("mean") != 0.0 or mask_stats.get("std") != 1.0):
            raise ValueError(f"{name}: validity channel must remain an unstandardised binary mask")


def feature_column(case: dict, idx, name: str, stats: dict) -> np.ndarray:
    if name in MASK_KEYS:
        base = name.removesuffix("_valid")
        _, valid = checked_pair(case[base], case[name], base)
        return valid[idx].astype(np.float64)
    values, valid = checked_pair(case[name], case[f"{name}_valid"], name)
    valid = valid[idx]; values = values[idx]
    result = np.zeros(len(values), dtype=np.float64)
    st = stats[name]
    result[valid] = (values[valid] - st["mean"]) / st["std"]
    if not np.isfinite(result).all() or np.any(np.abs(result) > np.finfo(np.float32).max):
        raise ValueError(f"{name}: normalised geometry cannot form finite float32 network inputs")
    return result


def validate_stats_file(path, names, split_path) -> None:
    """Frozen stats are opt-in and must bind the same train partition."""
    stats = json.loads(Path(path).read_text())
    split = json.loads(Path(split_path).read_text())
    if "train_cases" not in split:
        raise ValueError("longitudinal frozen statistics require explicit train_cases in split")
    validate_frozen_stats(stats, names, split["train_cases"])
