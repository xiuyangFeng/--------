"""Cycle-integrated wall fields (TAWSS / OSI) delivered by the three-head model family.

Shared by ``families`` (which attaches the extra fields to a prediction) and ``pipeline``
(which summarises and exports whatever extra wall scalar fields a prediction carries).
Nothing here depends on the model: it is label/threshold bookkeeping and point-cloud
statistics on arrays that already exist.

Definitions (frozen with the training labels, wss_v5/views/wall_cycle_v1):
    TAWSS = (1/80) Σ_k ‖τ_k‖ over frames 0..79 of the 0.8 s cycle           [Pa]
    OSI   = ½ · (1 − ‖Σ_k τ_k‖ / Σ_k ‖τ_k‖)                                   [0, 0.5], dimensionless
Clinical masks: low TAWSS < 0.4 Pa, oscillatory OSI > 0.1 (strong > 0.3), stagnation = both.

Derived indices (v0.13, computed point by point from the predicted TAWSS / OSI; no extra model output):
    RRT  = 1 / [(1 − 2·OSI) · TAWSS]    relative residence time                        [1/Pa]
    ECAP = OSI / TAWSS                   endothelial cell activation potential           [1/Pa]
Both divide by TAWSS (and RRT by 1 − 2·OSI), so they are evaluated with the floors TAWSS ≥ 0.01 Pa and
1 − 2·OSI ≥ 0.01; the floors only keep the numbers finite and never bind for the deployed predictions
(LV_GUO_YOU: TAWSS ≥ 0.09 Pa, OSI ≤ 0.39).  Their error has not been validated against CFD separately:
it inherits the TAWSS / OSI error, amplified where TAWSS is small.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .schema import field_descriptor

CYCLE_DEFINITION = {
    "frames": "0-79 of 81 (Fluent steps 1120-1278)", "weight": "1/80 each (frame 80 = frame 0 + T)",
    "period_s": 0.8, "peak_frame": {"index": 21, "step": 1162, "time_s": 0.21},
    "tawss": "time average of |wall shear stress| over one cycle, Pa",
    "osi": "0.5 * (1 - |time-averaged shear vector| / time-averaged |shear|), 0 (unidirectional) .. 0.5 (fully oscillatory)",
    "source": "wss_min_cycle_v1 labels derived from the 81-frame CFD wall shear vector; no extra CFD",
    "rrt": "1 / ((1 - 2*OSI) * TAWSS), relative residence time, 1/Pa; derived from the predicted TAWSS and OSI "
           "with TAWSS >= 0.01 Pa and 1 - 2*OSI >= 0.01",
    "ecap": "OSI / TAWSS, endothelial cell activation potential, 1/Pa; derived from the predicted TAWSS and OSI "
            "with TAWSS >= 0.01 Pa",
}
TAWSS_THRESHOLDS_PA = (0.4, 4.0, 7.0)      # low / high / very high, same triple as the peak-frame field
OSI_THRESHOLDS = (0.1, 0.2, 0.3)            # oscillatory / marked / strong
STAGNATION = {"tawss_lt_pa": 0.4, "osi_gt": 0.1}
OSI_MAX = 0.5
# Derived indices.  ECAP > 1.4 1/Pa is the thrombus-prone level used in the AAA literature; RRT has no agreed
# absolute threshold, so its three levels are a display grading only.  Both are editable in the report.
RRT_THRESHOLDS = (5.0, 10.0, 20.0)
ECAP_THRESHOLDS = (1.4, 2.8, 4.2)
DERIVED_TAWSS_FLOOR_PA = 0.01
DERIVED_OSI_FACTOR_FLOOR = 0.01            # floor of (1 − 2·OSI) in RRT
DERIVED_FROM = ("tawss", "osi")

FIELD_SPECS = {
    "tawss": {"label": "周期平均壁面切应力 TAWSS", "units": "Pa", "array_key": "tawss_pa", "thresholds": TAWSS_THRESHOLDS_PA,
              "threshold_labels": ("低 TAWSS <", "高 TAWSS >", "极高 >"), "log_scale": True, "clip": (0.0, None)},
    "osi": {"label": "振荡剪切指数 OSI", "units": "1", "array_key": "osi", "thresholds": OSI_THRESHOLDS,
            "threshold_labels": ("OSI >", "OSI >", "OSI >"), "log_scale": False, "clip": (0.0, OSI_MAX)},
    "rrt": {"label": "相对滞留时间 RRT", "units": "1/Pa", "array_key": "rrt_per_pa", "thresholds": RRT_THRESHOLDS,
            "threshold_labels": ("RRT >", "RRT >", "RRT >"), "log_scale": True, "clip": (0.0, None), "derived": True},
    "ecap": {"label": "内皮细胞激活势 ECAP", "units": "1/Pa", "array_key": "ecap_per_pa", "thresholds": ECAP_THRESHOLDS,
             "threshold_labels": ("ECAP >", "ECAP >", "ECAP >"), "log_scale": True, "clip": (0.0, None), "derived": True},
}
# Fields whose statistics count the area *above* the thresholds (TAWSS counts the low tail instead).
ABOVE_FIELDS = ("osi", "rrt", "ecap")


def derive_indices(tawss, osi) -> dict[str, np.ndarray]:
    """RRT and ECAP from TAWSS [Pa] and OSI arrays of any matching shape; NaN (uncovered vertex) stays NaN."""
    t = np.asarray(tawss, dtype=np.float64); o = np.asarray(osi, dtype=np.float64)
    if t.shape != o.shape:
        raise ValueError("TAWSS and OSI must have the same shape")
    with np.errstate(invalid="ignore"):
        t_safe = np.maximum(t, DERIVED_TAWSS_FLOOR_PA)
        o_safe = np.clip(o, 0.0, OSI_MAX)
        rrt = 1.0 / (np.maximum(1.0 - 2.0 * o_safe, DERIVED_OSI_FACTOR_FLOOR) * t_safe)
        ecap = o_safe / t_safe
    nan = ~(np.isfinite(t) & np.isfinite(o))
    rrt[nan] = np.nan; ecap[nan] = np.nan
    return {"rrt": rrt, "ecap": ecap}


def with_derived(extra: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """``extra_fields`` plus the derived ``rrt`` / ``ecap`` entries (``derived=True``) when TAWSS and OSI are present.

    Existing entries are kept as they are; a derived entry already present (a re-run) is recomputed.
    """
    out = {str(k): dict(v) for k, v in extra.items() if str(k) not in ("rrt", "ecap")}
    if all(k in out and "values" in out[k] for k in DERIVED_FROM):
        for key, values in derive_indices(out["tawss"]["values"], out["osi"]["values"]).items():
            out[key] = {"values": values, "derived": True, "derived_from": list(DERIVED_FROM)}
    return out


def descriptor(field_id: str, values: np.ndarray) -> dict[str, Any]:
    """Schema field descriptor plus the display block the report viewer uses to colour the field."""
    spec = FIELD_SPECS[field_id]
    v = np.asarray(values, dtype=np.float64)
    derived = bool(spec.get("derived"))
    d = field_descriptor(field_id, label=spec["label"], units=spec["units"], location="wall", kind="scalar",
                         array_key=spec["array_key"], time_indices=(0,), axis_order=("point",), statistics_key=field_id,
                         source="derived" if derived else "prediction")
    d["display"] = {"thresholds": [float(t) for t in spec["thresholds"]], "log_scale": bool(spec["log_scale"]),
                    "p99": float(np.quantile(v, 0.99)) if v.size else None, "max": float(v.max()) if v.size else None,
                    "range": [float(spec["clip"][0]), float(spec["clip"][1])] if spec["clip"][1] is not None else None,
                    "cycle": "time-integrated over one cardiac cycle (frames 0-79)",
                    "threshold_direction": "above" if field_id in ABOVE_FIELDS else "below"}
    if derived:
        d["derived_from"] = list(DERIVED_FROM)
        d["definition"] = CYCLE_DEFINITION[field_id]
    return d


def derived_display_arrays(fields: Mapping[str, Mapping[str, Any]], extra: Mapping[str, Mapping[str, Any]],
                           vertex_by_key: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Point and vertex arrays (keyed by ``array_key``) of the derived fields present in ``extra``.

    Vertex values apply the formula to the interpolated TAWSS / OSI vertex arrays (uncovered vertex → NaN), so
    a report built here, by ``rebuild_report`` or recomputed in the browser from an older page shows the same colours.
    """
    points: dict[str, np.ndarray] = {}; vertices: dict[str, np.ndarray] = {}
    derived = [k for k, item in extra.items() if item.get("derived")]
    if not derived:
        return points, vertices
    keys = {name: (fields.get(name) or {}).get("array_key") or FIELD_SPECS[name]["array_key"] for name in (*DERIVED_FROM, *derived)}
    if keys["tawss"] not in vertex_by_key or keys["osi"] not in vertex_by_key:
        raise ValueError("派生字段需要 TAWSS 与 OSI 的顶点数组")
    from_vertices = derive_indices(vertex_by_key[keys["tawss"]], vertex_by_key[keys["osi"]])
    for name in derived:
        points[keys[name]] = np.asarray(extra[name]["values"], dtype=np.float32)
        vertices[keys[name]] = from_vertices[name].astype(np.float32)
    return points, vertices


def _stats(v: np.ndarray) -> dict[str, float]:
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p10": float(np.quantile(v, .10)),
            "p90": float(np.quantile(v, .90)), "p99": float(np.quantile(v, .99)), "max": float(v.max()), "min": float(v.min()),
            "n_points": int(v.size)}


def scalar_field_summary(field_id: str, values: np.ndarray, segment_id: np.ndarray, branch_names: Mapping[str, str] | Mapping[int, str],
                         total_area_mm2: float, thresholds=None) -> dict[str, Any]:
    """Point-cloud statistics of one extra wall scalar field: global, threshold area fractions, per branch.

    ``segment_id`` is the per-point atlas segment; ``branch_names`` maps segment id -> name (either key type).
    Area = point fraction × input surface area, the same estimate the peak WSS statistics use.
    """
    spec = FIELD_SPECS[field_id]
    v = np.asarray(values, dtype=np.float64)
    seg = np.asarray(segment_id).astype(int)
    if v.ndim != 1 or v.shape != seg.shape or not np.isfinite(v).all():
        raise ValueError(f"{field_id}: values must be one finite number per prediction point")
    thr = tuple(float(t) for t in (thresholds or spec["thresholds"]))
    if len(thr) != 3 or not 0 <= thr[0] < thr[1] < thr[2]:
        raise ValueError(f"{field_id}: thresholds must be three increasing non-negative values")
    names = {str(k): str(n) for k, n in dict(branch_names).items()}
    low_like = field_id not in ABOVE_FIELDS
    frac = (lambda m: float(np.mean(m)))
    masks = {"low": v < thr[0], "high": v > thr[1], "very_high": v > thr[2]} if low_like else \
            {"above_t0": v > thr[0], "above_t1": v > thr[1], "above_t2": v > thr[2]}
    out: dict[str, Any] = {"field": field_id, "units": spec["units"], "definition": CYCLE_DEFINITION[field_id],
                           **_stats(v), "thresholds": list(thr),
                           "area_frac": {k: frac(m) for k, m in masks.items()},
                           "area_mm2": {k: float(total_area_mm2 * frac(m)) for k, m in masks.items()},
                           "area_method": "point_fraction_times_input_surface_area"}
    branch: dict[str, Any] = {}
    for sid in sorted(set(seg.tolist())):
        mk = seg == sid
        if int(mk.sum()) < 10:
            continue
        b = v[mk]
        branch[names.get(str(sid), str(sid))] = {"segment_id": int(sid), "n_points": int(mk.sum()), "area_mm2": float(total_area_mm2 * mk.mean()),
                                                 "mean": float(b.mean()), "median": float(np.median(b)), "p99": float(np.quantile(b, .99)), "max": float(b.max()),
                                                 **{f"frac_{k}": float(np.mean(m[mk])) for k, m in masks.items()}}
    out["per_branch"] = branch
    return out


def stagnation_summary(tawss: np.ndarray, osi: np.ndarray, segment_id: np.ndarray, branch_names: Mapping, total_area_mm2: float) -> dict[str, Any]:
    """Low-TAWSS ∧ high-OSI region (the classical atherogenic / thrombus-prone signature)."""
    t = np.asarray(tawss, dtype=np.float64); o = np.asarray(osi, dtype=np.float64); seg = np.asarray(segment_id).astype(int)
    if t.shape != o.shape or t.shape != seg.shape:
        raise ValueError("tawss, osi and segment_id must align")
    mask = (t < STAGNATION["tawss_lt_pa"]) & (o > STAGNATION["osi_gt"])
    names = {str(k): str(n) for k, n in dict(branch_names).items()}
    per = {}
    for sid in sorted(set(seg.tolist())):
        mk = seg == sid
        if int(mk.sum()) < 10:
            continue
        per[names.get(str(sid), str(sid))] = {"segment_id": int(sid), "frac": float(np.mean(mask[mk])), "area_mm2": float(total_area_mm2 * np.mean(mask & mk))}
    return {"definition": f"TAWSS < {STAGNATION['tawss_lt_pa']} Pa and OSI > {STAGNATION['osi_gt']}", "criteria": dict(STAGNATION),
            "area_frac": float(mask.mean()), "area_mm2": float(total_area_mm2 * mask.mean()), "n_points": int(mask.sum()), "per_branch": per}


def cycle_block(extra: Mapping[str, Mapping[str, Any]], segment_id: np.ndarray, branch_names: Mapping, total_area_mm2: float) -> dict[str, Any]:
    """The ``summary["cycle"]`` block: definition, per-field summaries and the stagnation region."""
    block: dict[str, Any] = {"definition": dict(CYCLE_DEFINITION), "fields": {}}
    for field_id, item in extra.items():
        if field_id in FIELD_SPECS:
            block["fields"][field_id] = scalar_field_summary(field_id, item["values"], segment_id, branch_names, total_area_mm2)
    if "tawss" in extra and "osi" in extra:
        block["stagnation"] = stagnation_summary(extra["tawss"]["values"], extra["osi"]["values"], segment_id, branch_names, total_area_mm2)
    return block


__all__ = ["ABOVE_FIELDS", "CYCLE_DEFINITION", "DERIVED_FROM", "ECAP_THRESHOLDS", "FIELD_SPECS", "OSI_MAX", "OSI_THRESHOLDS",
           "RRT_THRESHOLDS", "STAGNATION", "TAWSS_THRESHOLDS_PA", "cycle_block", "derive_indices", "derived_display_arrays", "descriptor",
           "scalar_field_summary", "stagnation_summary", "with_derived"]
