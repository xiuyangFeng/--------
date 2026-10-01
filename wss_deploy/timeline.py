"""Patient follow-up timeline (v0.12, contract ``ANALYSIS_CONTRACT.md`` §19.3).

Pure functions over finished job records plus their ``summary.json``: one *scan* is one input geometry
(``input_sha256``), so running the same STL through several releases gives one scan point.  Geometry
comes from ``summary.morphology`` and does not depend on the release; model quantities are keyed by
release and only ever connected within the same release.  Numbers are copied as stored (no recomputation).
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any, Callable, Iterable, Mapping

SCHEMA = "wss-deploy.timeline/v1"
MAX_PATIENT_ID = 80
MIN_GROWTH_DAYS = 30
FAMILY_LABELS = {"wall": "壁面 WSS", "wall_cycle": "WSS + TAWSS + OSI", "volume": "压力 + 速度体场"}
GEOMETRY_SERIES = (("max_diameter_mm", "管腔最大直径", "mm"), ("sac_volume_ml", "瘤体体积", "mL"),
                   ("sac_length_mm", "瘤体长度", "mm"), ("neck_diameter_mm", "瘤颈直径", "mm"),
                   ("neck_length_mm", "瘤颈长度", "mm"), ("lumen_volume_ml", "管腔体积", "mL"))
MODEL_SERIES = (("wss_p99_pa", "WSS p99", "Pa"), ("wss_low_frac", "低 WSS 面积占比", "1"),
                ("wss_high_frac", "高 WSS 面积占比", "1"), ("tawss_mean_pa", "TAWSS 均值", "Pa"),
                ("tawss_low_frac", "低 TAWSS 面积占比", "1"), ("osi_mean", "OSI 均值", "1"),
                ("osi_high_frac", "OSI > 0.1 面积占比", "1"), ("stagnation_frac", "滞留区面积占比", "1"),
                ("speed_p99_m_s", "速度 p99", "m/s"), ("aorta_delta_p_pa", "主动脉压降", "Pa"))
GROWTH_KEYS = ("max_diameter_mm", "sac_volume_ml")
FIXED_NOTES = ["年增长率只在两次扫描都填写了扫描日期时计算。", "模型量只在同一发布包内连线；几何量与发布包无关。"]


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def clean_patient_id(value: Any) -> str:
    """Trimmed patient id; ``ValueError`` (Chinese message) when empty, too long or not text."""
    if not isinstance(value, str):
        raise ValueError("患者编号必须是文本。")
    value = value.strip()
    if not value:
        raise ValueError("请提供患者编号。")
    if len(value) > MAX_PATIENT_ID or any(ord(c) < 32 or 127 <= ord(c) < 160 for c in value):
        raise ValueError(f"患者编号最多 {MAX_PATIENT_ID} 个字符，不能含控制字符。")
    return value


def family_key(job: Mapping[str, Any], summary: Mapping[str, Any] | None = None) -> str | None:
    """``wall`` / ``wall_cycle`` / ``volume`` from the finished summary, else from the bound release contract."""
    summary = _map(summary) or _map(job.get("summary"))
    fields = _map(summary.get("fields"))
    if not fields:
        fields = _map(_map(_map(job.get("model_release")).get("contract")).get("fields"))
    protocol = _map(_map(job.get("model_release")).get("contract")).get("protocol")
    if "velocity" in fields or "pressure" in fields or protocol == "single_frame_volume":
        return "volume"
    if summary.get("cycle") or "tawss" in fields or "osi" in fields or protocol == "single_frame_wss_cycle_multi":
        return "wall_cycle"
    if fields or protocol == "single_frame_wss":
        return "wall"
    if summary.get("peak") or summary.get("wss_field_pa"):   # pre-schema (2026-09-17) wall results
        return "wall"
    return None


def family_label(job: Mapping[str, Any], summary: Mapping[str, Any] | None = None) -> str | None:
    return FAMILY_LABELS.get(family_key(job, summary) or "")


def geometry_values(summary: Mapping[str, Any]) -> dict[str, Any]:
    """The §19.3 geometry block from ``summary.morphology`` (full or digested); empty when absent."""
    morphology = _map(summary.get("morphology"))
    if not morphology:
        return {}
    aorta = _map(morphology.get("aorta"))
    largest, sac, neck = _map(aorta.get("max")), _map(aorta.get("sac")), _map(aorta.get("neck"))
    present = sac.get("present")
    out = {"max_diameter_mm": _num(largest.get("max_diameter_mm")),
           "sac_present": bool(present) if present is not None else None,
           "sac_length_mm": _num(sac.get("length_mm")) if present else None,
           "sac_volume_ml": _num(sac.get("volume_ml")) if present else None,
           "neck_diameter_mm": _num(neck.get("diameter_mean_mm")) if neck.get("present") else None,
           "neck_length_mm": _num(neck.get("length_mm")) if neck.get("present") else None,
           "lumen_volume_ml": _num(morphology.get("lumen_volume_ml"))}
    return {key: value for key, value in out.items() if value is not None}


def _aorta_delta_p(summary: Mapping[str, Any]) -> float | None:
    for branch in _map(summary.get("morphology")).get("branches") or []:
        branch = _map(branch)
        if branch.get("segment_id") == 0 or branch.get("name") == "主动脉":
            value = _num(branch.get("delta_p_pa"))
            if value is not None:
                return value
    for item in _map(summary.get("findings")).get("items") or []:
        item = _map(item)
        if item.get("kind") == "pressure_drop" and item.get("branch") == "主动脉":
            return _num(item.get("value"))
    return None


def model_values(summary: Mapping[str, Any]) -> dict[str, float]:
    """Headline model quantities of one result (wall / cycle / volume); keys without a value are omitted."""
    peak, field = _map(summary.get("peak")), _map(summary.get("wss_field_pa"))
    cycle = _map(summary.get("cycle"))
    tawss = _map(_map(cycle.get("fields")).get("tawss"))
    osi = _map(_map(cycle.get("fields")).get("osi"))
    speed = _map(_map(summary.get("volume_statistics")).get("speed_m_s"))
    out = {"wss_p99_pa": _num(peak.get("p99_pa")), "wss_low_frac": _num(field.get("area_frac_low")),
           "wss_high_frac": _num(field.get("area_frac_high")),
           "tawss_mean_pa": _num(tawss.get("mean")), "tawss_low_frac": _num(_map(tawss.get("area_frac")).get("low")),
           "osi_mean": _num(osi.get("mean")), "osi_high_frac": _num(_map(osi.get("area_frac")).get("above_t0")),
           "stagnation_frac": _num(_map(cycle.get("stagnation")).get("area_frac")),
           "speed_p99_m_s": _num(speed.get("p99"))}
    if speed or "velocity" in _map(summary.get("fields")):
        out["aorta_delta_p_pa"] = _aorta_delta_p(summary)
    return {key: value for key, value in out.items() if value is not None}


def _created_date(job: Mapping[str, Any]) -> str | None:
    text = str(job.get("created_at") or "")
    try:
        return dt.date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return None


def _review(job: Mapping[str, Any]) -> str:
    return _map(job.get("review")).get("status") or "unreviewed"


def _sha(job: Mapping[str, Any], summary: Mapping[str, Any]) -> str | None:
    value = job.get("input_sha256") or _map(job.get("a")).get("input_sha256") or summary.get("input_sha256")
    return str(value) if value else None


def _growth(scans: list[dict], key: str, pair: tuple[int, int], basis: str) -> dict | None:
    first, last = scans[pair[0]], scans[pair[1]]
    if first["date_source"] != "scan_date" or last["date_source"] != "scan_date":
        return None
    start, end = dt.date.fromisoformat(first["date"]), dt.date.fromisoformat(last["date"])
    days = (end - start).days
    if days < MIN_GROWTH_DAYS:
        return None
    a, b = first["geometry"][key], last["geometry"][key]
    return {"per_year": (b - a) / days * 365.25, "delta": b - a, "days": days,
            "from": {"date": first["date"], "value": a, "input_sha256": first["input_sha256"]},
            "to": {"date": last["date"], "value": b, "input_sha256": last["input_sha256"]}, "basis": basis}


def build_timeline(patient_id: str, jobs: Iterable[Mapping[str, Any]],
                   load_summary: Callable[[Mapping[str, Any]], Mapping[str, Any] | None]) -> dict:
    """Timeline of one patient from already owner-filtered job records (any status; only ``done`` count).

    ``load_summary(job)`` returns the job's parsed ``summary.json`` (or None when missing / unreadable).
    """
    groups: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for job in jobs:
        if not isinstance(job, Mapping) or job.get("status") != "done":
            continue
        if str(job.get("patient_id") or "").strip() != patient_id:
            continue
        summary = _map(load_summary(job))
        sha = _sha(job, summary) or f"job:{job.get('id')}"
        groups.setdefault(sha, []).append((job, summary))
    notes: list[str] = list(FIXED_NOTES)
    scans: list[dict] = []
    for sha, members in groups.items():
        # Newest first: the newest run of a release, and the newest filled-in scan date, win.
        members.sort(key=lambda pair: (str(pair[0].get("created_at") or ""), str(pair[0].get("id") or "")), reverse=True)
        scan_date = next((str(job.get("scan_date")).strip() for job, _ in members if str(job.get("scan_date") or "").strip()), "")
        dates = {str(job.get("scan_date")).strip() for job, _ in members if str(job.get("scan_date") or "").strip()}
        if len(dates) > 1:
            notes.append(f"同一几何的任务填写了不同的扫描日期（{'、'.join(sorted(dates))}），按最新任务的 {scan_date} 计。")
        created = sorted(d for d in (_created_date(job) for job, _ in members) if d)
        date, source = (scan_date, "scan_date") if scan_date else ((created[0] if created else ""), "created_at")
        first = lambda key: next((str(job.get(key)).strip() for job, _ in members if str(job.get(key) or "").strip()), "")
        geometry: dict[str, Any] = {}
        models: dict[str, dict] = {}
        runs = []
        for job, summary in members:
            release = _map(job.get("model_release"))
            release_id = release.get("id") or summary.get("release")
            runs.append({"job_id": job.get("id"), "release_id": release_id, "release_label": family_label(job, summary),
                         "family": "volume" if family_key(job, summary) == "volume" else ("wall" if family_key(job, summary) else None),
                         "review": _review(job), "created_at": job.get("created_at")})
            if not geometry:
                geometry = geometry_values(summary)
            if release_id and release_id not in models:
                values = model_values(summary)
                if values:
                    models[str(release_id)] = values
        scans.append({"input_sha256": None if sha.startswith("job:") else sha, "date": date, "date_source": source,
                      "scan_label": first("scan_label"), "case_id": first("case_id"), "jobs": runs,
                      "geometry": geometry, "models": models,
                      "_order": (date or "9999-99-99", min((str(job.get("created_at") or "") for job, _ in members), default=""))})
    scans.sort(key=lambda scan: scan["_order"])
    for scan in scans:
        scan.pop("_order", None)
    if not scans:
        return {"schema_version": SCHEMA, "patient_id": patient_id, "n_scans": 0, "scans": [], "series": [], "growth": {},
                "notes": notes + ["没有找到该患者已完成的任务（或无权查看）。"]}
    undated = [index + 1 for index, scan in enumerate(scans) if scan["date_source"] != "scan_date"]
    if undated:
        notes.append(f"第 {'、'.join(map(str, undated))} 次扫描没有填写扫描日期，按任务创建日期排序，不计算年增长率。")

    def point(scan, value):
        return {"input_sha256": scan["input_sha256"], "date": scan["date"], "value": value}
    series = []
    for key, label, units in GEOMETRY_SERIES:
        points = [point(scan, scan["geometry"][key]) for scan in scans if key in scan["geometry"]]
        if points:
            series.append({"key": key, "label": label, "units": units, "group": "geometry", "points": points})
    release_labels: dict[str, str | None] = {}
    for scan in scans:
        for run in scan["jobs"]:
            if run["release_id"] and run["release_id"] not in release_labels:
                release_labels[run["release_id"]] = run["release_label"]
    for release_id in sorted({rid for scan in scans for rid in scan["models"]}):
        for key, label, units in MODEL_SERIES:
            points = [point(scan, scan["models"][release_id][key]) for scan in scans
                      if key in scan["models"].get(release_id, {})]
            if points:
                series.append({"key": key, "label": label, "units": units, "group": "model", "release_id": release_id,
                               "release_label": release_labels.get(release_id), "points": points})
    growth, recent = {}, {}
    short_gap = False
    for key in GROWTH_KEYS:
        having = [index for index, scan in enumerate(scans) if key in scan["geometry"]]
        if len(having) < 2:
            continue
        block = _growth(scans, key, (having[0], having[-1]), "first_last")
        if block:
            growth[key] = block
        elif scans[having[0]]["date_source"] == scans[having[-1]]["date_source"] == "scan_date":
            short_gap = True
        # The most recent interval only adds information when it differs from first → last (≥ 3 scans).
        if len(having) >= 3:
            block = _growth(scans, key, (having[-2], having[-1]), "last_two")
            if block:
                recent[key] = block
    if short_gap:
        notes.append(f"首末两次扫描相隔不足 {MIN_GROWTH_DAYS} 天，不计算年增长率。")
    out = {"schema_version": SCHEMA, "patient_id": patient_id, "n_scans": len(scans), "scans": scans, "series": series,
           "growth": growth, "notes": notes}
    if recent:
        out["growth_recent"] = recent
    return out


__all__ = ["FAMILY_LABELS", "MAX_PATIENT_ID", "build_timeline", "clean_patient_id", "family_key", "family_label",
           "geometry_values", "model_values"]
