"""Automatic reference wording for a finished result (contract §17.2).

``build_narrative(summary)`` turns the numbers already in ``summary.json`` into a short Chinese /
English description (three-head releases add one TAWSS / OSI / stagnation sentence, §19.2).  It states only quantities that exist, never invents a value, adds no
interpretation beyond the stored definitions, and always ends with the fixed disclaimer.  Reviewers
may replace the whole text through ``PUT /api/jobs/<id>/narrative``; the generated version is kept
next to the edit so the two can be compared.

2026-09-30 (C1 / C2, WORKSPACE_V2_CONTRACT.md §5.3): diameters are called lumen diameters (the input is the
lumen surface, without mural thrombus and wall) and the no-sac sentence says an aneurysm cannot be excluded;
a three-head result reads morphology → cycle quantities → one peak-frame sentence (the peak-frame low-WSS
fraction is no longer narrated); the hotspot sentence states its real definition (≥ the case's own p99).
"""
from __future__ import annotations

import math
from typing import Any, Mapping

NARRATIVE_SCHEMA = "wss-deploy.narrative/v1"
MAX_NARRATIVE_CHARS = 4000
DISCLAIMER_ZH = "以上为标准血流条件下固定收缩期单帧预测的参考描述，非诊断结论。"
DISCLAIMER_EN = ("The statements above describe a single fixed peak-systolic prediction frame under the standard "
                 "inflow condition, for reference only; they are not a diagnosis.")
CYCLE_DISCLAIMER_ZH = "以上为标准血流条件下单周期 TAWSS / OSI 与收缩期峰值 WSS 预测的参考描述，非诊断结论。"
CYCLE_DISCLAIMER_EN = ("The statements above describe the predicted single-cycle TAWSS / OSI and peak-systolic WSS "
                       "under the standard inflow condition, for reference only; they are not a diagnosis.")
AORTA_ZH, AORTA_EN = "主动脉", "aorta"
BRANCH_EN = {"主动脉": "aorta", "左髂总": "left common iliac", "左髂外": "left external iliac",
             "左髂内": "left internal iliac", "右髂总": "right common iliac",
             "右髂外": "right external iliac", "右髂内": "right internal iliac"}


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _d(value: Any) -> str | None:
    """Diameter / length: one decimal."""
    number = _num(value)
    return None if number is None else f"{number:.1f}"


def _pa(value: Any) -> str | None:
    number = _num(value)
    return None if number is None else f"{number:.1f}"


def _volume(value: Any) -> str | None:
    number = _num(value)
    return None if number is None else f"{number:.0f}"


def _percent(value: Any) -> str | None:
    number = _num(value)
    return None if number is None else f"{100.0 * number:.0f}"


def _ordinal(value: str) -> str:
    """English ordinal suffix for a whole-number percentile (1st, 2nd, 3rd, 4th, 11th ...)."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return f"{value}th"
    suffix = "th" if 11 <= number % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def family_of(summary: Mapping[str, Any]) -> str:
    fields = _map(summary.get("fields"))
    return "volume" if ("velocity" in fields or "pressure" in fields) else "wall"


def _english_branch(name: Any) -> str:
    return BRANCH_EN.get(str(name), str(name or ""))


# ----------------------------------------------------------------------------- sentences
def _morphology_sentences(summary: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    aorta = _map(_map(summary.get("morphology")).get("aorta"))
    largest = _map(aorta.get("max"))
    diameter = _d(largest.get("max_diameter_mm"))
    if diameter is None:
        return [], []
    where = _d(largest.get("distance_from_inlet_mm"))
    zh = [f"{AORTA_ZH}管腔最大直径 {diameter} mm（中心线垂直截面上管腔的最长径，不含附壁血栓与管壁）"]
    en = [f"Largest aortic lumen diameter {diameter} mm (longest chord of the cross-section perpendicular to the centreline, "
          "excluding mural thrombus and the wall)"]
    if where is not None:
        zh.append(f"位于入口下 {where} mm")
        en.append(f"{where} mm below the inlet")
    sac = _map(aorta.get("sac"))
    if sac.get("present"):
        parts_zh, parts_en = [], []
        length = _d(sac.get("length_mm"))
        if length is not None:
            parts_zh.append(f"瘤体长 {length} mm")
            parts_en.append(f"the sac is {length} mm long")
        volume = _volume(sac.get("volume_ml"))
        if volume is not None:
            parts_zh.append(f"体积约 {volume} mL")
            parts_en.append(f"about {volume} mL in volume")
        neck = _map(aorta.get("neck"))
        if neck.get("present"):
            neck_length, neck_diameter = _d(neck.get("length_mm")), _d(neck.get("diameter_mean_mm"))
            if neck_length is not None and neck_diameter is not None:
                parts_zh.append(f"近端瘤颈长 {neck_length} mm、平均直径 {neck_diameter} mm")
                parts_en.append(f"the proximal neck is {neck_length} mm long with a mean diameter of {neck_diameter} mm")
        sentence_zh = "；".join(["，".join(zh)] + parts_zh) + "。"
        sentence_en = "; ".join([", ".join(en)] + parts_en) + "."
    else:
        reference = _d(aorta.get("reference_diameter_mm"))
        tail_zh = "管腔未见瘤样扩张（< 1.5 × 参考直径"
        tail_en = "no aneurysmal dilatation of the lumen (< 1.5 × the reference diameter"
        tail_zh += f" {reference} mm）" if reference else "）"
        tail_en += f" of {reference} mm)" if reference else ")"
        tail_zh += "，不能据此排除动脉瘤"
        tail_en += "; an aneurysm cannot be excluded on this basis"
        sentence_zh = "，".join(zh) + "，" + tail_zh + "。"
        sentence_en = ", ".join(en) + "; " + tail_en + "."
    return [sentence_zh], [sentence_en]


def _hotspot_parts(summary: Mapping[str, Any]) -> tuple[str, str] | tuple[None, None]:
    """「峰值 WSS 最高的区域（不低于本例 p99，X Pa）N 处，最高 Y Pa 位于…」 from the high-WSS clusters.

    The clusters are defined by the case's own spatial p99 (analysis.findings_wall), not by 4 Pa.
    """
    items = [item for item in (_map(summary.get("findings")).get("items") or []) if isinstance(item, Mapping)]
    clusters = [item for item in items if item.get("kind") == "high_wss_cluster"]
    if not clusters:
        return None, None
    level = _pa(_map(summary.get("findings")).get("p99_threshold_pa"))
    if level is None:
        level = _pa(_map(summary.get("peak")).get("p99_pa"))
    strongest = max(clusters, key=lambda item: _num(item.get("value")) or float("-inf"))
    value = _pa(strongest.get("value"))
    where = str(strongest.get("branch") or "")
    tail_zh = f"，最高 {value} Pa" + (f" 位于{where}" if where else "") if value else ""
    tail_en = f", the strongest {value} Pa" + (f" in the {_english_branch(where)}" if where else "") if value else ""
    level_zh = f"（不低于本例 p99，{level} Pa）" if level is not None else "（不低于本例 p99）"
    level_en = f" (at or above this case's p99, {level} Pa)" if level is not None else " (at or above this case's p99)"
    return (f"峰值 WSS 最高的区域{level_zh}{len(clusters)} 处" + tail_zh,
            f"{len(clusters)} region(s) of highest peak WSS{level_en}" + tail_en)


def _cohort_phrase(summary: Mapping[str, Any]) -> tuple[str, str]:
    population = _map(_map(summary.get("reference_assessment")).get("population"))
    # ``percentile`` is already on a 0-100 scale (reference.evaluate), unlike the area fractions.
    rank = _num(population.get("percentile"))
    if rank is None:
        return "", ""
    percentile = f"{rank:.0f}"
    count = population.get("reference_count")
    if not isinstance(count, int):
        count = population.get("case_count")
    if isinstance(count, int):
        return (f"，处于 {count} 例参照人群第 {percentile} 百分位",
                f", at the {_ordinal(percentile)} percentile of the {count}-case reference cohort")
    return (f"，处于参照人群第 {percentile} 百分位", f", at the {_ordinal(percentile)} percentile of the reference cohort")


def _wall_sentences(summary: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Single-head peak-WSS result: (low WSS fraction; hotspots) then the p99 sentence (order unchanged)."""
    zh: list[str] = []
    en: list[str] = []
    field = _map(summary.get("wss_field_pa"))
    thresholds = field.get("thresholds_pa") if isinstance(field.get("thresholds_pa"), list) else [0.4, 4.0, 7.0]
    low_threshold = _num(thresholds[0]) if thresholds else 0.4
    items = [item for item in (_map(summary.get("findings")).get("items") or []) if isinstance(item, Mapping)]
    low_fraction = _percent(field.get("area_frac_low"))
    parts_zh, parts_en = [], []
    if low_fraction is not None and low_threshold is not None:
        low_items = [item for item in items if item.get("kind") == "low_wss_cluster"]
        where = str(low_items[0].get("branch")) if low_items and low_items[0].get("branch") else ""
        parts_zh.append(f"低 WSS（< {low_threshold:g} Pa）区占壁面 {low_fraction}%" + (f"，主要位于{where}" if where else ""))
        parts_en.append(f"Low WSS (< {low_threshold:g} Pa) covers {low_fraction}% of the wall"
                        + (f", mostly in the {_english_branch(where)}" if where else ""))
    hot_zh, hot_en = _hotspot_parts(summary)
    if hot_zh:
        parts_zh.append(hot_zh)
        parts_en.append(hot_en)
    if parts_zh:
        zh.append("；".join(parts_zh) + "。")
        en.append("; ".join(parts_en) + ".")
    p99 = _pa(_map(summary.get("peak")).get("p99_pa"))
    if p99 is not None:
        cohort_zh, cohort_en = _cohort_phrase(summary)
        zh.append(f"壁面 WSS 空间 p99 为 {p99} Pa{cohort_zh}。")
        en.append(f"The spatial p99 of wall shear stress is {p99} Pa{cohort_en}.")
    return zh, en


def _peak_frame_sentence(summary: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Three-head result (C2): one sentence on the peak frame — hotspots by the case p99, no low-WSS fraction."""
    hot_zh, hot_en = _hotspot_parts(summary)
    cohort_zh, cohort_en = _cohort_phrase(summary)
    if hot_zh:
        # the cohort rank belongs to the p99, not to the hottest region
        rank_zh = f"；p99 {cohort_zh[1:]}" if cohort_zh else ""
        rank_en = f"; the p99 is {cohort_en[2:]}" if cohort_en else ""
        return ([f"收缩期峰值帧：{hot_zh}{rank_zh}。"],
                [f"At the peak-systolic frame there are {hot_en}{rank_en}."])
    p99 = _pa(_map(summary.get("peak")).get("p99_pa"))
    if p99 is None:
        return [], []
    return ([f"收缩期峰值帧：壁面 WSS 空间 p99 为 {p99} Pa{cohort_zh}。"],
            [f"At the peak-systolic frame, the spatial p99 of wall shear stress is {p99} Pa{cohort_en}."])


def _largest_branch(rows: Mapping[str, Any], area) -> str:
    """Branch name whose ``area(row)`` is largest (positive); '' when none."""
    best, name = 0.0, ""
    for key, row in rows.items():
        value = _num(area(_map(row)))
        if value is not None and value > best:
            best, name = value, str(key)
    return name


def _cycle_sentences(summary: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """One sentence on TAWSS / OSI / stagnation from ``summary["cycle"]`` (contract §19.2); none without it."""
    cycle = _map(summary.get("cycle"))
    fields = _map(cycle.get("fields"))
    tawss, osi, stagnation = _map(fields.get("tawss")), _map(fields.get("osi")), _map(cycle.get("stagnation"))
    first_zh, first_en, parts_zh, parts_en = [], [], [], []
    mean = _num(tawss.get("mean"))
    if mean is not None:
        first_zh.append(f"周期平均 TAWSS 均值 {mean:.2f} Pa")
        first_en.append(f"Cycle-averaged TAWSS has a mean of {mean:.2f} Pa")
    thresholds = tawss.get("thresholds") if isinstance(tawss.get("thresholds"), list) and tawss.get("thresholds") else [0.4]
    low = _percent(_map(tawss.get("area_frac")).get("low"))
    low_threshold = _num(thresholds[0])
    if low is not None and low_threshold is not None:
        where = _largest_branch(_map(tawss.get("per_branch")),
                                lambda row: (_num(row.get("area_mm2")) or 0.0) * (_num(row.get("frac_low")) or 0.0))
        first_zh.append(f"低 TAWSS（< {low_threshold:g} Pa）区占壁面 {low}%" + (f"，以{where}为主" if where else ""))
        first_en.append(f"low TAWSS (< {low_threshold:g} Pa) covers {low}% of the wall"
                        + (f", mostly in the {_english_branch(where)}" if where else ""))
    if first_zh:
        parts_zh.append("，".join(first_zh))
        parts_en.append(", ".join(first_en))
    osi_thresholds = osi.get("thresholds") if isinstance(osi.get("thresholds"), list) and osi.get("thresholds") else [0.1]
    oscillatory = _percent(_map(osi.get("area_frac")).get("above_t0"))
    osi_threshold = _num(osi_thresholds[0])
    if oscillatory is not None and osi_threshold is not None:
        parts_zh.append(f"OSI > {osi_threshold:g} 占 {oscillatory}%")
        parts_en.append(f"OSI > {osi_threshold:g} covers {oscillatory}%")
    fraction = _percent(stagnation.get("area_frac"))
    if fraction is not None:
        criteria = _map(stagnation.get("criteria"))
        t_lt, o_gt = _num(criteria.get("tawss_lt_pa")), _num(criteria.get("osi_gt"))
        rule_zh = f"（TAWSS < {t_lt:g} Pa 且 OSI > {o_gt:g}）" if t_lt is not None and o_gt is not None else ""
        rule_en = f" (TAWSS < {t_lt:g} Pa and OSI > {o_gt:g})" if t_lt is not None and o_gt is not None else ""
        area = _num(stagnation.get("area_mm2"))
        area_zh = f"（约 {area / 100.0:.0f} cm²）" if area is not None else ""
        area_en = f" (about {area / 100.0:.0f} cm²)" if area is not None else ""
        where = _largest_branch(_map(stagnation.get("per_branch")), lambda row: row.get("area_mm2"))
        parts_zh.append(f"滞留区{rule_zh}占 {fraction}%{area_zh}" + (f"，主要位于{where}" if where else ""))
        parts_en.append(f"the stagnation region{rule_en} covers {fraction}%{area_en}"
                        + (f", mostly in the {_english_branch(where)}" if where else ""))
    if not parts_zh:
        return [], []
    return ["；".join(parts_zh) + "。"], ["; ".join(parts_en) + "."]


def _volume_sentences(summary: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    zh: list[str] = []
    en: list[str] = []
    items = [item for item in (_map(summary.get("findings")).get("items") or []) if isinstance(item, Mapping)]
    drop = next((item for item in items if item.get("kind") == "pressure_drop" and item.get("branch") == AORTA_ZH), None)
    speed_max = _map(_map(summary.get("volume_statistics")).get("speed_m_s")).get("max")
    fastest = next((item for item in items if item.get("kind") == "max_speed"), None)
    parts_zh, parts_en = [], []
    if drop is not None:
        value = _pa(drop.get("value"))
        if value is not None:
            parts_zh.append(f"{AORTA_ZH}近远端压差 {value} Pa")
            parts_en.append(f"The proximal-to-distal pressure difference along the {AORTA_EN} is {value} Pa")
    speed = _num(speed_max) if speed_max is not None else _num((fastest or {}).get("value"))
    if speed is not None:
        where = str((fastest or {}).get("branch") or "")
        parts_zh.append(f"最大速度 {speed:.2f} m/s" + (f" 位于{where}" if where else ""))
        parts_en.append(f"the peak speed is {speed:.2f} m/s" + (f" in the {_english_branch(where)}" if where else ""))
    if parts_zh:
        zh.append("，".join(parts_zh) + "。")
        en.append(", ".join(parts_en) + ".")
    regions = [item for item in items if item.get("kind") == "low_speed_region"]
    if regions:
        zh.append(f"低速（滞留）区 {len(regions)} 处，判据为速度低于全场第 10 百分位的连通簇。")
        en.append(f"{len(regions)} low-speed (stasis) region(s), defined as connected clusters below the 10th "
                  "percentile of the speed field.")
    return zh, en


def build_narrative(summary: Mapping[str, Any] | None, *, generated_at: str | None = None) -> dict:
    """The generated ``summary["narrative"]`` block (contract §17.2)."""
    summary = _map(summary)
    family = family_of(summary)
    zh, en = _morphology_sentences(summary)
    cycle_zh, cycle_en = _cycle_sentences(summary) if family == "wall" else ([], [])
    if family == "volume":
        body_zh, body_en = _volume_sentences(summary)
    elif cycle_zh:
        # C2: a three-head result reads morphology → cycle quantities → one peak-frame sentence.
        peak_zh, peak_en = _peak_frame_sentence(summary)
        body_zh, body_en = [*cycle_zh, *peak_zh], [*cycle_en, *peak_en]
    else:
        body_zh, body_en = _wall_sentences(summary)
    disclaimer_zh, disclaimer_en = (CYCLE_DISCLAIMER_ZH, CYCLE_DISCLAIMER_EN) if cycle_zh else (DISCLAIMER_ZH, DISCLAIMER_EN)
    zh, en = [*zh, *body_zh, disclaimer_zh], [*en, *body_en, disclaimer_en]
    if not generated_at:
        from .clock import now_iso
        generated_at = now_iso()
    stamp = generated_at
    return {"schema_version": NARRATIVE_SCHEMA, "generated_at": stamp, "family": family,
            "zh": zh, "en": en, "edited": None, "edited_by": None, "edited_at": None}


def merge_edit(auto: Mapping[str, Any] | None, edited: str | None, *, edited_by: str | None = None,
               edited_at: str | None = None) -> dict:
    """Generated block plus the reviewer's replacement text (empty / None restores the automatic text)."""
    block = dict(auto) if isinstance(auto, Mapping) else {}
    text = edited if isinstance(edited, str) and edited.strip() else None
    block["edited"] = text
    block["edited_by"] = edited_by if text else None
    block["edited_at"] = edited_at if text else None
    return block


def display_text(narrative: Mapping[str, Any] | None, lang: str = "zh") -> str:
    """One paragraph for the one-page report: the reviewer's text when present, else the generated lines."""
    block = _map(narrative)
    edited = block.get("edited")
    if isinstance(edited, str) and edited.strip():
        return edited.strip()
    lines = block.get(lang if lang in {"zh", "en"} else "zh")
    return " ".join(str(line) for line in lines) if isinstance(lines, list) else ""


__all__ = ["CYCLE_DISCLAIMER_EN", "CYCLE_DISCLAIMER_ZH", "DISCLAIMER_EN", "DISCLAIMER_ZH", "MAX_NARRATIVE_CHARS", "NARRATIVE_SCHEMA", "build_narrative",
           "display_text", "family_of", "merge_edit"]
