"""One-page (A4) printable report generated from ``summary.json`` (layout: contract §19.5).

Page 1 holds what a reader needs at a glance: header (institution template), case identity, the
reference conclusion, key numbers (TAWSS / OSI / stagnation cards for a three-head release),
pictures, morphology, the follow-up table (when a timeline is passed), the top findings and the
signature lines.  The appendix (printed from a new page) holds the merged branch table, finding
details, input check, trust coverage, limitations, timing, identity hashes and the terms used on
the page.  It is rendered on demand by the service (``GET /api/jobs/<id>/onepage``) and never
written to the job directory, so the exported result files stay exactly what the pipeline produced.
Everything is escaped with :func:`html.escape`; the page contains no scripts.

2026-09-30 (C line, WORKSPACE_V2_CONTRACT.md §5.3): lumen wording (C1); a three-head page leads with the cycle
quantities (C2); every key number carries its evidence tier 几何 / 模型 / 派生 and the appendix has a
「模型验证」 table read from the release's model card (C3); the limitations start with the standard inflow
condition and state that every diameter / length / volume is of the lumen (C4); the ensemble card reads
「多模型一致（一致不代表准确）」 while ``quality.label`` stays as stored (C5); numbers use three significant
digits and percentages whole numbers (U13); the page names the case by its display name — the patient id when
entered, else the case name, which comes from the file name and may be a person's name (C7).

2026-09-30 (second phase S5b, lane B): the page takes the workspace v2 look — v2 colour tokens and type, a ruled key-number
strip instead of tinted cards, status words with a dot (review state, severity, ensemble agreement), decision marks
coloured by state, a patient banner, light print styles with the footer held at the foot of sheet 1.  Presentation only:
every word and number on the page is unchanged (same text in the same order; the markup around it gained classes).
"""
from __future__ import annotations

import base64
import copy
import html
import json
import re
from pathlib import Path
from typing import Any, Mapping

SNAPSHOT_FILE = re.compile(r"snapshot_[a-z0-9_-]{1,32}\.png")
MAX_SNAPSHOT_IMAGES = 12
FIRST_PAGE_SNAPSHOTS = 4
FIRST_PAGE_FINDINGS = 5
TIMELINE_ROWS = 4
NO_SNAPSHOTS_HINT = "在工作区打开结果，点「导出」→「一页纸」生成配图，可为本页配图。"

# C4 (2026-09-30): the standard inflow condition comes first.  Protocol mean inflow 30.38 mL/s = cfd_auto/sanity.py
# PROTOCOL_Q_M3S; period 0.8 s = release.json cycle.period_s / cfd_auto journals.
LIMIT_FLOW = ("标准血流条件：所有病例用同一条入口流量波形（平均 30.38 mL/s，约 1.8 L/min；周期 0.8 s），不是该患者实测。"
              "预测的是这副几何在标准血流下的量；心输出量、心率不同时绝对值会变，空间分布的形态相对稳定；"
              "0.4 / 4 Pa 这类界值也以标准血流为前提。")
LIMIT_LUMEN = "所有直径、长度和体积都是管腔的：输入是管腔面，不含附壁血栓与管壁，通常小于 CT 报告的瘤体直径。"
LIMIT_SINGLE_FRAME = "预测对象是固定收缩期单帧（step 1162，约 0.21 s），不是全周期；没有 TAWSS、OSI 等周期量。"
LIMIT_CYCLE_FRAME = "峰值 WSS 为固定收缩期帧（step 1162，约 0.21 s）；TAWSS / OSI 为单周期（0.8 s、80 帧）积分量的直接回归预测，不是逐帧推演。"
LIMIT_PRESSURE = "压力为相对量（相对于该帧体积平均压力），不能解释为绝对血压；只有压差有意义。"
LIMIT_AREA = "面积与占比按预测点占比乘输入壁面面积估计，统计按点等权，不是体积或面积积分。"
LIMIT_DOMAIN = "模型只在腹主动脉—髂动脉五开口几何上训练；其它血管或超出参考范围的几何不适用。"
LIMIT_ORIENTATION = "输入 STL 不含患者方向；左右语义按解剖坐标架推断并经人工确认，请结合原始影像核对。"
LIMIT_CYCLE_REFERENCE = ("TAWSS / OSI 没有人群参照分位（该发布包没有 CV3 折外预测）；OSI 的预测一致性低于 TAWSS 与峰值 WSS，"
                         "滞留区与高 OSI 区的边界只作定位参考。")
# Full historical list (single-frame wording, every family); ``limitations(summary)`` picks the lines per field.
LIMITATIONS = (LIMIT_FLOW, LIMIT_SINGLE_FRAME, LIMIT_PRESSURE, LIMIT_LUMEN, LIMIT_AREA, LIMIT_DOMAIN, LIMIT_ORIENTATION)
FOOTER_STATEMENT = "仅供研究参考，不作临床诊断依据"

REVIEW_LABELS = {"unreviewed": "未审阅", "reviewed": "已审阅签字", "reopened": "已重新打开"}
REVIEW_TONES = {"reviewed": "ok", "reopened": "warn", "unreviewed": "idle"}      # S5b: ws_ui.reviewInfo tones
# S5b: the workspace mark and the print icon (ws_icons.js 'logo' / 'print', 16 px grid, 1.5 px stroke), inline and
# decorative; the page stays script-free.
_SVG = ('<svg class="{cls}" viewBox="0 0 16 16" width="{size}" height="{size}" fill="none" stroke="currentColor" stroke-width="1.5" '
        'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">{body}</svg>')
LOGO_SVG = _SVG.format(cls="logo", size=22, body='<path d="M8 1.25v3.1"/><ellipse cx="8" cy="6.55" rx="2.55" ry="2.2"/><path d="M8 8.75v1.35"/>'
                                                '<path d="M8 10.1 4.4 14.75"/><path d="M8 10.1l3.6 4.65"/><path d="M5.75 12.35 6.9 14.75"/>')
PRINT_SVG = _SVG.format(cls="ic", size=16, body='<path d="M4 5.75V2.25h8v3.5"/><rect x="1.75" y="5.75" width="12.5" height="5.75" rx="1"/>'
                                               '<rect x="4" y="9.5" width="8" height="4.25"/>')
SEVERITY_LABELS = {"attention": "关注", "note": "提示", "info": "参考"}
QUALITY_DISPLAY = {"good": "多模型一致", "review": "多模型分歧偏大，建议复核", "poor": "多模型分歧大，建议复核"}
QUALITY_CAVEAT = "一致不代表准确"
TIER_NOTE = "来源档：几何 = 从输入表面量出，不经过模型；模型 = 模型预测（与 CFD 的一致性见附录「模型验证」）；派生 = 由预测量算出，没有单独验证。"
UNNAMED_CASE = "未命名病例"
SEVERITY_ORDER = {"attention": 0, "note": 1, "info": 2}
GEOMETRY_KINDS = {"max_diameter", "min_radius"}
TRUST_LABELS = {"interpolation_uncovered": "插值无支撑", "rough_surface": "表面粗糙", "geometry_out_of_range": "几何越界",
                "low_sample_support": "采样支撑弱", "near_opening": "靠近切口"}
DECISION_LABELS = {"confirmed": "☑ 已确认", "rejected": "✕ 已驳回", None: "☐ 未判定"}
RELIABILITY_NOTE = re.compile(r"^.+：(\d+ 站重新定向，\d+ 站截面不可靠未计入直径统计|没有可靠截面)")
TIMELINE_METRICS = {"wss_p99_pa": ("WSS p99", "Pa", 1), "stagnation_frac": ("滞留区占比", "%", 0),
                    "tawss_mean_pa": ("TAWSS 均值", "Pa", 2), "speed_p99_m_s": ("速度 p99", "m/s", 2),
                    "aorta_delta_p_pa": ("主动脉 ΔP", "Pa", 0)}


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _num(value: Any, digits: int = 2, suffix: str = "") -> str:
    if isinstance(value, bool):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number or number in (float("inf"), float("-inf")):
        return "—"
    return f"{number:.{digits}f}{suffix}"


def _sig(value: Any, suffix: str = "", sig: int = 3) -> str:
    """U13: three significant digits (whole numbers from 100 up, no exponent); ``—`` for a missing value."""
    if isinstance(value, bool):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number or number in (float("inf"), float("-inf")):
        return "—"
    if number == 0:
        text = "0"
    elif abs(number) >= 10 ** (sig - 1):
        text = f"{number:.0f}"
    else:
        text = f"{number:#.{sig}g}"
        if "e" in text:
            text = f"{number:.{sig - 1}e}"
        text = text.rstrip(".")
    return text + suffix


def _pct(value: Any, digits: int | None = None) -> str:
    """U13: whole-number percentages; one decimal below 1 % (``digits`` is kept for older callers, ignored)."""
    if isinstance(value, bool) or value is None:
        return "—"
    try:
        number = 100.0 * float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number or number in (float("inf"), float("-inf")):
        return "—"
    if number != 0 and abs(number) < 1:
        return f"{number:.1f}%"
    return f"{number:.0f}%"


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _g(key: str, text: str) -> str:
    """A glossary-marked term (already escaped text); ``show_glossary=used`` lists exactly these keys."""
    return f'<span class="g" data-gloss="{_e(key)}">{text}</span>'


def family_of(summary: Mapping[str, Any]) -> str:
    fields = _map(summary.get("fields"))
    return "volume" if ("velocity" in fields or "pressure" in fields) else "wall"


def has_cycle(summary: Mapping[str, Any]) -> bool:
    """A three-head result: ``summary["cycle"]`` carries TAWSS or OSI statistics."""
    fields = _map(_map(summary.get("cycle")).get("fields"))
    return bool(fields.get("tawss") or fields.get("osi"))


def model_count(summary: Mapping[str, Any], job: Mapping[str, Any] | None = None) -> int | None:
    """Ensemble size: ``run_parameters.seed_count``, else the job's release ``models_count``."""
    for value in (_map(summary.get("run_parameters")).get("seed_count"), _map(_map(job).get("model_release")).get("models_count")):
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def display_name(summary: Mapping[str, Any], job: Mapping[str, Any] | None = None) -> str:
    """C7: the job's ``display_name``, else the summary's, else patient id → case id; ``未命名病例`` without any."""
    from .schema import display_name as shown
    job = _map(job)
    meta = _map(summary.get("case_metadata"))
    for value in (job.get("display_name"), summary.get("display_name"),
                  shown(summary.get("case_id") or job.get("case_id"), meta.get("patient_id") or job.get("patient_id"))):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return UNNAMED_CASE


def model_card_for(summary: Mapping[str, Any], job: Mapping[str, Any] | None = None) -> dict | None:
    """The model card of the result's release (``model_cards.load``; repository copy), or None."""
    from . import model_cards
    try:
        return model_cards.load(_release_id(summary, job) or None)
    except Exception:  # noqa: BLE001 — a missing / unreadable card only removes the validation table
        return None


DEFAULT_TIERS = {"wss": "model", "tawss": "model", "osi": "model", "rrt": "derived", "ecap": "derived",
                 "stagnation": "derived", "max_diameter": "geometry", "sac_volume": "geometry", "lumen_volume": "geometry",
                 "pressure": "model", "speed": "model", "wall_pressure": "model", "delta_p": "derived",
                 "streamlines": "derived"}
TIER_LABELS = {"geometry": "几何", "model": "模型", "derived": "派生"}


def _tier(card: Mapping[str, Any] | None, field_id: str) -> str:
    """C3: the evidence-tier tag of a key number (from the model card, else the documented default)."""
    tiers = _map(_map(card).get("field_tiers"))
    tier = tiers.get(field_id) if tiers.get(field_id) in TIER_LABELS else DEFAULT_TIERS.get(field_id)
    return f' <span class="tier">{TIER_LABELS[tier]}</span>' if tier in TIER_LABELS else ""


def limitations(summary: Mapping[str, Any]) -> list[str]:
    """Limitation lines for the fields this result actually carries (§19.2), plus the summary's own notes.

    C4 (2026-09-30): the standard inflow condition comes first and a lumen line follows the frame lines."""
    cycle = has_cycle(summary)
    items = [LIMIT_FLOW, LIMIT_CYCLE_FRAME if cycle else LIMIT_SINGLE_FRAME]
    if family_of(summary) == "volume":
        items.append(LIMIT_PRESSURE)
    items += [LIMIT_LUMEN, LIMIT_AREA, LIMIT_DOMAIN, LIMIT_ORIENTATION]
    if cycle:
        items.append(LIMIT_CYCLE_REFERENCE)
    items += [str(item) for item in (summary.get("notes") or []) if isinstance(item, str)]
    return items


def _release_id(summary: Mapping[str, Any], job: Mapping[str, Any] | None = None) -> str:
    release = _map(summary.get("model_release"))
    value = (release.get("registry_id") or release.get("release") or release.get("name") or summary.get("release")
             or _map(_map(job).get("model_release")).get("id"))
    return str(value or "")


def _release_short(summary: Mapping[str, Any], job: Mapping[str, Any] | None = None) -> str:
    rid = _release_id(summary, job)
    what = "压力 + 速度" if family_of(summary) == "volume" else ("WSS + TAWSS + OSI" if has_cycle(summary) else "壁面 WSS")
    return f"{rid.split('_')[0]} · {what}" if rid else what


def _model_name(summary: Mapping[str, Any], job: Mapping[str, Any] | None = None, card: Mapping[str, Any] | None = None) -> str:
    """U14: the reader-facing model name (model card ``display_name``); the release code only without a card."""
    name = _map(card).get("display_name")
    return str(name) if isinstance(name, str) and name.strip() else _release_short(summary, job)


def _template(job_dir: Path | str | None, template: Mapping[str, Any] | None) -> dict:
    from . import report_template as RT
    if isinstance(template, Mapping):
        try:
            return RT.validate({key: value for key, value in template.items() if key != "schema_version"})
        except ValueError:
            return copy.deepcopy(RT.DEFAULTS)
    return RT.load_for_job(Path(job_dir)) if job_dir else copy.deepcopy(RT.DEFAULTS)


def _table(headers: list[str], rows: list[list[str]], *, klass: str = "", raw_headers: bool = False,
           row_classes: list[str] | None = None) -> str:
    if not rows:
        return '<p class="muted">无数据</p>'
    head = "".join(f"<th>{h if raw_headers else _e(h)}</th>" for h in headers)
    classes = list(row_classes or [])
    tr = lambda i: f'<tr class="{_e(classes[i])}">' if i < len(classes) and classes[i] else "<tr>"
    body = "".join(tr(i) + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for i, row in enumerate(rows))
    return f'<table class="{_e(klass)}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _kv(pairs: list[tuple[str, str]]) -> str:
    return '<dl class="kv">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in pairs) + "</dl>"


def _columns(count: int) -> int:
    """S5b: columns of a key-number strip — one row up to five numbers, else rows of three (6) or four."""
    return count if count <= 5 else (3 if count == 6 else 4)


def _card_tone(label: str, value: str) -> str:
    """S5b: the ensemble-agreement card reads as a status word (dot + text) like the workspace status line."""
    if 'data-gloss="quality_grade"' not in label:
        return ""
    return " q ok" if value == _e(QUALITY_DISPLAY["good"]) else (" q warn" if "复核" in value else " q")


def _cards(cards: list[tuple[str, str, str]], *, klass: str = "") -> str:
    """Cards: (label HTML, value HTML, note HTML); callers escape their text.  S5b: a ruled strip of key numbers
    (``c<n>`` = columns, see :func:`_columns`)."""
    return f'<div class="cards c{_columns(len(cards))} {klass}">' + "".join(
        f'<div class="card{_card_tone(label, value)}"><span class="label">{label}</span><strong>{value}</strong><small>{note}</small></div>'
        for label, value, note in cards) + "</div>"


def _value_text(value: Any, units: Any) -> str:
    """A finding value with its units, three significant digits (U13)."""
    units = str(units or "")
    text = _sig(value)
    return text if units in ("", "1") or text == "—" else f"{text} {units}"


# ----------------------------------------------------------------------------- page 1
def _header(summary: Mapping[str, Any], job: Mapping[str, Any], template: Mapping[str, Any], title: str) -> str:
    review = _map(summary.get("review")) or _map(job.get("review"))
    status = str(review.get("status") or "unreviewed")
    institution = " · ".join(str(x) for x in (template.get("institution"), template.get("department")) if x)
    frame = ("单周期 TAWSS / OSI 与收缩期峰值 WSS 预测（标准血流条件）" if has_cycle(summary)
             else "固定收缩期单帧预测（标准血流条件）")
    inst = f'<p class="inst">{_e(institution)}</p>' if institution else ""
    tone = REVIEW_TONES.get(status, "idle")      # S5b: status word with a dot, the workspace review tones
    return (f'<header class="top">{LOGO_SVG}<div class="head-main">{inst}<h1>{_e(title)}</h1>'
            f'<p class="sub">{_e(frame)} · {_e(FOOTER_STATEMENT)}</p></div>'
            f'<div class="status">{_g("review_status", "审阅状态")}<br><strong class="{tone}">'
            f'{_e(REVIEW_LABELS.get(status, status))}</strong><br>{_e(_short_time(summary.get("created_at") or job.get("created_at")))}</div></header>')


def _short_time(value: Any) -> str:
    text = str(value or "")
    return text.replace("T", " ")[:16] if text else "—"


def _full_time(value: Any) -> str:
    """``YYYY-MM-DD HH:MM:SS`` from the v0.15 ISO form (offset dropped) or the older plain local form."""
    text = str(value or "")
    return text.replace("T", " ")[:19] if text else "—"


def _identity_line(summary: Mapping[str, Any], job: Mapping[str, Any], card: Mapping[str, Any] | None = None) -> str:
    """C7: the page names the case by its display name — the patient id when entered (then the file-derived case
    name is not repeated on page 1), else the case name."""
    meta = _map(summary.get("case_metadata"))
    scan = " / ".join(x for x in (str(meta.get("scan_label") or job.get("scan_label") or ""),
                                  str(meta.get("scan_date") or job.get("scan_date") or "")) if x)
    patient = str(meta.get("patient_id") or job.get("patient_id") or "").strip()
    first = ("患者编号", patient) if patient else ("病例", display_name(summary, job))
    parts = [first, ("扫描", scan),
             (_g("release", "模型"), _model_name(summary, job, card)), ("生成", _short_time(summary.get("created_at") or job.get("created_at")))]
    # S5b: a patient banner; the " · " separators stay in the text (drawn as thin rules).
    return '<p class="idline">' + '<span class="sep"> · </span>'.join(
        f'<span class="idf">{label if label.startswith("<") else _e(label)} <b>{_e(value)}</b></span>' for label, value in parts if value) + "</p>"


def _narrative_section(summary: Mapping[str, Any]) -> str:
    """「结论（参考）」: the reviewer's text when present (badged), else the generated sentences."""
    block = _map(summary.get("narrative"))
    edited = block.get("edited")
    lines = block.get("zh") if isinstance(block.get("zh"), list) else []
    if not (isinstance(edited, str) and edited.strip()) and not lines:
        return ""
    if isinstance(edited, str) and edited.strip():
        body = "".join(f"<p>{_e(line)}</p>" for line in edited.strip().splitlines() if line.strip())
        badge = '<span class="badge">审阅人已修改</span>'
        who = block.get("edited_by")
        note = f'<p class="muted">{_e(str(who))} · {_e(str(block.get("edited_at") or ""))}</p>' if who else ""
    else:
        body = "".join(f"<p>{_e(line)}</p>" for line in lines)
        badge = '<span class="badge auto">自动生成</span>'
        note = ""
    return f'<h2>结论（参考）{badge}<small>{_g("narrative", "由本页数字自动组织，只陈述存在的量，未做诊断判断")}</small></h2><div class="narrative">{body}{note}</div>'


def _quality_card(summary: Mapping[str, Any], job: Mapping[str, Any]) -> tuple[str, str, str]:
    """C5: shown as 「多模型一致（一致不代表准确）」; ``quality.label`` in summary.json is not changed."""
    from .quality import model_count_phrase
    quality = _map(summary.get("quality"))
    level = str(quality.get("level") or "")
    note = "；".join(str(x) for x in quality.get("reasons") or []) or f"{model_count_phrase(model_count(summary, job))}离散度在常规范围内"
    value = QUALITY_DISPLAY.get(level) or quality.get("label") or quality.get("level") or "未评估"
    return (_g("quality_grade", "多模型一致性"), _e(value), _e(f"{QUALITY_CAVEAT}；{note}"))


def _wall_cards(summary: Mapping[str, Any], job: Mapping[str, Any], card: Mapping[str, Any] | None = None,
                *, peak_prefix: str = "") -> list[tuple[str, str, str]]:
    peak = _map(summary.get("peak"))
    field = _map(summary.get("wss_field_pa"))
    thresholds = field.get("thresholds_pa") if isinstance(field.get("thresholds_pa"), list) else [0.4, 4.0, 7.0]
    area_low = field.get("area_low_mm2")
    population = _map(_map(summary.get("reference_assessment")).get("population"))
    rank = population.get("percentile")
    p99_note = (_e(f"人群第 {_num(rank, 0)} 百分位（{population.get('reference_count') or '—'} 例）") if rank is not None
                else "预测点云空间第 99 百分位")
    tier = _tier(card, "wss")
    lead = lambda text: f"{peak_prefix}{' ' if peak_prefix and text[:1].isascii() else ''}{text}"
    return [
        (_g("p99", _e(lead("WSS 全场 p99"))) + tier, _sig(peak.get("p99_pa"), " Pa"), p99_note),
        (_g("max", _e(lead("全场最大值"))) + tier, _sig(peak.get("max_pa"), " Pa"),
         _e(f"{peak.get('branch') or '—'}，距入口 {_sig(peak.get('s_from_inlet_mm'))} mm；单点值，仅作参考")),
        (_g("area_fraction", f"{_e(lead('低 WSS 面积'))} (&lt; {_e(f'{float(thresholds[0]):g}')} Pa)") + tier, _pct(field.get("area_frac_low")),
         _sig(area_low / 100.0 if isinstance(area_low, (int, float)) else None, " cm²")),
        (_g("thresholds", f"{_e(lead('高 WSS 面积'))} (&gt; {_e(f'{float(thresholds[1]):g}')} Pa)") + tier, _pct(field.get("area_frac_high")),
         _e(f"> {float(thresholds[2]):g} Pa：{_pct(field.get('area_frac_very_high'))}")),
    ]


def _largest(rows: Mapping[str, Any], area) -> str:
    best, name = 0.0, ""
    for key, row in rows.items():
        try:
            value = float(area(_map(row)))
        except (TypeError, ValueError):
            continue
        if value > best:
            best, name = value, str(key)
    return name


def _cycle_cards(summary: Mapping[str, Any], card: Mapping[str, Any] | None = None) -> list[tuple[str, str, str]]:
    """The §19.2 cards (+ the v0.13 RRT / ECAP card); every number comes from ``summary["cycle"]``."""
    cycle = _map(summary.get("cycle"))
    fields = _map(cycle.get("fields"))
    tawss, osi, stagnation = _map(fields.get("tawss")), _map(fields.get("osi")), _map(cycle.get("stagnation"))
    cards = []
    if tawss:
        t = tawss.get("thresholds") if isinstance(tawss.get("thresholds"), list) and tawss.get("thresholds") else [0.4]
        cards.append((_g("tawss", "TAWSS 均值") + _tier(card, "tawss"), _sig(tawss.get("mean"), " Pa"),
                      _e(f"低 TAWSS (< {float(t[0]):g} Pa) {_pct(_map(tawss.get('area_frac')).get('low'))}；p99 {_sig(tawss.get('p99'))} Pa")))
    if osi:
        t = osi.get("thresholds") if isinstance(osi.get("thresholds"), list) and len(osi.get("thresholds")) == 3 else [0.1, 0.2, 0.3]
        frac = _map(osi.get("area_frac"))
        cards.append((_g("osi", "OSI 均值") + _tier(card, "osi"), _sig(osi.get("mean")),
                      _e(f"OSI > {t[0]:g}：{_pct(frac.get('above_t0'))}；> {t[2]:g}：{_pct(frac.get('above_t2'))}")))
    rrt, ecap = _map(fields.get("rrt")), _map(fields.get("ecap"))
    if rrt or ecap:
        # v0.13 derived indices share one card (1/Pa); thresholds come from the summary block.
        def above(block, fallback):
            t = block.get("thresholds") if isinstance(block.get("thresholds"), list) and block.get("thresholds") else [fallback]
            return f"> {float(t[0]):g}：{_pct(_map(block.get('area_frac')).get('above_t0'))}"
        value = " / ".join(_sig(block.get("mean")) for block in (rrt, ecap) if block)
        note = "；".join(part for part in ((f"RRT {above(rrt, 5.0)}" if rrt else ""), (f"ECAP {above(ecap, 1.4)}" if ecap else "")) if part)
        title = _g("rrt", "RRT") + " / " + _g("ecap", "ECAP") + " 均值" if rrt and ecap else (_g("rrt", "RRT 均值") if rrt else _g("ecap", "ECAP 均值"))
        cards.append((title + _tier(card, "rrt" if rrt else "ecap"), value + " Pa⁻¹", _e(note)))
    if stagnation:
        area = stagnation.get("area_mm2")
        where = _largest(_map(stagnation.get("per_branch")), lambda row: row.get("area_mm2"))
        cards.append((_g("stagnation", "滞留区面积") + _tier(card, "stagnation"), _sig(area / 100.0 if isinstance(area, (int, float)) else None, " cm²"),
                      _e(f"占壁面 {_pct(stagnation.get('area_frac'))}" + (f"；主要位于{where}" if where else "") + "；TAWSS < 0.4 Pa 且 OSI > 0.1")))
    return cards


def _drops(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [item for item in _findings(summary) if item.get("kind") == "pressure_drop"]


def _volume_cards(summary: Mapping[str, Any], card: Mapping[str, Any] | None = None) -> list[tuple[str, str, str]]:
    stats = _map(summary.get("volume_statistics"))
    speed, interior, wall = _map(stats.get("speed_m_s")), _map(stats.get("pressure_interior_pa")), _map(stats.get("pressure_wall_pa"))
    cards = [(_g("speed", "体内速度 p99") + _tier(card, "speed"), _sig(speed.get("p99"), " m/s"),
              _e(f"均值 {_sig(speed.get('mean'))}，最大 {_sig(speed.get('max'))} m/s"))]
    aorta = next((item for item in _drops(summary) if item.get("branch") == "主动脉"), None)
    if aorta is not None:
        value = aorta.get("value")
        mmhg = _sig(float(value) / 133.322) if isinstance(value, (int, float)) else "—"
        cards.append((_g("delta_p", "主动脉近远端压差") + _tier(card, "delta_p"), _sig(value, " Pa"), _e(f"≈ {mmhg} mmHg；近端 10% − 远端 10%")))
    cards += [(_g("relative_pressure", "体内相对压力范围") + _tier(card, "pressure"),
               _e(f"{_sig(interior.get('min'))} ～ {_sig(interior.get('max'))} Pa"), "相对于该帧体积平均压力"),
              ("壁面相对压力范围" + _tier(card, "wall_pressure"), _e(f"{_sig(wall.get('min'))} ～ {_sig(wall.get('max'))} Pa"), "壁面查询点，插值到顶点")]
    lines = _map(summary.get("streamlines"))
    if lines:
        cards.append((_g("streamlines", "流线") + _tier(card, "streamlines"), _e(f"{lines.get('line_count', '—')} 条"), "固定帧稳态积分，非粒子轨迹"))
    return cards


def _numbers_section(summary: Mapping[str, Any], job: Mapping[str, Any], card: Mapping[str, Any] | None = None) -> str:
    """「关键数字」: C2 — a three-head result leads with the cycle quantities, then the peak frame; C3 — tiers."""
    if family_of(summary) == "volume":
        cards = _volume_cards(summary, card)
    elif has_cycle(summary):
        # C2: the peak-frame low-WSS share is not a headline next to the low-TAWSS share (the tables keep it).
        peak = [c for c in _wall_cards(summary, job, card, peak_prefix="峰值帧") if 'data-gloss="area_fraction"' not in c[0]]
        cards = _cycle_cards(summary, card) + peak + [_quality_card(summary, job)]
    else:
        cards = _wall_cards(summary, job, card) + [_quality_card(summary, job)]
    return "<h2>关键数字</h2>" + _cards(cards) + f'<p class="muted tiers">{_e(TIER_NOTE)}</p>'


def _reliability(morphology: Mapping[str, Any]) -> tuple[str, list[str], list[str]]:
    """One-line section reliability, the per-branch notes (appendix) and the remaining notes (page 1)."""
    rows = [row for row in (morphology.get("branches") or []) if isinstance(row, Mapping)]
    aorta = _map(morphology.get("aorta"))
    counted = rows or ([aorta] if aorta else [])
    reoriented = sum(int(row.get("n_reoriented") or 0) for row in counted)
    excluded = sum(int(row.get("n_excluded") or 0) for row in counted)
    notes = [str(item) for item in (morphology.get("notes") or []) if isinstance(item, str)]
    detail = [note for note in notes if RELIABILITY_NOTE.match(note)]
    other = [note for note in notes if not RELIABILITY_NOTE.match(note)]
    line = f"截面可靠性：{reoriented} 站重定向，{excluded} 站不可靠（已排除）" if counted else ""
    return line, detail, other


def _morphology_section(summary: Mapping[str, Any], card: Mapping[str, Any] | None = None) -> str:
    """「瘤体形态」: largest section, sac, neck, lumen volume and the reference diameter (all of the lumen, C1)."""
    morphology = _map(summary.get("morphology"))
    aorta = _map(morphology.get("aorta"))
    largest, sac, neck = _map(aorta.get("max")), _map(aorta.get("sac")), _map(aorta.get("neck"))
    if not largest:
        return ""
    tier = _tier(card, "max_diameter")
    cards = [
        (_g("max_diameter", "管腔最大直径") + tier, _sig(largest.get("max_diameter_mm"), " mm"),
         _e(f"距入口 {_sig(largest.get('distance_from_inlet_mm'))} mm；等效直径 {_sig(largest.get('equivalent_diameter_mm'))} mm；不含附壁血栓与管壁")),
        (_g("aneurysm_sac", "瘤体") + tier, (_e(_sig(sac.get("length_mm"), " mm")) if sac.get("present") else "管腔未见"),
         _e(f"体积约 {_sig(sac.get('volume_ml'))} mL；判据 ≥ {_sig(sac.get('threshold_mm'))} mm（管腔）" if sac.get("present")
            else f"管腔最大等效直径 < 1.5 × 参考直径 {_sig(aorta.get('reference_diameter_mm'))} mm；不能据此排除动脉瘤")),
        (_g("aneurysm_neck", "近端瘤颈") + tier, (_e(_sig(neck.get("length_mm"), " mm")) if neck.get("present") else "未给出"),
         _e(f"平均管腔直径 {_sig(neck.get('diameter_mean_mm'))} mm（{_sig(neck.get('diameter_min_mm'))}–{_sig(neck.get('diameter_max_mm'))}）"
            if neck.get("present") else "瘤体近端无持续 < 1.2 × 参考直径的区段")),
        (_g("lumen_volume", "全腔体积") + tier, _sig(morphology.get("lumen_volume_ml"), " mL"),
         _g("reference_diameter", _e(f"参考直径 {_sig(aorta.get('reference_diameter_mm'))} mm（等效直径第 10 百分位）"))),
    ]
    line, _, other = _reliability(morphology)
    notes = "；".join(([line] if line else []) + other)
    return (f'<h2>瘤体形态（管腔）<small>输入是管腔面，不含附壁血栓与管壁；逐分支明细与方法见附录</small></h2>{_cards(cards, klass="morph")}'
            + (f'<p class="muted">{_e(notes)}</p>' if notes else ""))


def _findings(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    items = _map(summary.get("findings")).get("items")
    if not isinstance(items, list):
        return []
    items = [item for item in items if isinstance(item, Mapping)]
    return sorted(items, key=lambda item: (item.get("rank") if isinstance(item.get("rank"), (int, float)) else 1e9))


def _findings_review(summary: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    """Reviewer decisions (contract §11.10): ``findings.review.items`` keyed by finding id, plus manual additions."""
    review = _map(_map(summary.get("findings")).get("review"))
    decisions = review.get("items") if isinstance(review.get("items"), Mapping) else {}
    added = [item for item in (review.get("added") or []) if isinstance(item, Mapping)]
    return decisions, added


def _severity(item: Mapping[str, Any]) -> str:
    severity = str(item.get("severity") or "")
    label = "几何" if item.get("kind") in GEOMETRY_KINDS else SEVERITY_LABELS.get(severity, severity)
    if label and item.get("grading") == "no_reference":
        label += "（无队列参照，不定级）"      # U11: high-WSS items without a same-protocol cohort reference
    geo = " geo" if item.get("kind") in GEOMETRY_KINDS else ""
    return f'<span class="sev {_e(severity)}{geo}">{_e(label)}</span>' if label else ""


def _decision(decision: Any) -> str:
    """S5b: the reviewer's mark (☑ / ✕ / ☐ + word) coloured by state; the text is :data:`DECISION_LABELS`."""
    decision = decision if isinstance(decision, str) else None
    key = decision if decision in ("confirmed", "rejected") else "none"
    return f'<span class="dec {key}">{_e(DECISION_LABELS.get(decision, "☐ 未判定"))}</span>'


def _row_class(decision: Any, *, manual: bool = False) -> str:
    return "manual" if manual else (f"d-{decision}" if decision in ("confirmed", "rejected") else "")


def top_findings(summary: Mapping[str, Any], limit: int = FIRST_PAGE_FINDINGS) -> list[Mapping[str, Any]]:
    """Page-1 findings: one per kind first (severity, then flow before geometry, then rank), then the rest.

    Rejected items never appear; the single-point maximum is deferred because it repeats the hottest cluster.
    """
    decisions, _ = _findings_review(summary)
    items = [item for item in _findings(summary) if _map(decisions.get(str(item.get("id")))).get("decision") != "rejected"]
    key = lambda item: (SEVERITY_ORDER.get(str(item.get("severity")), 3) + (0.5 if item.get("kind") in GEOMETRY_KINDS else 0.0),
                        item.get("rank") if isinstance(item.get("rank"), (int, float)) else 1e9)
    ordered = sorted(items, key=key)
    chosen, kinds = [], set()
    for item in ordered:
        if len(chosen) < limit and item.get("kind") not in kinds and item.get("kind") != "max_wss":
            chosen.append(item); kinds.add(item.get("kind"))
    for item in ordered:
        if len(chosen) < limit and item not in chosen:
            chosen.append(item)
    return sorted(chosen, key=key)


def _top_findings_section(summary: Mapping[str, Any]) -> str:
    decisions, added = _findings_review(summary)
    chosen = top_findings(summary)
    if not chosen and not added:
        return ""
    decided = [_map(decisions.get(str(item.get("id")))).get("decision") for item in chosen]
    rows = [[_e(item.get("id") or ""), _e(item.get("label") or item.get("kind") or ""), _e(item.get("branch") or "—"),
             _e(_value_text(item.get("value"), item.get("units"))) if item.get("value") is not None else "—", _severity(item),
             _decision(decision)] for item, decision in zip(chosen, decided)]
    rows += [[_e(item.get("id") or ""), _e("【人工】" + str(item.get("text") or item.get("label") or "")), _e(item.get("branch") or "—"),
              "—", _severity(item), _decision("confirmed")] for item in added]
    classes = [_row_class(decision) for decision in decided] + [_row_class("confirmed", manual=True)] * len(added)
    total = len(_findings(summary))
    more = f"共 {total} 条自动发现，完整列表与口径见附录" if total > len(chosen) else "口径见附录"
    return (f'<h2>重点发现<small>{_e(more)}</small></h2>'
            + _table(["#", "发现", "分支", "数值", "级别", _g("finding_decision", "判定")], rows, klass="top", raw_headers=True, row_classes=classes))


def _snapshot_figures(job_dir: Path | None, snapshots: Any) -> list[str]:
    """Pictures exported from the 3-D report, embedded as data URIs so the page stays one file."""
    document = _snapshots_document(job_dir, snapshots)
    items = [item for item in (document.get("items") or []) if isinstance(item, Mapping)]
    figures = []
    for item in items[:MAX_SNAPSHOT_IMAGES]:
        name = str(item.get("file") or "")
        if not SNAPSHOT_FILE.fullmatch(name) or job_dir is None:
            continue
        root = Path(job_dir).resolve()
        path = (root / name).resolve()
        if path.parent != root or not path.is_file():
            continue
        try:
            data = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            continue
        caption = item.get("caption") or item.get("view") or item.get("name") or ""
        figures.append(f'<figure><img alt="{_e(caption)}" src="data:image/png;base64,{data}"><figcaption>{_e(caption)}</figcaption></figure>')
    return figures


def _snapshots_document(job_dir: Path | None, snapshots: Any) -> Mapping[str, Any]:
    """The stored snapshots manifest: passed in directly, or read from the job directory."""
    if isinstance(snapshots, Mapping):
        return snapshots
    if job_dir is None:
        return {}
    path = Path(job_dir) / "snapshots.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, Mapping) else {}


def _shots(figures: list[str], *, first_page: bool = False) -> str:
    """Page 1 keeps its pictures in one row (1–2 → two columns, 3 → three, 4 → four) so it still fits one A4
    sheet; a 2 × 2 grid of useful size would not.  Appendix pictures use three columns."""
    columns = (2 if len(figures) <= 2 else min(len(figures), 4)) if first_page else 3
    return f'<div class="shots cols{columns}">' + "".join(figures) + "</div>"


def _snapshots_section(figures: list[str]) -> str:
    # v0.14 (F6): without pictures the slot is a one-line screen hint (never printed); the appendix flows up.
    if not figures:
        return f'<div class="noprint snap-hint"><h2>配图</h2><p class="muted">{_e(NO_SNAPSHOTS_HINT)}</p></div>'
    return "<h2>配图</h2>" + _shots(figures[:FIRST_PAGE_SNAPSHOTS], first_page=True)


def _timeline_section(timeline: Mapping[str, Any] | None, summary: Mapping[str, Any]) -> str:
    """「随访变化」 (contract §19.3): dates, max diameter, sac volume, this release's main metric, growth rates."""
    data = _map(timeline)
    scans = [scan for scan in (data.get("scans") or []) if isinstance(scan, Mapping)]
    if len(scans) < 2:
        return ""
    rid = _release_id(summary)
    wanted = (("speed_p99_m_s", "aorta_delta_p_pa") if family_of(summary) == "volume"
              else ("wss_p99_pa", "stagnation_frac") if has_cycle(summary) else ("wss_p99_pa",))
    models = [_map(_map(scan.get("models")).get(rid)) for scan in scans]
    metrics = [key for key in wanted if any(model.get(key) is not None for model in models)]
    headers = ["扫描", _g("max_diameter", "管腔最大直径 mm"), _g("aneurysm_sac", "瘤体体积 mL")]
    headers += [_e(f"{TIMELINE_METRICS[key][0]}{'' if TIMELINE_METRICS[key][1] == '%' else ' ' + TIMELINE_METRICS[key][1]}") for key in metrics]
    rows = []
    shown = list(zip(scans, models))[-TIMELINE_ROWS:]
    for scan, model in shown:
        geometry = _map(scan.get("geometry"))
        date = str(scan.get("date") or "—") + ("" if scan.get("date_source") == "scan_date" else "（建档）")
        label = str(scan.get("scan_label") or "")
        current = bool(summary.get("input_sha256")) and scan.get("input_sha256") == summary.get("input_sha256")
        cell = _e(date + (f" {label}" if label else "")) + (' <span class="badge auto">本次</span>' if current else "")
        values = []
        for key in metrics:
            _, units, digits = TIMELINE_METRICS[key]
            values.append(_pct(model.get(key), 0) if units == "%" else _sig(model.get(key)))
        rows.append([cell, _sig(geometry.get("max_diameter_mm")),
                     _sig(geometry.get("sac_volume_ml")) if geometry.get("sac_present", True) else "未见", *values])
    growth_parts = []
    for block_key, label in (("growth", "首末"), ("growth_recent", "最近两次")):
        growth = _map(data.get(block_key))
        for key, name, units in (("max_diameter_mm", "管腔最大直径", "mm/年"), ("sac_volume_ml", "瘤体体积", "mL/年")):
            item = _map(growth.get(key))
            rate = item.get("per_year")
            if isinstance(rate, (int, float)) and not isinstance(rate, bool):
                span = f"{_map(item.get('from')).get('date') or '?'} → {_map(item.get('to')).get('date') or '?'}"
                growth_parts.append(f"{name} {rate:+.1f} {units}（{label}，{span}）")
    growth_text = ("年增长率：" + "；".join(growth_parts)) if growth_parts else "；".join(
        str(note) for note in (data.get("notes") or []) if isinstance(note, str))
    more = f"共 {len(scans)} 次扫描，显示最近 {len(shown)} 次" if len(scans) > len(shown) else f"共 {len(scans)} 次扫描"
    return (f'<h2>{_g("follow_up_timeline", "随访变化")}<small>{_e(more)}</small></h2>'
            + _table(headers, rows, klass="num", raw_headers=True)
            + (f'<p class="muted">{_g("growth_rate", _e(growth_text))}</p>' if growth_text else ""))


def _signature_section(summary: Mapping[str, Any], job: Mapping[str, Any], template: Mapping[str, Any]) -> str:
    lines = [str(x) for x in (template.get("signature_lines") or []) if str(x).strip()]
    review = _map(summary.get("review")) or _map(job.get("review"))
    record = ""
    if str(review.get("status") or "") in REVIEW_LABELS and review.get("status") != "unreviewed":
        parts = [REVIEW_LABELS[str(review["status"])]] + [str(review[key]) for key in ("by", "at", "note") if review.get(key)]
        if review.get("version") is not None:
            parts.append(f"锁定版本 {review['version']}")
        record = f'<p class="muted record {REVIEW_TONES.get(str(review["status"]), "idle")}">电子审阅记录：{_e(" · ".join(parts))}</p>'
    if not lines:
        return record
    body = "".join(f'<div class="line"><span class="who">{_e(line)}</span><span class="blank"></span>'
                   f'<span class="who">日期</span><span class="blank short"></span></div>' for line in lines)
    return f'<section class="sign">{body}</section>{record}'


def _footer(summary: Mapping[str, Any], job: Mapping[str, Any], template: Mapping[str, Any], page: str,
            card: Mapping[str, Any] | None = None) -> str:
    note = str(template.get("footer_note") or "").strip()
    note_html = ("；" + "<br>".join(_e(line) for line in note.splitlines() if line.strip())) if note else ""
    return (f'<footer class="pf"><span>{_e(FOOTER_STATEMENT)}{note_html}</span>'
            f'<span>{_e(_model_name(summary, job, card))} · 任务 {_e(job.get("id") or "—")} · {_e(page)}</span></footer>')


# ----------------------------------------------------------------------------- appendix
def _branch_table(summary: Mapping[str, Any]) -> str:
    """Merged 分支统计 + 血管分支表 (+ 几何 stenosis index); columns follow the result family."""
    family = family_of(summary)
    morphology = {str(row.get("name")): row for row in (_map(summary.get("morphology")).get("branches") or []) if isinstance(row, Mapping)}
    per_branch, geometry = _map(summary.get("per_branch")), _map(summary.get("geometry") or summary.get("branch_geometry"))
    cycle = _map(summary.get("cycle"))
    tawss_b = _map(_map(_map(cycle.get("fields")).get("tawss")).get("per_branch"))
    osi_b = _map(_map(_map(cycle.get("fields")).get("osi")).get("per_branch"))
    stag_b = _map(_map(cycle.get("stagnation")).get("per_branch"))
    drops = {str(item.get("branch")): item for item in _drops(summary)}
    names = list(dict.fromkeys([*morphology, *(per_branch if family == "wall" else drops), *geometry]))
    headers = ["分支", "长度 mm", _g("tortuosity", "扭曲度"), _g("max_diameter", "管腔直径 mm"), "狭窄指数"]
    if family == "wall":
        headers += ["面积 cm²", "WSS 均值", "WSS p99", "WSS 最大", "低 WSS", "高 WSS"]
        if tawss_b or osi_b or stag_b:
            headers += [_g("tawss", "TAWSS 均值"), _g("osi", "OSI 均值"), _g("stagnation", "滞留区")]
    else:
        headers += [_g("delta_p", "ΔP Pa"), "ΔP mmHg", _g("speed", "速度均值 m/s"), "速度最大 m/s"]
    rows = []
    for name in names:
        m, g, w = _map(morphology.get(name)), _map(geometry.get(name)), _map(per_branch.get(name))
        length = m.get("length_mm") if m.get("length_mm") is not None else g.get("length_mm")
        tortuosity = m.get("tortuosity") if m.get("tortuosity") is not None else g.get("tortuosity")
        if m.get("diameter_min_mm") is not None:
            diameter = f"{_sig(m.get('diameter_min_mm'))}–{_sig(m.get('diameter_max_mm'))}"
        else:
            diameter = f"{_sig(2 * g['radius_min_mm'] if isinstance(g.get('radius_min_mm'), (int, float)) else None)}–{_sig(g.get('max_diameter_mm'))}"
        row = [_e(name), _sig(length), _sig(tortuosity), _e(diameter), _sig(g.get("stenosis_index"))]
        if family == "wall":
            area = w.get("area_mm2")
            row += [_sig(area / 100.0 if isinstance(area, (int, float)) else None),
                    _sig(w.get("wss_mean_pa") if w.get("wss_mean_pa") is not None else m.get("wss_mean_pa")),
                    _sig(w.get("wss_p99_pa") if w.get("wss_p99_pa") is not None else m.get("wss_p99_pa")), _sig(w.get("wss_max_pa")),
                    _pct(w.get("frac_low") if w.get("frac_low") is not None else m.get("area_frac_low"), 0),
                    _pct(w.get("frac_high") if w.get("frac_high") is not None else m.get("area_frac_high"), 0)]
            if tawss_b or osi_b or stag_b:
                row += [_sig(_map(tawss_b.get(name)).get("mean")), _sig(_map(osi_b.get(name)).get("mean")),
                        _pct(_map(stag_b.get(name)).get("frac"), 0)]
        else:
            drop = _map(drops.get(name))
            value = drop.get("value") if drop.get("value") is not None else m.get("delta_p_pa")
            row += [_sig(value), _sig(float(value) / 133.322) if isinstance(value, (int, float)) else "—",
                    _sig(m.get("speed_mean_m_s")), _sig(m.get("speed_max_m_s"))]
        rows.append(row)
    notes = ["直径为中心线垂直截面上管腔的最长径（即最大 Feret 直径；没有形态测量时为中心线内切直径），不含附壁血栓与管壁；扭曲度 = 中心线弧长 / 两端直线距离；分叉附近的截面会同时切到母血管，该处直径偏大。"]
    if family == "volume" and drops:
        first = next(iter(drops.values()))
        notes.append(f"各分支近远端压差：{first.get('definition') or '近端与远端弧长段相对压力均值之差'}")
    return ('<h2>分支统计</h2>' + _table(headers, rows, klass="num wide", raw_headers=True)
            + "".join(f'<p class="muted">{_e(note)}</p>' for note in notes))


def _finding_row(item: Mapping[str, Any], decision: Mapping[str, Any] | None, *, manual: bool = False) -> list[str]:
    label = item.get("label") or item.get("text") or item.get("kind") or ""
    note = (decision or {}).get("note") or ""
    definition = item.get("definition") or (item.get("text") if manual else "") or ""
    value = item.get("value")
    return [_e(item.get("id") or ""), _e(("【人工】" if manual else "") + str(label)), _e(item.get("branch") or "—"),
            _e(_value_text(value, item.get("units"))) if value is not None else "—", _severity(item),
            _decision((decision or {}).get("decision")),
            _e((definition + ("；备注：" + note if note else "")) if not manual else (note or definition))]


def _findings_section(summary: Mapping[str, Any]) -> str:
    items = _findings(summary)
    decisions, added = _findings_review(summary)
    if not items and not added:
        return ""
    headers = ["#", "发现", "分支", "数值", "级别", "判定", "口径 / 备注"]
    body, rejected = [], []
    for item in items:
        decision = _map(decisions.get(str(item.get("id"))))
        (rejected if decision.get("decision") == "rejected" else body).append((item, decision))
    rows = [_finding_row(item, decision) for item, decision in body]
    rows += [_finding_row(item, {"decision": "confirmed", "note": ""}, manual=True) for item in added]
    classes = [_row_class(decision.get("decision")) for _, decision in body] + [_row_class("confirmed", manual=True)] * len(added)
    out = '<h2>发现详情</h2>' + _table(headers, rows, klass="findings", row_classes=classes)
    if decisions or added:
        out += '<p class="muted">☑ 审阅人已确认；☐ 尚未判定；【人工】为审阅人手动添加；驳回项单列在下方。</p>'
    if rejected:
        out += ('<h3>附录：已驳回的自动发现</h3>'
                + _table(headers, [_finding_row(item, decision) for item, decision in rejected], klass="findings", row_classes=["d-rejected"] * len(rejected)))
    legacy = [item for item in (_map(_map(summary.get("findings")).get("review")).get("legacy") or []) if isinstance(item, Mapping)]
    if legacy:
        # 2026-09-30: decisions made under the previous listing rules that no longer match a finding; kept, not applied.
        rows = [[_e(item.get("id") or ""), _e(str(item.get("label") or item.get("kind") or "")), _e(item.get("branch") or "—"),
                 _e(_value_text(item.get("value"), item.get("units"))) if item.get("value") is not None else "—",
                 _decision(item.get("decision")), _e(str(item.get("reason") or "") + ("；备注：" + str(item["note"]) if item.get("note") else ""))]
                for item in legacy]
        out += ('<h3>规则更新前的判定<small>发现规则更新后找不到对应发现，保留原判定但不套用</small></h3>'
                + _table(["原编号", "发现", "分支", "数值", "原判定", "说明"], rows, klass="findings legacy"))
    return out


def _annotations_section(summary: Mapping[str, Any]) -> str:
    items = [item for item in (_map(summary.get("annotations")).get("items") or []) if isinstance(item, Mapping)]
    if not items:
        return ""
    rows = [[_e(item.get("id") or ""), _e(item.get("branch") or "—"), _num(item.get("s_from_root_mm"), 0), _e(item.get("text") or "")]
            for item in items]
    return '<h2>标注</h2>' + _table(["编号", "分支", "弧长 mm", "文字"], rows, klass="findings")


def _input_section(summary: Mapping[str, Any]) -> str:
    ic = _map(summary.get("input_check"))
    quality = _map(ic.get("quality"))
    bbox = ic.get("bbox_size_mm") if isinstance(ic.get("bbox_size_mm"), list) else []
    assessment = _map(summary.get("reference_assessment"))
    population = _map(assessment.get("population"))
    geometry_status = {"pass": "在参考范围内", "review": "部分超出参考范围，请复核", "unknown": "未评估"}.get(str(assessment.get("status") or "unknown"), "未评估")
    if population.get("percentile") is not None:
        population_text = f"第 {_num(population.get('percentile'), 0)} 百分位（{population.get('reference_count') or '—'} 例折外预测）"
    else:
        population_text = "未提供" + (f"（{'；'.join(str(r) for r in population.get('reasons') or [])}）" if population.get("reasons") else "")
    pairs = [
        ("单位", _e(ic.get("unit") or ic.get("resolved_units") or "—")),
        ("顶点 / 面片", _e(f"{ic.get('vertices', '—')} / {ic.get('faces', '—')}")),
        ("壁面面积", _num(float(ic["area_mm2"]) / 100.0, 1, " cm²") if isinstance(ic.get("area_mm2"), (int, float)) else "—"),
        ("包围盒 (mm)", _e(" × ".join(_num(v, 0) for v in bbox) if bbox else "—")),
        ("开口数 / 连通片", _e(f"{ic.get('openings', '—')} / {ic.get('components', '—')}")),
        ("质量评分卡", _e(quality.get("grade_label") or quality.get("grade") or "—")),
        (_g("trust_geometry_out_of_range", "几何参考范围"), _e(geometry_status)),
        (_g("population_percentile", "人群分位"), _e(population_text)),
    ]
    flags = [str(f) for f in (ic.get("flags") or [])]
    return "<h2>输入检查与参考范围</h2>" + _kv(pairs) + (f'<p class="muted">{_e("；".join(flags))}</p>' if flags else "")


def _morphology_details(summary: Mapping[str, Any]) -> str:
    morphology = _map(summary.get("morphology"))
    if not _map(morphology.get("aorta")).get("max"):
        return ""
    method = _map(morphology.get("method"))
    station = morphology.get("station_mm") or 1.0
    equivalent = str(method.get("equivalent_diameter") or "2·sqrt(面积/π)")
    equivalent = equivalent if equivalent.startswith("等效直径") else f"等效直径 = {equivalent}"
    explanation = (f"沿中心线每 {_num(station, 0)} mm 取一站，以局部切线为法向切壁面网格；"
                   f"管腔最大直径 = {method.get('max_diameter') or '轮廓上最远两点的距离'}；{equivalent}。"
                   "输入是管腔面，所有直径、长度和体积都是管腔的，不含附壁血栓与管壁。")
    line, detail, _ = _reliability(morphology)
    return ('<h2>瘤体形态（管腔）方法与截面可靠性</h2>'
            + f'<p class="muted">{_g("equivalent_diameter", _e(explanation))}</p>'
            + (f'<p class="muted">{_e(line)}：{_e("；".join(detail))}</p>' if detail else (f'<p class="muted">{_e(line)}</p>' if line else "")))


def _trust_section(summary: Mapping[str, Any]) -> str:
    trust = _map(summary.get("trust"))
    fractions = _map(trust.get("fractions"))
    if not fractions:
        return ""
    labels = {str(s.get("bit")): s.get("label") for s in trust.get("sources") or [] if isinstance(s, Mapping)}
    bits = _map(trust.get("bits"))
    rows = []
    for key, value in fractions.items():
        bit = next((b for b, name in bits.items() if name == key), None)
        label = _e(labels.get(str(bit)) or TRUST_LABELS.get(str(key), key))
        rows.append([_g(f"trust_{key}", label) if re.fullmatch(r"[a-z_]+", str(key)) else label, _pct(value)])
    return ('<h2>可信区域</h2><p class="muted">下列比例是模型结果中应谨慎解读的区域占比（按顶点或采样点计），不代表其它位置的预测准确率。</p>'
            + _table(["原因", "占比"], rows, klass="num narrow"))


def _timing_section(summary: Mapping[str, Any]) -> str:
    timing = _map(summary.get("timing_s"))
    names = {"ingest": "输入检查", "centerline": "中心线", "smooth_resample": "平滑重采样", "features": "几何特征",
             "volume_features": "体场特征", "inference_5_models": "模型推理", "inference_volume": "模型推理", "morphology": "形态测量",
             "metrics_and_interpolation": "统计与插值", "streamlines_and_interpolation": "流线与插值", "export": "导出", "total": "合计",
             "precompute": "后台几何预计算（确认出口期间）"}
    # v0.14: stages served from the background geometry precompute (summary.geometry_cache.reused) take ≈ 0 s.
    kind_stage = {"mesh": "smooth_resample", "resample": "smooth_resample", "pointgeom": "features", "morph": "morphology", "morphvol": "morphology"}
    reused = _map(summary.get("geometry_cache")).get("reused")
    cached = {kind_stage[k] for k in (reused if isinstance(reused, list) else []) if k in kind_stage}
    parts = [f"{names.get(k, k)} {_num(v, 1)} s{'（已预计算）' if k in cached else ''}" for k, v in timing.items() if isinstance(v, (int, float))]
    device = summary.get("device")
    gpu = summary.get("gpu")
    device_text = f"{device}{' · ' + str(gpu) if gpu else ''}" if device else ""
    return f'<h2>计算耗时</h2><p class="muted">{_e("；".join(parts))}{_e("（" + device_text + "）") if device_text else ""}</p>'


def _identity_section(summary: Mapping[str, Any], job: Mapping[str, Any]) -> str:
    meta = _map(summary.get("case_metadata"))
    release = _map(summary.get("model_release"))
    contract = _map(summary.get("feature_contract"))
    tags = meta.get("tags") if isinstance(meta.get("tags"), list) else []
    frame = "收缩期峰值帧（step 1162，约 0.21 s）" + ("；TAWSS / OSI：单周期 0.8 s、80 帧" if has_cycle(summary) else "")
    pairs = [
        # C7 (2026-09-30): the case name comes from the uploaded file name and may be a person's name — it is
        # not an anonymous id; the page is titled by the display name (patient id when entered).
        ("显示名", _e(display_name(summary, job))),
        ("病例名（取自上传文件名，可能含姓名）", _e(summary.get("case_id") or job.get("case_id") or "—")),
        ("患者编号", _e(meta.get("patient_id") or job.get("patient_id") or "—")),
        ("扫描标签 / 日期", _e(" / ".join(x for x in (meta.get("scan_label") or job.get("scan_label") or "", meta.get("scan_date") or job.get("scan_date") or "") if x) or "—")),
        ("标签", _e("、".join(str(t) for t in tags) or "—")),
        ("任务编号", _e(job.get("id") or "—")),
        ("生成时间", _e(_full_time(summary.get("created_at")))),
        (_g("release", "发布包"), _e(_release_id(summary, job) or "—")),
        ("模型数", _e(model_count(summary, job) or "—")),
        ("发布包指纹", _e(str(release.get("fingerprint") or summary.get("release_hash") or "—")[:16])),
        (_g("feature_contract", "特征程序合同"), _e(f"{contract.get('version') or '—'} · {str(contract.get('source_hash') or '—')[:16]}")),
        (_g("run_identity", "运行身份"), _e(str(summary.get("run_identity") or job.get("run_identity") or "—")[:16])),
        (_g("cycle_period" if has_cycle(summary) else "fixed_frame", "时间帧"), _e(frame)),
    ]
    return "<h2>身份与哈希</h2>" + _kv(pairs)


def _validation_section(card: Mapping[str, Any] | None, summary: Mapping[str, Any]) -> str:
    """C3: 「模型验证」 — one row per quantity from the release's model card (tier, held-out agreement, usage)."""
    from .model_cards import validation_rows
    if not isinstance(card, Mapping):
        return ('<h2>模型验证</h2><p class="muted">该发布包没有模型说明卡；与 CFD 的一致性未在本页列出。'
                '多模型一致只说明几个模型之间差异小，一致不代表准确。</p>')
    validation = _map(card.get("validation"))
    holdout = validation.get("holdout_n")
    agree_head = f"与 CFD 的一致性（{holdout} 例留出）" if isinstance(holdout, int) else "与 CFD 的一致性"
    rows = [[_e(row["label"]), f'<span class="tier">{_e(row["tier_label"])}</span>', _e(row.get("agreement") or "—"),
             _e(row.get("usage") or "—")] for row in validation_rows(card)]
    training = _map(card.get("training"))
    parts = []
    if isinstance(training.get("n_train"), int):
        cohorts = "、".join(str(c) for c in training.get("cohorts") or [])
        parts.append(f"训练 {training['n_train']} 例" + (f"（{cohorts}）" if cohorts else "")
                     + (f"，留出 {training['n_holdout']} 例" if isinstance(training.get("n_holdout"), int) else ""))
    notes = [str(validation.get("r2_note") or ""), str(validation.get("note") or ""), "；".join(parts),
             "多模型一致只说明几个模型之间差异小，一致不代表准确；与 CFD 的一致性以上表为准。",
             str(card.get("usage_note") or "")]
    protocol = [str(item) for item in card.get("protocol") or [] if isinstance(item, str)]
    weak = [str(item) for item in card.get("weaknesses") or [] if isinstance(item, str)]
    body = _table(["量", "来源", agree_head, "怎么用"], rows, klass="validation", raw_headers=False)
    body += "".join(f'<p class="muted">{_e(note)}</p>' for note in notes if note.strip())
    if protocol:
        body += '<h3>CFD 协议假设</h3><ul class="limits">' + "".join(f"<li>{_e(item)}</li>" for item in protocol) + "</ul>"
    if weak:
        body += '<h3>已知薄弱处</h3><ul class="limits">' + "".join(f"<li>{_e(item)}</li>" for item in weak) + "</ul>"
    title = " · ".join(str(x) for x in (card.get("display_name"), card.get("version_date")) if x)
    return f'<h2>模型验证<small>{_e(title)}</small></h2>' + body


def _glossary_section(mode: str, used: list[str]) -> str:
    from .glossary import GLOSSARY
    if mode == "none":
        return ""
    keys = list(GLOSSARY) if mode == "all" else [key for key in dict.fromkeys(used) if key in GLOSSARY]
    if not keys:
        return ""
    rows = [[_e(GLOSSARY[key].get("zh") or key), _e(GLOSSARY[key].get("zh_desc") or "")] for key in keys]
    note = "本页用到的术语" if mode == "used" else "全部术语"
    return f'<h2>术语说明<small>{note}</small></h2>' + _table(["术语", "解释"], rows, klass="glossary")


# ----------------------------------------------------------------------------- page
CSS = """
/* S5b (2026-09-30): the workspace v2 look on paper — v2 tokens and type, ruled strips instead of tinted cards, status words
   with a dot, one accent.  --ink-3 / --line are a step darker than on screen so 8 px notes and hairlines survive printing. */
:root{--ink:#101828;--ink-2:#475467;--ink-3:#6b7587;--line:#e4e7ec;--line-2:#d0d5dd;--panel-2:#f7f8fa;--bg:#f2f4f7;
--accent:#0e6e8a;--accent-2:#0a566c;--warn:#c4570d;--error:#b42318;--ok:#0f8a5f;
--font:"Inter","SF Pro Text","Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,"PingFang SC","Microsoft YaHei","Noto Sans CJK SC",sans-serif}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:10px/1.45 var(--font);font-variant-numeric:tabular-nums;-webkit-font-smoothing:antialiased}
.page{width:210mm;min-height:297mm;margin:0 auto 10mm;background:#fff;padding:12mm 13mm 9mm;border:1px solid var(--line);box-shadow:0 1px 2px rgba(16,24,40,.05);display:flex;flex-direction:column}
.page>footer.pf{margin-top:auto}.page>:nth-last-child(2){margin-bottom:10px}
header.top{display:grid;grid-template-columns:auto minmax(0,1fr) auto;column-gap:10px;align-items:end;padding-bottom:7px;border-bottom:1.5px solid var(--ink)}
header.top .logo{align-self:center;color:var(--accent)}
.inst{margin:0 0 1px;font-size:9.5px;font-weight:600;color:var(--accent-2);letter-spacing:.02em}
h1{margin:0;font-size:17px;line-height:1.25;font-weight:700;letter-spacing:-.005em}
.sub{margin:2px 0 0;font-size:9px;color:var(--ink-3)}
.status{text-align:right;font-size:8.5px;color:var(--ink-3);line-height:1.65;white-space:nowrap}
.status strong{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:600;color:var(--ink-2)}
.status strong::before,.card.q strong::before,.record::before,.sev::before{content:"";flex:none;width:7px;height:7px;border-radius:50%;border:1.5px solid var(--ink-3);box-sizing:border-box}
.status strong.ok{color:var(--ok)}.status strong.ok::before{background:var(--ok);border-color:var(--ok)}
.status strong.warn{color:var(--warn)}.status strong.warn::before{background:var(--warn);border-color:var(--warn)}
.idline{display:flex;flex-wrap:wrap;align-items:center;row-gap:2px;margin:6px 0 0;padding:5px 10px;background:var(--panel-2);font-size:9px;color:var(--ink-3)}
.idline .idf{white-space:nowrap}.idline b{margin-left:2px;color:var(--ink);font-size:10px;font-weight:600}
.idline .sep{display:inline-block;flex:none;width:1px;height:10px;margin:0 10px;background:var(--line-2);font-size:0;line-height:0;overflow:hidden}
h2{display:flex;align-items:baseline;gap:6px;margin:12px 0 4px;font-size:11.5px;font-weight:700;color:var(--ink);break-after:avoid;page-break-after:avoid}
h2 small{margin-left:auto;font-size:8px;font-weight:400;color:var(--ink-3);text-align:right}
h3{display:flex;align-items:baseline;gap:6px;font-size:10px;font-weight:700;margin:9px 0 3px;color:var(--ink-2);break-after:avoid;page-break-after:avoid}
h3 small{font-size:8px;font-weight:400;color:var(--ink-3)}
.muted{color:var(--ink-3);font-size:9px;margin:3px 0}
.g{text-decoration:underline dotted var(--line-2);text-underline-offset:2px}
.badge{display:inline-block;padding:0 5px;border:1px solid currentColor;border-radius:2px;font-size:8px;font-weight:500;line-height:1.5;color:var(--warn);vertical-align:1px;white-space:nowrap}
.badge.auto{color:var(--ink-3);border-color:var(--line-2)}
.narrative{padding:1px 0 1px 10px;border-left:2px solid var(--accent)}.narrative p{margin:0 0 2px;font-size:10.5px;line-height:1.65}.narrative p.muted{font-size:8.5px;margin-top:3px}
.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));margin:2px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.cards.c1{grid-template-columns:minmax(0,1fr)}.cards.c2{grid-template-columns:repeat(2,minmax(0,1fr))}.cards.c3{grid-template-columns:repeat(3,minmax(0,1fr))}.cards.c5{grid-template-columns:repeat(5,minmax(0,1fr))}
.card{min-width:0;padding:6px 10px 7px;border-left:1px solid var(--line)}
.cards.c1 .card,.cards.c2 .card:nth-child(2n+1),.cards.c3 .card:nth-child(3n+1),.cards.c4 .card:nth-child(4n+1),.cards.c5 .card:nth-child(5n+1){border-left:0;padding-left:0}
.cards.c3 .card:nth-child(n+4),.cards.c4 .card:nth-child(n+5){border-top:1px solid var(--line)}
.card .label{display:block;font-size:8.5px;font-weight:500;color:var(--ink-3)}
.card strong{display:block;margin:2px 0 1px;font-size:16px;line-height:1.2;font-weight:700;letter-spacing:-.01em;color:var(--ink)}
.card small{display:block;font-size:8px;line-height:1.4;color:var(--ink-3);overflow-wrap:anywhere}
.card.q strong{display:flex;align-items:center;gap:5px;margin:5px 0 3px;font-size:12px;font-weight:600;letter-spacing:0}
.card.q.ok strong{color:var(--ok)}.card.q.ok strong::before{background:var(--ok);border-color:var(--ok)}
.card.q.warn strong{color:var(--warn)}.card.q.warn strong::before{background:var(--warn);border-color:var(--warn)}
.tier{display:inline-block;margin-left:4px;padding:0 3px;border:1px solid var(--line-2);border-radius:2px;font-size:7.5px;line-height:1.4;font-weight:500;color:var(--ink-2);background:none;vertical-align:1px;white-space:nowrap}
p.tiers{font-size:8px}
.shots{display:grid;gap:6px 8px;margin:3px 0}.shots.cols2{grid-template-columns:repeat(2,1fr)}.shots.cols3{grid-template-columns:repeat(3,1fr)}.shots.cols4{grid-template-columns:repeat(4,1fr)}
.shots figure{margin:0;min-width:0;page-break-inside:avoid;break-inside:avoid}.shots img{display:block;width:100%;max-width:100%;height:auto;max-height:40mm;object-fit:contain;border:1px solid var(--line);border-radius:2px;background:#fff}
.shots.cols2 img{max-height:41mm}.shots.cols4 img{max-height:31mm}
.shots figcaption{margin-top:2px;font-size:8px;color:var(--ink-3);overflow-wrap:anywhere}
table{width:100%;border-collapse:collapse;font-size:9px;margin:2px 0}
th,td{padding:3px 8px 3px 0;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th:last-child,td:last-child{padding-right:0}
th{font-size:8px;font-weight:500;color:var(--ink-3);white-space:nowrap;border-bottom-color:var(--line-2)}
table.num td:not(:first-child),table.num th:not(:first-child){text-align:right;font-variant-numeric:tabular-nums}
table.wide{font-size:8px}table.wide td,table.wide th{padding:2px 5px 2px 0}table.narrow{width:auto;min-width:45%;align-self:flex-start}
table.top td{vertical-align:middle}table.top td:nth-child(1){width:8mm;color:var(--ink-3);font-size:8.5px}table.top td:nth-child(2){font-weight:500;color:var(--ink)}table.top td:nth-child(3){color:var(--ink-2)}
table.top td:nth-child(4),table.top th:nth-child(4){text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}table.top td:nth-child(n+5){white-space:nowrap}
table.findings td:nth-child(2){min-width:30mm}table.findings td:nth-child(7){font-size:8px;color:var(--ink-2)}table.findings td:nth-child(n+3):nth-child(-n+6){white-space:nowrap}table.findings.legacy td:nth-child(6){white-space:normal;font-size:8px;color:var(--ink-2)}
tr.manual td:nth-child(2){color:var(--accent-2)}tr.d-rejected td{color:var(--ink-3)}
table.glossary{font-size:8px}table.glossary td:first-child{width:24%;white-space:nowrap;color:var(--ink)}
table.validation td:nth-child(2){white-space:nowrap}td>.tier:first-child{margin-left:0}
.sev{display:inline-flex;align-items:center;gap:4px;font-size:8.5px;color:var(--ink-2);white-space:nowrap}.sev::before{width:6px;height:6px;border-width:1.3px}
.sev.attention{color:var(--warn)}.sev.attention::before{background:var(--warn);border-color:var(--warn)}.sev.note::before{border-color:var(--ink-2)}
.sev.info{color:var(--ink-3)}.sev.info::before{background:var(--line-2);border-color:var(--line-2)}.sev.geo::before{border-radius:1px}
.dec{white-space:nowrap;color:var(--ink-3)}.dec.confirmed{color:var(--ok);font-weight:600}.dec.rejected{color:var(--ink-2)}
dl.kv{display:grid;grid-template-columns:auto 1fr auto 1fr;gap:2px 12px;margin:2px 0;font-size:9px}dl.kv dt{color:var(--ink-3);white-space:nowrap}dl.kv dd{margin:0;overflow-wrap:anywhere}
ol.limits,ul.limits{margin:2px 0 0 16px;padding:0;font-size:9px;line-height:1.5;color:var(--ink-2)}ol.limits li,ul.limits li{margin:1px 0}
.sign{display:grid;grid-template-columns:repeat(2,1fr);gap:12px 24px;margin:18px 0 2px;font-size:9.5px;break-inside:avoid}.sign .line{display:flex;align-items:flex-end;gap:6px}
.sign .who{white-space:nowrap;color:var(--ink-2)}.sign .blank{flex:1;border-bottom:1px solid var(--ink);height:16px}.sign .blank.short{flex:0 0 24mm}
.record{display:flex;align-items:center;gap:6px}.record.ok{color:var(--ok)}.record.ok::before{background:var(--ok);border-color:var(--ok)}.record.warn{color:var(--warn)}.record.warn::before{background:var(--warn);border-color:var(--warn)}
footer.pf{border-top:1px solid var(--line);padding-top:4px;font-size:8px;color:var(--ink-3);display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}
.appendix-title{margin:0 0 2px;padding-bottom:6px;font-size:14px;border-bottom:1.5px solid var(--ink)}
.toolbar{position:sticky;top:0;z-index:2;display:flex;justify-content:flex-end;padding:10px max(12px,calc((100% - 210mm) / 2));background:rgba(242,244,247,.92)}
.toolbar button{display:inline-flex;align-items:center;gap:6px;height:30px;padding:0 14px;border:1px solid var(--accent);border-radius:6px;background:var(--accent);color:#fff;font:500 12px/1 var(--font);cursor:pointer}
.toolbar button:hover{background:var(--accent-2);border-color:var(--accent-2)}.toolbar button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}.toolbar .ic{flex:none}
@media (pointer:coarse){.toolbar button{min-height:44px}}
@media screen and (max-width:840px){.page{width:auto;min-height:0;margin:0 0 12px;padding:16px;border-left:0;border-right:0}
.cards.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.cards .card{border-left:0!important;padding-left:0!important;border-top:1px solid var(--line)!important}
.cards .card:nth-child(-n+2){border-top:0!important}.cards .card:nth-child(2n){padding-left:10px!important;border-left:1px solid var(--line)!important}
.shots.cols3,.shots.cols4{grid-template-columns:repeat(2,1fr)}dl.kv{grid-template-columns:auto 1fr}table.wide,table.findings{display:block;overflow-x:auto}.sign{grid-template-columns:1fr}}
@page{size:A4;margin:10mm 10mm 11mm;@bottom-right{content:"第 " counter(page) " 页";font-size:8px;color:#6b7587}}
@media print{body{background:#fff}.page{width:auto;min-height:0;margin:0;padding:0;border:0;box-shadow:none;display:block}.page>footer.pf{margin-top:8px}.page+.page{break-before:page;page-break-before:always}
.page.first{display:flex;min-height:272mm}.page.first>footer.pf{margin-top:auto}
h2{margin:9px 0 3px}th,td{padding-top:2px;padding-bottom:2px}.sign{margin-top:14px}
.toolbar,.noprint{display:none}tr,.card,.sign,figure,.more-shots{break-inside:avoid;page-break-inside:avoid}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
/* v0.14 (F6): no pictures on page 1 → the appendix continues right below page 1 instead of leaving half a page blank */
.snap-hint{display:flex;align-items:baseline;gap:10px;margin:8px 0 2px}.snap-hint h2{margin:0;flex:0 0 auto}.snap-hint p{margin:0}
body.flow .page.first{min-height:0;margin-bottom:0;padding-bottom:5mm}body.flow .page.first>footer.pf{display:none}
body.flow .page.appendix{margin-top:0;padding-top:5mm;min-height:0;border-top:1px dashed var(--line-2)}
@media print{body.flow .page+.page{break-before:auto;page-break-before:auto}body.flow .page.first{display:block}body.flow .page.appendix{border-top:0;padding-top:4mm}}
"""


def _auto_title(summary: Mapping[str, Any]) -> str:
    if family_of(summary) == "volume":
        return "压力与速度体场一页纸报告"
    return "壁面 WSS / TAWSS / OSI 一页纸报告" if has_cycle(summary) else "壁面 WSS 一页纸报告"


def render_onepage(summary: Mapping[str, Any], job: Mapping[str, Any] | None = None, *,
                   job_dir: Path | str | None = None, snapshots: Mapping[str, Any] | None = None,
                   template: Mapping[str, Any] | None = None, timeline: Mapping[str, Any] | None = None,
                   model_card: Mapping[str, Any] | None = None) -> str:
    """Render the A4 page (+ appendix) as a self-contained HTML string (no scripts, no external resources).

    ``job_dir`` enables the 配图 section (PNGs listed in ``snapshots.json`` are embedded as data URIs) and
    selects the institution template (``report_template.load_for_job``); ``template`` overrides it.
    ``timeline`` is the §19.3 patient timeline; with ≥ 2 scans page 1 carries a 「随访变化」 table.
    ``model_card`` (C3) overrides the release's card (``model_cards.load``) for the tiers and the 「模型验证」 table.
    """
    summary = _map(summary)
    job = _map(job)
    template = _template(job_dir, template)
    title = template.get("report_title") or _auto_title(summary)
    case_name = display_name(summary, job)          # C7: patient id when entered, else the case name
    card = model_card if isinstance(model_card, Mapping) else model_card_for(summary, job)
    figures = _snapshot_figures(Path(job_dir) if job_dir else None, snapshots)
    first = "".join([
        _header(summary, job, template, title), _identity_line(summary, job, card), _narrative_section(summary),
        _numbers_section(summary, job, card), _snapshots_section(figures), _morphology_section(summary, card),
        _timeline_section(timeline, summary), _top_findings_section(summary),
        _signature_section(summary, job, template), _footer(summary, job, template, "第 1 页", card),
    ])
    pages = [f'<article class="page first">{first}</article>']
    if template.get("appendix", True):
        more = figures[FIRST_PAGE_SNAPSHOTS:]
        body = "".join([
            f'<h1 class="appendix-title">附录 · {_e(case_name)}</h1>', _branch_table(summary), _findings_section(summary),
            _annotations_section(summary), ('<section class="more-shots"><h2>更多配图</h2>' + _shots(more) + "</section>") if more else "",
            _input_section(summary), _morphology_details(summary), _trust_section(summary),
            _validation_section(card, summary),
            '<h2>局限性声明</h2><ol class="limits">' + "".join(f"<li>{_e(item)}</li>" for item in limitations(summary)) + "</ol>",
            _timing_section(summary), _identity_section(summary, job),
        ])
        used = re.findall(r'data-gloss="([a-z0-9_]+)"', first + body)
        body += _glossary_section(str(template.get("show_glossary") or "used"), used)
        pages.append(f'<article class="page appendix">{body}{_footer(summary, job, template, "附录", card)}</article>')
    # The handler text must stay exactly "window.print()": the one-page CSP allows only its hash (server.ONEPAGE_HANDLER_HASH).
    toolbar = f'<div class="toolbar"><button type="button" onclick="window.print()">{PRINT_SVG}打印 / 保存 PDF</button></div>'
    # Layout with pictures is unchanged: page 1 is a full A4 sheet and the appendix starts on a new one.
    body_class = ' class="flow"' if not figures and len(pages) > 1 else ""
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_e(title)} · {_e(case_name)}</title><style>{CSS}</style></head><body{body_class}>{toolbar}{"".join(pages)}</body></html>')


__all__ = ["DECISION_LABELS", "LIMITATIONS", "LIMIT_FLOW", "LIMIT_LUMEN", "NO_SNAPSHOTS_HINT", "QUALITY_CAVEAT", "QUALITY_DISPLAY",
           "display_name", "family_of", "has_cycle", "limitations", "model_card_for", "model_count", "render_onepage", "top_findings"]
