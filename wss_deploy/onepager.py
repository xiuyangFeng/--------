"""One-page (A4) printable report generated from ``summary.json``.

The page carries no 3-D content: identity and provenance, input check, geometry table, key
numbers, findings, trust coverage, review state and the fixed limitations statement.  It is
rendered on demand by the service (``GET /api/jobs/<id>/onepage``) and never written to the
job directory, so the exported result files stay exactly what the pipeline produced.
Everything is escaped with :func:`html.escape`; the page contains no scripts.
"""
from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path
from typing import Any, Mapping

SNAPSHOT_FILE = re.compile(r"snapshot_[a-z0-9_-]{1,32}\.png")
MAX_SNAPSHOT_IMAGES = 12
NO_SNAPSHOTS_HINT = "在三维报告「视图」菜单点「生成一页纸配图」可为本页配图。"

LIMITATIONS = (
    "预测对象是固定收缩期单帧（step 1162，约 0.21 s），不是全周期；没有 TAWSS、OSI 等周期量。",
    "压力为相对量（相对于该帧体积平均压力），不能解释为绝对血压；只有压差有意义。",
    "面积与占比按预测点占比乘输入壁面面积估计，统计按点等权，不是体积或面积积分。",
    "模型只在腹主动脉—髂动脉五开口几何上训练；其它血管或超出参考范围的几何不适用。",
    "输入 STL 不含患者方向；左右语义按解剖坐标架推断并经人工确认，请结合原始影像核对。",
)

REVIEW_LABELS = {"unreviewed": "未审阅", "reviewed": "已审阅签字", "reopened": "已重新打开"}
SEVERITY_LABELS = {"attention": "关注", "note": "提示", "info": "几何"}
TRUST_LABELS = {"interpolation_uncovered": "插值无支撑", "rough_surface": "表面粗糙", "geometry_out_of_range": "几何越界",
                "low_sample_support": "采样支撑弱", "near_opening": "靠近切口"}


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _num(value: Any, digits: int = 2, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number or number in (float("inf"), float("-inf")):
        return "—"
    return f"{number:.{digits}f}{suffix}"


def _pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{100.0 * float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def family_of(summary: Mapping[str, Any]) -> str:
    fields = _map(summary.get("fields"))
    return "volume" if ("velocity" in fields or "pressure" in fields) else "wall"


def _table(headers: list[str], rows: list[list[str]], *, klass: str = "") -> str:
    if not rows:
        return '<p class="muted">无数据</p>'
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<table class="{_e(klass)}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _kv(pairs: list[tuple[str, str]]) -> str:
    return '<dl class="kv">' + "".join(f"<dt>{_e(k)}</dt><dd>{v}</dd>" for k, v in pairs) + "</dl>"


def _identity_section(summary: Mapping[str, Any], job: Mapping[str, Any]) -> str:
    meta = _map(summary.get("case_metadata"))
    release = _map(summary.get("model_release"))
    contract = _map(summary.get("feature_contract"))
    tags = meta.get("tags") if isinstance(meta.get("tags"), list) else []
    pairs = [
        ("匿名病例编号", _e(summary.get("case_id") or job.get("case_id") or "—")),
        ("匿名患者编号", _e(meta.get("patient_id") or job.get("patient_id") or "—")),
        ("扫描标签 / 日期", _e(" / ".join(x for x in (meta.get("scan_label") or job.get("scan_label") or "", meta.get("scan_date") or job.get("scan_date") or "") if x) or "—")),
        ("标签", _e("、".join(str(t) for t in tags) or "—")),
        ("任务编号", _e(job.get("id") or "—")),
        ("生成时间", _e(summary.get("created_at") or "—")),
        ("发布包", _e(release.get("registry_id") or release.get("release") or release.get("name") or summary.get("release") or "—")),
        ("发布包指纹", _e(str(release.get("fingerprint") or summary.get("release_hash") or "—")[:16])),
        ("特征程序合同", _e(f"{contract.get('version') or '—'} · {str(contract.get('source_hash') or '—')[:16]}")),
        ("运行身份", _e(str(summary.get("run_identity") or job.get("run_identity") or "—")[:16])),
        ("时间帧", _e(", ".join(str(f.get("label") or f.get("step")) for f in summary.get("time_axis") or [] if isinstance(f, Mapping)) or "固定收缩期单帧")),
    ]
    return _kv(pairs)


def _input_section(summary: Mapping[str, Any]) -> str:
    ic = _map(summary.get("input_check"))
    quality = _map(ic.get("quality"))
    bbox = ic.get("bbox_size_mm") if isinstance(ic.get("bbox_size_mm"), list) else []
    pairs = [
        ("单位", _e(ic.get("unit") or ic.get("resolved_units") or "—")),
        ("顶点 / 面片", _e(f"{ic.get('vertices', '—')} / {ic.get('faces', '—')}")),
        ("壁面面积", _num(float(ic["area_mm2"]) / 100.0, 1, " cm²") if ic.get("area_mm2") is not None else "—"),
        ("包围盒 (mm)", _e(" × ".join(_num(v, 0) for v in bbox) if bbox else "—")),
        ("开口数 / 连通片", _e(f"{ic.get('openings', '—')} / {ic.get('components', '—')}")),
        ("质量评分卡", _e(quality.get("grade_label") or quality.get("grade") or "—")),
    ]
    flags = [str(f) for f in (ic.get("flags") or [])]
    extra = f'<p class="muted">{_e("；".join(flags))}</p>' if flags else ""
    return _kv(pairs) + extra


def _geometry_section(summary: Mapping[str, Any]) -> str:
    geometry = _map(summary.get("geometry"))
    rows = []
    for name, row in geometry.items():
        row = _map(row)
        rows.append([_e(name), _num(row.get("length_mm"), 0), _num(row.get("radius_min_mm"), 1),
                     _num(row.get("radius_median_mm"), 1), _num(row.get("max_diameter_mm"), 1),
                     _num(row.get("stenosis_index"), 2), _num(row.get("tortuosity"), 2)])
    return _table(["分支", "长度 mm", "最小半径", "中位半径", "最大直径", "狭窄指数", "迂曲度"], rows, klass="compact")


def _wall_numbers(summary: Mapping[str, Any]) -> str:
    peak = _map(summary.get("peak"))
    field = _map(summary.get("wss_field_pa"))
    thresholds = field.get("thresholds_pa") if isinstance(field.get("thresholds_pa"), list) else [0.4, 4.0, 7.0]
    cards = [
        ("全场 p99", _num(peak.get("p99_pa"), 2, " Pa"), "预测点云空间第 99 百分位"),
        ("全场最大值", _num(peak.get("max_pa"), 2, " Pa"), _e(f"位置：{peak.get('branch') or '—'}，距入口 {_num(peak.get('s_from_inlet_mm'), 0)} mm")),
        (f"低 WSS 面积 (< {_num(thresholds[0], 1)} Pa)", _pct(field.get("area_frac_low")), _num(field.get("area_low_mm2", 0) / 100.0 if field.get("area_low_mm2") is not None else None, 0, " cm²")),
        (f"高 WSS 面积 (> {_num(thresholds[1], 0)} Pa)", _pct(field.get("area_frac_high")), _e(f"> {_num(thresholds[2], 0)} Pa：{_pct(field.get('area_frac_very_high'))}")),
    ]
    quality = _map(summary.get("quality"))
    cards.append(("集成质量", _e(quality.get("label") or quality.get("level") or "未评估"), _e("；".join(quality.get("reasons") or []) or "五模型离散度在常规范围内")))
    return _cards(cards)


def _wall_branches(summary: Mapping[str, Any]) -> str:
    rows = []
    for name, row in _map(summary.get("per_branch")).items():
        row = _map(row)
        rows.append([_e(name), _num(row.get("wss_mean_pa"), 2), _num(row.get("wss_p99_pa"), 2), _num(row.get("wss_max_pa"), 1),
                     _pct(row.get("frac_low"), 0), _pct(row.get("frac_high"), 0), _num(row.get("area_mm2", 0) / 100.0 if row.get("area_mm2") is not None else None, 0)])
    return _table(["分支", "均值 Pa", "p99 Pa", "最大 Pa", "低 WSS", "高 WSS", "面积 cm²"], rows, klass="compact")


def _volume_numbers(summary: Mapping[str, Any]) -> str:
    stats = _map(summary.get("volume_statistics"))
    speed, interior, wall = _map(stats.get("speed_m_s")), _map(stats.get("pressure_interior_pa")), _map(stats.get("pressure_wall_pa"))
    cards = [
        ("体内速度 p99", _num(speed.get("p99"), 3, " m/s"), _e(f"均值 {_num(speed.get('mean'), 3)}，最大 {_num(speed.get('max'), 3)} m/s")),
        ("体内相对压力范围", _e(f"{_num(interior.get('min'), 0)} ～ {_num(interior.get('max'), 0)} Pa"), "相对于该帧体积平均压力"),
        ("壁面相对压力范围", _e(f"{_num(wall.get('min'), 0)} ～ {_num(wall.get('max'), 0)} Pa"), "壁面查询点，Gaussian 插值到顶点"),
    ]
    lines = _map(summary.get("streamlines"))
    if lines:
        cards.append(("流线", _e(f"{lines.get('line_count', '—')} 条"), "固定帧稳态积分，非粒子轨迹"))
    drops = [item for item in _findings(summary) if item.get("kind") == "pressure_drop"]
    out = _cards(cards)
    if drops:
        rows = [[_e(item.get("branch") or item.get("label")), _num(item.get("value"), 1), _num(float(item["value"]) / 133.322, 2) if item.get("value") is not None else "—",
                 _e(item.get("definition") or "")] for item in drops]
        out += "<h3>各分支近远端压差</h3>" + _table(["分支", "ΔP Pa", "ΔP mmHg", "口径"], rows, klass="compact")
    return out


def _narrative_section(summary: Mapping[str, Any]) -> str:
    """「结论（参考）」: the reviewer's text when present (badged), else the generated sentences."""
    from .narrative import display_text
    block = _map(summary.get("narrative"))
    edited = block.get("edited")
    lines = block.get("zh") if isinstance(block.get("zh"), list) else []
    if not (isinstance(edited, str) and edited.strip()) and not lines:
        return ""
    if isinstance(edited, str) and edited.strip():
        body = "".join(f"<p>{_e(line)}</p>" for line in edited.strip().splitlines() if line.strip())
        badge = '<span class="badge">审阅人已修改</span>'
        who = _map(summary.get("narrative")).get("edited_by")
        note = f'<p class="muted">{_e(str(who))} · {_e(str(block.get("edited_at") or ""))}</p>' if who else ""
    else:
        body = "".join(f"<p>{_e(line)}</p>" for line in lines)
        badge = '<span class="badge auto">自动生成</span>'
        note = '<p class="muted">由本页数字自动组织，只陈述存在的量，未做诊断判断。</p>'
    return f'<h2>结论（参考）{badge}</h2><div class="narrative">{body}{note}</div>'


def _morphology_section(summary: Mapping[str, Any]) -> str:
    """「瘤体形态」: largest section, sac, neck, lumen volume and the reference diameter."""
    morphology = _map(summary.get("morphology"))
    aorta = _map(morphology.get("aorta"))
    largest, sac, neck = _map(aorta.get("max")), _map(aorta.get("sac")), _map(aorta.get("neck"))
    if not largest:
        return ""
    method = _map(morphology.get("method"))
    station = morphology.get("station_mm") or 1.0
    cards = [
        ("最大直径", _num(largest.get("max_diameter_mm"), 1, " mm"),
         _e(f"距入口 {_num(largest.get('distance_from_inlet_mm'), 0)} mm；等效直径 {_num(largest.get('equivalent_diameter_mm'), 1)} mm")),
        ("瘤体", (_e(f"{_num(sac.get('length_mm'), 0)} mm") if sac.get("present") else "未见"),
         _e(f"体积约 {_num(sac.get('volume_ml'), 0)} mL；判据 ≥ {_num(sac.get('threshold_mm'), 1)} mm" if sac.get("present")
            else f"主动脉最大等效直径 < 1.5 × 参考直径 {_num(aorta.get('reference_diameter_mm'), 1)} mm")),
        ("近端瘤颈", (_e(f"{_num(neck.get('length_mm'), 0)} mm") if neck.get("present") else "未给出"),
         _e(f"平均直径 {_num(neck.get('diameter_mean_mm'), 1)} mm（{_num(neck.get('diameter_min_mm'), 1)}–{_num(neck.get('diameter_max_mm'), 1)}）"
            if neck.get("present") else "瘤体近端无持续 < 1.2 × 参考直径的区段")),
        ("全腔体积", _num(morphology.get("lumen_volume_ml"), 0, " mL"),
         _e(f"参考直径 {_num(aorta.get('reference_diameter_mm'), 1)} mm（等效直径第 10 百分位）")),
    ]
    explanation = _e(f"沿中心线每 {_num(station, 0)} mm 取一站，以局部切线为法向切壁面网格；"
                     f"{method.get('max_diameter') or '最大直径为轮廓上最远两点的距离'}，"
                     f"{method.get('equivalent_diameter') or '等效直径 = 2·sqrt(面积/π)'}。")
    notes = [str(item) for item in (morphology.get("notes") or []) if isinstance(item, str)]
    warning = f'<p class="muted">{_e("；".join(notes))}</p>' if notes else ""
    return f'<h2>瘤体形态</h2>{_cards(cards)}<p class="muted">{explanation}</p>{warning}'


def _morphology_branches(summary: Mapping[str, Any]) -> str:
    """「血管分支表」 (contract §17.1): length, tortuosity, diameter range and the family's own column."""
    morphology = _map(summary.get("morphology"))
    branches = [row for row in (morphology.get("branches") or []) if isinstance(row, Mapping)]
    if not branches:
        return ""
    family = family_of(summary)
    if family == "wall":
        extra = ["p99 Pa", "均值 Pa", "低 WSS", "高 WSS"]
        cells = lambda row: [_num(row.get("wss_p99_pa"), 2), _num(row.get("wss_mean_pa"), 2),
                             _pct(row.get("area_frac_low"), 0), _pct(row.get("area_frac_high"), 0)]
    else:
        extra = ["ΔP Pa", "速度均值 m/s", "速度最大 m/s", ""]
        cells = lambda row: [_num(row.get("delta_p_pa"), 1), _num(row.get("speed_mean_m_s"), 3),
                             _num(row.get("speed_max_m_s"), 3), ""]
    rows = [[_e(row.get("name") or row.get("segment_id")), _num(row.get("length_mm"), 0), _num(row.get("tortuosity"), 2),
             _e(f"{_num(row.get('diameter_min_mm'), 1)}–{_num(row.get('diameter_max_mm'), 1)}"),
             _num(row.get("diameter_mean_mm"), 1), *cells(row)] for row in branches]
    headers = ["分支", "长度 mm", "扭曲度", "直径范围 mm", "平均直径 mm", *extra]
    return ('<h2>血管分支表</h2>' + _table(headers, rows, klass="compact")
            + '<p class="muted">直径为壁面网格截面的最大 Feret 直径；扭曲度 = 中心线弧长 / 两端直线距离。'
              '分叉附近的截面会同时切到母血管，该处直径偏大。</p>')


def _findings(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    items = _map(summary.get("findings")).get("items")
    if not isinstance(items, list):
        return []
    items = [item for item in items if isinstance(item, Mapping)]
    return sorted(items, key=lambda item: (item.get("rank") if isinstance(item.get("rank"), (int, float)) else 1e9))


DECISION_LABELS = {"confirmed": "☑ 已确认", "rejected": "✕ 已驳回", None: "☐ 未判定"}


def _findings_review(summary: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    """Reviewer decisions (contract §11.10): ``findings.review.items`` keyed by finding id, plus manual additions."""
    review = _map(_map(summary.get("findings")).get("review"))
    decisions = review.get("items") if isinstance(review.get("items"), Mapping) else {}
    added = [item for item in (review.get("added") or []) if isinstance(item, Mapping)]
    return decisions, added


def _finding_row(item: Mapping[str, Any], decision: Mapping[str, Any] | None, *, manual: bool = False) -> list[str]:
    value = item.get("value")
    units = item.get("units") or ""
    label = item.get("label") or item.get("text") or item.get("kind") or ""
    note = (decision or {}).get("note") or ""
    definition = item.get("definition") or (item.get("text") if manual else "") or ""
    return [_e(item.get("id") or ""), _e(("【人工】" if manual else "") + str(label)), _e(item.get("branch") or "—"),
            _e(f"{_num(value, 2)} {units}".strip()) if value is not None else "—",
            _e(SEVERITY_LABELS.get(str(item.get("severity")), item.get("severity") or "")),
            _e(DECISION_LABELS.get((decision or {}).get("decision"), "☐ 未判定")),
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
    rows = [_finding_row(item, decision) for item, decision in body[:14]]
    rows += [_finding_row(item, {"decision": "confirmed", "note": ""}, manual=True) for item in added]
    more = f'<p class="muted">另有 {len(body) - 14} 项未列出，见三维报告。</p>' if len(body) > 14 else ""
    out = '<h2>发现列表</h2>' + _table(headers, rows, klass="compact findings") + more
    if decisions or added:
        out += '<p class="muted">☑ 审阅人已确认；☐ 尚未判定；【人工】为审阅人手动添加。驳回项见附录。</p>'
    if rejected:
        out += '<h3>附录：已驳回的自动发现</h3>' + _table(headers, [_finding_row(item, decision) for item, decision in rejected], klass="compact findings")
    return out


def _annotations_section(summary: Mapping[str, Any]) -> str:
    items = [item for item in (_map(summary.get("annotations")).get("items") or []) if isinstance(item, Mapping)]
    if not items:
        return ""
    rows = [[_e(item.get("id") or ""), _e(item.get("branch") or "—"), _num(item.get("s_from_root_mm"), 0), _e(item.get("text") or "")]
            for item in items]
    return '<h2>标注</h2>' + _table(["编号", "分支", "弧长 mm", "文字"], rows, klass="compact findings")


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


def _snapshots_section(job_dir: Path | None, snapshots: Any) -> str:
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
    if not figures:
        return f'<h2>配图</h2><p class="muted">{_e(NO_SNAPSHOTS_HINT)}</p>'
    columns = 2 if len(figures) <= 2 else 3
    return f'<h2>配图</h2><div class="shots cols{columns}">' + "".join(figures) + "</div>"


def _glossary_section() -> str:
    from .glossary import GLOSSARY
    rows = [[_e(term.get("zh") or key), _e(term.get("zh_desc") or "")] for key, term in GLOSSARY.items()]
    return '<h2>术语说明</h2>' + _table(["术语", "解释"], rows, klass="compact findings glossary")


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
        rows.append([_e(labels.get(str(bit)) or TRUST_LABELS.get(str(key), key)), _pct(value)])
    return '<h2>可信区域</h2><p class="muted">下列比例是模型结果中应谨慎解读的区域占比（按顶点或采样点计）。</p>' + _table(["原因", "占比"], rows, klass="compact")


def _review_section(summary: Mapping[str, Any], job: Mapping[str, Any]) -> str:
    review = _map(summary.get("review")) or _map(job.get("review"))
    status = str(review.get("status") or "unreviewed")
    pairs = [("状态", _e(REVIEW_LABELS.get(status, status)))]
    if review.get("by"):
        pairs.append(("审阅人", _e(review.get("by"))))
    if review.get("at"):
        pairs.append(("时间", _e(review.get("at"))))
    if review.get("note"):
        pairs.append(("备注", _e(review.get("note"))))
    if review.get("version") is not None:
        pairs.append(("锁定版本", _e(review.get("version"))))
    klass = "review-box " + ("ok" if status == "reviewed" else "warn")
    return f'<div class="{klass}">{_kv(pairs)}</div>'


def _timing_section(summary: Mapping[str, Any]) -> str:
    timing = _map(summary.get("timing_s"))
    names = {"ingest": "输入检查", "centerline": "中心线", "smooth_resample": "平滑重采样", "features": "几何特征",
             "volume_features": "体场特征", "inference_5_models": "模型推理", "inference_volume": "模型推理",
             "metrics_and_interpolation": "统计与插值", "streamlines_and_interpolation": "流线与插值", "export": "导出", "total": "合计"}
    parts = [f"{names.get(k, k)} {_num(v, 1)} s" for k, v in timing.items() if isinstance(v, (int, float))]
    device = summary.get("device")
    gpu = summary.get("gpu")
    device_text = f"{device}{' · ' + str(gpu) if gpu else ''}" if device else ""
    return f'<p class="muted">{_e("；".join(parts))}{_e("（" + device_text + "）") if device_text else ""}</p>'


def _cards(cards: list[tuple[str, str, str]]) -> str:
    return '<div class="cards">' + "".join(
        f'<div class="card"><span class="label">{_e(label)}</span><strong>{value}</strong><small>{note}</small></div>'
        for label, value, note in cards) + "</div>"


CSS = """
:root{--ink:#20374d;--muted:#62798d;--line:#d8e2ea;--soft:#f4f7fa;--ok:#236956;--warn:#8b6519}
*{box-sizing:border-box}body{margin:0;background:#eef3f7;color:var(--ink);font:11.5px/1.45 system-ui,-apple-system,"Microsoft YaHei",sans-serif}
.page{width:210mm;min-height:297mm;margin:10mm auto;background:#fff;padding:12mm 13mm;box-shadow:0 4px 18px #16334b14}
header{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;border-bottom:2px solid var(--ink);padding-bottom:6px;margin-bottom:8px}
h1{margin:0;font-size:17px}h2{font-size:12.5px;margin:10px 0 4px;color:#2c4a66;border-bottom:1px solid var(--line);padding-bottom:2px}h3{font-size:11.5px;margin:8px 0 3px;color:#3f5a72}
.eyebrow{font-size:9.5px;letter-spacing:1.2px;text-transform:uppercase;color:#176ea2;font-weight:650}
.muted{color:var(--muted);font-size:10.5px;margin:3px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:6px 14px}
dl.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 8px;margin:0;font-size:10.5px}dl.kv dt{color:var(--muted);white-space:nowrap}dl.kv dd{margin:0;overflow-wrap:anywhere}
table{width:100%;border-collapse:collapse;font-size:10.5px;margin:3px 0}th,td{padding:2.5px 5px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:var(--soft);color:var(--muted);font-weight:500;white-space:nowrap}
table.compact td:not(:first-child),table.compact th:not(:first-child){text-align:right;font-variant-numeric:tabular-nums}table.findings td,table.findings th{text-align:left}table.glossary{font-size:9.5px;page-break-inside:auto}table.glossary td:first-child{white-space:nowrap;color:#2c4a66}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin:4px 0}.card{background:var(--soft);border-radius:6px;padding:6px 8px;min-width:0}.card .label{display:block;font-size:9.5px;color:var(--muted)}.card strong{display:block;font-size:14px;margin:1px 0}.card small{display:block;font-size:9px;color:var(--muted);overflow-wrap:anywhere}
.shots{display:grid;gap:5px 7px;margin:4px 0}.shots.cols2{grid-template-columns:repeat(2,1fr)}.shots.cols3{grid-template-columns:repeat(3,1fr)}
.shots figure{margin:0;min-width:0;page-break-inside:avoid;break-inside:avoid}.shots img{display:block;width:100%;max-width:100%;height:auto;border:1px solid var(--line);border-radius:4px;background:#fff}
.shots figcaption{font-size:9.5px;color:var(--muted);margin-top:2px;overflow-wrap:anywhere}
.narrative{border:1px solid var(--line);border-left:4px solid #176ea2;border-radius:6px;padding:6px 9px;background:#f7fbfd}
.narrative p{margin:2px 0;font-size:11px}
.badge{display:inline-block;margin-left:6px;padding:0 6px;border-radius:8px;font-size:9px;font-weight:600;vertical-align:middle;background:#fdf0dc;color:#8b6519}
.badge.auto{background:#e8f1f7;color:#176ea2}
.review-box{border:1px solid var(--line);border-left:4px solid var(--warn);border-radius:6px;padding:6px 9px;background:#fffaf0}.review-box.ok{border-left-color:var(--ok);background:#f0f8f5}
ol.limits{margin:3px 0 0 16px;padding:0;font-size:10px;color:#3f5a72}ol.limits li{margin:1px 0}
footer{margin-top:8px;border-top:1px solid var(--line);padding-top:4px;font-size:9.5px;color:var(--muted);display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}
.toolbar{width:210mm;margin:8mm auto 0;display:flex;justify-content:flex-end}.toolbar button{font:inherit;padding:6px 12px;border:1px solid #b9cbd8;border-radius:7px;background:#fff;color:#176ea2;cursor:pointer}
@page{size:A4;margin:10mm}@media print{body{background:#fff}.page{width:auto;min-height:0;margin:0;padding:0;box-shadow:none}.toolbar{display:none}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""


def render_onepage(summary: Mapping[str, Any], job: Mapping[str, Any] | None = None, *,
                   job_dir: Path | str | None = None, snapshots: Mapping[str, Any] | None = None) -> str:
    """Render the A4 page as a self-contained HTML string (no scripts, no external resources).

    ``job_dir`` enables the 配图 section: the PNGs listed in ``snapshots.json`` are embedded as data
    URIs.  Without it (or without pictures) the section degrades to a one-line hint.
    """
    summary = _map(summary)
    job = _map(job)
    family = family_of(summary)
    title = "壁面 WSS 一页纸报告" if family == "wall" else "压力与速度体场一页纸报告"
    case_id = summary.get("case_id") or job.get("case_id") or "匿名病例"
    release = _map(summary.get("model_release"))
    contract = _map(summary.get("feature_contract"))
    review = _map(summary.get("review")) or _map(job.get("review"))
    review_status = REVIEW_LABELS.get(str(review.get("status") or "unreviewed"), "未审阅")
    numbers = _wall_numbers(summary) if family == "wall" else _volume_numbers(summary)
    branches = ("<h2>分支统计</h2>" + _wall_branches(summary)) if family == "wall" else ""
    notes = summary.get("notes") if isinstance(summary.get("notes"), list) else []
    body = f"""<div class="toolbar"><button type="button" onclick="window.print()">打印 / 保存 PDF</button></div>
<article class="page">
<header><div><p class="eyebrow">{_e(title)}</p><h1>{_e(case_id)}</h1><p class="muted">固定收缩期单帧预测 · 仅供研究参考，不作临床诊断依据</p></div>
<div style="text-align:right"><p class="muted">复核状态：<strong>{_e(review_status)}</strong></p><p class="muted">{_e(summary.get('created_at') or '')}</p></div></header>
<div class="grid2"><section><h2>病例与版本</h2>{_identity_section(summary, job)}</section><section><h2>输入检查</h2>{_input_section(summary)}</section></div>
{_narrative_section(summary)}
<h2>关键数字</h2>{numbers}
{_snapshots_section(Path(job_dir) if job_dir else None, snapshots)}
{_morphology_section(summary)}
{branches}
{_morphology_branches(summary)}
<h2>几何</h2>{_geometry_section(summary)}
{_findings_section(summary)}
{_annotations_section(summary)}
{_trust_section(summary)}
<h2>复核</h2>{_review_section(summary, job)}
<h2>局限性声明</h2><ol class="limits">{"".join(f"<li>{_e(item)}</li>" for item in LIMITATIONS)}{"".join(f"<li>{_e(item)}</li>" for item in notes if isinstance(item, str))}</ol>
<h2>计算耗时</h2>{_timing_section(summary)}
{_glossary_section()}
<footer><span>发布包 {_e(release.get('registry_id') or release.get('release') or summary.get('release') or '—')} · 指纹 {_e(str(release.get('fingerprint') or summary.get('release_hash') or '—')[:12])}</span><span>特征合同 {_e(contract.get('version') or '—')} · {_e(str(contract.get('source_hash') or '—')[:12])}</span><span>任务 {_e(job.get('id') or '—')}</span></footer>
</article>"""
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_e(title)} · {_e(case_id)}</title><style>{CSS}</style></head><body>{body}</body></html>')


__all__ = ["DECISION_LABELS", "LIMITATIONS", "NO_SNAPSHOTS_HINT", "family_of", "render_onepage"]
