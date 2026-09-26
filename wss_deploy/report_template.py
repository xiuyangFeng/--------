"""Institution report template (v0.12, contract ``ANALYSIS_CONTRACT.md`` §19.4).

One JSON file per jobs root, ``<jobs_root>/report_template.json``, read by the one-page report (and the bundle
that embeds it).  Missing or damaged files fall back to :data:`DEFAULTS`, so a fresh install needs nothing.
Only presentation lives here: institution / department names, the title override, an extra footer note, the
signature lines and how much glossary / appendix the one-pager prints.  No clinical thresholds.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path

LOG = logging.getLogger(__name__)
TEMPLATE_FILE = "report_template.json"
SCHEMA = "wss-deploy.report-template/v1"
GLOSSARY_MODES = ("used", "all", "none")
DEFAULTS = {
    "schema_version": SCHEMA,
    "institution": "",                      # 机构名称，印在一页纸页眉
    "department": "",                       # 科室 / 课题组
    "report_title": "",                     # 覆盖一页纸标题；空 = 按结果族自动
    "footer_note": "",                      # 追加在固定声明「仅供研究参考，不作临床诊断依据」之后
    "signature_lines": ["报告人", "审阅人"],  # 签字栏；空列表 = 不印签字栏
    "show_glossary": "used",                # used = 只列本页出现的术语；all = 全部；none = 不印
    "appendix": True,                       # 分支表、发现详情、计算耗时放到附录页（打印时另起一页）
}
_TEXT_LIMITS = {"institution": 120, "department": 120, "report_title": 80, "footer_note": 400}


def validate(data) -> dict:
    """Return a normalised template or raise ``ValueError`` with a message fit for the UI / CLI."""
    if not isinstance(data, dict):
        raise ValueError("报告模板必须是 JSON 对象。")
    unknown = sorted(set(data) - set(DEFAULTS))
    if unknown:
        raise ValueError("报告模板含未知字段：" + "、".join(unknown))
    out = copy.deepcopy(DEFAULTS)
    for key, limit in _TEXT_LIMITS.items():
        if key in data:
            value = data[key]
            if value is None:
                value = ""
            if not isinstance(value, str):
                raise ValueError(f"{key} 必须是文字。")
            value = " ".join(value.replace("\r", " ").split("\n")).strip() if key != "footer_note" else value.replace("\r", "").strip()
            if len(value) > limit:
                raise ValueError(f"{key} 不能超过 {limit} 个字符。")
            out[key] = value
    if "signature_lines" in data:
        lines = data["signature_lines"]
        if lines is None:
            lines = []
        if not isinstance(lines, list) or not all(isinstance(x, str) for x in lines):
            raise ValueError("signature_lines 必须是文字列表。")
        lines = [x.strip() for x in lines if x.strip()]
        if len(lines) > 4 or any(len(x) > 20 for x in lines):
            raise ValueError("签字栏最多 4 项，每项不超过 20 个字符。")
        out["signature_lines"] = lines
    if "show_glossary" in data:
        if data["show_glossary"] not in GLOSSARY_MODES:
            raise ValueError("show_glossary 只能是 used、all 或 none。")
        out["show_glossary"] = data["show_glossary"]
    if "appendix" in data:
        if not isinstance(data["appendix"], bool):
            raise ValueError("appendix 必须是 true 或 false。")
        out["appendix"] = data["appendix"]
    out["schema_version"] = SCHEMA
    return out


def template_path(jobs_root) -> Path:
    return Path(jobs_root) / TEMPLATE_FILE


def load(jobs_root) -> dict:
    """Template for a jobs root; defaults when the file is absent or invalid (never raises)."""
    path = template_path(jobs_root)
    if not path.is_file():
        return copy.deepcopy(DEFAULTS)
    try:
        return validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        LOG.warning("报告模板 %s 无效，使用默认值：%s", path, exc)
        return copy.deepcopy(DEFAULTS)


def load_for_job(job_dir) -> dict:
    """Template for a job directory: ``<root>/<job>`` or a trashed ``<root>/.trash/<job>``."""
    job_dir = Path(job_dir)
    for root in (job_dir.parent, job_dir.parent.parent):
        if template_path(root).is_file():
            return load(root)
    return copy.deepcopy(DEFAULTS)


def save(jobs_root, data) -> dict:
    """Validate then atomically replace ``report_template.json``; returns the stored template."""
    value = validate(data)
    path = template_path(jobs_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return value
