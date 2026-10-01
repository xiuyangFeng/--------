"""Single-file offline package of a finished result (``WORKSPACE_V2_CONTRACT.md`` §2 ``/offline`` and §6.3 离线模式).

The page is the workspace shell ``static/v2/index.html`` with every external reference removed and the code inlined
in ``bundle.json`` order: the ``styles`` as one ``<style>``, then the ``legacy_scripts`` (three.js, OrbitControls,
report_common.js from ``static/``) and the ``scripts`` minus ``offline_exclude``.  The data travels in four JSON
script elements the viewer core reads (``core_data.createEmbeddedSource``):

* ``wssv2-manifest`` — the §3 manifest with every array ``url`` set to ``"embedded"``;
* ``wssv2-arrays``   — ``{key: base64}``; the report's own base64 strings are reused as they are, so the numbers
  are the ones embedded in ``report.html`` (only RRT / ECAP, speed and the streamline concatenation are encoded here);
* ``wssv2-offline``  — ``{exported_at, hide_name, bookmarks, view, review, …}``;
* ``wss-glossary``   — the classic reports' glossary (``glossary.glossary_document()``) for the term tips.

JSON is escaped with ``report._script_json`` (``<``, ``>``, ``&`` and the line separators), inline code has
``</script`` / ``</style`` broken up, so neither a case name nor a bookmark note can end an element early.  With
``hide_name`` the display name, case id, patient id and upload file name become 「病例」 everywhere in the page.
Nothing here requests a URL: the page works from ``file://``.
"""
from __future__ import annotations

import html as _html
import json
import re
from pathlib import Path

from . import __version__
from .paths import STATIC_DIR
from .glossary import glossary_document
from .report import _script_json
from .v2_data import PLACEHOLDER_NAME, JobData, arrays_b64, hide_names

OFFLINE_SCHEMA = "wss-deploy.v2-offline/v1"
V2_DIR = STATIC_DIR / "v2"
MAX_BOOKMARKS = 200
_NAME_RE = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_SCRIPT_SRC = re.compile(r"<script\b[^>]*\bsrc\s*=[^>]*>\s*</script\s*>", re.I | re.S)
_LINK = re.compile(r"<link\b[^>]*>", re.I | re.S)
_CSP_META = re.compile(r"<meta\b[^>]*http-equiv\s*=\s*[\"']?content-security-policy[^>]*>", re.I | re.S)
_BASE = re.compile(r"<base\b[^>]*>", re.I | re.S)
_TITLE = re.compile(r"<title\b[^>]*>.*?</title\s*>", re.I | re.S)


class OfflineBuildError(RuntimeError):
    """The offline package cannot be built (bundle list incomplete, a listed file missing)."""


def load_bundle(v2_dir: Path | None = None) -> dict:
    path = Path(v2_dir or V2_DIR) / "bundle.json"
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OfflineBuildError(f"离线打包缺少可读的 bundle.json：{exc.__class__.__name__}") from exc
    if not isinstance(bundle, dict):
        raise OfflineBuildError("bundle.json 不是 JSON 对象。")
    return bundle


def _names(bundle: dict, key: str) -> list[str]:
    value = bundle.get(key) or []
    if not isinstance(value, list) or any(not isinstance(name, str) or not _NAME_RE.fullmatch(name) or name in {".", ".."}
                                          for name in value):
        raise OfflineBuildError(f"bundle.json 的 {key} 必须是文件名列表（不含目录）。")
    return list(value)


def _inline_js(text: str) -> str:
    # ``</script`` cannot occur in JavaScript outside strings, comments and regular expressions, where ``<\/`` means ``</``.
    return re.sub(r"</(script)", r"<\\/\1", text, flags=re.I)


def _inline_css(text: str) -> str:
    return re.sub(r"</(style)", r"<\\/\1", text, flags=re.I)


def collect_sources(bundle: dict, *, static_dir: Path | None = None, v2_dir: Path | None = None) -> dict:
    """``{"page", "styles", "scripts"}`` texts for the offline page; every missing file is named in one error."""
    static_dir, v2_dir = Path(static_dir or STATIC_DIR), Path(v2_dir or V2_DIR)
    exclude = set(_names(bundle, "offline_exclude"))
    wanted = ([("page", v2_dir / "index.html")]
              + [("style", v2_dir / name) for name in _names(bundle, "styles")]
              + [("script", static_dir / name) for name in _names(bundle, "legacy_scripts")]
              + [("script", v2_dir / name) for name in _names(bundle, "scripts") if name not in exclude])
    missing = [path.name if path.parent == v2_dir else f"static/{path.name}" for _, path in wanted if not path.is_file()]
    if missing:
        raise OfflineBuildError("离线打包缺少 bundle.json 列出的文件：" + "、".join(missing))
    out: dict = {"page": "", "styles": [], "scripts": [], "script_names": []}
    for kind, path in wanted:
        text = path.read_text(encoding="utf-8")
        if kind == "page":
            out["page"] = text
        elif kind == "style":
            out["styles"].append(text)
        else:
            out["scripts"].append(text)
            out["script_names"].append(path.name)
    return out


def _shell(page: str, title: str) -> str:
    """The workspace page without external references; falls back to a bare document when it has no head / body."""
    page = _SCRIPT_SRC.sub("", page)
    page = _LINK.sub("", page)
    page = _CSP_META.sub("", page)
    page = _BASE.sub("", page)
    page = _TITLE.sub("", page, count=1)
    if not re.search(r"</head\s*>", page, re.I) or not re.search(r"</body\s*>", page, re.I):
        page = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1"></head><body></body></html>')
    head = f"<title>{_html.escape(title)}</title>"
    return re.sub(r"</head\s*>", lambda _m: head + "\n</head>", page, count=1, flags=re.I)


def _insert_before_body_end(page: str, block: str) -> str:
    ends = list(re.finditer(r"</body\s*>", page, re.I))
    at = ends[-1].start()
    return page[:at] + block + page[at:]


def build_offline_html(data: JobData, manifest: dict, *, record: dict | None = None, hide_name: bool = False,
                       bookmarks: list | None = None, view: dict | None = None, exported_at: str | None = None,
                       static_dir: Path | None = None, v2_dir: Path | None = None) -> str:
    """The self-contained page (a string) for ``manifest`` of ``data``'s job."""
    bundle = load_bundle(v2_dir)
    sources = collect_sources(bundle, static_dir=static_dir, v2_dir=v2_dir)
    bookmarks = list(bookmarks or [])[:MAX_BOOKMARKS]
    view = view if isinstance(view, dict) else None
    manifest = json.loads(json.dumps(manifest, ensure_ascii=False, allow_nan=False))   # private deep copy
    if hide_name:
        manifest, (bookmarks, view) = hide_names(manifest, record, extra=[bookmarks, view])
    arrays = arrays_b64(data, keys=list(manifest.get("arrays") or {}))
    for entry in (manifest.get("arrays") or {}).values():
        if isinstance(entry, dict):
            entry["url"] = "embedded"
    if not exported_at:
        from .clock import now_iso
        exported_at = now_iso()
    job = manifest.get("job") or {}
    review = job.get("review") if isinstance(job.get("review"), dict) else {}
    offline = {"schema": OFFLINE_SCHEMA, "exported_at": exported_at, "hide_name": bool(hide_name),
               "bookmarks": bookmarks, "view": view,
               "review": {k: review.get(k) for k in ("status", "by", "at", "version")},
               "viewer_version": bundle.get("viewer_version"), "deploy_version": __version__,
               "job_id": job.get("id"), "data_version": manifest.get("data_version")}
    name = (PLACEHOLDER_NAME if hide_name else str(job.get("display_name") or "").strip()) or PLACEHOLDER_NAME
    page = _shell(sources["page"], f"WSS 离线报告 · {name}")
    style = "<style>\n" + "\n".join(_inline_css(text) for text in sources["styles"]) + "\n</style>\n"
    page = re.sub(r"</head\s*>", lambda _m: style + "</head>", page, count=1, flags=re.I)
    data_block = ('<script type="application/json" id="wssv2-manifest">' + _script_json(manifest) + "</script>\n"
                  + '<script type="application/json" id="wssv2-arrays">' + _script_json(arrays) + "</script>\n"
                  + '<script type="application/json" id="wssv2-offline">' + _script_json(offline) + "</script>\n"
                  # phase 3 lane 3 (W63): the term tips read the classic glossary; offline they read this copy
                  + '<script type="application/json" id="wss-glossary">' + _script_json(glossary_document()) + "</script>\n")
    code = "".join(f"<script>/* {script_name} */\n{_inline_js(text)}\n</script>\n"
                   for script_name, text in zip(sources["script_names"], sources["scripts"]))
    return _insert_before_body_end(page, "\n" + data_block + code)


def offline_filename(manifest: dict, *, hide_name: bool, date: str | None = None) -> str:
    """``WSS_<显示名或「病例」>_<yyyymmdd>.html`` (characters a file system rejects become ``_``)."""
    if date is None:
        from .clock import strftime
        date = strftime("%Y%m%d")
    job = manifest.get("job") or {}
    name = PLACEHOLDER_NAME if hide_name else (str(job.get("display_name") or "").strip() or PLACEHOLDER_NAME)
    name = re.sub(r"[\\/:*?\"<>|\x00-\x1f\x7f\s]+", "_", name).strip("._") or PLACEHOLDER_NAME
    return f"WSS_{name[:80]}_{date}.html"


__all__ = ["OFFLINE_SCHEMA", "OfflineBuildError", "build_offline_html", "collect_sources", "load_bundle", "offline_filename"]
