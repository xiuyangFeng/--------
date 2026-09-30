"""Bundle download (C15): every white-listed result file of a job in one zip, plus the one-page summary and a README.

Text files are deflated; ``field.npz`` (already compressed) is stored.  Nothing is written into the job
directory: the one-page HTML is rendered on the fly from summary.json.

v0.14: the service writes bundles to a temporary file (:func:`write_job_bundle` / :func:`write_multi_bundle`)
instead of building them in memory; :func:`job_bundle` / :func:`multi_bundle` keep the byte-returning API.

S7 (PHASE3_LANES.md §3 lane 6 item 6): the page to open offline is the workspace's single-file offline report
(``v2_offline.build_offline_html``, passed in as ``offline_html``) under :data:`OFFLINE_NAME`; it replaces the classic
``report.html`` in the zip (that file stays in the job directory as the workspace's data source).  When the service
could not build the offline page, the classic ``report.html`` is packed as before.
"""
from __future__ import annotations

from . import clock as _clock
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Mapping

STORED_SUFFIXES = {".npz", ".zip", ".gz", ".png"}
SIDECARS = ("annotations.json", "findings_review.json", "snapshots.json", "narrative.json")
SNAPSHOT_FILE = re.compile(r"snapshot_[a-z0-9_-]{1,32}\.png")
MAX_BUNDLE_JOBS = 50
OFFLINE_NAME = "offline_report.html"
CLASSIC_REPORT = "report.html"


def _safe(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "_", str(name or "")).strip("._")
    return cleaned[:80] or fallback


def readme_text(job: Mapping[str, Any], summary: Mapping[str, Any], files: list[str]) -> str:
    release = summary.get("model_release") if isinstance(summary.get("model_release"), Mapping) else {}
    review = job.get("review") if isinstance(job.get("review"), Mapping) else {}
    # §19.2: an M1 result carries single-cycle TAWSS / OSI, so the fixed "no TAWSS / OSI" caveat is written per result.
    if isinstance(summary.get("cycle"), Mapping) and summary.get("cycle"):
        scope = ("- 峰值 WSS 为固定收缩期单帧（step 1162，约 0.21 s）；TAWSS / OSI 为单周期（0.8 s、80 帧）积分量的直接回归预测，"
                 "不是逐帧推演；滞留区 = TAWSS < 0.4 Pa 且 OSI > 0.1。")
    else:
        scope = "- 预测对象是固定收缩期单帧（step 1162，约 0.21 s），不是全周期；没有 TAWSS / OSI。"
    lines = [
        "WSS 部署工具 · 结果打包",
        f"病例编号：{summary.get('case_id') or job.get('case_id') or '—'}",
        f"任务编号：{job.get('id') or '—'}",
        f"发布包：{release.get('registry_id') or release.get('id') or summary.get('release') or '—'}",
        f"发布包指纹：{release.get('fingerprint') or summary.get('release_hash') or '—'}",
        f"输入 SHA256：{summary.get('input_sha256') or job.get('input_sha256') or '—'}",
        f"run_identity：{summary.get('run_identity') or job.get('run_identity') or '—'}",
        f"审阅状态：{review.get('status') or 'unreviewed'}{'（' + str(review.get('by')) + '）' if review.get('by') else ''}",
        f"生成时间：{_clock.now_iso()}",
        "",
        "口径说明：",
        scope,
        "- 主指标是预测点云的空间 p99；最大值只作参考并标出位置。",
        "- 面积占比 = 点占比 × 输入壁面面积（估计值）；压力是相对量，只有压差有意义。",
        offline_line(files),
        "- onepage.html 为 A4 一页纸；summary.json / run_manifest.json 记录来源链与文件哈希。",
        "- annotations.json / findings_review.json（若存在）是审阅人的标注与发现判定。",
        "- narrative.json（若存在）是自动结论（auto）与审阅人改写（edited）；一页纸「结论（参考）」以改写优先。",
        "- summary.json 的 morphology 是壁面网格每 1 mm 一站的截面测量（最大 Feret / 等效直径、瘤体与瘤颈、体积）。",
        "- snapshots.json 与 snapshot_*.png（若存在）是工作区「导出 → 一页纸」生成的配图，已内嵌在 onepage.html 中。",
        "",
        "文件：",
        *[f"- {name}" for name in files],
    ]
    return "\n".join(lines) + "\n"


def offline_line(files: list[str]) -> str:
    """README line about the page that opens without the service (S7: the workspace offline page, else the classic one)."""
    if OFFLINE_NAME in files:
        return (f"- {OFFLINE_NAME} 是单文件离线报告（与工作区同一个查看器）：用浏览器直接打开，不需要网络和服务，"
                "可切换字段、看截面、读数和导出图片。")
    if CLASSIC_REPORT in files:
        return f"- {CLASSIC_REPORT} 可离线打开（three.js 内嵌，旧版报告页）。"
    return "- 本包没有可离线打开的报告页。"


def bundle_files(job_dir: Path, allowed: set[str]) -> list[str]:
    """Names that go into a job's zip: white-listed files, sidecars and snapshot pictures that exist."""
    job_dir = Path(job_dir).resolve()
    names = []
    extras = {path.name for path in job_dir.glob("snapshot_*.png") if SNAPSHOT_FILE.fullmatch(path.name)}
    for name in sorted(set(allowed) | set(SIDECARS) | extras):
        candidate = (job_dir / name).resolve()
        if candidate.parent == job_dir and candidate.is_file():
            names.append(name)
    return names


def estimate_bytes(job_dir: Path, allowed: set[str]) -> int:
    """Upper-bound size of a job's zip before compression (sum of the member files)."""
    job_dir = Path(job_dir).resolve()
    total = 0
    for name in bundle_files(job_dir, allowed):
        try:
            total += (job_dir / name).stat().st_size
        except OSError:
            pass
    return total


def job_bundle(job_dir: Path, job: Mapping[str, Any], allowed: set[str], *, onepage_html: str | None = None,
               offline_html: str | None = None) -> bytes:
    """Zip bytes of one finished job: white-listed files that exist, sidecars, onepage.html, the offline page and
    README.txt (with ``offline_html`` the classic report.html is left out)."""
    buffer = io.BytesIO()
    write_job_bundle(buffer, job_dir, job, allowed, onepage_html=onepage_html, offline_html=offline_html)
    return buffer.getvalue()


def write_job_bundle(target, job_dir: Path, job: Mapping[str, Any], allowed: set[str], *, onepage_html: str | None = None,
                     offline_html: str | None = None) -> None:
    """Write one job's zip to ``target`` (a path or a seekable binary file); same members as :func:`job_bundle`."""
    job_dir = Path(job_dir).resolve()
    summary_path = job_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    except (OSError, ValueError):
        summary = {}
    if not isinstance(summary, Mapping):
        summary = {}
    names = bundle_files(job_dir, allowed)
    if offline_html:
        names = [name for name in names if name != CLASSIC_REPORT]
    with zipfile.ZipFile(target, "w") as archive:
        for name in names:
            method = zipfile.ZIP_STORED if Path(name).suffix in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            archive.write(job_dir / name, arcname=name, compress_type=method)
        listed = list(names)
        if onepage_html:
            archive.writestr("onepage.html", onepage_html.encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED)
            listed.append("onepage.html")
        if offline_html:
            archive.writestr(OFFLINE_NAME, offline_html.encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED)
            listed.append(OFFLINE_NAME)
        listed.append("README.txt")
        archive.writestr("README.txt", readme_text(job, summary, listed).encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED)


def bundle_name(job: Mapping[str, Any]) -> str:
    return f"{_safe(job.get('case_id'), 'case')}_{_safe(job.get('id'), 'job')}.zip"


def _unique_names(names):
    seen = set()
    for name in names:
        unique = name
        counter = 2
        while unique in seen:
            unique = f"{Path(name).stem}_{counter}.zip"
            counter += 1
        seen.add(unique)
        yield unique


def multi_bundle(bundles: list[tuple[str, bytes]]) -> bytes:
    """Outer zip holding one already-compressed zip per job (stored, not re-deflated)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for unique, (_, data) in zip(_unique_names(name for name, _ in bundles), bundles):
            archive.writestr(unique, data, compress_type=zipfile.ZIP_STORED)
    return buffer.getvalue()


def write_multi_bundle(target, parts: list[tuple[str, Path]]) -> None:
    """Outer zip written to ``target`` from per-job zips on disk (stored, streamed file by file)."""
    with zipfile.ZipFile(target, "w", allowZip64=True) as archive:
        for unique, (_, path) in zip(_unique_names(name for name, _ in parts), parts):
            archive.write(path, arcname=unique, compress_type=zipfile.ZIP_STORED)


__all__ = ["CLASSIC_REPORT", "MAX_BUNDLE_JOBS", "OFFLINE_NAME", "SIDECARS", "offline_line", "bundle_files", "bundle_name", "estimate_bytes", "job_bundle", "multi_bundle",
           "readme_text", "write_job_bundle", "write_multi_bundle"]
