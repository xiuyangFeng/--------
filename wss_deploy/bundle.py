"""Bundle download (C15): every white-listed result file of a job in one zip, plus the one-page summary and a README.

Text files are deflated; ``field.npz`` (already compressed) is stored.  Nothing is written into the job
directory: the one-page HTML is rendered on the fly from summary.json.
"""
from __future__ import annotations

import datetime as dt
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


def _safe(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "_", str(name or "")).strip("._")
    return cleaned[:80] or fallback


def readme_text(job: Mapping[str, Any], summary: Mapping[str, Any], files: list[str]) -> str:
    release = summary.get("model_release") if isinstance(summary.get("model_release"), Mapping) else {}
    review = job.get("review") if isinstance(job.get("review"), Mapping) else {}
    lines = [
        "WSS 部署工具 · 结果打包",
        f"病例编号：{summary.get('case_id') or job.get('case_id') or '—'}",
        f"任务编号：{job.get('id') or '—'}",
        f"发布包：{release.get('registry_id') or release.get('id') or summary.get('release') or '—'}",
        f"发布包指纹：{release.get('fingerprint') or summary.get('release_hash') or '—'}",
        f"输入 SHA256：{summary.get('input_sha256') or job.get('input_sha256') or '—'}",
        f"run_identity：{summary.get('run_identity') or job.get('run_identity') or '—'}",
        f"审阅状态：{review.get('status') or 'unreviewed'}{'（' + str(review.get('by')) + '）' if review.get('by') else ''}",
        f"生成时间：{dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "口径说明：",
        "- 预测对象是固定收缩期单帧（step 1162，约 0.21 s），不是全周期；没有 TAWSS / OSI。",
        "- 主指标是预测点云的空间 p99；最大值只作参考并标出位置。",
        "- 面积占比 = 点占比 × 输入壁面面积（估计值）；压力是相对量，只有压差有意义。",
        "- report.html 可离线打开（three.js 内嵌）；onepage.html 为 A4 一页纸；summary.json / run_manifest.json 记录来源链与文件哈希。",
        "- annotations.json / findings_review.json（若存在）是审阅人的标注与发现判定。",
        "- narrative.json（若存在）是自动结论（auto）与审阅人改写（edited）；一页纸「结论（参考）」以改写优先。",
        "- summary.json 的 morphology 是壁面网格每 1 mm 一站的截面测量（最大 Feret / 等效直径、瘤体与瘤颈、体积）。",
        "- snapshots.json 与 snapshot_*.png（若存在）是三维报告导出的一页纸配图，已内嵌在 onepage.html 中。",
        "",
        "文件：",
        *[f"- {name}" for name in files],
    ]
    return "\n".join(lines) + "\n"


def job_bundle(job_dir: Path, job: Mapping[str, Any], allowed: set[str], *, onepage_html: str | None = None) -> bytes:
    """Zip bytes of one finished job: white-listed files that exist, sidecars, onepage.html and README.txt."""
    job_dir = Path(job_dir).resolve()
    summary_path = job_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    except (OSError, ValueError):
        summary = {}
    if not isinstance(summary, Mapping):
        summary = {}
    names = []
    extras = {path.name for path in job_dir.glob("snapshot_*.png") if SNAPSHOT_FILE.fullmatch(path.name)}
    for name in sorted(set(allowed) | set(SIDECARS) | extras):
        candidate = (job_dir / name).resolve()
        if candidate.parent == job_dir and candidate.is_file():
            names.append(name)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            method = zipfile.ZIP_STORED if Path(name).suffix in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            archive.write(job_dir / name, arcname=name, compress_type=method)
        listed = list(names)
        if onepage_html:
            archive.writestr("onepage.html", onepage_html.encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED)
            listed.append("onepage.html")
        listed.append("README.txt")
        archive.writestr("README.txt", readme_text(job, summary, listed).encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED)
    return buffer.getvalue()


def bundle_name(job: Mapping[str, Any]) -> str:
    return f"{_safe(job.get('case_id'), 'case')}_{_safe(job.get('id'), 'job')}.zip"


def multi_bundle(bundles: list[tuple[str, bytes]]) -> bytes:
    """Outer zip holding one already-compressed zip per job (stored, not re-deflated)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        seen = set()
        for name, data in bundles:
            unique = name
            counter = 2
            while unique in seen:
                unique = f"{Path(name).stem}_{counter}.zip"
                counter += 1
            seen.add(unique)
            archive.writestr(unique, data, compress_type=zipfile.ZIP_STORED)
    return buffer.getvalue()


__all__ = ["MAX_BUNDLE_JOBS", "SIDECARS", "bundle_name", "job_bundle", "multi_bundle", "readme_text"]
