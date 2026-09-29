"""Single-file offline package (WORKSPACE_V2_CONTRACT.md §2 /offline, §6.3 离线模式)."""
from __future__ import annotations

import base64
import json
import re
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pytest

from wss_deploy import v2_data as V
from wss_deploy import v2_offline as O
from wss_deploy.paths import PROJECT_ROOT, STATIC_DIR

DEV_ROOT = PROJECT_ROOT / "outputs" / "wss_deploy_jobs"
M1_JOB = DEV_ROOT / "20260929_230442_a7273f1670e4"
VOLUME_JOB = DEV_ROOT / "20260929_230450_d1428792b846"
dev = pytest.mark.skipif(not (M1_JOB / "report.html").is_file() or not (VOLUME_JOB / "report.html").is_file(),
                         reason="development job copies are not present")

INDEX = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="script-src 'self'">
<title>WSS 工作区</title>
<link rel="stylesheet" href="/static/v2/v2.css"><link rel="icon" href="/favicon.ico">
</head><body><div id="wssv2-app"></div>
<script src="/static/three.min.js"></script>
<script src="/static/v2/core_util.js"></script>
<script src="/static/v2/ws_api.js"></script>
</body></html>
"""


def _fake_bundle(tmp_path: Path, *, drop=()) -> tuple[Path, Path]:
    static = tmp_path / "static"; v2 = static / "v2"; v2.mkdir(parents=True)
    bundle = {"schema": "wss-deploy.v2-bundle/v1", "viewer_version": "v2.test",
              "legacy_scripts": ["three.min.js", "report_common.js"], "scripts": ["core_util.js", "ws_api.js", "ws_main.js"],
              "offline_exclude": ["ws_api.js"], "styles": ["v2.css"], "pages": ["index.html"], "assets": [], "dev": [],
              "generated": ["example_report.html"]}
    (v2 / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    files = {v2 / "index.html": INDEX, v2 / "v2.css": "body{color:#1b2430}/* </style> */",
             static / "three.min.js": "var THREE={};", static / "report_common.js": "var WssReportCommon={};",
             v2 / "core_util.js": "var WSSV2=window.WSSV2||{};WSSV2.util={s:'</script>'};",
             v2 / "ws_api.js": "WSSV2.api={online:true};", v2 / "ws_main.js": "WSSV2.main=1;"}
    for path, text in files.items():
        if path.name not in drop:
            path.write_text(text, encoding="utf-8")
    return static, v2


def _scripts(html: str, script_id: str):
    match = re.search(r'<script type="application/json" id="' + script_id + r'">(.*?)</script>', html, re.S)
    assert match, script_id
    return json.loads(match.group(1))


@pytest.fixture(autouse=True)
def _fresh_cache():
    V.clear_cache()
    yield
    V.clear_cache()


@dev
def test_offline_page_embeds_manifest_arrays_and_inline_code_without_external_references(tmp_path):
    static, v2 = _fake_bundle(tmp_path)
    record = json.loads((M1_JOB / "job.json").read_text(encoding="utf-8"))
    data = V.load(M1_JOB)
    manifest = V.build_manifest(data, record)
    html = O.build_offline_html(data, manifest, record=record, bookmarks=[{"id": "b1", "name": "瘤颈"}], view={"field": "tawss"},
                                exported_at="2026-09-30T10:00:00+08:00", static_dir=static, v2_dir=v2)
    assert html.lstrip().lower().startswith("<!doctype html>")
    # no network: no src / href to anything, no CSP meta that would block the inline code
    assert not re.search(r"<script[^>]*\ssrc=", html, re.I) and "<link" not in html.lower()
    assert "Content-Security-Policy" not in html and 'id="wssv2-app"' in html
    embedded = _scripts(html, "wssv2-manifest")
    assert all(entry["url"] == "embedded" for entry in embedded["arrays"].values())
    arrays = _scripts(html, "wssv2-arrays")
    assert set(arrays) == set(manifest["arrays"])
    for key in arrays:
        assert base64.b64decode(arrays[key]) == data.report.raw(key), key
    # embedded strings are reused as they are (byte-identical base64 of report.html)
    assert arrays["mv"] is data.report.specs["mv"].b64 or arrays["mv"] == data.report.specs["mv"].b64
    offline = _scripts(html, "wssv2-offline")
    assert offline["exported_at"] == "2026-09-30T10:00:00+08:00" and offline["hide_name"] is False
    assert offline["bookmarks"] == [{"id": "b1", "name": "瘤颈"}] and offline["view"] == {"field": "tawss"}
    assert offline["review"]["status"] == manifest["job"]["review"]["status"] and offline["viewer_version"] == "v2.test"
    # legacy scripts first, then the v2 scripts in bundle order, minus offline_exclude
    order = [m.group(1) for m in re.finditer(r"<script>/\* ([\w.-]+) \*/", html)]
    assert order == ["three.min.js", "report_common.js", "core_util.js", "ws_main.js"]
    assert "WSSV2.api={online:true}" not in html
    assert "s:'<\\/script>'" in html and "</style> */" not in html          # element ends broken up
    assert "<title>WSS 离线报告 · LV_GUO_YOU</title>" in html
    assert html.index('id="wssv2-manifest"') < html.index("/* core_util.js */")


@dev
@pytest.mark.parametrize("job_dir", [M1_JOB, VOLUME_JOB], ids=["wall", "volume"])
def test_hide_name_leaves_no_case_name_anywhere_in_the_page(tmp_path, job_dir):
    static, v2 = _fake_bundle(tmp_path)
    record = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    record["patient_id"] = "PATIENT-4711"
    data = V.load(job_dir)
    manifest = V.build_manifest(data, record)
    assert manifest["job"]["display_name"] == "PATIENT-4711"
    html = O.build_offline_html(data, manifest, record=record, hide_name=True,
                                bookmarks=[{"name": "LV_GUO_YOU 瘤颈", "note": "PATIENT-4711 复查"}], static_dir=static, v2_dir=v2)
    for name in ("LV_GUO_YOU", "PATIENT-4711", record["source_filename"]):
        assert name not in html, name
    embedded = _scripts(html, "wssv2-manifest")
    job = embedded["job"]
    assert job["display_name"] == job["case_id"] == job["patient_id"] == job["source_filename"] == "病例"
    assert _scripts(html, "wssv2-offline")["hide_name"] is True
    assert _scripts(html, "wssv2-offline")["bookmarks"] == [{"name": "病例 瘤颈", "note": "病例 复查"}]
    assert "<title>WSS 离线报告 · 病例</title>" in html
    # numbers are untouched by the scrub
    plain = V.build_manifest(data, record)
    assert embedded["fields"] == json.loads(json.dumps(plain["fields"]))
    assert O.offline_filename(embedded, hide_name=True, date="20260930") == "WSS_病例_20260930.html"


def test_offline_filename_uses_the_display_name_and_is_file_system_safe():
    assert O.offline_filename({"job": {"display_name": "LV_GUO_YOU"}}, hide_name=False, date="20260930") == "WSS_LV_GUO_YOU_20260930.html"
    assert O.offline_filename({"job": {"display_name": "a/b:c d"}}, hide_name=False, date="20260930") == "WSS_a_b_c_d_20260930.html"
    assert O.offline_filename({"job": {"display_name": ""}}, hide_name=False, date="20260930") == "WSS_病例_20260930.html"
    assert re.fullmatch(r"WSS_X_\d{8}\.html", O.offline_filename({"job": {"display_name": "X"}}, hide_name=False))


def test_a_listed_but_missing_file_is_named_in_the_error(tmp_path):
    static, v2 = _fake_bundle(tmp_path, drop=("ws_main.js", "three.min.js"))
    with pytest.raises(O.OfflineBuildError) as error:
        O.collect_sources(O.load_bundle(v2), static_dir=static, v2_dir=v2)
    assert "ws_main.js" in str(error.value) and "static/three.min.js" in str(error.value)
    # an excluded file may be missing
    static, v2 = _fake_bundle(tmp_path / "b", drop=("ws_api.js",))
    assert O.collect_sources(O.load_bundle(v2), static_dir=static, v2_dir=v2)["script_names"][-1] == "ws_main.js"
    bad = json.loads((v2 / "bundle.json").read_text(encoding="utf-8")); bad["scripts"] = ["../app.js"]
    (v2 / "bundle.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(O.OfflineBuildError):
        O.collect_sources(O.load_bundle(v2), static_dir=static, v2_dir=v2)
    (v2 / "bundle.json").unlink()
    with pytest.raises(O.OfflineBuildError):
        O.load_bundle(v2)


def _real_bundle_complete() -> bool:
    try:
        O.collect_sources(O.load_bundle())
        return True
    except O.OfflineBuildError:
        return False


@dev
@pytest.mark.skipif(not _real_bundle_complete() or shutil.which("node") is None,
                    reason="the v2 scripts of bundle.json are not all present yet (routes 4 / 5 in progress) or node missing")
def test_the_real_bundle_builds_and_every_inline_script_parses(tmp_path):
    record = json.loads((M1_JOB / "job.json").read_text(encoding="utf-8"))
    data = V.load(M1_JOB)
    html = O.build_offline_html(data, V.build_manifest(data, record), record=record)
    blocks = re.findall(r"<script>(/\* [\w.-]+ \*/.*?)</script>", html, re.S)
    assert len(blocks) == len(O.collect_sources(O.load_bundle())["script_names"])
    for index, block in enumerate(blocks):
        path = tmp_path / f"block_{index}.js"; path.write_text(block, encoding="utf-8")
        result = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, (block[:60], result.stderr[-400:])
