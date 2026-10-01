"""Phase 3 lane 6 — S7 旧页面下线 (PHASE3_LANES.md §3 lane 6).

* server routes: ``/``, ``/compare`` and page visits of the classic report redirect into the workspace; the classic
  workbench / comparison files are gone; ``/ops`` and ``/support`` stay;
* ``ws_legacy.js``: classic addresses (``?job=``, ``#job=``, the classic ``#view=`` state) become workspace routes and
  the workspace's reproducible-view state (the classic encoders of report.py and volume_viewer.js produce the input);
* the workspace no longer links a classic page; 运维中心 / 反馈问题 entries; the loading-failure fallback;
* bundle.zip carries the workspace single-file offline page instead of the classic report.html;
* versioned asset URLs (``?v=<build>``) served as immutable, the page itself revalidated;
* devshot / rehearse on the workspace addresses.
"""
from __future__ import annotations

import http.client
import io
import json
import re
import shutil
import subprocess
import threading
import zipfile
from pathlib import Path

import pytest

from tests._c_helpers import Service, call, finished
from tests.test_v2_workspace_js import _HARNESS, WS_FILES, _need_node
from wss_deploy import server as S
from wss_deploy.jobs import JobManager
from wss_deploy.paths import STATIC_DIR
from wss_deploy.server import ServiceHTTPServer, SessionStore

V2 = STATIC_DIR / "v2"
JOB = "20260920_173929_a9ec139d6cdd"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / JOB
RETIRED = ("index.html", "app.js", "compare.html", "compare.js", "batch_export.js", "workbench_core.js")


def _get(api, path, headers=None):
    api.request("GET", path, headers=headers or {})
    response = api.getresponse()
    return response.status, response.read(), response


# ----------------------------------------------------------------------------------------------- routes
def test_classic_addresses_redirect_into_the_workspace(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        # no session needed: the workspace asks for a login itself
        for path, location in (("/", "/v2/"), ("/?x=1", "/v2/?x=1"),
                               ("/compare?left=A_1&right=B-2", "/v2/#/job/A_1?v=compare&cmp=B-2"),
                               ("/compare?left=A_1&right=A_1", "/v2/#/job/A_1"), ("/compare?left=A_1", "/v2/#/job/A_1"),
                               ("/compare?left=../x&right=B", "/v2/"), ("/compare", "/v2/"),
                               ("/compare?left=A%3Cscript%3E&right=B", "/v2/")):
            status, body, response = _get(api, path)
            assert status == 302 and response.getheader("Location") == location, path
            assert body == b"" and response.getheader("Set-Cookie") is None
        for name in RETIRED:
            assert _get(api, f"/static/{name}")[0] == 404, name
            assert not (STATIC_DIR / name).exists(), name
        # kept: the operations console and the support page with their stylesheet, the report scripts
        for path in ("/ops", "/ops/", "/support", "/support/", "/static/app.css", "/static/ops.js", "/static/support.js",
                     "/static/three.min.js", "/static/report_common.js", "/static/volume_viewer.js", "/static/glossary.json"):
            assert _get(api, path)[0] == 200, path
        # their 「返回工作台」 now points at the workspace
        for name in ("ops.html", "support.html"):
            html = (STATIC_DIR / name).read_text(encoding="utf-8")
            back = re.search(r'<a href="([^"]+)" class="text-button ops-back">返回工作台</a>', html)
            assert back and back[1] == "/v2/", name
            assert 'class="brand" href="/v2/"' in html and 'href="/"' not in html, name
    finally:
        api.close(); service.close()


def test_classic_report_is_redirected_only_for_page_visits(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        cookie, _, owner = service.session(api)
        job = finished(service.manager, owner=owner, case_id="R")
        jid = job["id"]
        visit = {"Cookie": cookie, "Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate", "Accept": "text/html,*/*"}
        for path in (f"/api/jobs/{jid}/report", f"/jobs/{jid}/report.html", f"/api/jobs/{jid}/files/report.html"):
            status, _, response = _get(api, path, visit)
            assert status == 302 and response.getheader("Location") == f"/v2/?job={jid}", path
        # an older browser without fetch metadata: an HTML Accept is a visit of /report, not of the files route
        status, _, response = _get(api, f"/api/jobs/{jid}/report", {"Cookie": cookie, "Accept": "text/html"})
        assert status == 302 and response.getheader("Location") == f"/v2/?job={jid}"
        status, body, _ = _get(api, f"/api/jobs/{jid}/files/report.html", {"Cookie": cookie, "Accept": "text/html"})
        assert status == 200 and b"wss-report-meta" in body
        # programmatic reads and embeds keep the file (it is the workspace's data source)
        for headers in ({"Cookie": cookie}, {"Cookie": cookie, "Sec-Fetch-Dest": "empty"}, {"Cookie": cookie, "Sec-Fetch-Dest": "iframe", "Accept": "text/html"}):
            status, body, _ = _get(api, f"/api/jobs/{jid}/report", headers)
            assert status == 200 and b"wss-report-meta" in body, headers
        status, _, _ = _get(api, f"/jobs/{jid}/summary.json", visit)                 # other files of the old route: unchanged
        assert status == 200
        # the redirect does not need a session (the workspace logs in, then opens the job)
        status, _, response = _get(api, f"/api/jobs/{jid}/report", {"Sec-Fetch-Dest": "document"})
        assert status == 302 and response.getheader("Location") == f"/v2/?job={jid}"
        status, _, _ = _get(api, "/api/jobs/not..valid/report", visit)
        assert status == 404
    finally:
        api.close(); service.close()


def test_workspace_page_names_its_files_with_the_build_and_they_are_immutable(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        status, body, response = _get(api, "/v2/")
        assert status == 200 and response.getheader("Set-Cookie")
        build = S.static_build_id("v2")
        assert re.fullmatch(r"[0-9a-f]{12}", build)
        page = body.decode("utf-8")
        disk = (V2 / "index.html").read_text(encoding="utf-8")
        refs = re.findall(r'(?:src|href)="(/static/[^"]+)"', page)
        disk_refs = re.findall(r'(?:src|href)="(/static/[^"]+)"', disk)
        assert len(refs) == len(disk_refs) > 40 and all(ref.endswith("?v=" + build) for ref in refs)
        assert refs == [ref + "?v=" + build for ref in disk_refs]                  # same order, only the version added
        assert "?v=" not in disk                                                    # the file (and the offline package) unchanged
        assert response.getheader("Cache-Control") == S.STATIC_CACHE              # the page itself is revalidated
        etag = response.getheader("ETag")
        status, body, response = _get(api, "/v2/", {"If-None-Match": etag})
        assert status == 304 and body == b""
        # the build the page names is the one a logged-in page compares for 「页面文件已更新」
        session = _get(api, "/api/session")[2].getheader("Set-Cookie").split(";", 1)[0]
        status, health, _ = call(api, "GET", "/api/health", {"Cookie": session})
        assert status == 200 and health["ui_build_v2"] == build
        for path in ("/static/v2/ws_shell.js", "/static/v2/v2.css", "/static/three.min.js", "/static/volume_viewer.js"):
            status, _, response = _get(api, f"{path}?v={build}")
            assert status == 200 and response.getheader("Cache-Control") == S.IMMUTABLE_CACHE, path
            for variant in (path, f"{path}?v=000000000000", f"{path}?x=1"):     # no or another version: revalidated as before
                status, _, response = _get(api, variant)
                assert status == 200 and response.getheader("Cache-Control") == S.STATIC_CACHE, variant
        for path in ("/static/app.css", "/static/ops.js", "/static/glossary.json"):   # not part of the v2 build
            status, _, response = _get(api, f"{path}?v={build}")
            assert status == 200 and response.getheader("Cache-Control") == S.STATIC_CACHE, path
    finally:
        api.close(); service.close()


def test_v2_page_rewrite_follows_the_index_file(tmp_path, monkeypatch):
    v2 = tmp_path / "v2"; v2.mkdir()
    static = tmp_path / "static"; static.mkdir()
    (v2 / "bundle.json").write_text(json.dumps({"legacy_scripts": ["three.min.js"], "scripts": ["a.js"], "styles": ["v2.css"], "pages": ["index.html"]}), encoding="utf-8")
    (v2 / "a.js").write_text("var a=1;", encoding="utf-8"); (v2 / "v2.css").write_text("b{}", encoding="utf-8")
    (static / "three.min.js").write_text("var T;", encoding="utf-8")
    index = v2 / "index.html"
    index.write_text('<link rel="stylesheet" href="/static/v2/v2.css"><script src="/static/three.min.js" defer></script>'
                     '<script src="/static/v2/a.js" defer></script><a href="/">x</a><a href="/static/v2/help.html">h</a>', encoding="utf-8")
    monkeypatch.setattr(S, "V2_DIR", v2); monkeypatch.setattr(S, "STATIC_DIR", static)
    monkeypatch.setattr(S, "_STATIC_BUILD", {}); monkeypatch.setattr(S, "_V2_BUNDLE", {}); monkeypatch.setattr(S, "_V2_PAGE", {})
    body, etag = S.v2_page(index)
    build = S.static_build_id("v2")
    text = body.decode("utf-8")
    assert f'href="/static/v2/v2.css?v={build}"' in text and f'src="/static/three.min.js?v={build}"' in text and f'src="/static/v2/a.js?v={build}"' in text
    assert '<a href="/">' in text and 'href="/static/v2/help.html"' in text      # only scripts and styles are versioned
    assert S.v2_versioned("a.js", legacy=False) and S.v2_versioned("three.min.js", legacy=True)
    assert not S.v2_versioned("app.css", legacy=True) and not S.v2_versioned("absent.js", legacy=False)
    import os
    (v2 / "a.js").write_text("var a=2;", encoding="utf-8")
    os.utime(v2 / "a.js", ns=(os.stat(v2 / "a.js").st_atime_ns, os.stat(v2 / "a.js").st_mtime_ns + 10_000_000))
    body2, etag2 = S.v2_page(index)
    assert S.static_build_id("v2") != build and etag2 != etag and f"?v={S.static_build_id('v2')}" in body2.decode("utf-8")


# ----------------------------------------------------------------------------------------------- bundle.zip
class _FixtureServer:
    """The checked-in LV_GUO_YOU job on a loopback server whose session owns it (as test_report_freshness)."""
    def __init__(self, tmp_path: Path):
        if not (FIXTURE / "report.html").is_file():
            pytest.fail(f"test fixture job missing: {FIXTURE}", pytrace=False)
        root = tmp_path / "jobs"; (root / JOB).mkdir(parents=True)
        for name in ("job.json", "summary.json", "report.html", "run_manifest.json"):
            shutil.copy2(FIXTURE / name, root / JOB / name)
        owner = json.loads((root / JOB / "job.json").read_text(encoding="utf-8"))["owner"]
        self.manager = JobManager(root, stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {})
        self.sessions = SessionStore(tmp_path / "sessions.json", shared=False, legacy_owner=owner)
        self.server = ServiceHTTPServer(("127.0.0.1", 0), self.manager, self.sessions)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        api = self.api(); api.request("GET", "/api/session"); response = api.getresponse(); response.read(); api.close()
        self.cookie = response.getheader("Set-Cookie").split(";", 1)[0]
    def api(self):
        return http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=60)
    def close(self):
        self.server.shutdown(); self.server.server_close(); self.manager.close()


def test_bundle_carries_the_workspace_offline_page_instead_of_the_classic_report(tmp_path):
    from wss_deploy.bundle import OFFLINE_NAME
    service = _FixtureServer(tmp_path)
    api = service.api()
    try:
        status, data, response = _get(api, f"/api/jobs/{JOB}/bundle.zip", {"Cookie": service.cookie})
        assert status == 200 and response.getheader("Content-Type") == "application/zip"
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
            assert OFFLINE_NAME in names and "report.html" not in names and {"summary.json", "onepage.html", "README.txt"} <= names
            page = archive.read(OFFLINE_NAME).decode("utf-8")
            readme = archive.read("README.txt").decode("utf-8")
        assert page.lstrip().lower().startswith("<!doctype html>") and 'id="wssv2-manifest"' in page and 'id="wssv2-arrays"' in page
        assert not re.search(r"<script[^>]*\ssrc=", page, re.I)                    # opens from file:// without the service
        assert "ws_legacy.js" not in page and "/* ws_shell.js */" in page
        manifest = json.loads(re.search(r'<script type="application/json" id="wssv2-manifest">(.*?)</script>', page, re.S)[1])
        assert manifest["job"]["id"] == JOB and all(entry["url"] == "embedded" for entry in manifest["arrays"].values())
        assert f"- {OFFLINE_NAME}" in readme and "单文件离线报告" in readme and "- report.html" not in readme
        assert (service.manager.root / JOB / "report.html").is_file()               # the job keeps its data source
        # several jobs: the per-job zips carry it too
        status, payload, _ = call(api, "GET", "/api/session", {"Cookie": service.cookie})
        headers = {"Cookie": service.cookie, "X-CSRF-Token": payload["csrf_token"], "Content-Type": "application/json"}
        status, data, _ = call(api, "POST", "/api/jobs/bundle", headers, {"ids": [JOB]})
        assert status == 200
        with zipfile.ZipFile(io.BytesIO(data)) as outer:
            inner = zipfile.ZipFile(io.BytesIO(outer.read(outer.namelist()[0])))
            assert OFFLINE_NAME in inner.namelist() and "report.html" not in inner.namelist()
    finally:
        api.close(); service.close()


def test_bundle_keeps_the_classic_report_when_the_offline_page_cannot_be_built(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        cookie, _, owner = service.session(api)
        job = finished(service.manager, owner=owner, case_id="F")        # a report.html without arrays: no workspace data
        status, data, _ = _get(api, f"/api/jobs/{job['id']}/bundle.zip", {"Cookie": cookie})
        assert status == 200
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            readme = archive.read("README.txt").decode("utf-8")
        assert "report.html" in names and "offline_report.html" not in names and "旧版报告页" in readme
    finally:
        api.close(); service.close()


def test_job_bundle_writer_swaps_the_offline_page_in(tmp_path):
    from wss_deploy.bundle import OFFLINE_NAME, job_bundle, offline_line
    from wss_deploy.server import OUTPUT_FILES
    job_dir = tmp_path / "job"; job_dir.mkdir()
    (job_dir / "summary.json").write_text("{}", encoding="utf-8"); (job_dir / "report.html").write_text("<html>classic</html>", encoding="utf-8")
    data = job_bundle(job_dir, {"id": "j"}, OUTPUT_FILES, onepage_html="<html>one</html>", offline_html="<!doctype html><p>v2</p>")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert sorted(archive.namelist()) == sorted(["summary.json", "onepage.html", OFFLINE_NAME, "README.txt"])
        assert archive.getinfo(OFFLINE_NAME).compress_type == zipfile.ZIP_DEFLATED
        assert archive.read(OFFLINE_NAME) == "<!doctype html><p>v2</p>".encode("utf-8")
        readme = archive.read("README.txt").decode("utf-8")
    assert f"- {OFFLINE_NAME}" in readme and "工作区「导出 → 一页纸」" in readme and "三维报告" not in readme
    assert "旧版" in offline_line(["report.html"]) and "没有" in offline_line([])


# ----------------------------------------------------------------------------------------------- ws_legacy.js (Node)
LEGACY = V2 / "ws_legacy.js"
NODE_PRELUDE = """
globalThis.document={getElementById:()=>null};
const out=x=>console.log(JSON.stringify(x));
const L=require(@@legacy@@);
"""


def _node(script: str, **extra) -> dict:
    _need_node()
    program = NODE_PRELUDE.replace("@@legacy@@", json.dumps(str(LEGACY))) + "(async()=>{\n" + script + "\n})().catch(e=>{console.error(e&&e.stack||e);process.exit(1);});"
    for key, value in extra.items():
        program = program.replace("@@" + key + "@@", json.dumps(str(value)))
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True, timeout=120)
    if result.returncode:
        raise AssertionError(result.stderr[-4000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_legacy_script_parses_and_loads_between_the_shell_and_the_start():
    _need_node()
    subprocess.run(["node", "--check", str(LEGACY)], check=True, capture_output=True)
    bundle = json.loads((V2 / "bundle.json").read_text(encoding="utf-8"))
    scripts = bundle["scripts"]
    assert scripts.index("ws_figure.js") < scripts.index("ws_shell.js") < scripts.index("ws_legacy.js") < scripts.index("ws_main.js")
    assert "ws_legacy.js" in bundle["offline_exclude"]                          # online only
    assert '<script src="/static/v2/ws_legacy.js" defer></script>' in (V2 / "index.html").read_text(encoding="utf-8")


def test_classic_addresses_parse():
    out = _node("""
      const P=(s,h)=>L.parseAddress(s,h);
      out({
        query: P('?job=20260929_230442_a7273f1670e4',''), queryView: P('?job=J_1','#view=eyJhIjoxfQ'),
        workbench: P('','#job=a_1'), workbenchCombined: P('','#view=x&job=b-2'), workspace: P('?job=J','#/job/K?v=compare'),
        badQuery: P('?job=../x',''), badHash: P('','#job=../x'), none: P('',''), plain: P('?x=1',''),
        lastQuery: L.jobFromSearch('?job=A&job=B'), firstHash: L.classicHash('#job=A&job=B'),
        strip: [L.searchWithoutJob('?job=A&x=1'), L.searchWithoutJob('?job=A'), L.searchWithoutJob('')],
        route: [L.routeHash('J_1','tawss'), L.routeHash('J_1',null)]
      });
    """)
    assert out["query"] == {"jobId": "20260929_230442_a7273f1670e4", "view": None, "keepHash": False, "classic": True}
    assert out["queryView"] == {"jobId": "J_1", "view": "eyJhIjoxfQ", "keepHash": False, "classic": True}
    assert out["workbench"]["jobId"] == "a_1" and out["workbench"]["classic"] is True
    assert out["workbenchCombined"]["jobId"] == "b-2" and out["workbenchCombined"]["view"] == "x"   # as workbench_core.parseJobHash
    assert out["workspace"] == {"jobId": "J", "view": None, "keepHash": True, "classic": True}
    assert out["badQuery"]["jobId"] is None and out["badQuery"]["classic"] is True               # still stripped from the address
    assert out["badHash"]["jobId"] is None and out["none"]["classic"] is False and out["plain"]["classic"] is False
    assert out["lastQuery"] == "B" and out["firstHash"]["job"] == "A"
    assert out["strip"] == ["?x=1", "", ""] and out["route"] == ["#/job/J_1?f=tawss", "#/job/J_1"]


def _wall_core(tmp_path: Path) -> Path:
    from wss_deploy.report import REPORT_CORE_JS
    path = tmp_path / "wall_core.js"
    path.write_text(REPORT_CORE_JS, encoding="utf-8")
    return path


WALL_STATE = {  # report.py currentViewState() of a TAWSS result opened with the classic defaults, then changed
    "schema_version": "wss-deploy.view/v1", "family": "wall", "run_identity": "0123456789abcdefXYZ",
    "camera": {"position": [10, -290, 40], "target": [10, 10, 40], "up": [0, 0, 1]},
    "field": "tawss", "mode": "wss", "colormap": "rainbow", "range": {"mode": "case", "min": 0, "max": 2.5, "case_max": 2.5},
    "bands": 0, "log": False, "units": "Pa", "thresholds_pa": [0.4, 4, 7], "field_thresholds": {}, "opacity": 1,
    "overlay": {"trust": True, "contours": False, "stagnation": False}, "slice": {"clip": 1},
    "highlight": {"branch": -1, "top_pct": 1, "top": False, "peak": True, "feature": "wss"},
    "labels": {"findings": 0, "branches": False, "max_diameter": True}, "measurements": [], "annotations": {"items": []},
    "probe_log": [], "branches_hidden": [2], "preset_name": None, "lang": "zh",
    "export": {"scale": 2, "background": "white", "colorbar": "overlay", "ui": False, "lang": "zh"}, "montage": {"views": ["front"], "columns": 2}}


def test_classic_wall_link_round_trip(tmp_path):
    program = """
      require(@@core@@);
      const C=globalThis.WssReportCore, S=@@state@@, X=@@changed@@;
      const hash=C.viewHash(S);
      const a=L.parseAddress('?job=J1', hash), st=L.decodeClassicView(a.view);
      const ctx={family:'wall', fields:['wss','tawss','osi','rrt','ecap'], logFields:['wss','tawss','rrt','ecap'], branches:[0,1,2,3]};
      const def=L.classicToV2(st, ctx), none=L.classicToV2(st, {}), chg=L.classicToV2(L.decodeClassicView(C.encodeView(X)), ctx);
      const logOn=L.classicToV2(Object.assign({}, S, {log:true}), ctx), osi=L.classicToV2(Object.assign({}, S, {field:'osi', log:false}), ctx);
      const cl=L.classicToV2(Object.assign({}, S, {mode:'cl', field:'wss', branches_hidden:[0,1,2,3]}), ctx);
      const missing=L.classicToV2(Object.assign({}, S, {field:'ecap'}), {fields:['wss']});
      const fixedOther=L.classicToV2(Object.assign({}, S, {range:{mode:'fixed', max:3, field:'wss'}}), ctx);
      let bad=null; try { L.decodeClassicView(C.encodeView({schema_version:'other/v9'})); } catch (e) { bad=String(e.message); }
      let junk=null; try { L.decodeClassicView('%%%'); } catch (e) { junk='throws'; }
      out({def, none, chg, logOn:logOn.state.w, osi:osi.state.w, cl:cl.state, missing:missing.state.f||null, fixedOther:fixedOther.state.w, bad, junk});
    """
    changed = dict(WALL_STATE, colormap="viridis", bands=8, units="dyn", mode="cloud", thresholds_pa=[0.5, 4, 7], opacity=0.6,
                   overlay={"trust": False, "contours": True, "stagnation": True}, slice={"clip": 0.4},
                   highlight={"branch": 3, "top_pct": 2, "top": True, "peak": True, "feature": "th", "finding": "F2"},
                   labels={"findings": 3, "branches": True, "max_diameter": True}, measurements=[{"points": [[0, 0, 0]]}],
                   probe_log=[{"id": "P1"}], lang="en", range={"mode": "fixed", "min": 0, "max": 3.5, "field": "tawss"}, log=True)
    program = program.replace("@@state@@", json.dumps(WALL_STATE, ensure_ascii=False)).replace("@@changed@@", json.dumps(changed, ensure_ascii=False))
    out = _node(program, core=_wall_core(tmp_path))
    d = out["def"]
    assert d["family"] == "wall" and d["hiddenBranches"] == [2]
    st = d["state"]
    assert st["v"] == 1 and st["run"] == "0123456789abcdef" and st["f"] == "tawss"
    assert st["w"] == "adaptive-linear"          # the classic wall report opens linear; TAWSS is log in the workspace
    assert st["L"] == {"trust": True} and st["br"] == [0, 1, 3]
    # the same framing with the workspace's 30° lens: the camera moves back along the same line of sight
    import math
    k = math.tan(math.radians(35 / 2)) / math.tan(math.radians(30 / 2))
    assert st["cam"]["t"] == [10, 10, 40] and st["cam"]["u"] == [0, 0, 1] and st["cam"]["fov"] == 30
    assert st["cam"]["p"] == pytest.approx([10, 10 - 300 * k, 40], abs=1e-3)
    assert "sl" not in st and d["dropped"] == []                                   # classic defaults: nothing to report
    assert "br" not in out["none"]["state"] and out["none"]["hiddenBranches"] == [2]  # without the manifest: after opening
    c = out["chg"]
    assert c["state"]["w"] == {"range": [0, 3.5]} and c["state"]["L"] == {"points": True, "trust": False}
    assert c["dropped"] == ["等值线", "滞留区叠加", "固定色标的对数刻度", "配色", "分段色带", "dyn/cm² 单位", "阈值", "壁面透明度", "剖切",
                            "高亮最高区域", "高亮分支", "点云特征着色", "选中的发现", "自动标注", "测量", "探针记录", "英文标注"]
    assert out["logOn"] == "adaptive" and out["osi"] == "adaptive"               # OSI is linear either way
    assert out["cl"]["L"] == {"centerline": True, "trust": True} and out["cl"]["br"] is None and out["cl"]["f"] == "wss"
    assert out["missing"] is None                                                   # a field the result lacks is left out
    assert out["fixedOther"] == "adaptive-linear"                                   # a fixed range set for another field
    assert out["bad"] and out["junk"] == "throws"


VOLUME_STATE = {  # volume_viewer.js captureView() on the 横截面 page
    "schema_version": "wss-deploy.view/v1", "family": "volume", "run_identity": "fedcba9876543210ABC",
    "camera": {"position": [0, 0, 400], "target": [0, 0, 100], "up": [0, 1, 0]},
    "field": "velocity", "mode": "slice", "colormap": "rainbow", "range": {"mode": "case", "min": 0, "max": 1.2}, "bands": 0, "log": False,
    "units": "m/s", "units_by_field": {"pressure": "Pa", "velocity": "m/s"}, "pressure_units": "Pa", "speed_units": "m/s", "thresholds_pa": None,
    "opacity": 0.1, "overlay": {"trust": False, "contours": False},
    "slice": {"basis": "centerline", "branch": "2", "position": 35, "pitch": 10, "yaw": -5, "thickness": 3, "offset_u": 1.5, "offset_v": -2,
              "picks": [], "cut": True, "module": "cut-negative", "fill": True, "series": None,
              "color_range": {"mode": "manual", "min": 0.1, "max": 0.9}, "quantity": "normal", "arrows": True},
    "highlight": {"finding": None, "region": None}, "streamlines": {"width": 1, "density": "1", "thin": False}, "vectors": False,
    "montage": {"views": ["front"], "columns": 2}, "measurements": [], "annotations": {"items": []}, "probe_log": [], "branches_hidden": [],
    "preset_name": None, "lang": "zh", "export": {}, "labels": {"findings": 0, "branches": False, "max_diameter": True},
    "findings_review": {"items": {}, "added": []}}


def test_classic_volume_link_converts_to_the_workspace_state():
    program = """
      require(@@common@@); require(@@vv@@);
      const V=globalThis.VolumeViewerCore, S=@@state@@;
      const text=V.encodeView(S);
      const st=L.decodeClassicView(text);
      const ctx={family:'volume', fields:['speed','pressure','wall_pressure'], logFields:[], branches:[0,2,3]};
      const slice=L.classicToV2(st, ctx);
      const variant=o=>L.classicToV2(Object.assign({}, S, o), ctx);
      const old=JSON.parse(JSON.stringify(S)); delete old.slice.color_range; delete old.slice.arrows; delete old.slice.quantity;
      out({same: JSON.stringify(V.decodeView(text))===JSON.stringify(st), slice,
        pressure: variant({field:'pressure'}).state.sl.quantity,
        pick: variant({slice:Object.assign({}, S.slice, {basis:'pick', picks:[[0,0,0],[1,1,1]]})}),
        axis: variant({slice:Object.assign({}, S.slice, {basis:'z'})}).dropped,
        pos: variant({slice:Object.assign({}, S.slice, {module:'cut-positive'})}).state.sl.cut,
        uncut: variant({slice:Object.assign({}, S.slice, {cut:false})}).state.sl.cut,
        old: L.classicToV2(old, ctx).state.sl,
        wall: variant({mode:'wall', field:'pressure', slice:Object.assign({}, S.slice, {cut:false})}).state, cutWall: !!variant({mode:'wall'}).state.sl, lines: variant({mode:'streamlines'}).state, cloud: variant({mode:'cloud', field:'pressure'}).state,
        drops: variant({mode:'streamlines', log:true, units_by_field:{pressure:'mmHg', velocity:'m/s'}, opacity:0.3, streamlines:{width:2, density:'3', thin:true},
          vectors:true, highlight:{finding:'F1', region:{branch:'0', smin:10, smax:50}}, overlay:{trust:true}}).dropped});
    """
    program = program.replace("@@state@@", json.dumps(VOLUME_STATE, ensure_ascii=False))
    out = _node(program, common=STATIC_DIR / "report_common.js", vv=STATIC_DIR / "volume_viewer.js")
    assert out["same"] is True
    s = out["slice"]
    assert s["family"] == "volume" and s["dropped"] == []
    st = s["state"]
    assert st["f"] == "speed" and st["w"] == "adaptive" and st["L"] == {"streamlines": False, "trust": False} and st["run"] == "fedcba9876543210"
    assert st["cam"]["fov"] == 30 and st["cam"]["t"] == [0, 0, 100]
    import math
    k = math.tan(math.radians(42 / 2)) / math.tan(math.radians(30 / 2))
    assert st["cam"]["p"] == pytest.approx([0, 0, 100 + 300 * k], abs=1e-3)
    assert st["sl"] == {"basis": "centerline", "pick": None, "picks": [], "shift": 0, "segment": 2, "fraction": 0.35, "pitch": 10, "yaw": -5,
                        "offU": 1.5, "offV": -2, "thickness": 3, "range": "manual", "manual": {"min": 0.1, "max": 0.9}, "quantity": "normal",
                        "arrows": True, "fill": True, "cut": "neg"}
    assert out["pressure"] == "pressure" and out["pos"] == "pos" and out["uncut"] == "none"
    pick = out["pick"]
    assert pick["dropped"] == ["点选截面"] and pick["state"]["sl"]["basis"] == "centerline" and pick["state"]["sl"]["fraction"] == 0.5
    assert pick["state"]["sl"]["pitch"] == 0 and pick["state"]["sl"]["offU"] == 0
    assert out["axis"] == ["按坐标轴放的截面"]
    # a state written before v0.12 (no colour keys) replays as it looked: whole-field range, speed, no arrows
    assert out["old"]["range"] == "global" and out["old"]["quantity"] == "speed" and out["old"]["arrows"] is False
    assert out["wall"]["f"] == "wall_pressure" and "sl" not in out["wall"]
    assert out["cutWall"] is True                     # a cut vessel keeps its section (the classic page showed the slice panel)
    assert out["lines"]["f"] == "speed" and out["lines"]["L"]["streamlines"] is True and out["lines"]["L"]["interior"] is False
    assert out["cloud"]["f"] == "pressure" and out["cloud"]["L"]["interior"] is True and out["cloud"]["L"]["streamlines"] is False
    assert out["drops"] == ["对数色标", "单位", "外壁透明度", "流线粗细和密度", "速度箭头", "区域统计", "选中的发现"]


def test_boot_rewrite_and_the_state_applied_when_the_result_opens(tmp_path):
    program = """
      require(@@core@@);
      const C=globalThis.WssReportCore, S=@@state@@;
      const calls=[];
      const win=(search, hash)=>({location:{search, hash, pathname:'/v2/'}, history:{replaceState:(a,b,url)=>calls.push(url)}});
      const r1=L.rewrite(win('?job=J1&x=1', C.viewHash(S))), p1=L.pending();
      const r2=L.rewrite(win('', '#job=J2')), p2=L.pending();
      const r3=L.rewrite(win('', '#/job/J3?v=compare')), p3=L.pending();
      const r4=L.rewrite(win('?job=J4', '#view=@@@')), p4=L.pending();
      const r5=L.rewrite(win('?job=J5', '#view=bm90IGpzb24')), p5=L.pending();
      const r6=L.rewrite(win('?job=J6', '#/job/J6?f=osi'));
      // after the result opened: the manifest completes the conversion (shown branches), ws_figure applies it
      L.rewrite(win('?job=J1', C.viewHash(Object.assign({}, S, {colormap:'turbo'}))));
      const applied=[], toasts=[];
      globalThis.WSSV2.figure={encodeState:s=>Buffer.from(JSON.stringify(s)).toString('base64'), decodeState:t=>JSON.parse(Buffer.from(t,'base64').toString()),
        applyView:(api, st)=>{ applied.push(st); return Promise.resolve(['链接来自这份结果的另一次计算，已按可用项恢复']); }};
      const manifest={result:{family:'wall'}, fields:[{id:'wss'},{id:'tawss', display:{log_scale:true}},{id:'osi'}], geometry:{branches:[{id:0},{id:2},{id:4}]}};
      const api=(id)=>{ const cur={jobId:id, manifest}; return {cur:()=>cur, ui:{toast:(t,o)=>toasts.push([t,o.kind])}}; };
      const other=await L.applyPending(api('OTHER'));
      L.rewrite(win('?job=J1', C.viewHash(Object.assign({}, S, {colormap:'turbo'}))));
      const res=await L.applyPending(api('J1'));
      const again=await L.applyPending(api('J1'));
      L.rewrite(win('?job=J4', '#view=@@@'));
      await L.applyPending(api('J4'));
      const hooks=(globalThis.WSSV2.ext||[]).filter(x=>x.id==='legacy').length;
      out({r1, p1:!!p1&&p1.jobId, r2, p2, r3, p3, r4:r4.url, p4, r5:r5.url, p5, r6, calls, other, applied, toasts, res:!!res, again, hooks});
    """
    program = program.replace("@@state@@", json.dumps(WALL_STATE, ensure_ascii=False))
    out = _node(program, core=_wall_core(tmp_path))
    assert out["r1"]["url"] == "/v2/?x=1#/job/J1?f=tawss" and out["p1"] == "J1"
    assert out["r2"]["url"] == "/v2/#/job/J2" and out["p2"] is None               # the workbench link: a route, nothing pending
    assert out["r3"] is None and out["p3"] is None                                  # already a workspace route: untouched
    assert out["r4"] == "/v2/#/job/J4" and out["p4"] == {"jobId": "J4", "classic": None, "unreadable": True}
    assert out["r5"] == "/v2/#/job/J5" and out["p5"]["unreadable"] is True
    assert out["r6"]["url"] == "/v2/#/job/J6?f=osi"                                  # only ?job= removed
    assert out["calls"][0] == "/v2/?x=1#/job/J1?f=tawss"
    assert out["other"] is None                                                     # another result opened first: dropped
    assert len(out["applied"]) == 1
    st = out["applied"][0]
    assert st["v"] == 1 and st["f"] == "tawss" and st["w"] == "adaptive-linear" and st["br"] == [0, 4]   # branch 2 hidden
    assert out["res"] is True and out["again"] is None and out["hooks"] == 1
    assert out["toasts"][0] == ["已按旧版链接打开。链接来自这份结果的另一次计算，已按可用项恢复；旧版链接里的配色没有带过来。", "info"]
    assert out["toasts"][1] == ["旧版链接里的视图无法读取，已按默认打开。", "info"]


def test_classic_state_waits_for_the_saved_view_restore(tmp_path):
    """The shell restores this browser's saved reading position asynchronously after onResult; the classic link must be
    applied after it, or the restore overwrites camera and shown branches (seen in the sandbox walk-through)."""
    program = """
      require(@@core@@);
      const C=globalThis.WssReportCore, S=@@state@@;
      const handlers=[], applied=[];
      const viewer={on:(n,f)=>{ handlers.push(f); return ()=>{ handlers.length=0; }; }};
      globalThis.WSSV2.store={readView:run=>run==='R1'?{run_identity:'R1', viewer:{camera:{position:[1,2,3]}}}:null};
      globalThis.WSSV2.figure={applyView:(api, st)=>{ applied.push(Date.now()); return Promise.resolve([]); }};
      const manifest={result:{family:'wall'}, fields:[{id:'tawss'}], geometry:{branches:[{id:0},{id:2}]}};
      const cur={jobId:'J1', runIdentity:'R1', manifest};
      const api={cur:()=>cur, viewer:()=>viewer, ui:{toast(){}}};
      L.rewrite({location:{search:'?job=J1', hash:C.viewHash(S), pathname:'/v2/'}, history:{replaceState(){}}});
      const p=L.applyPending(api);
      await new Promise(r=>setTimeout(r, 50));
      const before=applied.length, subscribed=handlers.length;
      const t=Date.now(); handlers.forEach(f=>f({what:'branches'}));
      await p;
      // no saved view: applied at once
      const cur2={jobId:'J2', runIdentity:'R2', manifest};
      L.rewrite({location:{search:'?job=J2', hash:C.viewHash(S), pathname:'/v2/'}, history:{replaceState(){}}});
      await L.applyPending({cur:()=>cur2, viewer:()=>viewer, ui:{toast(){}}});
      out({before, subscribed, after:applied.length, unsubscribed:handlers.length===0, order:applied[0]>=t});
    """.replace("@@state@@", json.dumps(WALL_STATE, ensure_ascii=False))
    out = _node(program, core=_wall_core(tmp_path))
    assert out == {"before": 0, "subscribed": 1, "after": 2, "unsubscribed": True, "order": True}


def _harness(scenario: str, files: list[str], *, before: str = "") -> dict:
    _need_node()
    program = (_HARNESS.replace("__WS__", json.dumps([str(V2 / f) for f in files])).replace("__OFFLINE__", "false")
               + before + "\n(async () => {\n" + scenario + "\n})().catch(e => { errors.push('scenario: ' + (e && e.stack || e)); done({}); });\n")
    result = subprocess.run(["node", "-e", program], text=True, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise AssertionError(result.stderr[-4000:])
    lines = [line for line in result.stdout.strip().splitlines() if line.startswith("{")]
    assert lines, result.stdout[-2000:] + result.stderr[-2000:]
    return json.loads(lines[-1])


# Workspace harness files in bundle order, with ws_figure (applies the state) and ws_legacy (after the shell).
LEGACY_FILES = [f for f in WS_FILES if f not in {"ws_shell.js", "ws_main.js"}] + ["ws_figure.js", "ws_shell.js", "ws_legacy.js", "ws_main.js"]


def test_a_classic_report_link_opens_the_result_in_the_workspace(tmp_path):
    from wss_deploy.report import REPORT_CORE_JS
    state = dict(WALL_STATE, branches_hidden=[2], bands=6)
    before = ("\nglobal.WSSV2_LEGACY_AUTOREWRITE = true;\n" + REPORT_CORE_JS + "\n"
              "global.location.search = '?job=A'; global.location.hash = WssReportCore.viewHash(" + json.dumps(state, ensure_ascii=False) + ");\n")
    out = _harness("""
      const seen = [];
      await boot();
      const F = ns.figure, orig = F.applyView;
      await wait(400);
      done({hash: global.location.hash, field: shellState().cur && shellState().cur.field, win: shellState().cur && shellState().cur.window,
            toast: textOf(byId('ws-toast')), replaced: calls.filter(c => c.startsWith('REPLACE ')), applyStates: vcalls.filter(v => v.endsWith(':applyState')).length});
    """, LEGACY_FILES, before=before)
    assert out["errors"] == [], out["errors"]
    assert out["replaced"][0] == "REPLACE /v2/#/job/A?f=tawss"
    assert out["hash"].startswith("#/job/A") and "view=" not in out["hash"]
    assert out["field"] == "tawss" and out["win"] == "adaptive-linear"
    assert "GET /api/v2/jobs/A/manifest" in out["calls"]
    assert out["applyStates"] >= 1                                                  # camera / shown branches through the viewer
    assert "已按旧版链接打开" in out["toast"] and "分段色带" in out["toast"] and "另一次计算" in out["toast"]


# ----------------------------------------------------------------------------------------------- shell entries
def test_menus_offer_ops_and_support_and_no_classic_page():
    out = _harness("""
      await boot();
      await hashTo('#/job/A', 200);
      const avatar = walk(app(), e => e.className === 'top-avatar')[0];
      fire(avatar, 'click'); await wait(20);
      const items = walk(byId('ws-menu'), e => String(e.className).includes('menu-item')).map(e => [textOf(e), e.href || e.attrs.href || null, e.target || e.attrs.target || null]);
      fire(avatar, 'click'); await wait(10);
      const help = walk(app(), e => e.attrs && e.attrs['aria-label'] === '帮助' || e.title === '帮助')[0];
      fire(help, 'click'); await wait(20);
      const helpItems = walk(byId('ws-menu'), e => String(e.className).includes('menu-item')).map(e => [textOf(e), e.href || e.attrs.href || null]);
      fire(help, 'click'); await wait(10);
      // the 工具 tab: no 经典报告 section any more
      const tools = walk(app(), e => e.dataset && e.dataset.tab === 'tools' || textOf(e) === '工具')[0];
      if (tools) fire(tools, 'click');
      await wait(40);
      // loading failure: 重试 and the data package instead of 「在经典报告中打开」
      canned['/api/v2/jobs/B/manifest'] = {status: 500, body: {error: {message: '坏了'}}};
      await hashTo('#/job/B', 200);
      const msg = byClass(app(), 'stage-msg')[0];
      const actions = walk(msg, e => e.tagName === 'BUTTON' || e.tagName === 'A').map(e => [textOf(e), e.href || e.attrs.href || null]);
      done({items, helpItems, text: visibleText(), actions, msg: textOf(msg)});
    """, WS_FILES)
    assert out["errors"] == [], out["errors"]
    labels = [row[0] for row in out["items"]]
    assert "经典工作台" not in labels and ["运维中心", "/ops", "_blank"] in out["items"]
    assert labels.index("运维中心") < labels.index("退出登录")
    assert ["反馈问题", "/support"] in out["helpItems"]
    assert "经典报告" not in out["text"] and "经典工作台" not in out["text"]
    assert ["重试", None] in out["actions"] and ["下载数据包", "/api/jobs/B/bundle.zip"] in out["actions"]
    assert "经典" not in out["msg"]


def test_ops_entry_only_for_administrators_of_a_shared_service():
    out = _harness("""
      canned['/api/session'] = {body: {authenticated: true, shared: true, login: 'password', csrf_token: 'c', username: 'doc', role: 'user'}};
      await boot();
      const avatar = walk(app(), e => e.className === 'top-avatar')[0];
      fire(avatar, 'click'); await wait(20);
      const menu = byId('ws-menu');
      const labels = byClass(menu, 'menu-label').map(textOf);
      const seps = walk(menu, e => String(e.className) === 'menu-sep').length;
      const kids = menu.children.map(e => String(e.className));
      const doubled = kids.some((c, i) => c === 'menu-sep' && kids[i + 1] === 'menu-sep');
      done({labels, doubled, seps});
    """, WS_FILES)
    assert out["errors"] == [], out["errors"]
    assert "运维中心" not in out["labels"] and "退出登录" in out["labels"] and out["doubled"] is False


def test_no_workspace_source_links_a_classic_page():
    pattern = re.compile(r"""urls\.report\b|href: '/'|href="/"|'/#job=|"/#job=|/compare\?|在经典报告中打开|经典工作台""")
    offenders = {}
    for path in sorted(list(V2.glob("*.js")) + list(V2.glob("*.html"))):
        if path.name in {"example_report.html"}:
            continue
        text = path.read_text(encoding="utf-8")
        # comments may name the retired pages; code, markup and user-facing text may not
        text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m[0].count("\n"), text, flags=re.S)
        for number, line in enumerate(text.splitlines(), 1):
            line = re.sub(r"(^|\s)//.*$", "", line)
            if pattern.search(line):
                offenders.setdefault(path.name, []).append(number)
    # phase 3 lane 1 removes the input page's 「在经典工作台中处理」 (ws_shell.js showInput) and lane 2 the login footer
    # link (showLogin); both are theirs to delete (PHASE3_LANES.md §2).  Nothing else may remain.
    allowed = {"ws_shell.js"}
    assert set(offenders) <= allowed, offenders
    shell = (V2 / "ws_shell.js").read_text(encoding="utf-8").splitlines()
    for number in offenders.get("ws_shell.js", []):
        line = shell[number - 1]
        assert "在经典工作台中处理" in line or "login-foot" in line, (number, line)
    api = (V2 / "ws_api.js").read_text(encoding="utf-8")
    assert "report: function" not in api and "classic: function (id) { return id ? '/v2/#/job/'" in api
    assert "经典" not in (V2 / "ws_main.js").read_text(encoding="utf-8").split("*/", 1)[1]
    assert "经典" not in (V2 / "index.html").read_text(encoding="utf-8") and "经典" not in (V2 / "help_quickstart.html").read_text(encoding="utf-8")


def test_shortcut_help_names_the_changed_keys():
    out = _harness("""
      await boot();
      await key('?');
      done({dlg: textOf(byId('ws-dialog'))});
    """, WS_FILES)
    assert out["errors"] == [], out["errors"]
    for text in ("和旧版不同", "G 是沿血管游标", "O 打开一页纸", "复位视角按 0", "R 不再复位"):
        assert text in out["dlg"], text


# ----------------------------------------------------------------------------------------------- tools and texts
def test_devshot_logs_in_through_the_workspace_form():
    from wss_deploy import devshot
    browser = devshot.Browser.__new__(devshot.Browser)
    seen = {"go": [], "js": []}
    state = {"form": True}
    browser.go = lambda url, wait=3.0: seen["go"].append(url)
    def js(script, args=None):
        seen["js"].append((script, args))
        if "requestSubmit" in script:
            ok = state["form"]; state["form"] = False; return ok
        return state["form"]
    browser.js = js
    import time as _time
    sleep = _time.sleep
    try:
        _time.sleep = lambda *_: None
        assert browser.login("http://127.0.0.1:8826/", "admin", "pw") is True
        assert seen["go"] == ["http://127.0.0.1:8826/v2/"]
        script, args = seen["js"][0]
        assert "wsl-pass" in script and "wsl-user" in script and args == ["admin", "pw"]
        state["form"] = False
        assert browser.login("http://127.0.0.1:8826", "admin", "pw") is False     # no form: already logged in
        state["form"] = True
        browser.js = lambda script, args=None: True                               # the form stays: wrong password
        with pytest.raises(RuntimeError):
            browser.login("http://127.0.0.1:8826", "admin", "bad")
    finally:
        _time.sleep = sleep
    source = Path(devshot.__file__).read_text(encoding="utf-8")
    assert "login-panel" not in source and "/#job=" not in source and "/compare?left" not in source


def test_python_texts_point_at_the_workspace():
    from wss_deploy.onepager import NO_SNAPSHOTS_HINT
    package = STATIC_DIR.parent
    assert "三维报告" not in NO_SNAPSHOTS_HINT and "「导出」→「一页纸」" in NO_SNAPSHOTS_HINT
    assert "三维报告中的截面环" not in (package / "morphology.py").read_text(encoding="utf-8")
    cli = (package / "cli.py").read_text(encoding="utf-8")
    assert "{base}/v2/#/job/{job['id']}" in cli and "/#job=" not in cli
    service = (package / "service.py").read_text(encoding="utf-8")
    assert service.count(":{port}/v2/") + service.count(":{info['port']}/v2/") + service.count(":{cfg['port']}/v2/") == 3
    assert not re.search(r"""address(?:=|\s=\s)f"http://[^"]*:\{[^}]+\}/\"""", service)
    rehearse = (package / "rehearse.py").read_text(encoding="utf-8")
    assert '"/v2/"' in rehearse and "/api/v2/jobs/{job_id}/manifest" in rehearse and "打开报告与一页纸" not in rehearse
