"""§19.4: institution template — GET/PUT /api/report-template (loopback or admin only) and `cli template`."""
from __future__ import annotations

import json

import pytest

from wss_deploy import report_template as T
from wss_deploy.cli import main as cli_main
from wss_deploy.users import UserStore
from tests._c_helpers import Service, call


def test_loopback_get_put_merges_and_validates(tmp_path):
    service = Service(tmp_path)
    api = service.api()
    try:
        cookie, headers, _ = service.session(api)
        status, payload, _ = call(api, "GET", "/api/report-template", {"Cookie": cookie})
        assert status == 200 and payload["editable"] is True and payload["template"] == T.DEFAULTS
        status, payload, _ = call(api, "PUT", "/api/report-template", headers, {"institution": "某某医院", "signature_lines": ["报告人"]})
        assert status == 200 and payload["template"]["institution"] == "某某医院" and payload["template"]["signature_lines"] == ["报告人"]
        assert payload["template"]["show_glossary"] == "used"          # untouched keys keep their values
        status, payload, _ = call(api, "PUT", "/api/report-template", headers, {"department": "血管外科"})
        assert payload["template"]["institution"] == "某某医院" and payload["template"]["department"] == "血管外科"
        stored = json.loads((service.manager.root / "report_template.json").read_text(encoding="utf-8"))
        assert stored["department"] == "血管外科"
        for bad in ({"show_glossary": "some"}, {"unknown": 1}, {"signature_lines": ["a"] * 5}):
            status, payload, _ = call(api, "PUT", "/api/report-template", headers, bad)
            assert status == 400 and payload["error"]["message"]
        status, _, _ = call(api, "PUT", "/api/report-template", {"Cookie": cookie, "Content-Type": "application/json"}, {"institution": "x"})
        assert status == 403                                               # CSRF token required
        status, payload, _ = call(api, "GET", "/api/report-template", {"Cookie": cookie})
        assert payload["template"]["department"] == "血管外科"
    finally:
        api.close(); service.close()


def test_shared_mode_only_admin_may_edit(tmp_path):
    users = UserStore(tmp_path / "users.json")
    users.add("alice", "alice-secret")
    users.add("root", "root-secret", admin=True)
    service = Service(tmp_path, shared=True, token="legacy", users=users)
    api = service.api()
    try:
        status, payload, cookie, headers = service.login(api, {"username": "alice", "password": "alice-secret"})
        assert status == 200
        status, payload, _ = call(api, "GET", "/api/report-template", {"Cookie": cookie})
        assert status == 200 and payload["editable"] is False
        status, payload, _ = call(api, "PUT", "/api/report-template", headers, {"institution": "x"})
        assert status == 403 and "管理员" in payload["error"]["message"]
        status, payload, cookie, headers = service.login(api, {"username": "root", "password": "root-secret"})
        status, payload, _ = call(api, "PUT", "/api/report-template", headers, {"institution": "中心医院", "appendix": False})
        assert status == 200 and payload["editable"] is True and payload["template"]["appendix"] is False
        status, _, _ = call(api, "GET", "/api/report-template", {})
        assert status == 401                                               # no session, no template
    finally:
        api.close(); service.close()


def test_cli_template_show_and_set(tmp_path, capsys):
    root = tmp_path / "jobs"
    assert cli_main(["template", "show", "--jobs-root", str(root)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["exists"] is False and shown["template"] == T.DEFAULTS
    assert cli_main(["template", "set", "--jobs-root", str(root), "--institution", "某医院", "--department", "血管外科",
                     "--title", "WSS 报告", "--footer", "内部使用", "--signatures", "报告人,审阅人,主任", "--glossary", "all",
                     "--appendix", "off"]) == 0
    saved = json.loads(capsys.readouterr().out)["template"]
    assert saved == {**T.DEFAULTS, "institution": "某医院", "department": "血管外科", "report_title": "WSS 报告", "footer_note": "内部使用",
                     "signature_lines": ["报告人", "审阅人", "主任"], "show_glossary": "all", "appendix": False}
    assert T.load(root) == saved
    assert cli_main(["template", "set", "--jobs-root", str(root), "--signatures", ""]) == 0
    assert T.load(root)["signature_lines"] == [] and T.load(root)["institution"] == "某医院"
    assert cli_main(["template", "set", "--jobs-root", str(root), "--reset"]) == 0
    assert T.load(root) == T.DEFAULTS
    with pytest.raises(SystemExit):
        cli_main(["template", "set", "--jobs-root", str(root), "--signatures", "一,二,三,四,五"])
