"""Developer screenshot tool ``wss_deploy.devshot`` (v0.12): error filtering, sandbox guard, optional real browser run."""
import os
import shutil
from pathlib import Path

import pytest

from wss_deploy import devshot


def test_console_filter_keeps_page_errors_and_drops_browser_chrome():
    lines = ["JavaScript error: resource://gre/modules/XULStore.sys.mjs, line 60: Error: Can't find profile directory.",
             'console.error: "boom"', "JavaScript error: http://127.0.0.1:8799/x.html, line 1: ReferenceError: f is not defined",
             "console.log: fine"]
    kept = [l for l in lines if devshot._ERROR_RE.search(l) and not devshot._BROWSER_RE.search(l)]
    assert kept == lines[1:3]


def test_sandbox_refuses_to_run_on_the_source_jobs_root(tmp_path):
    with pytest.raises(SystemExit):
        devshot.sandbox(["x"], tmp_path, 1, source=tmp_path)


def test_sandbox_requires_existing_jobs(tmp_path):
    (tmp_path / "src").mkdir()
    with pytest.raises(SystemExit):
        devshot.sandbox(["missing"], tmp_path / "dst", 1, source=tmp_path / "src")


@pytest.mark.skipif(os.environ.get("WSS_DEPLOY_DEVSHOT") != "1" or not shutil.which("firefox"),
                    reason="set WSS_DEPLOY_DEVSHOT=1 on a node with Firefox to run the real headless browser")
def test_real_browser_captures_page_errors_and_webgl(tmp_path):
    page = tmp_path / "p.html"
    page.write_text("<html><body><canvas id=c></canvas><script>window.gl=!!document.getElementById('c').getContext('webgl');"
                    "console.error('boom-console');</script></body></html>", encoding="utf-8")
    with devshot.Browser(width=640, height=480, marionette_port=2839) as browser:
        browser.go(page.as_uri(), 2)
        assert browser.js("return window.gl") is True
        out = browser.shot(tmp_path / "p.png")
        assert Path(out).stat().st_size > 1000
        assert any("boom-console" in line for line in browser.errors())


def test_login_spec_and_sandbox_users_file(tmp_path):
    """v0.15: ``--login user:password`` → a users.json of the sandbox's own (administrator), nowhere else."""
    assert devshot.parse_login(None) is None and devshot.parse_login("admin:pa:ss") == ("admin", "pa:ss")
    for bad in ("admin", ":x", "admin:"):
        with pytest.raises(SystemExit):
            devshot.parse_login(bad)
    from wss_deploy.users import UserStore
    path = devshot.write_sandbox_users(tmp_path, "admin:sandbox-pass-1")
    assert path == tmp_path / "users.json"
    store = UserStore(path)
    assert store.names() == ["admin"] and store.verify("admin", "sandbox-pass-1")["role"] == "admin"
    devshot.write_sandbox_users(tmp_path, "admin:another-pass")          # existing user: left as it is
    assert UserStore(path).verify("admin", "sandbox-pass-1")["username"] == "admin"
    assert "is_loopback = lambda host: False" in devshot._SHARED_LAUNCHER and "127.0.0.1" in devshot._SHARED_LAUNCHER
