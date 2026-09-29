"""The preview must refuse existing data and must never offer computation."""
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tests.ops_demo import prepare_root


def test_preview_refuses_unmarked_data_and_symlinks(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    record = existing / "job.json"
    record.write_text('{"original": true}')
    with pytest.raises(ValueError, match="非验收目录"):
        prepare_root(existing)
    link = tmp_path / "link"
    link.symlink_to(existing, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        prepare_root(link)
    assert json.loads(record.read_text()) == {"original": True}
    assert sorted(p.name for p in existing.iterdir()) == ["job.json"]


def test_preview_authentication_no_compute_and_restart(tmp_path):
    root = tmp_path / "preview"
    command = [sys.executable, "-m", "tests.ops_demo", "--root", str(root), "--port", "0", "--hours", "0.1"]
    credentials = None
    for _ in range(2):
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        api = None
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError(process.stderr.read().decode())
                preview = root / "preview.json"
                if preview.exists():
                    info = json.loads(preview.read_text())
                    if info["pid"] == process.pid:
                        break
                time.sleep(.1)
            else:
                raise AssertionError("preview did not start")
            from urllib.parse import urlsplit
            api = http.client.HTTPConnection("127.0.0.1", urlsplit(info["ops"]).port, timeout=5)
            api.request("GET", "/ops")
            response = api.getresponse()
            assert response.status == 200 and "独立验收实例" in response.read().decode()
            saved = json.loads((root / "credentials.json").read_text())
            if credentials is not None:
                assert saved == credentials
            credentials = saved
            assert os.stat(root / "credentials.json").st_mode & 0o777 == 0o600
            admin = next(account for account in saved["accounts"] if account["role"] == "admin")
            api.request("POST", "/api/session", json.dumps({key: admin[key] for key in ("username", "password")}), {"Content-Type": "application/json"})
            response = api.getresponse()
            session = json.loads(response.read())
            assert response.status == 200
            cookie = response.getheader("Set-Cookie").split(";", 1)[0]
            assert cookie.startswith("wss_ops_acceptance=")
            headers = {"Cookie": cookie, "X-CSRF-Token": session["csrf_token"], "Content-Type": "application/json"}
            api.request("POST", "/api/jobs", b"{}", headers)
            response = api.getresponse()
            assert response.status == 403
            assert "不启动算例计算" in response.read().decode()
            api.request("GET", "/api/ops/overview", headers=headers)
            response = api.getresponse()
            overview = json.loads(response.read())
            assert response.status == 200
            assert overview["counts"]["users"] == 3
            assert overview["counts"]["jobs"] == 3 and overview["counts"]["trash"] == 1
            assert not overview["health"]["worker_alive"]
            api.request("POST", "/api/session/logout", "{}", headers)
            response = api.getresponse()
            assert response.status == 200
            response.read()
        finally:
            if api:
                api.close()
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            process.stderr.close()
