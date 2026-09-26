"""§19.6: `cli doctor` — every check is ✓/⚠/✗ with advice, and the exit code is 1 exactly when something failed."""
from __future__ import annotations

import json
import socket

import pytest

from wss_deploy import doctor as D
from wss_deploy.cli import main as cli_main


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_run_checks_structure_and_exit_code(tmp_path):
    root = tmp_path / "jobs"
    root.mkdir()
    rows = D.run_checks(root, port=_free_port(), skip={"torch", "vmtk"})
    keys = [row["key"] for row in rows]
    for key in ("features", "glossary", "jobs_root", "disk", "service", "login", "reports"):
        assert key in keys, key
    assert any(key.startswith("release:") for key in keys) and any(key.startswith("reference:") for key in keys)
    assert all(row["status"] in {"ok", "info", "warn", "fail"} and row["label"] and row["detail"] for row in rows)   # v0.15: ℹ 信息
    by_key = {row["key"]: row for row in rows}
    assert by_key["jobs_root"]["status"] == "ok" and by_key["service"]["status"] == "warn" and "service start" in by_key["service"]["advice"]
    assert by_key["reports"]["status"] == "ok"
    assert all(row["status"] == "ok" for row in rows if row["key"].startswith("release:"))
    assert D.exit_code(rows) == (1 if any(row["status"] == "fail" for row in rows) else 0)
    text = D.format_rows(rows)
    assert "✓" in text and "共" in text


def test_failures_are_reported_with_advice(tmp_path, monkeypatch):
    import wss_features
    monkeypatch.setattr(wss_features, "contract", lambda: {"version": "v", "source_hash": "a" * 64, "recorded_hash": "b" * 64, "matches_record": False})
    rows = D.check_feature_contract()
    assert rows[0]["status"] == "fail" and "record" in rows[0]["advice"]
    monkeypatch.setattr("wss_deploy.glossary.glossary_json", lambda: "{}")
    rows = D.check_glossary()
    assert rows[0]["status"] == "fail" and "python -m wss_deploy.glossary" in rows[0]["advice"]
    monkeypatch.setattr("wss_deploy.service.disk_free_gb", lambda path: 3.0)
    assert {row["key"]: row["status"] for row in D.check_jobs_root(tmp_path)}["disk"] == "fail"
    monkeypatch.setattr("wss_deploy.service.disk_free_gb", lambda path: 12.0)
    assert {row["key"]: row["status"] for row in D.check_jobs_root(tmp_path)}["disk"] == "warn"
    assert D.exit_code([{"status": "ok"}, {"status": "warn"}]) == 0 and D.exit_code([{"status": "fail"}]) == 1


def test_reference_sidecar_and_service_states(tmp_path, monkeypatch):
    release = tmp_path / "REL"
    release.mkdir()
    assert D._reference_check("REL", release)["status"] == "warn"
    (release / "reference.json").write_text(json.dumps({"geometry_reference": {"release": "OTHER"}}))
    assert D._reference_check("REL", release)["status"] == "fail"
    (release / "reference.json").write_text(json.dumps({"geometry_reference": {"release": "REL"}}))
    row = D._reference_check("REL", release)
    assert row["status"] == "ok" and "几何范围" in row["detail"]
    base = {"running": True, "managed": False, "healthy": True, "legacy": False, "pid": 42, "version": "0.13.0", "address": "http://0.0.0.0:8765/",
            "shared": True, "port": 8765}
    monkeypatch.setattr("wss_deploy.service.status", lambda root, port=None: dict(base))
    rows = {row["key"]: row for row in D.check_service(tmp_path)}
    assert rows["service"]["status"] == "warn" and "service upgrade" in rows["service"]["advice"]
    assert rows["login"]["status"] == "warn" and "user add" in rows["login"]["advice"]
    (tmp_path / "users.json").write_text("{}")
    monkeypatch.setattr("wss_deploy.service.status", lambda root, port=None: dict(base, managed=True))
    monkeypatch.setattr(D, "_modules_newer_than_process", lambda pid: [])
    rows = {row["key"]: row for row in D.check_service(tmp_path)}
    assert rows["service"]["status"] == "ok" and rows["login"]["status"] == "ok"
    # modules edited after the process started: the running service still has the old code
    monkeypatch.setattr(D, "_modules_newer_than_process", lambda pid: ["wss_deploy/volume_report.py"])
    row = {row["key"]: row for row in D.check_service(tmp_path)}["service"]
    assert row["status"] == "warn" and "volume_report.py" in row["detail"] and "service upgrade" in row["advice"]
    monkeypatch.setattr("wss_deploy.service.status", lambda root, port=None: dict(base, managed=True, healthy=False))
    assert {row["key"]: row["status"] for row in D.check_service(tmp_path)}["service"] == "fail"
    monkeypatch.setattr("wss_deploy.service.status", lambda root, port=None: dict(base, legacy=True, version=None))
    assert {row["key"]: row["status"] for row in D.check_service(tmp_path)}["service"] == "warn"


@pytest.mark.parametrize("fmt", ["json", "text"])
def test_cli_doctor(tmp_path, capsys, fmt):
    args = ["doctor", "--jobs-root", str(tmp_path), "--port", str(_free_port()), "--skip", "torch,vmtk,releases"]
    code = cli_main(args + (["--json"] if fmt == "json" else []))
    out = capsys.readouterr().out
    if fmt == "json":
        payload = json.loads(out)
        assert payload["exit_code"] == code and {row["key"] for row in payload["checks"]} >= {"features", "glossary", "service"}
    else:
        assert "服务" in out and ("✓" in out or "⚠" in out)
    assert code in (0, 1)


def test_environment_checks_run(monkeypatch):
    """torch / VMTK checks against the real environments (no CUDA context is created)."""
    rows = {row["key"]: row for row in D.check_torch() + D.check_vmtk()}
    assert rows["torch"]["status"] == "ok" and rows["cuda"]["status"] in {"ok", "warn"}
    assert rows["vessel_geom"]["status"] == "ok" and rows["vmtk"]["status"] == "ok"
    monkeypatch.setattr("wss_deploy.paths.VMTK_PYTHON", "/nonexistent/python")
    assert {row["key"]: row["status"] for row in D.check_vmtk()}["vmtk"] == "fail"


def test_lock_git_and_access_log_checks(tmp_path, monkeypatch):
    """v0.14: writer lock (held / released / missing), git_dirty and the access.log path."""
    from wss_deploy import schema
    from wss_deploy import service as S
    root = tmp_path / "jobs"
    root.mkdir()
    assert D.check_lock(root)[0]["status"] == "ok" and "没有进程" in D.check_lock(root)[0]["detail"]
    lock = S.acquire_lock(root, purpose="serve")
    try:
        row = D.check_lock(root)[0]
        assert row["status"] == "ok" and "由服务" in row["detail"]
    finally:
        lock.release()
    other = S.acquire_lock(root, purpose="jobs claim")
    try:
        assert D.check_lock(root)[0]["status"] == "warn"                 # a maintenance command blocks serve
    finally:
        other.release()
    assert "flock 已随进程释放" in D.check_lock(root)[0]["detail"]
    assert D.check_access_log(root)[0]["status"] == "ok" and "尚未生成" in D.check_access_log(root)[0]["detail"]
    (root / S.ACCESS_LOG).write_text("x\n")
    assert "轮转" in D.check_access_log(root)[0]["detail"]
    monkeypatch.setattr(schema, "code_provenance", lambda refresh=False: {"deploy_version": "0.14", "git_describe": "abc-dirty", "git_dirty": True, "source_hash": "f" * 64})
    assert D.check_git()[0]["status"] == "warn" and "未提交" in D.check_git()[0]["detail"]
    monkeypatch.setattr(schema, "code_provenance", lambda refresh=False: {"deploy_version": "0.14", "git_describe": "abc", "git_dirty": False, "source_hash": "f" * 64})
    assert D.check_git()[0]["status"] == "ok"
    keys = [row["key"] for row in D.run_checks(root, port=_free_port(), skip={"torch", "vmtk", "releases", "features", "glossary", "service", "reports",
                                                                            "exposure", "cache", "tz"})]
    assert keys == ["jobs_root", "disk", "lock", "access_log", "log_rotation", "git"]     # v0.15: logs group adds log_rotation
