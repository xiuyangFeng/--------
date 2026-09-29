"""CLI account maintenance must appear in the independent operations journal."""
import json

from wss_deploy.cli import main
from wss_deploy.operations import OperationsStore
from wss_deploy.users import UserStore


def test_cli_account_events_persist_without_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin", type("Input", (), {"isatty": staticmethod(lambda: False)})())
    monkeypatch.setenv("WSS_DEPLOY_PASSWORD", "private-cli-password-1")
    args = ["--jobs-root", str(tmp_path)]
    assert main(["user", "add", "admin", "--admin", *args]) == 0
    assert main(["user", "add", "alice", *args]) == 0
    monkeypatch.setenv("WSS_DEPLOY_PASSWORD", "private-cli-password-2")
    assert main(["user", "passwd", "alice", *args]) == 0
    assert main(["user", "role", "alice", "--admin", *args]) == 0
    assert main(["user", "disable", "alice", *args]) == 0
    events, total = OperationsStore(tmp_path).events(owner="alice")
    assert total == 4
    assert {row["action"] for row in events} == {"user_add", "user_passwd", "user_role", "user_disable"}
    assert all(row["source"] == "cli" and row["actor"].startswith("cli:") for row in events)
    encoded = json.dumps(events)
    assert "private-cli-password" not in encoded and "password_hash" not in encoded


def test_broken_operations_database_does_not_prevent_account_recovery(tmp_path, monkeypatch, capsys):
    users = UserStore(tmp_path / "users.json")
    users.add("admin", "old-password", admin=True)
    monkeypatch.setattr("sys.stdin", type("Input", (), {"isatty": staticmethod(lambda: False)})())
    monkeypatch.setenv("WSS_DEPLOY_PASSWORD", "new-private-password")
    def unavailable(*args, **kwargs):
        raise OSError("database unavailable")
    monkeypatch.setattr("wss_deploy.operations.OperationsStore", unavailable)
    assert main(["user", "passwd", "admin", "--jobs-root", str(tmp_path)]) == 0
    assert users.authenticate("admin", "new-private-password")
    assert "运维审计库不可用" in capsys.readouterr().err
    log = (tmp_path / "server.log").read_text()
    assert "user_passwd" in log and "new-private-password" not in log
