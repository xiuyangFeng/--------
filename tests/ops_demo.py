"""Isolated, persistent operations acceptance preview; synthetic data only.

Run from the repository root:
    python -m tests.ops_demo --root outputs/wss_ops_acceptance --port 18765

Never starts compute workers. Reuses the real HTTP/auth/storage/UI code and test
fixtures only for synthetic source records. Credentials are generated once in
the private preview directory, never accepted on the command line or printed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import secrets
import signal
import threading

from tests._c_helpers import manager, finished
from wss_deploy import server as http
from wss_deploy.jobs import JobError, atomic_json
from wss_deploy.operations import OperationsStore
from wss_deploy.service import acquire_lock
from wss_deploy.users import UserStore

SCHEMA = "wss-ops-acceptance/v1"
BANNER = ('<section class="card"><strong>独立验收实例 · 合成数据</strong>'
          '<p>此页面用于运维功能验收，不含真实病例。算例计算工作线程未启动；'
          '健康区显示计算服务未就绪属于预期。工单、审计、账号及 STL 留存使用实际接口。</p></section>')


class PreviewHandler(http.Handler):
    def _send_path(self, path, ctype, **kwargs):
        if Path(path).name in {"ops.html", "support.html"}:
            content = Path(path).read_text(encoding="utf-8")
            content = content.replace('<main class="ops-main">', '<main class="ops-main">' + BANNER, 1)
            return self._send(200, content.encode(), ctype, compress=True)
        return super()._send_path(path, ctype, **kwargs)

    def _post(self):
        path = self._path()
        # Allow real account/ticket/archive flows and synthetic deletion tests,
        # but never accept an upload or start a computation in this preview.
        if not (path in {"/api/session", "/api/session/logout"} or path.startswith(("/api/ops/", "/api/support/"))
                or re.fullmatch(r"/api/(?:jobs|trash)/[A-Za-z0-9_-]+/(?:delete|restore|purge)", path)):
            raise JobError("验收实例不启动算例计算；请使用工单、素材库和审计功能。", 403)
        return super()._post()

    def _put(self):
        raise JobError("验收实例不修改算例或工作台配置。", 403)


def prepare_root(root: Path) -> Path:
    if root.is_symlink():
        raise ValueError("验收目录不能是符号链接。")
    root = root.resolve()
    marker = root / "acceptance.json"
    if root.exists() and any(root.iterdir()):
        if not marker.is_file() or marker.is_symlink() or json.loads(marker.read_text()).get("schema") != SCHEMA:
            raise ValueError("拒绝使用非空的非验收目录；请指定一个新的独立目录。")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    return root


def seed(root, mgr, store, users):
    credentials_path = root / "credentials.json"
    if credentials_path.exists():
        return
    if users.load() or mgr.jobs:
        raise ValueError("验收数据已存在但口令文件丢失，请使用新的空目录。")
    accounts = []
    for name, display, admin in (("ops_admin", "验收管理员", True),
                                 ("researcher_a", "验收用户 A", False), ("researcher_b", "验收用户 B", False)):
        password = secrets.token_urlsafe(18)
        users.add(name, password, admin=admin, display_name=display)
        accounts.append({"username": name, "password": password, "role": "admin" if admin else "user"})
        store.record_event({"action": "user_add", "actor": "acceptance_seed", "owner": name, "source": "demo"})
    a = finished(mgr, owner="researcher_a", case_id="DEMO_A_合成算例", with_files=False,
                 content=b"solid synthetic_A\nendsolid synthetic_A\n")
    b = finished(mgr, owner="researcher_b", case_id="DEMO_B_合成算例", with_files=False,
                 content=b"solid synthetic_B\nendsolid synthetic_B\n")
    trash = finished(mgr, owner="researcher_a", case_id="DEMO_TRASH_回收站", with_files=False,
                     content=b"solid synthetic_trash\nendsolid synthetic_trash\n")
    failed = finished(mgr, owner="researcher_b", case_id="DEMO_FAILED_合成失败", with_files=False,
                      content=b"solid synthetic_failed\nendsolid synthetic_failed\n")
    with mgr.lock:
        item = mgr.jobs[failed["id"]]
        item.update(status="failed", phase="验收示例：几何检查失败", error={"message": "合成失败样例，用于检查运维错误展示。", "category": "input", "retryable": False})
        mgr._event(item, "failed")
    mgr.delete(trash["id"], "researcher_a", {"version": trash["version"]})
    store.create_ticket(owner="researcher_a", title="验收：核查 STL 几何", description="合成工单，请受理、回复并关闭。", job_id=a["id"])
    store.create_ticket(owner="researcher_b", title="验收：结果导出咨询", description="合成工单，用于验证用户之间的隔离。", job_id=b["id"], priority="high")
    atomic_json(credentials_path, {"notice": "仅供本独立验收实例使用，不是正式服务账号。", "accounts": accounts})
    os.chmod(credentials_path, 0o600)


def serve(root: Path, port: int, hours: float):
    os.umask(0o077)
    root = prepare_root(root)
    lock = acquire_lock(root, purpose="ops_acceptance", host="127.0.0.1", port=port)
    stop = threading.Event()
    mgr = server = None
    thread = None
    try:
        atomic_json(root / "acceptance.json", {"schema": SCHEMA})
        store = OperationsStore(root)
        users = UserStore(root / "users.json")
        mgr = manager(root)
        mgr.audit_sink = store.record_event
        seed(root, mgr, store, users)
        # A distinct cookie avoids replacing a production login on another
        # port of the same host. This constant changes only this process.
        http.COOKIE_NAME = "wss_ops_acceptance"
        sessions = http.SessionStore(root / "sessions.json", shared=True, users=users)
        server = http.ServiceHTTPServer(("127.0.0.1", port), mgr, sessions, operations=store)
        server.RequestHandlerClass = PreviewHandler
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        atomic_json(root / "preview.json", {"pid": os.getpid(), "ops": base + "/ops", "support": base + "/support",
                                            "expires_after_hours": hours, "synthetic_data_only": True})
        print(f"独立验收页面：{base}/ops\n用户工单：{base}/support\n账号文件：{root / 'credentials.json'}", flush=True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.set())
        stop.wait(hours * 3600)
    finally:
        if thread is not None:
            server.shutdown()
            thread.join(timeout=5)
        if server is not None:
            server.server_close()
        if mgr is not None:
            mgr.close()
        lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="空目录或之前的专用验收目录")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--hours", type=float, default=48, help="自动退出前保留小时数，默认48")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535 or not 0 < args.hours <= 168:
        parser.error("port 必须为 0–65535；hours 必须大于 0 且不超过 168。")
    serve(args.root, args.port, args.hours)


if __name__ == "__main__":
    main()
