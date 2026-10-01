"""Headless-browser screenshots for UI checks (developer tool, v0.12; no selenium / geckodriver needed).

Drives the system Firefox through its Marionette socket with software WebGL (Mesa llvmpipe), so the three.js
reports render on a GPU-less node.  Console output of the page is captured, and JavaScript errors are
reported, so a layout change can be checked without asking a person for a screenshot.

    python -m wss_deploy.devshot http://127.0.0.1:8799/ out.png [--width 1440 --height 900 --wait 4 --full]
        [--js "document.querySelector('[data-job-id]').click()" --after 6]   # run script, wait, then shoot
    python -m wss_deploy.devshot suite --base http://127.0.0.1:8799 --jobs <id,…> --out <dir>
        # standard page set of the workspace /v2/ (home / results / one-pagers / compare / a classic report address /
        # narrow widths) + index.json of JS errors
    python -m wss_deploy.devshot sandbox --jobs 20260922_173705_27065942b465,... --dir <scratch>/jobs --port 8799
        # copy finished jobs from outputs/wss_deploy_jobs and start a loopback server on the copy (CPU only)
    python -m wss_deploy.devshot sandbox ... --login admin:sandbox-pass     # v0.15: user-name login mode (like the
        # production service) on the loopback copy: writes the sandbox's own users.json; then pass the same
        # ``--login admin:sandbox-pass`` to ``suite`` or to a single shot so the browser logs in first

Use a private port pair per developer: the HTTP port of the sandbox server and ``--marionette-port``.
Never point it at the production jobs root: the sandbox server runs the worker (trash purge, resume).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

_PREFS = {
    "browser.shell.checkDefaultBrowser": False, "datareporting.policy.dataSubmissionEnabled": False,
    "toolkit.telemetry.reportingpolicy.firstRun": False, "browser.startup.homepage_override.mstone": "ignore",
    "intl.accept_languages": "zh-CN,zh", "dom.disable_open_during_load": False,
    # software WebGL on a node without X / GPU
    "webgl.force-enabled": True, "webgl.disabled": False, "webgl.out-of-process": False,
    "webgl.disable-fail-if-major-performance-caveat": True, "webgl.force-layers-readback": True,
    "gfx.x11-egl.force-enabled": True, "layers.gpu-process.enabled": False, "webgl.prefer-native-gl": False,
    "layers.acceleration.force-enabled": False, "gfx.webrender.software": True,
    # page console → Firefox stdout (captured into the log file)
    "devtools.console.stdout.content": True, "browser.dom.window.dump.enabled": True,
}
_BROWSER_RE = re.compile(r"(resource://|chrome://|moz-extension://|XULStore|addons\.xpi|dummy-system-addons)")  # Firefox's own chrome, not the page
_ERROR_RE = re.compile(r"(JavaScript error|TypeError|ReferenceError|SyntaxError|RangeError|console\.error)", re.I)


class Browser:
    """Minimal Marionette client: navigate, run script, screenshot, switch into iframes."""

    def __init__(self, *, width=1440, height=900, marionette_port=2828, log_path=None, prefs=None):
        self.profile = tempfile.mkdtemp(prefix="wss_devshot_")
        values = dict(_PREFS, **{"marionette.port": int(marionette_port)}, **(prefs or {}))
        with open(Path(self.profile) / "user.js", "w", encoding="utf-8") as fh:
            for key, value in values.items():
                fh.write(f"user_pref({json.dumps(key)}, {json.dumps(value)});\n")
        env = dict(os.environ, MOZ_HEADLESS="1", MOZ_HEADLESS_WIDTH=str(width), MOZ_HEADLESS_HEIGHT=str(height),
                   LIBGL_ALWAYS_SOFTWARE="1", MOZ_WEBGL_FORCE_EGL="1", GALLIUM_DRIVER="llvmpipe")
        for key in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
            env.pop(key, None)
        self.log_path = Path(log_path) if log_path else Path(self.profile) / "console.log"
        self._log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(["firefox", "--marionette", "--headless", "--no-remote", "--profile", self.profile],
                                     env=env, stdout=self._log, stderr=subprocess.STDOUT)
        deadline = time.time() + 60
        while True:
            try:
                self.sock = socket.create_connection(("127.0.0.1", int(marionette_port)), timeout=5)
                break
            except OSError:
                if time.time() > deadline or self.proc.poll() is not None:
                    self.close()
                    raise RuntimeError(f"Firefox Marionette 未能在端口 {marionette_port} 启动（端口被占用或 Firefox 缺失）")
                time.sleep(0.4)
        self.sock.settimeout(300)
        self._buf, self._id = b"", 0
        self._recv()
        self.cmd("WebDriver:NewSession", {"capabilities": {"alwaysMatch": {"acceptInsecureCerts": True}}})
        self.cmd("WebDriver:SetWindowRect", {"width": int(width), "height": int(height)})

    def _recv(self):
        while b":" not in self._buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RuntimeError("Marionette 连接已关闭")
            self._buf += chunk
        size, rest = self._buf.split(b":", 1)
        size = int(size)
        while len(rest) < size:
            rest += self.sock.recv(1 << 20)
        message, self._buf = rest[:size], rest[size:]
        return json.loads(message)

    def cmd(self, name, params=None):
        self._id += 1
        data = json.dumps([0, self._id, name, params or {}]).encode()
        self.sock.sendall(str(len(data)).encode() + b":" + data)
        while True:
            reply = self._recv()
            if isinstance(reply, list) and len(reply) == 4 and reply[1] == self._id:
                if reply[2]:
                    raise RuntimeError(f"{name}: {reply[2]}")
                return reply[3]

    def resize(self, width, height):
        return self.cmd("WebDriver:SetWindowRect", {"width": int(width), "height": int(height)})

    def go(self, url, wait=3.0):
        self.cmd("WebDriver:Navigate", {"url": url})
        time.sleep(wait)

    def js(self, script, args=None):
        """Run ``script`` as a function body (use ``return``); returns the JSON-able value."""
        reply = self.cmd("WebDriver:ExecuteScript", {"script": script, "args": args or []})
        return reply.get("value") if isinstance(reply, dict) else reply

    def shot(self, path, full=False):
        reply = self.cmd("WebDriver:TakeScreenshot", {"full": bool(full), "hash": False})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(base64.b64decode(reply["value"]))
        return str(path)

    def frame(self, css=None):
        """Enter the iframe matching ``css``; ``None`` returns to the parent document."""
        if css is None:
            return self.cmd("WebDriver:SwitchToParentFrame")
        element = self.cmd("WebDriver:FindElement", {"using": "css selector", "value": css})["value"]
        return self.cmd("WebDriver:SwitchToFrame", {"element": list(element.values())[0], "focus": True})

    # S7: the workspace login form (ws_shell.showLogin: #wsl-user in user-name mode, #wsl-pass = password or token).
    LOGIN_JS = ("var p=document.getElementById('wsl-pass');if(!p||!p.form)return false;var u=document.getElementById('wsl-user');"
                "if(u)u.value=arguments[0];p.value=arguments[1];p.form.requestSubmit();return true;")

    def login(self, base, user, password, wait=4.0):
        """Log in through the workspace's own form at ``/v2/``.  Returns False when the page shows no login form (already
        logged in, or a local service without login); raises when the form is still there after ``wait`` seconds."""
        self.go(base.rstrip("/") + "/v2/", 3)
        if not self.js(self.LOGIN_JS, [user, password]):
            return False
        time.sleep(wait)
        if self.js("return Boolean(document.getElementById('wsl-pass'));"):
            raise RuntimeError("登录失败：工作区的登录表单仍在（用户名或口令不对？）")
        return True

    def errors(self):
        """JavaScript error lines printed to the console so far."""
        self._log.flush()
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return [line for line in text.splitlines() if _ERROR_RE.search(line) and not _BROWSER_RE.search(line)]

    def close(self):
        try:
            self.cmd("WebDriver:DeleteSession")
        except Exception:
            pass
        if getattr(self, "proc", None):
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except Exception:
                self.proc.kill()
        try:
            self._log.close()
        except Exception:
            pass
        shutil.rmtree(self.profile, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def parse_login(spec: str | None) -> tuple[str, str] | None:
    """``"user:password"`` → ``(user, password)``; ``None`` / empty → ``None``."""
    if not spec:
        return None
    user, sep, password = str(spec).partition(":")
    if not sep or not user or not password:
        raise SystemExit("--login 的格式是 用户名:口令")
    return user, password


def write_sandbox_users(target: Path, spec: str) -> Path:
    """Create (or extend) the sandbox's own ``users.json`` with one administrator; never touches another directory."""
    from .users import UserStore
    user, password = parse_login(spec)
    store = UserStore(Path(target) / "users.json")
    if user not in store.load():
        store.add(user, password, admin=True)
    return store.path


# Shared (user-name) mode on a loopback address: ``serve`` decides ``shared = not is_loopback(host)``; the sandbox
# keeps the 127.0.0.1 bind and only flips that decision (sandbox use only).
_SHARED_LAUNCHER = ("import sys; from wss_deploy import server; server.is_loopback = lambda host: False; "
                    "server.serve(host='127.0.0.1', port=int(sys.argv[2]), jobs_root=sys.argv[1], device='cpu')")


def sandbox(jobs: list[str], target: Path, port: int, *, source: Path | None = None, owner: str | None = None, login: str | None = None) -> subprocess.Popen:
    """Copy finished jobs into ``target`` and start a loopback, CPU-only server on it (returns the process).

    ``login="user:password"`` starts it in user-name login mode (the production setting) with a users.json of
    its own; the copied jobs stay with their original owner, so log in as that user (normally ``admin``)."""
    from .paths import PROJECT_ROOT
    source = Path(source or PROJECT_ROOT / "outputs/wss_deploy_jobs")
    target = Path(target).resolve()
    if target == source.resolve():
        raise SystemExit("沙箱目录不能是正式任务目录。")
    target.mkdir(parents=True, exist_ok=True)
    owners = set()
    for job_id in jobs:
        src = source / job_id
        if not (src / "job.json").is_file():
            raise SystemExit(f"找不到任务 {job_id}")
        if not (target / job_id).exists():
            shutil.copytree(src, target / job_id)
        owners.add(json.loads((src / "job.json").read_text(encoding="utf-8")).get("owner"))
    owner = owner or next((o for o in owners if o), None)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(PROJECT_ROOT))
    env.pop("WSS_DEPLOY_LEGACY_OWNER", None)
    if owner and not login:
        env["WSS_DEPLOY_LEGACY_OWNER"] = owner
    log = open(target / "sandbox_server.log", "a", encoding="utf-8")
    if login:
        write_sandbox_users(target, login)
        command = [sys.executable, "-u", "-c", _SHARED_LAUNCHER, str(target), str(port)]
    else:
        command = [sys.executable, "-u", "-m", "wss_deploy.cli", "serve", "--host", "127.0.0.1", "--port", str(port),
                   "--jobs-root", str(target), "--device", "cpu"]
    proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT), start_new_session=True)
    for _ in range(60):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            break
        except OSError:
            time.sleep(0.5)
    return proc


def suite(base: str, jobs: list[str], out: Path, *, marionette_port: int = 2828, report_wait: float = 25.0, login: str | None = None) -> dict:
    """Capture the standard page set against a running (sandbox) server; returns and writes ``index.json``.

    Pages (S7: the workspace ``/v2/``; the classic workbench, report and compare pages are retired): home (1440 / 1024
    / 390 wide), every job's result page and one-pager, the comparison of the first two jobs, and the classic report
    address of the first job (it must land on the workspace; ``url`` is recorded).  Each entry records the JavaScript
    errors printed while that page was open, so ``errors`` must be empty before a UI change is handed over.
    """
    base = base.rstrip("/")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def record(name, browser, before, **extra):
        path = browser.shot(out / f"{name}.png", full=extra.pop("full", False))
        rows.append({"name": name, "file": Path(path).name, "errors": browser.errors()[before:], **extra})

    def url(browser):
        try:
            return browser.cmd("WebDriver:GetCurrentURL").get("value")
        except Exception:
            return None

    home = base + "/v2/"
    credentials = parse_login(login)
    with Browser(width=1440, height=900, marionette_port=marionette_port) as browser:
        if credentials:   # v0.15: user-name mode — the login page is part of the set, then log in through the form
            n = len(browser.errors()); browser.go(home, 3); record("login", browser, n)
            browser.login(base, *credentials)
        n = len(browser.errors()); browser.go(home, 4); record("home", browser, n)
        for job in jobs:
            n = len(browser.errors()); browser.go(f"{home}#/job/{job}", report_wait); record(f"result_{job}", browser, n)
            n = len(browser.errors()); browser.go(f"{base}/api/jobs/{job}/onepage", 4); record(f"onepage_{job}", browser, n, full=True)
        if len(jobs) >= 2:
            n = len(browser.errors()); browser.go(f"{home}#/job/{jobs[0]}?v=compare&cmp={jobs[1]}", report_wait + 10); record("compare", browser, n)
        if jobs:   # S7: a classic report address lands on the workspace result
            n = len(browser.errors()); browser.go(f"{base}/api/jobs/{jobs[0]}/report", report_wait); record("classic_report_address", browser, n, url=url(browser))
        for width, height in ((1024, 768), (390, 844)):
            browser.resize(width, height); n = len(browser.errors()); browser.go(home, 4); record(f"home_{width}", browser, n)
    index = {"base": base, "jobs": jobs, "pages": rows, "pages_with_errors": [r["name"] for r in rows if r["errors"]]}
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return index


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["sandbox"]:
        ap = argparse.ArgumentParser(prog="wss_deploy.devshot sandbox")
        ap.add_argument("--jobs", required=True, help="逗号分隔的已完成任务 id（从正式目录复制）")
        ap.add_argument("--dir", required=True, help="沙箱任务目录（放在自己的 scratch 下）")
        ap.add_argument("--port", type=int, default=8799)
        ap.add_argument("--source", default=None)
        ap.add_argument("--login", default=None, help="用户名:口令 —— 以用户名登录模式启动（沙箱自己的 users.json，管理员）")
        args = ap.parse_args(argv[1:])
        proc = sandbox([j for j in args.jobs.split(",") if j], Path(args.dir), args.port, source=Path(args.source) if args.source else None, login=args.login)
        print(json.dumps({"pid": proc.pid, "url": f"http://127.0.0.1:{args.port}/v2/", "jobs_root": str(Path(args.dir).resolve()),
                          "stop": f"kill {proc.pid}"}, ensure_ascii=False))
        return 0
    if argv[:1] == ["suite"]:
        ap = argparse.ArgumentParser(prog="wss_deploy.devshot suite")
        ap.add_argument("--base", required=True, help="沙箱服务地址，如 http://127.0.0.1:8799")
        ap.add_argument("--jobs", required=True, help="逗号分隔的任务 id（前两个用于并排比较）")
        ap.add_argument("--out", required=True); ap.add_argument("--marionette-port", type=int, default=2828)
        ap.add_argument("--report-wait", type=float, default=25.0)
        ap.add_argument("--login", default=None, help="用户名:口令 —— 用户名登录模式的沙箱先登录（并截登录页）")
        args = ap.parse_args(argv[1:])
        index = suite(args.base, [j for j in args.jobs.split(",") if j], Path(args.out), marionette_port=args.marionette_port, report_wait=args.report_wait,
                      login=args.login)
        print(json.dumps({"pages": len(index["pages"]), "pages_with_errors": index["pages_with_errors"], "out": str(Path(args.out).resolve())}, ensure_ascii=False))
        return 3 if index["pages_with_errors"] else 0
    ap = argparse.ArgumentParser(prog="wss_deploy.devshot")
    ap.add_argument("url"); ap.add_argument("out")
    ap.add_argument("--width", type=int, default=1440); ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--wait", type=float, default=4.0, help="导航后等待秒数（工作区结果页的三维约 10–30 s）")
    ap.add_argument("--js", default=None, help="截图前执行的脚本（函数体）"); ap.add_argument("--after", type=float, default=2.0)
    ap.add_argument("--full", action="store_true", help="整页截图"); ap.add_argument("--marionette-port", type=int, default=2828)
    ap.add_argument("--login", default=None, help="用户名:口令 —— 先在工作区 /v2/ 登录（用户名登录模式的沙箱）")
    args = ap.parse_args(argv)
    with Browser(width=args.width, height=args.height, marionette_port=args.marionette_port) as browser:
        parts = urlsplit(args.url)
        credentials = parse_login(args.login)
        if credentials and parts.scheme in ("http", "https"):
            browser.login(f"{parts.scheme}://{parts.netloc}", *credentials)
        elif parts.scheme in ("http", "https") and parts.path.startswith("/api/"):
            browser.go(f"{parts.scheme}://{parts.netloc}/v2/", 3)   # result files need the session cookie of the workspace
        browser.go(args.url, args.wait)
        if args.js:
            print(json.dumps({"js": browser.js(args.js)}, ensure_ascii=False))
            time.sleep(args.after)
        print(browser.shot(args.out, full=args.full))
        errors = browser.errors()
        if errors:
            print("\n".join(["JS 错误："] + errors[:40]), file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
