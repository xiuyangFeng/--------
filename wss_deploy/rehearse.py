"""``python -m wss_deploy.cli service rehearse`` (v0.15 O5): try the code on disk against copies of real jobs
before ``service upgrade`` — one command, nothing of the production service or its jobs root is touched.

1. copy 1–2 finished jobs (default: the newest ones with a report; ``--job`` picks them) from the jobs root into a
   temporary directory (``geometry_cache/`` and ``history/`` are left out; the source is only read);
2. write a temporary ``users.json`` with a random administrator;
3. start ``serve`` from the code on disk (the service's python / cwd / saved environment) on 127.0.0.1 and a
   random free port, CPU only (``CUDA_VISIBLE_DEVICES=``), no model preloading unless ``--preload``, no
   background precompute, and ``WSS_DEPLOY_TRUST_PROXY=1`` so the loopback service requires login exactly like the
   shared production service;
4. smoke test with ``urllib``: ``/api/ready`` (anonymous) → login → job list → job detail → the workspace page
   ``/v2/`` (scripts named with their build), each job's workspace data (``/api/v2/jobs/<id>/manifest``, read from its
   report.html), the classic report address landing on the workspace (S7) and the one-pager → ``/api/ready`` as
   administrator (the ``summary`` block);
5. stop the server (SIGTERM, then SIGKILL) and delete the directory (``--keep`` keeps it for inspection).

Every step reports pass / fail and seconds; on failure the last lines of the server log are included.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .paths import PROJECT_ROOT

REHEARSAL_USER = "rehearsal"
COPY_IGNORE = ("geometry_cache", "history", ".report_ui_stage_*")
DEFAULT_TIMEOUT_S = 180.0
STOP_TIMEOUT_S = 20.0
DROP_ENV = ("WSS_DEPLOY_TOKEN", "WSS_DEPLOY_PASSWORD", "WSS_DEPLOY_LEGACY_OWNER", "WSS_DEPLOY_RELEASE_ROOT_OVERRIDE")


class RehearsalError(RuntimeError):
    pass


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _job(path: Path) -> dict:
    try:
        value = json.loads((Path(path) / "job.json").read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def pick_jobs(source: Path, requested=None, count: int = 2) -> list[str]:
    """``requested`` ids (each must be a finished job with a report), else the ``count`` newest such jobs."""
    source = Path(source)
    if requested:
        out = []
        for job_id in requested:
            job_dir = (source / job_id).resolve()
            if job_dir.parent != source.resolve() or _job(job_dir).get("status") != "done" or not (job_dir / "report.html").is_file():
                raise RehearsalError(f"任务 {job_id} 不是 {source} 中已完成且有报告的任务。")
            out.append(job_id)
        return out
    rows = []
    for path in source.glob("*/job.json"):
        job_dir = path.parent
        if job_dir.name.startswith(".") or not (job_dir / "report.html").is_file():
            continue
        job = _job(job_dir)
        if job.get("status") == "done":
            rows.append((float(job.get("created_ts") or 0), job_dir.name))
    rows.sort(reverse=True)
    return [name for _, name in rows[:max(0, int(count))]]


def child_env(cfg: dict, *, tz_name: str | None, preload: bool) -> dict:
    """The service's saved environment with the rehearsal overrides (CPU, login required, nothing heavy)."""
    from .service import ENV_KEYS, ENV_PREFIX, kept_env
    env = {key: value for key, value in os.environ.items() if not (key.startswith(ENV_PREFIX) or key in ENV_KEYS)}
    env.update(kept_env(cfg.get("env") or {}))
    for key in DROP_ENV:
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES="", WSS_DEPLOY_TRUST_PROXY="1", WSS_DEPLOY_PRECOMPUTE="0",
               WSS_DEPLOY_PRELOAD="1" if preload else "0", WSS_DEPLOY_WARMUP="1" if preload else "0",
               PYTHONUNBUFFERED="1")
    if tz_name:
        env["WSS_DEPLOY_TZ"] = tz_name
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    return env


def launch(workdir: Path, port: int, cfg: dict, env: dict) -> subprocess.Popen:
    """Start ``serve`` on the copy (own session, so the stop signal reaches only it)."""
    python = cfg.get("python") or sys.executable
    argv = [python, "-u", "-m", "wss_deploy.cli", "serve", "--host", "127.0.0.1", "--port", str(port), "--jobs-root", str(workdir),
            "--device", "cpu", "--log-file", str(workdir / "server.log")]
    console = open(workdir / "server.console.log", "ab")
    try:
        return subprocess.Popen(argv, cwd=cfg.get("cwd") or str(PROJECT_ROOT), env=env, stdin=subprocess.DEVNULL, stdout=console,
                                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    finally:
        console.close()


def stop_process(proc, timeout: float = STOP_TIMEOUT_S) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(10)
    except OSError:
        pass


def _tail(path: Path, lines: int = 30) -> list[str]:
    from .service import _tail as tail
    return tail(path, lines).splitlines()


def smoke(base: str, *, user: str, password: str, jobs: list[str], timeout: float, proc=None, step=None) -> None:
    """The login → list → report → ready sequence against ``base``; ``step(name, fn)`` records each part."""
    from .service import ServiceClient, ServiceError, http_opener
    import urllib.error
    import urllib.request

    client = ServiceClient(base, timeout=60)

    def ready_anonymous():
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            if proc is not None and proc.poll() is not None:
                raise RehearsalError(f"服务进程已退出（返回码 {proc.returncode}）")
            try:
                status, payload = client.request("GET", "/api/ready")
            except ServiceError as exc:
                last = str(exc)
                time.sleep(0.5)
                continue
            if status == 200 and isinstance(payload, dict) and payload.get("ok"):
                if set(payload) - {"ok", "version"}:
                    raise RehearsalError(f"未登录的 /api/ready 返回了多余字段：{sorted(set(payload) - {'ok', 'version'})}")
                return f"版本 {payload.get('version')}"
            last = f"HTTP {status}"
            time.sleep(0.5)
        raise RehearsalError(f"{timeout:.0f} 秒内 /api/ready 未就绪（最后：{last}）")

    def login():
        status, payload = client.request("GET", "/api/session")
        if status != 200 or not isinstance(payload, dict):
            raise RehearsalError(f"GET /api/session 返回 HTTP {status}")
        if payload.get("login") != "password":
            raise RehearsalError(f"临时服务的登录方式是 {payload.get('login')!r}，不是用户名登录（WSS_DEPLOY_TRUST_PROXY 未生效？）")
        session = client.connect(user=user, password=password)
        if session.get("role") != "admin":
            raise RehearsalError(f"登录成功但角色是 {session.get('role')!r}")
        return f"{user}（管理员）"

    def listing():
        payload = client.jobs(all_owners=True)
        ids = {job.get("id") for job in payload.get("jobs") or []}
        missing = [job_id for job_id in jobs if job_id not in ids]
        if missing:
            raise RehearsalError(f"任务列表缺少 {missing}")
        for job_id in jobs:
            status, detail = client.request("GET", f"/api/jobs/{job_id}")
            if status != 200 or not isinstance(detail, dict) or detail.get("status") != "done":
                raise RehearsalError(f"GET /api/jobs/{job_id} 返回 HTTP {status}")
        return f"{len(ids)} 个任务，详情 {len(jobs)} 个"

    def fetch(path: str) -> tuple[int, str, bytes, str]:
        """A page visit (``Accept: text/html``, redirects followed): status, type, body and the final address."""
        headers = {"Accept": "text/html"}
        if client.cookie:
            headers["Cookie"] = client.cookie
        request = urllib.request.Request(base.rstrip("/") + path, headers=headers)
        try:
            with http_opener().open(request, timeout=120) as response:
                return response.status, response.headers.get("Content-Type") or "", response.read(), response.geturl()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers.get("Content-Type") or "", exc.read(), exc.geturl()

    def workspace():
        status, ctype, page, _ = fetch("/v2/")
        if status != 200 or not ctype.startswith("text/html") or b'id="ws-app"' not in page:
            raise RehearsalError(f"GET /v2/ 返回 HTTP {status}（{ctype}，{len(page)} 字节）")
        if b".js?v=" not in page:
            raise RehearsalError("/v2/ 的脚本地址没有带构建号（?v=）")
        sizes = []
        for job_id in jobs:
            status, manifest = client.request("GET", f"/api/v2/jobs/{job_id}/manifest")
            if status != 200 or not isinstance(manifest, dict) or not manifest.get("fields") or not manifest.get("arrays"):
                raise RehearsalError(f"GET /api/v2/jobs/{job_id}/manifest 返回 HTTP {status}")
            sizes.append(len(json.dumps(manifest, ensure_ascii=False)))
            # S7: the classic report address is a visit of the workspace now
            status, _, _, final = fetch(f"/api/jobs/{job_id}/report")
            if status != 200 or not final.endswith(f"/v2/?job={job_id}"):
                raise RehearsalError(f"旧报告地址 /api/jobs/{job_id}/report 没有转到工作区（HTTP {status}，{final}）")
            status, ctype, body, _ = fetch(f"/api/jobs/{job_id}/onepage")
            if status != 200 or not ctype.startswith("text/html") or b"<html" not in body[:4096].lower():
                raise RehearsalError(f"GET /api/jobs/{job_id}/onepage 返回 HTTP {status}（{ctype}，{len(body)} 字节）")
            sizes.append(len(body))
        return f"工作区页面、结果数据与一页纸 {len(jobs)} 个任务，{sum(sizes) / 1e6:.1f} MB"

    def ready_admin():
        status, payload = client.request("GET", "/api/ready")
        if status != 200 or not isinstance(payload, dict) or not payload.get("ok"):
            failing = [k for k, v in ((payload or {}).get("checks") or {}).items() if isinstance(v, dict) and v.get("ok") is False] \
                if isinstance(payload, dict) else []
            raise RehearsalError(f"/api/ready 返回 HTTP {status}" + (f"，未通过：{failing}" if failing else ""))
        summary = payload.get("summary")
        if not isinstance(summary, dict) or summary.get("login") != "password":
            raise RehearsalError("管理员的 /api/ready 没有 summary 块")
        return f"就绪；summary：版本 {summary.get('version')}，登录 {summary.get('login')}，队列 {summary.get('queue')}"

    if not jobs:
        raise RehearsalError("没有可用的已完成任务")
    step("就绪（未登录）", ready_anonymous)
    step("登录", login)
    step("任务列表与详情", listing)
    step("打开工作区与一页纸", workspace)
    step("就绪（管理员摘要）", ready_admin)


def rehearse(source: Path, *, jobs=None, count: int = 2, workdir=None, keep: bool = False, preload: bool = False,
             timeout: float = DEFAULT_TIMEOUT_S, port: int | None = None, out=print, launcher=None) -> dict:
    """Run the rehearsal; returns ``{ok, steps: [{name, ok, seconds, detail}], seconds, port, dir, jobs, kept}``.

    ``launcher(workdir, port, cfg, env) -> Popen-like`` replaces :func:`launch` (tests)."""
    from . import clock
    from .service import load_config
    from .users import UserStore
    source = Path(source).resolve()
    started = time.perf_counter()
    steps: list[dict] = []
    result = {"ok": False, "source": str(source), "steps": steps, "jobs": [], "dir": None, "port": None, "kept": bool(keep)}
    proc = None
    work = None
    failed = False

    def step(name, fn):
        nonlocal failed
        if failed:
            steps.append({"name": name, "ok": None, "seconds": 0.0, "detail": "未执行（前一步失败）"})
            return
        t0 = time.perf_counter()
        try:
            detail = fn()
            steps.append({"name": name, "ok": True, "seconds": round(time.perf_counter() - t0, 2), "detail": detail or ""})
            out(f"✓ {name}（{time.perf_counter() - t0:.1f} s）{('：' + detail) if detail else ''}")
        except Exception as exc:  # noqa: BLE001 — every failure is a finding of the rehearsal
            failed = True
            text = str(exc) if isinstance(exc, (RehearsalError,)) else f"{type(exc).__name__}: {exc}"
            steps.append({"name": name, "ok": False, "seconds": round(time.perf_counter() - t0, 2), "detail": text})
            out(f"✗ {name}（{time.perf_counter() - t0:.1f} s）：{text}")

    try:
        cfg = load_config(source)
        picked = pick_jobs(source, jobs, count)
        result["jobs"] = picked
        work = Path(tempfile.mkdtemp(prefix="wss_rehearse_", dir=str(workdir) if workdir else None))
        result["dir"] = str(work)
        out(f"演练目录 {work}；复制任务 {', '.join(picked) or '（无）'}")

        def copy():
            if not picked:
                raise RehearsalError(f"{source} 没有已完成且有报告的任务可复制")
            total = 0
            for job_id in picked:
                shutil.copytree(source / job_id, work / job_id, symlinks=True, ignore=shutil.ignore_patterns(*COPY_IGNORE))
                total += sum(p.stat().st_size for p in (work / job_id).rglob("*") if p.is_file())
            return f"{len(picked)} 个任务，{total / 1e6:.1f} MB"
        step("复制任务", copy)
        password = secrets.token_urlsafe(18)

        def users():
            UserStore(work / "users.json").add(REHEARSAL_USER, password, admin=True, display_name="上线演练")
            return f"临时管理员 {REHEARSAL_USER}"
        step("临时账号", users)
        port = port or free_port()
        result["port"] = port
        env = child_env(cfg, tz_name=clock.resolve(source)["name"], preload=preload)

        def start():
            nonlocal proc
            proc = (launcher or launch)(work, port, cfg, env)
            return f"PID {proc.pid}，http://127.0.0.1:{port}/（仅回环，CPU）"
        step("启动临时服务", start)
        smoke(f"http://127.0.0.1:{port}", user=REHEARSAL_USER, password=password, jobs=picked, timeout=timeout, proc=proc, step=step)
    except RehearsalError as exc:
        steps.append({"name": "准备", "ok": False, "seconds": 0.0, "detail": str(exc)})
        out(f"✗ 准备：{exc}")
    finally:
        stop_process(proc)
        ok = bool(steps) and all(row["ok"] for row in steps)
        if not ok and work is not None:
            result["server_log_tail"] = _tail(work / "server.log") or _tail(work / "server.console.log")
            if result["server_log_tail"]:
                out("临时服务日志最后几行：\n" + "\n".join("  " + line for line in result["server_log_tail"]))
        if work is not None and not keep:
            shutil.rmtree(work, ignore_errors=True)
        result.update(ok=ok, seconds=round(time.perf_counter() - started, 1))
    out(("演练通过" if result["ok"] else "演练失败") + f"（{result['seconds']:.1f} s）" +
        ("" if keep or work is None else "；临时目录已删除") + (f"；临时目录保留在 {work}" if keep and work is not None else ""))
    return result


__all__ = ["REHEARSAL_USER", "RehearsalError", "child_env", "free_port", "launch", "pick_jobs", "rehearse", "smoke", "stop_process"]
