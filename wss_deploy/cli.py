"""CLI.  run:      python -m wss_deploy.cli run case.stl --out jobs/case [--outlets auto | "3=out-le,4=out-li,5=out-re,6=out-ri"] [--device auto|cuda|cpu]
          serve:    python -m wss_deploy.cli serve --port 8765 --jobs-root outputs/wss_deploy_jobs [--log-file server.log] [--token-file PATH]
                    (持有 <jobs-root>/.service.lock 写锁；请求日志写 <jobs-root>/access.log)
          service:  python -m wss_deploy.cli service start|stop|restart|upgrade|status|logs|token [--jobs-root …] [--port …] [--drain [秒]]
                    (upgrade = 预检（导入 + doctor）→ 停服务 → 启动 → 新服务后台重建需要新分析的任务并刷新过期报告；更新代码后用它)
                    (restart --env CUDA_VISIBLE_DEVICES=1 把服务固定到 1 号 GPU，写入 service.json)
                    python -m wss_deploy.cli service rehearse [--job ID] [--keep] [--preload] [--json]
                    (上线演练：复制 1–2 个已完成任务到临时目录，回环随机端口 + CPU 起新代码，登录 → 列表 → 报告 → /api/ready 烟测后清理)
          doctor:   python -m wss_deploy.cli doctor [--json] [--jobs-root …] [--host 0.0.0.0]   (✗ 硬错 / ⚠ 警告 / ℹ 信息；✗ 阻止 upgrade)
          reports:  python -m wss_deploy.cli reports refresh [--job ID | --all] [--check]
          template: python -m wss_deploy.cli template show | set --institution … --department … --title … --footer … --signatures 报告人,审阅人 --glossary used|all|none --appendix on|off
          submit:   python -m wss_deploy.cli submit a.stl b.stl [--server http://127.0.0.1:8765] [--units mm] [--patient-id …] [--user name]
          user:     python -m wss_deploy.cli user add|passwd|disable|role|list <name> [--admin|--user] [--jobs-root …]   (password read from the terminal)
                    (passwd / disable / role 立即结束该用户所有已登录会话)
          jobs:     python -m wss_deploy.cli jobs list [--status …] [--json] [--server …]
                    python -m wss_deploy.cli jobs claim --owner <old owner> --user <name> [--jobs-root …] [--force]   (service stopped)
                    python -m wss_deploy.cli jobs du [--top N] [--json]   (只读：按任务列出占用)
                    python -m wss_deploy.cli jobs prune-cache --older-than DAYS [--yes] [--force] [--json]
                    (只清 geometry_cache/ 与 .tmp/ 这类可重算文件；不加 --yes 只列清单)"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path


def _read_password(prompt: str = "口令：", *, confirm: bool = False) -> str:
    """Never take the password from argv (it would land in shell history and ``ps``)."""
    env = os.environ.get("WSS_DEPLOY_PASSWORD")
    if not sys.stdin.isatty():
        if env is None: raise SystemExit("非交互模式请通过环境变量 WSS_DEPLOY_PASSWORD 提供口令。")
        return env
    import getpass
    value = getpass.getpass(prompt)
    if confirm and getpass.getpass("再输入一次：") != value: raise SystemExit("两次输入的口令不一致。")
    return value


def _jobs_root(value) -> Path:
    from .paths import PROJECT_ROOT
    return Path(value) if value else PROJECT_ROOT / "outputs/wss_deploy_jobs"


def _print_json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=1))


def _offline_lock(root: Path, *, force: bool, purpose: str):
    """J3: offline writers refuse while the service (or another maintenance command) holds the jobs-root lock."""
    from .service import ServiceLockError, offline_lock
    try:
        return offline_lock(root, force=force, purpose=purpose, out=lambda text: print(text, file=sys.stderr))
    except ServiceLockError as exc:
        raise SystemExit(f"错误：{exc}") from None


def _configure_request_logs(root: Path) -> None:
    """J8: ``wss_deploy.access`` → ``<jobs_root>/access.log`` (rotating, 20 MB × 5, not in server.log);
    ``wss_deploy.audit`` (job state changes, logins) propagates to the application log (server.log)."""
    import logging
    from logging.handlers import RotatingFileHandler
    from . import server as srv
    from .service import ACCESS_LOG
    access = logging.getLogger("wss_deploy.access")
    if not any(getattr(handler, "_wss_access", False) for handler in access.handlers):
        root.mkdir(parents=True, exist_ok=True)
        handler = getattr(srv, "PrivateRotatingFileHandler", RotatingFileHandler)(root / ACCESS_LOG, maxBytes=getattr(srv, "LOG_MAX_BYTES", 20 * 1024 * 1024),
                                      backupCount=getattr(srv, "LOG_BACKUPS", 5), encoding="utf-8")
        from .clock import LogFormatter   # O1: same zone as server.log and the job records
        handler.setFormatter(LogFormatter(getattr(srv, "LOG_FORMAT", "%(asctime)s %(levelname)s %(name)s %(message)s"),
                                          datefmt=getattr(srv, "LOG_DATEFMT", "%Y-%m-%dT%H:%M:%S%z")))
        handler._wss_access = True
        access.addHandler(handler)
    access.setLevel(logging.INFO)
    access.propagate = False
    # ``wss_deploy.audit`` needs nothing here: it propagates to the root logger (INFO, server.log) set up by serve().


def _serve(args) -> int:
    from .paths import PROJECT_ROOT
    from .service import ServiceLockError, acquire_lock
    root = (Path(args.jobs_root) if args.jobs_root else PROJECT_ROOT / "outputs/wss_deploy_jobs").resolve()
    token = args.token
    if args.token_file:
        try:
            token = Path(args.token_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            print(f"错误：无法读取令牌文件 {args.token_file}：{exc}", file=sys.stderr); return 1
        if not token:
            print(f"错误：令牌文件 {args.token_file} 为空。", file=sys.stderr); return 1
    elif args.token:
        print("警告：--token 会出现在进程命令行（ps 可见）；请改用 --token-file PATH 或环境变量 WSS_DEPLOY_TOKEN。"
              "service.json / status 输出中已隐去该值。", file=sys.stderr)
    from .server import apply_umask_from_env
    apply_umask_from_env()     # v0.15: WSS_DEPLOY_UMASK before the lock file and access.log are created
    try:
        lock = acquire_lock(root, purpose="serve", host=args.host, port=args.port)
    except ServiceLockError as exc:
        print(f"错误：{exc}", file=sys.stderr); return 1
    try:
        _configure_request_logs(root)
        from .server import serve
        from .service import install_crash_logging
        install_crash_logging()        # O6: uncaught thread exceptions → server.log with a diagnostic id
        try:
            serve(host=args.host, port=args.port, jobs_root=root, device=args.device, token=token, log_file=args.log_file)
        except BaseException as exc:
            if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                import logging, secrets
                logging.getLogger("wss_deploy.service").critical("Service process crashed; diagnostic_id=%s", secrets.token_hex(6), exc_info=True)
            raise
    finally:
        lock.release()
    return 0


class _AuditLog:
    """v0.15: offline commands that change accounts or ownership (``user add|passwd|disable|role``, ``jobs claim``)
    write ``wss_deploy.audit`` lines (``cli {json}``) to the service log — ``service.json`` ``log_file``, default
    ``<jobs_root>/server.log`` — created 0600 and never rotated from here.  Passwords are never logged."""
    def __init__(self, root: Path):
        self.root, self.handler, self.logger, self.level = Path(root), None, None, None
        self.operations = None
    def __enter__(self):
        import logging
        from . import server as srv
        try:
            from .operations import OperationsStore
            self.operations = OperationsStore(self.root)
        except Exception:
            # Account recovery must still work if the operations disk/database
            # is damaged. Keep the original service-log audit and say so.
            print("警告：运维审计库不可用；本次账号操作仍尝试写入服务日志。", file=sys.stderr)
        try:
            from .service import load_config
            path = Path(load_config(self.root).get("log_file") or self.root / "server.log")
        except Exception:  # noqa: BLE001 — an unreadable service.json must not block user management
            path = self.root / "server.log"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handler = srv.PrivateRotatingFileHandler(path, maxBytes=0, backupCount=0, encoding="utf-8")
        except OSError as exc:
            print(f"警告：无法写入审计日志 {path}：{exc}", file=sys.stderr)
            return self
        try:
            from .clock import LogFormatter as formatter
        except ImportError:
            formatter = logging.Formatter
        self.handler.setFormatter(formatter(srv.LOG_FORMAT, datefmt=srv.LOG_DATEFMT))
        self.logger = logging.getLogger("wss_deploy.audit"); self.level = self.logger.level
        self.logger.addHandler(self.handler); self.logger.setLevel(logging.INFO)
        return self
    def line(self, action: str, **fields) -> None:
        import getpass, logging
        try: actor = getpass.getuser()
        except Exception: actor = str(os.getuid())  # noqa: BLE001
        logging.getLogger("wss_deploy.audit").info("cli %s", json.dumps({"action": action, "actor": "cli:" + actor, **fields},
                                                                         ensure_ascii=False, separators=(",", ":"), default=str))
        if self.operations is not None:
            try:
                self.operations.record_event({"action": action, "actor": "cli:" + actor,
                    "owner": fields.get("target") or fields.get("user"),
                    "job": (fields.get("jobs") or [None])[0], "source": "cli", "details": fields})
            except Exception:
                print("警告：操作已完成，但未能写入运维审计库；请检查服务日志和存储状态。", file=sys.stderr)
    def __exit__(self, *_exc):
        if self.handler is not None:
            self.logger.removeHandler(self.handler); self.logger.setLevel(self.level); self.handler.close()
        return False


# ------------------------------------------------------------------------------------------ service / doctor
def _service(args) -> int:
    from . import service as S
    root = _jobs_root(args.jobs_root)
    overrides = dict(host=args.host, port=args.port, device=args.device, env_items=args.env, unset_env=args.unset_env, python=args.python)
    drain_s = args.drain
    try:
        if args.action == "status":
            info = S.status(root, args.port)
            if args.json: _print_json(info)
            else: print(S.format_status(info))
            return 0 if info["running"] else 3
        if args.action == "start":
            S.start(root, timeout=args.timeout, **overrides); return 0
        if args.action == "stop":
            S.stop(root, args.port, timeout=args.stop_timeout, drain_s=drain_s); return 0
        if args.action == "restart":
            S.restart(root, args.port, timeout=args.timeout, stop_timeout=args.stop_timeout, drain_s=drain_s,
                      **{k: v for k, v in overrides.items() if k != "port"}); return 0
        if args.action == "upgrade":
            result = S.upgrade(root, args.port, rebuild=args.rebuild or (), auto_rebuild=not args.no_auto_rebuild, timeout=args.timeout,
                               stop_timeout=args.stop_timeout, drain_s=drain_s, preflight_check=not args.skip_preflight,
                               maintenance_timeout=args.maintenance_timeout, **{k: v for k, v in overrides.items() if k != "port"})
            return 1 if result["upgrade"]["rebuild_failed"] or result["upgrade"]["refresh_failed"] else 0
        if args.action == "logs":
            return S.logs(root, lines=args.lines, follow=args.follow)
        if args.action == "rehearse":
            from .rehearse import rehearse
            quiet = (lambda *_a, **_k: None) if args.json else print
            result = rehearse(root, jobs=args.job, count=args.count, workdir=args.dir, keep=args.keep, preload=args.preload,
                              timeout=180.0 if args.timeout == 60.0 else args.timeout, out=quiet)   # --timeout 缺省 60 → 演练 180 s
            if args.json: _print_json(result)
            return 0 if result["ok"] else 1
        return S.token(root, rotate=args.rotate)
    except S.ServiceError as exc:
        print(f"错误：{exc}", file=sys.stderr); return 1


def _doctor(args) -> int:
    from .doctor import exit_code, format_rows, run_checks
    rows = run_checks(_jobs_root(args.jobs_root), port=args.port, skip={s.strip() for s in (args.skip or "").split(",") if s.strip()},
                      host=args.host)
    if args.json: _print_json({"checks": rows, "exit_code": exit_code(rows)})
    else: print(format_rows(rows))
    return exit_code(rows)


def _reports(args) -> int:
    from . import report_freshness as F
    root = _jobs_root(args.jobs_root).resolve()
    if args.job:
        dirs = []
        for job_id in args.job:
            job_dir = (root / job_id).resolve()
            if job_dir.parent != root or not (job_dir / "job.json").is_file(): raise SystemExit(f"找不到任务 {job_id}。")
            if F._job_status(job_dir) != "done" or not (job_dir / "report.html").is_file(): raise SystemExit(f"任务 {job_id} 还没有完成的报告。")
            dirs.append(job_dir)
    else:
        dirs = F.done_jobs(root)
    current = F.ui_fingerprint()
    stale = [d for d in dirs if F.is_stale(d, current)]
    if args.check or not (args.job or args.all):
        # read-only listing: no lock needed
        for job_dir in stale:
            marker = F.read_marker(job_dir)
            print(f"{job_dir.name}  {'已记录模板 ' + str(marker.get('fingerprint'))[:12] if marker else '无 report_ui.json'}")
        print(f"当前模板指纹 {current[:12]}；{len(dirs)} 个报告中 {len(stale)} 个需要刷新。")
        if not args.check: print("刷新请加 --all 或 --job <任务号>（服务运行时刷新同样安全；升级服务时也会自动刷新）。")
        return 0
    targets = dirs if args.force else stale
    failed = 0
    lock = _offline_lock(root, force=args.force, purpose="reports refresh") if targets else None
    try:
        for job_dir in targets:
            try:
                result = F.refresh(job_dir, source="cli")
                print(f"✓ {job_dir.name}（{result.get('report_bytes', 0) / 1e6:.1f} MB）")
            except Exception as exc:  # noqa: BLE001 — report every job, keep going
                failed += 1; print(f"✗ {job_dir.name}：{type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        if lock is not None:
            lock.release()
    print(f"已刷新 {len(targets) - failed} 个，失败 {failed} 个（未过期的 {len(dirs) - len(stale)} 个跳过）。" if not args.force else f"已刷新 {len(targets) - failed} 个，失败 {failed} 个。")
    return 1 if failed else 0


def _template(args) -> int:
    from . import report_template as T
    root = _jobs_root(args.jobs_root)
    if args.action == "show":
        _print_json({"file": str(T.template_path(root)), "exists": T.template_path(root).is_file(), "template": T.load(root)}); return 0
    data = dict(T.DEFAULTS) if args.reset else T.load(root)
    for key, value in (("institution", args.institution), ("department", args.department), ("report_title", args.title),
                       ("footer_note", args.footer), ("show_glossary", args.glossary)):
        if value is not None: data[key] = value
    if args.signatures is not None:
        data["signature_lines"] = [item.strip() for item in args.signatures.replace("，", ",").split(",") if item.strip()]
    if args.appendix is not None:
        data["appendix"] = args.appendix == "on"
    try:
        saved = T.save(root, data)
    except ValueError as exc:
        raise SystemExit(f"报告模板无效：{exc}") from exc
    _print_json({"file": str(T.template_path(root)), "template": saved})
    print("已保存；一页纸下次打开即生效（无需重启服务）。", file=sys.stderr)
    return 0


# ------------------------------------------------------------------------------------------ HTTP client commands
def _client(args):
    from .service import ServiceClient, ServiceError, read_token
    client = ServiceClient(args.server)
    password = None
    if args.user:
        password = _read_password(f"{args.user} 的口令：")
    token = os.environ.get("WSS_DEPLOY_TOKEN") or (read_token(_jobs_root(args.jobs_root)) if not args.user else None)
    try:
        session = client.connect(user=args.user, password=password, token=token)
    except ServiceError as exc:
        raise SystemExit(f"错误：{exc}") from exc
    return client, session


def _submit(args) -> int:
    from .service import ServiceError
    paths = [Path(p) for p in args.stl]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing: raise SystemExit("找不到文件：" + "、".join(missing))
    client, session = _client(args)
    fields = {"units": args.units, "release_id": args.release_id, "patient_id": args.patient_id, "scan_label": args.scan_label,
              "scan_date": args.scan_date, "tags": args.tags, "notes": args.notes, "on_duplicate": args.on_duplicate, "device": args.device}
    try:
        results = client.submit(paths, fields)
    except ServiceError as exc:
        raise SystemExit(f"错误：{exc}") from exc
    base = args.server.rstrip("/")
    created = failed = duplicates = 0
    rows = []
    for item in results:
        name = Path(item.get("file") or "?").name
        if item.get("job"):
            job = item["job"]; created += 1
            reused = f"，复用 {item['reused_from']} 的中心线" if item.get("reused_from") else ""
            print(f"✓ {name} → 任务 {job['id']}（病例 {job.get('case_id')}，{job.get('phase') or job.get('status')}{reused}）  {base}/v2/#/job/{job['id']}")
        elif item.get("duplicate"):
            duplicates += 1
            existing = item.get("existing") or []
            newest = existing[0]["job_id"] if existing else "?"
            print(f"⚠ {name}：同一几何已有 {len(existing)} 个任务（最新 {newest}）；用 --on-duplicate reuse 复用中心线，或 force 重新计算。")
        else:
            failed += 1
            print(f"✗ {name}：{(item.get('error') or {}).get('message') or '未知错误'}")
        rows.append(item)
    print(f"提交 {len(paths)} 个：新建 {created}，重复 {duplicates}，失败 {failed}。")
    if not args.units:
        print("提示：未指定 --units，任务会停在「请确认输入」，需在网页确认单位后继续。")
    login = session.get("login")
    if login == "token":
        print("提示：令牌登录的任务属于这个命令行会话；网页里看不到它们时，请启用用户名登录（cli user add）后用 --user 提交，或停服务后用 `python -m wss_deploy.cli jobs claim --owner <旧 owner> --user <用户名>` 认领。")
    elif login == "none":
        print("提示：本机回环模式下任务属于命令行这次会话；若网页会话看不到，可用 WSS_DEPLOY_LEGACY_OWNER 固定 owner 启动服务。")
    if args.json: _print_json(rows)
    return 1 if failed else (2 if duplicates and not created else 0)


def _configure_clock(args) -> None:
    """O1: every timestamp this command writes uses the zone of the jobs root it works on (service.json env.TZ); a
    ``service start|restart|upgrade --env TZ=…`` counts before it is saved."""
    from . import clock
    root = getattr(args, "jobs_root", None)
    if getattr(args, "cmd", None) == "serve":
        root = root or str(_jobs_root(None))
    pending = None
    for item in getattr(args, "env", None) or []:
        if isinstance(item, str) and item.startswith("TZ="):
            pending = item[3:] or None
    clock.configure(Path(root).resolve() if root else None, tz=pending)


def _jobs_disk(args) -> int:
    """O4: ``jobs du`` (read only) and ``jobs prune-cache`` (list only unless ``--yes``)."""
    from . import housekeeping as H
    root = _jobs_root(args.jobs_root).resolve()
    if not root.is_dir(): raise SystemExit(f"任务目录不存在：{root}")
    if args.action == "du":
        info = H.usage(root)
        if args.json:
            if args.top: info = dict(info, jobs=info["jobs"][:args.top])
            _print_json(info)
        else: print(H.format_usage(info, top=args.top))
        return 0
    if args.older_than is None: raise SystemExit("jobs prune-cache 需要 --older-than DAYS（只清早于这么多天的 geometry_cache/ 与 .tmp/ 文件）。")
    try:
        plan = H.prune_plan(root, args.older_than)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    execute = args.yes and not args.dry_run
    if not execute or not plan["items"]:
        if args.json: _print_json({**plan, "executed": False})
        else:
            print(H.format_plan(plan))
            print("未删除任何文件：确认后加 --yes 执行。" if plan["items"] else "没有需要清理的文件。")
        return 0
    lock = _offline_lock(root, force=args.force, purpose="jobs prune-cache")
    try:
        result = H.prune_execute(root, plan, by="cli jobs prune-cache")
    finally:
        if lock is not None: lock.release()
    if args.json: _print_json({**plan, "executed": True, "result": result})
    else:
        print(H.format_plan(plan))
        print(f"已删除 {result['files']} 个文件，释放 {H.human(result['bytes'])}" + (f"，失败 {result['failed']} 个" if result["failed"] else "")
              + f"；记录在 {root / H.PRUNE_LOG}。")
    return 1 if result["failed"] else 0


def _jobs_list(args) -> int:
    from .service import ServiceError
    client, _ = _client(args)
    try:
        payload = client.jobs(status=args.status, all_owners=args.all)
    except ServiceError as exc:
        raise SystemExit(f"错误：{exc}") from exc
    jobs = payload.get("jobs") or []
    if args.json: _print_json(jobs); return 0
    if not jobs: print("没有任务。"); return 0
    from .service import _width
    status_cn = {"done": "完成", "failed": "失败", "cancelled": "已取消", "interrupted": "中断", "queued": "排队", "running": "计算中",
                 "awaiting_input": "待确认输入", "awaiting_confirmation": "待确认出口"}
    pad = lambda text, width: str(text) + " " * max(1, width - _width(str(text)))
    print(pad("任务号", 30) + pad("病例", 18) + pad("状态", 12) + pad("结果", 20) + pad("患者", 12) + "创建时间")
    for job in jobs:
        created = str(job.get("created_at") or "")[:16].replace("T", " ")
        label = job.get("family_label") or job.get("family") or ""
        print(pad(job.get("id", ""), 30) + pad(str(job.get("case_id") or "")[:16], 18) + pad(status_cn.get(job.get("status"), job.get("status")), 12)
              + pad(label[:18], 20) + pad(str(job.get("patient_id") or "")[:10], 12) + created)
    print(f"共 {payload.get('total', len(jobs))} 个。")
    return 0


# ------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wss_deploy"); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("stl"); r.add_argument("--out", required=True); r.add_argument("--outlets", default="auto"); r.add_argument("--inlet", type=int, default=None)
    r.add_argument("--units", choices=("mm", "cm", "m"), default="mm"); r.add_argument("--accept-auto", action="store_true", help="仅用于已人工核对 proposal 的自动映射")
    r.add_argument("--smooth-mm", type=float, default=1.0); r.add_argument("--spacing-mm", type=float, default=0.5); r.add_argument("--device", default="auto"); r.add_argument("--case-id", default=None)
    r.add_argument("--release-root", default=None); r.add_argument("--release-id", default=None, help="发布包标识；省略使用默认发布包")
    r.add_argument("--stage-a-only", action="store_true", help="only ingest + centreline + naming proposal (no GPU needed)")
    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8765); s.add_argument("--jobs-root", default=None); s.add_argument("--device", default="auto")
    s.add_argument("--token-file", default=None, metavar="PATH", help="共享访问令牌所在文件（推荐；令牌不进命令行）")
    s.add_argument("--token", default=None, help="（不推荐：ps 可见）共享访问令牌；请改用 --token-file 或 WSS_DEPLOY_TOKEN")
    s.add_argument("--log-file", default=None, help="写入轮转日志文件（20 MB × 5，时间带时区偏移）；省略则输出到标准错误")
    sv = sub.add_parser("service", help="服务管理：start|stop|restart|upgrade|status|logs|token（service.json / service.pid / .service_token 在任务目录）")
    sv.add_argument("action", choices=("start", "stop", "restart", "upgrade", "status", "logs", "token", "rehearse")); sv.add_argument("--jobs-root", default=None)
    sv.add_argument("--job", action="append", default=None, metavar="ID", help="rehearse：复制这个已完成任务（可重复；默认取最新 2 个）")
    sv.add_argument("--count", type=int, default=2, help="rehearse：未指定 --job 时复制的任务数（默认 2）")
    sv.add_argument("--keep", action="store_true", help="rehearse：保留临时目录（默认结束即删除）")
    sv.add_argument("--dir", default=None, help="rehearse：临时目录的父目录（默认系统临时目录）")
    sv.add_argument("--preload", action="store_true", help="rehearse：同时在 CPU 上预加载全部发布包（慢，验证权重可加载）")
    sv.add_argument("--rebuild", action="append", default=None, metavar="JOB", help="upgrade：停服务期间对该任务完整重建（可重复）")
    sv.add_argument("--no-auto-rebuild", action="store_true", help="upgrade：不自动重建缺少新版分析结果的任务")
    sv.add_argument("--host", default=None, help="start/restart：监听地址（共享给他人用 0.0.0.0），写入 service.json"); sv.add_argument("--port", type=int, default=None)
    sv.add_argument("--device", default=None, choices=("auto", "cuda", "cpu")); sv.add_argument("--python", default=None, help="服务使用的 Python 解释器")
    sv.add_argument("--env", action="append", default=None, metavar="KEY=VALUE",
                    help="保存到 service.json 的环境变量（WSS_DEPLOY_* / CUDA_VISIBLE_DEVICES / TZ）；"
                         "例：service restart --env CUDA_VISIBLE_DEVICES=1 把服务固定到 1 号 GPU，--env CUDA_VISIBLE_DEVICES= 只用 CPU")
    sv.add_argument("--unset-env", action="append", default=None, metavar="KEY"); sv.add_argument("--timeout", type=float, default=60.0, help="等待健康检查的秒数")
    sv.add_argument("--stop-timeout", type=float, default=15.0); sv.add_argument("--json", action="store_true"); sv.add_argument("-n", "--lines", type=int, default=80)
    sv.add_argument("--follow", "-f", action="store_true"); sv.add_argument("--rotate", action="store_true", help="token：生成新令牌（restart 后生效）")
    sv.add_argument("--drain", nargs="?", type=float, const=600.0, default=None, metavar="SECONDS",
                    help="stop/restart/upgrade：先让服务不再开始新计算，等待正在计算的任务结束（默认最多 600 秒）；排队的任务重启后自动继续")
    sv.add_argument("--skip-preflight", action="store_true", help="upgrade：跳过预检（导入新代码 + doctor），仅用于紧急情况")
    sv.add_argument("--maintenance-timeout", type=float, default=900.0, help="upgrade：等待新服务后台重建 / 刷新报告的秒数（超时后服务继续执行）")
    d = sub.add_parser("doctor", help="环境自检"); d.add_argument("--json", action="store_true"); d.add_argument("--jobs-root", default=None); d.add_argument("--port", type=int, default=None)
    d.add_argument("--skip", default="", help="逗号分隔跳过的检查组：torch,vmtk,releases,features,glossary,jobs_root,exposure,lock,logs,cache,tz,git,service,reports")
    d.add_argument("--host", default=None, help="按这个绑定地址判断共享登录（默认读 service.json；upgrade 预检会传入 --host 覆盖值）")
    rp = sub.add_parser("reports", help="报告模板刷新"); rp.add_argument("action", choices=("refresh",)); rp.add_argument("--job", action="append", default=None)
    rp.add_argument("--all", action="store_true"); rp.add_argument("--check", action="store_true", help="只列出过期报告")
    rp.add_argument("--force", action="store_true", help="未过期也刷新；服务持有写锁时也执行（升级服务时本就会自动刷新）")
    rp.add_argument("--jobs-root", default=None)
    t = sub.add_parser("template", help="机构报告模板（一页纸页眉、签字栏、术语、附录）"); t.add_argument("action", choices=("show", "set")); t.add_argument("--jobs-root", default=None)
    t.add_argument("--institution", default=None); t.add_argument("--department", default=None); t.add_argument("--title", default=None); t.add_argument("--footer", default=None)
    t.add_argument("--signatures", default=None, help="逗号分隔，如 报告人,审阅人；空字符串 = 不印签字栏"); t.add_argument("--glossary", choices=("used", "all", "none"), default=None)
    t.add_argument("--appendix", choices=("on", "off"), default=None); t.add_argument("--reset", action="store_true", help="先恢复默认值再应用本次参数")
    sm = sub.add_parser("submit", help="经 HTTP 把 STL 提交给运行中的服务（每批 ≤ 20）"); sm.add_argument("stl", nargs="+")
    sm.add_argument("--server", default="http://127.0.0.1:8765"); sm.add_argument("--release-id", default=None); sm.add_argument("--units", choices=("mm", "cm", "m"), default=None)
    sm.add_argument("--patient-id", default=None); sm.add_argument("--scan-label", default=None); sm.add_argument("--scan-date", default=None); sm.add_argument("--tags", default=None, help="逗号分隔")
    sm.add_argument("--notes", default=None); sm.add_argument("--on-duplicate", choices=("ask", "reuse", "force"), default="ask"); sm.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None)
    sm.add_argument("--user", default=None, help="共享模式用户名（口令取 WSS_DEPLOY_PASSWORD 或终端输入）"); sm.add_argument("--jobs-root", default=None, help="读取 .service_token 的任务目录（服务机器上）")
    sm.add_argument("--json", action="store_true")
    u = sub.add_parser("user", help="共享模式的用户名登录（users.json）")
    u.add_argument("action", choices=("add", "passwd", "disable", "role", "list"),
                   help="add 新建用户；passwd 改口令（该用户所有已登录会话立即失效，需重新登录）；"
                        "disable 禁用（该用户所有已登录会话立即失效）；role 改角色，配合 --admin 或 --user（该用户所有会话失效）；list 列出")
    u.add_argument("name", nargs="?", default=None)
    role = u.add_mutually_exclusive_group()
    role.add_argument("--admin", action="store_true", help="add / role：管理员（可用 ?all=1 查看全部任务，只读）")
    role.add_argument("--user", dest="plain_user", action="store_true", help="role：改为普通用户")
    u.add_argument("--display-name", default=""); u.add_argument("--jobs-root", default=None)
    u.add_argument("--force", action="store_true", help="disable / role：允许停用或降级最后一位启用的管理员")
    j = sub.add_parser("jobs", help="任务：list 经 HTTP 列出；claim 在服务停止时迁移 owner；du 占用；prune-cache 清可重算缓存")
    j.add_argument("action", choices=("claim", "list", "du", "prune-cache"))
    j.add_argument("--top", type=int, default=None, help="du：只列最大的 N 个任务")
    j.add_argument("--older-than", type=float, default=None, metavar="DAYS", help="prune-cache：只清早于这么多天的文件（必填）")
    j.add_argument("--dry-run", action="store_true", help="prune-cache：只列清单（不带 --yes 时也是只列清单）")
    j.add_argument("--yes", action="store_true", help="prune-cache：确认删除")
    j.add_argument("--owner", default=None, help="claim：旧会话 owner（见 .sessions.json 或任务 job.json）；`none` = 没有 owner 的命令行任务"); j.add_argument("--user", default=None, help="claim：接收任务的用户名；list：共享模式登录用户名")
    j.add_argument("--jobs-root", default=None); j.add_argument("--server", default="http://127.0.0.1:8765"); j.add_argument("--status", default=None)
    j.add_argument("--json", action="store_true"); j.add_argument("--all", action="store_true", help="管理员查看全部任务")
    j.add_argument("--force", action="store_true", help="claim：服务持有写锁时仍执行（运行中的服务可能覆盖结果；建议先停服务）")
    args = ap.parse_args(argv)
    _configure_clock(args)
    if args.cmd == "run":
        from .pipeline import stage_a, stage_b, run_all
        out = Path(args.out)
        if args.stage_a_only:
            a = stage_a(Path(args.stl), out, inlet=args.inlet, units=args.units); print(json.dumps({"stage": a["stage"], "input_check": a["input_check"], "proposal": (a["proposal"] or {}).get("mapping", {}), "flags": (a["proposal"] or {}).get("flags", []), "timing_s": a["timing_s"]}, ensure_ascii=False, indent=1)); return 0
        from .registry import ReleaseRegistry, ReleaseError
        try:
            registry = ReleaseRegistry(args.release_root, device=args.device)
            rel = registry.load(args.release_id)
        except ReleaseError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"release {rel.name} loaded on {rel.device} in {rel.load_seconds:.1f}s", file=sys.stderr)
        mapping = None
        if args.outlets != "auto":
            try:
                mapping = dict(kv.split("=", 1) for kv in args.outlets.split(","))
            except ValueError as exc:
                raise SystemExit("--outlets 格式应为 segment_id=out-le,...") from exc
        elif not args.accept_auto:
            a = stage_a(Path(args.stl), out, inlet=args.inlet, units=args.units)
            print(json.dumps({"stage": a["stage"], "message": "请人工核对出口映射后使用 --outlets，或在确认后显式加 --accept-auto。", "proposal": (a["proposal"] or {}).get("mapping", {}), "flags": (a["proposal"] or {}).get("flags", []), "input_check": a["input_check"]}, ensure_ascii=False, indent=1))
            return 2
        meta = run_all(Path(args.stl), out, rel, mapping=mapping, inlet=args.inlet,
                       smooth_mm=args.smooth_mm, spacing_mm=args.spacing_mm,
                       case_id=args.case_id, units=args.units,
                       allow_unvalidated_auto=bool(args.accept_auto and mapping is None))
        print(json.dumps({"case_id": meta["case_id"], "peak": meta.get("peak"), "volume_statistics": meta.get("volume_statistics"), "timing_s": meta["timing_s"], "report": str(out / "report.html")}, ensure_ascii=False, indent=1)); return 0
    if args.cmd == "serve":
        return _serve(args)
    if args.cmd == "service":
        return _service(args)
    if args.cmd == "doctor":
        return _doctor(args)
    if args.cmd == "reports":
        return _reports(args)
    if args.cmd == "template":
        return _template(args)
    if args.cmd == "submit":
        return _submit(args)
    if args.cmd == "user":
        from .jobs import JobError
        from .users import UserStore
        users_root = _jobs_root(args.jobs_root)
        store = UserStore(users_root / "users.json")
        try:
            if args.action == "list":
                users = store.load(); print(json.dumps([store.public(name, row) for name, row in sorted(users.items())], ensure_ascii=False, indent=1)); return 0
            if not args.name: raise SystemExit("请给出用户名。")
            if args.action == "role" and not (args.admin or args.plain_user): raise SystemExit("user role 需要 --admin 或 --user。")
            with _AuditLog(users_root) as audit:        # v0.15: every account change leaves an audit line (no password)
                if args.action == "add":
                    row = store.add(args.name, _read_password(confirm=True), admin=args.admin, display_name=args.display_name)
                    audit.line("user_add", target=args.name, role=row["role"]); print(json.dumps(row, ensure_ascii=False)); return 0
                if args.action == "passwd":
                    store.set_password(args.name, _read_password(confirm=True)); audit.line("user_passwd", target=args.name)
                    print(f"已更新 {args.name} 的口令；该用户已登录的会话均已失效。"); return 0
                if args.action == "role":
                    role_name = "admin" if args.admin else "user"
                    store.set_role(args.name, role_name, keep_admin=not args.force)   # v0.15: never the last admin unless --force
                    audit.line("user_role", target=args.name, role=role_name, forced=bool(args.force))
                    print(f"{args.name} 的角色为 {'管理员' if role_name == 'admin' else '普通用户'}；该用户已登录的会话均已失效。"); return 0
                store.disable(args.name, keep_admin=not args.force); audit.line("user_disable", target=args.name, forced=bool(args.force))
                print(f"已禁用 {args.name}；该用户已登录的会话均已失效。")
                if not store.exists():   # v0.15: no enabled user left → a shared service falls back to token login
                    print("警告：已没有启用的用户；共享模式下服务将退回令牌登录（.service_token / WSS_DEPLOY_TOKEN），"
                          "需要时先 `user add` 或删除令牌文件。", file=sys.stderr)
                return 0
        except JobError as exc:
            raise SystemExit(str(exc)) from exc
    if args.cmd == "jobs":
        if args.action == "list":
            return _jobs_list(args)
        if args.action in ("du", "prune-cache"):
            return _jobs_disk(args)
        if not args.owner or not args.user: raise SystemExit("jobs claim 需要 --owner 与 --user。")
        from .jobs import JobError, JobManager
        root = _jobs_root(args.jobs_root)
        lock = _offline_lock(root, force=args.force, purpose="jobs claim")
        try:
            # offline: nothing is marked interrupted / re-queued and no trash is purged (J3)
            manager = JobManager(root, stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {}, offline=True)
            with _AuditLog(root) as audit:     # v0.15: the per-job ``owner_claimed`` lines and a summary reach server.log
                old_owner = None if args.owner in ("none", "-") else args.owner
                result = manager.claim_owner(old_owner, args.user)
                import hashlib
                audit.line("jobs_claim", user=args.user, claimed=result.get("claimed"), jobs=result.get("job_ids"),
                           previous_owner="h:" + hashlib.sha256(old_owner.encode("utf-8")).hexdigest()[:12] if old_owner else None)
        except JobError as exc:
            raise SystemExit(str(exc)) from exc
        finally:
            if lock is not None:
                lock.release()
        print(json.dumps(result, ensure_ascii=False)); return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
