"""CLI.  run:   python -m wss_deploy.cli run case.stl --out jobs/case [--outlets auto | "3=out-le,4=out-li,5=out-re,6=out-ri"] [--device auto|cuda|cpu]
          serve: python -m wss_deploy.cli serve --port 8765 --jobs-root outputs/wss_deploy_jobs
          user:  python -m wss_deploy.cli user add|passwd|disable|list <name> [--admin] [--jobs-root …]   (password read from the terminal)
          jobs:  python -m wss_deploy.cli jobs claim --owner <old owner> --user <name> [--jobs-root …]    (service stopped)"""
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wss_deploy"); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("stl"); r.add_argument("--out", required=True); r.add_argument("--outlets", default="auto"); r.add_argument("--inlet", type=int, default=None)
    r.add_argument("--units", choices=("mm", "cm", "m"), default="mm"); r.add_argument("--accept-auto", action="store_true", help="仅用于已人工核对 proposal 的自动映射")
    r.add_argument("--smooth-mm", type=float, default=1.0); r.add_argument("--spacing-mm", type=float, default=0.5); r.add_argument("--device", default="auto"); r.add_argument("--case-id", default=None)
    r.add_argument("--release-root", default=None); r.add_argument("--release-id", default=None, help="发布包标识；省略使用默认发布包")
    r.add_argument("--stage-a-only", action="store_true", help="only ingest + centreline + naming proposal (no GPU needed)")
    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8765); s.add_argument("--jobs-root", default=None); s.add_argument("--device", default="auto"); s.add_argument("--token", default=None, help="非本机共享访问所需令牌")
    u = sub.add_parser("user", help="共享模式的用户名登录（users.json）"); u.add_argument("action", choices=("add", "passwd", "disable", "list")); u.add_argument("name", nargs="?", default=None)
    u.add_argument("--admin", action="store_true", help="管理员可用 ?all=1 查看全部任务（只读）"); u.add_argument("--display-name", default=""); u.add_argument("--jobs-root", default=None)
    j = sub.add_parser("jobs", help="任务维护（服务停止时执行）"); j.add_argument("action", choices=("claim",)); j.add_argument("--owner", required=True, help="旧会话 owner（见 .sessions.json 或任务 job.json）")
    j.add_argument("--user", required=True, help="接收任务的用户名"); j.add_argument("--jobs-root", default=None)
    args = ap.parse_args(argv)
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
        from .server import serve
        serve(host=args.host, port=args.port, jobs_root=Path(args.jobs_root) if args.jobs_root else None, device=args.device, token=args.token); return 0
    if args.cmd == "user":
        from .jobs import JobError
        from .users import UserStore
        store = UserStore(_jobs_root(args.jobs_root) / "users.json")
        try:
            if args.action == "list":
                users = store.load(); print(json.dumps([store.public(name, row) for name, row in sorted(users.items())], ensure_ascii=False, indent=1)); return 0
            if not args.name: raise SystemExit("请给出用户名。")
            if args.action == "add":
                row = store.add(args.name, _read_password(confirm=True), admin=args.admin, display_name=args.display_name); print(json.dumps(row, ensure_ascii=False)); return 0
            if args.action == "passwd":
                store.set_password(args.name, _read_password(confirm=True)); print(f"已更新 {args.name} 的口令。"); return 0
            store.disable(args.name); print(f"已禁用 {args.name}。"); return 0
        except JobError as exc:
            raise SystemExit(str(exc)) from exc
    if args.cmd == "jobs":
        from .jobs import JobError, JobManager
        try:
            manager = JobManager(_jobs_root(args.jobs_root), stage_a_fn=lambda *_a, **_k: {}, stage_b_fn=lambda *_a, **_k: {})
            result = manager.claim_owner(args.owner, args.user)
        except JobError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(result, ensure_ascii=False)); return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
