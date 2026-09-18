"""CLI.  run:   python -m wss_deploy.cli run case.stl --out jobs/case [--outlets auto | "3=out-le,4=out-li,5=out-re,6=out-ri"] [--device auto|cuda|cpu]
          serve: python -m wss_deploy.cli serve --port 8765 --jobs-root outputs/wss_deploy_jobs"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wss_deploy"); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("stl"); r.add_argument("--out", required=True); r.add_argument("--outlets", default="auto"); r.add_argument("--inlet", type=int, default=None)
    r.add_argument("--units", choices=("mm", "cm", "m"), default="mm"); r.add_argument("--accept-auto", action="store_true", help="仅用于已人工核对 proposal 的自动映射")
    r.add_argument("--smooth-mm", type=float, default=1.0); r.add_argument("--spacing-mm", type=float, default=0.5); r.add_argument("--device", default="auto"); r.add_argument("--case-id", default=None)
    r.add_argument("--stage-a-only", action="store_true", help="only ingest + centreline + naming proposal (no GPU needed)")
    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8765); s.add_argument("--jobs-root", default=None); s.add_argument("--device", default="auto"); s.add_argument("--token", default=None, help="非本机共享访问所需令牌")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        from .pipeline import stage_a, stage_b, run_all
        out = Path(args.out)
        if args.stage_a_only:
            a = stage_a(Path(args.stl), out, inlet=args.inlet, units=args.units); print(json.dumps({"stage": a["stage"], "input_check": a["input_check"], "proposal": (a["proposal"] or {}).get("mapping", {}), "flags": (a["proposal"] or {}).get("flags", []), "timing_s": a["timing_s"]}, ensure_ascii=False, indent=1)); return 0
        from .infer import Release
        rel = Release(device=args.device); print(f"release {rel.name} loaded on {rel.device} in {rel.load_seconds:.1f}s", file=sys.stderr)
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
        meta = run_all(Path(args.stl), out, rel, mapping=mapping, inlet=args.inlet, smooth_mm=args.smooth_mm, spacing_mm=args.spacing_mm, case_id=args.case_id)
        print(json.dumps({"case_id": meta["case_id"], "peak": meta["peak"], "timing_s": meta["timing_s"], "report": str(out / "report.html")}, ensure_ascii=False, indent=1)); return 0
    if args.cmd == "serve":
        from .server import serve
        serve(host=args.host, port=args.port, jobs_root=Path(args.jobs_root) if args.jobs_root else None, device=args.device, token=args.token); return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
