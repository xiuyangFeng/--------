"""Write the two example files the input guide and the workspace link to (``static/v2/example_aaa.stl`` and
``static/v2/example_report.html``).

Both are derived from a patient geometry, so they are **not** kept in git (see ``.gitignore``); run this once on the
machine that serves the workbench, after deploying the code:

    PYTHONPATH=. python -m wss_deploy.v2_examples \
        --stl outputs/wss_deploy_golden/20260920_baseline/20260920_144135_673ccd0e36b1/input.stl \
        --job-dir outputs/wss_deploy_jobs/<finished three-head job>

* ``example_aaa.stl``: the binary STL with its 80-byte header replaced by a neutral text (the header of the source
  file carries the case name); the triangles are copied byte for byte.
* ``example_report.html``: the single-file offline report of ``--job-dir`` with every case name hidden (the same
  builder as ``POST /api/v2/jobs/<id>/offline`` with ``hide_name``), served by ``/v2/example`` to logged-in users.

Either argument can be omitted to write only the other file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .paths import STATIC_DIR

V2_DIR = STATIC_DIR / "v2"
EXAMPLE_STL = "example_aaa.stl"
EXAMPLE_REPORT = "example_report.html"
STL_HEADER = b"WSS example: aorto-iliac lumen surface, 5 openings, units mm"


def write_example_stl(source: Path, target: Path | None = None) -> Path:
    """Copy a binary STL with a neutral header; refuses ASCII STL (its first line would carry the solid's name)."""
    data = Path(source).read_bytes()
    if len(data) < 84 or data[:5].lower() == b"solid" and b"facet" in data[:1024].lower():
        raise ValueError("示例 STL 必须是二进制 STL")
    count = int.from_bytes(data[80:84], "little")
    if len(data) != 84 + 50 * count:
        raise ValueError(f"二进制 STL 长度与三角面数 {count} 不符")
    target = Path(target) if target else V2_DIR / EXAMPLE_STL
    target.write_bytes(STL_HEADER.ljust(80, b" ") + data[80:])
    return target


def write_example_report(job_dir: Path, target: Path | None = None) -> Path:
    """The offline report of a finished job with the case name hidden."""
    from . import v2_data, v2_offline
    job_dir = Path(job_dir)
    record = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    if record.get("status") != "done":
        raise ValueError(f"任务 {job_dir.name} 还没有完成")
    data = v2_data.load(job_dir)
    release_id = (record.get("model_release") or {}).get("id") if isinstance(record.get("model_release"), dict) else None
    manifest = v2_data.build_manifest(data, record, card=v2_data.load_card(release_id))
    page = v2_offline.build_offline_html(data, manifest, record=record, hide_name=True)
    target = Path(target) if target else V2_DIR / EXAMPLE_REPORT
    target.write_text(page, encoding="utf-8")
    return target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wss_deploy.v2_examples", description=__doc__.split("\n\n")[0])
    ap.add_argument("--stl", type=Path, help="二进制 STL（例如黄金参照里的 input.stl）")
    ap.add_argument("--job-dir", type=Path, help="已完成任务的目录（建议用三头结果）")
    args = ap.parse_args(argv)
    if not args.stl and not args.job_dir:
        ap.error("至少给出 --stl 或 --job-dir")
    out = {}
    if args.stl:
        out["stl"] = str(write_example_stl(args.stl))
    if args.job_dir:
        out["report"] = str(write_example_report(args.job_dir))
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
