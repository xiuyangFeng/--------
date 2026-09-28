"""Read-only protection for the case library: nothing Fluent reads or writes may live under data/ or data_new/."""
from __future__ import annotations

import re
from pathlib import Path

LIBRARY_ROOTS = (Path("/public/newhome/cy/Digital_twin/GNN/data"), Path("/public/newhome/cy/Digital_twin/GNN/data_new"))
ABS_PATH_RE = re.compile(r"/public/[^\s\"')]+")


def assert_inside(path: str | Path, workdir: str | Path) -> Path:
    p, w = Path(path).resolve(), Path(workdir).resolve()
    for root in LIBRARY_ROOTS:
        if p == root or root in p.parents or w == root or root in w.parents:
            raise PermissionError(f"refusing to touch the case library: {p}")
    if p != w and w not in p.parents:
        raise PermissionError(f"{p} is outside the work directory {w}")
    return p


def check_journal(text: str, workdir: str | Path) -> None:
    """Every absolute path a journal mentions must be inside the work directory (or the Fluent install)."""
    for m in ABS_PATH_RE.findall(text):
        if m.startswith("/public/newapps/"):
            continue
        assert_inside(m, workdir)
