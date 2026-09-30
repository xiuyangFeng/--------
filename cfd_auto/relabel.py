"""Put a refined-step run (schedule ``dt2``: dt / r for n x r native steps) into the library layout: native step N ->
library label N / r (same flow time and cardiac phase as library step N / r).

  exports     ascii/ and ascii_in/: kept frames (``frames.json``) renamed <prefix>-{N/r:04d}; anything else is an error
  monitors    Global_conditions/*-rfile.out: rows of native steps N % r == 0 kept and relabelled (original -> *.raw)
  transcript  Global_conditions/Fluent_cfdauto_*.out: step blocks of N % r != 0 dropped, 'time step = N' -> N / r
              (original -> *.raw), so cfd_auto.sanity reads it like a library run

Idempotent: a finished relabel leaves RELABELLED.json and is not repeated. (Generalises the 2026-09-29
``_recover/relabel_dt2.py`` used for WANG_CAI-0/before.)

    python -m cfd_auto.relabel <work dir>
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from cfd_auto import schedule

NUM = re.compile(r"^(.*)-(\d{4})$")


def relabel(work: Path) -> dict:
    work = Path(work)
    done = work / "RELABELLED.json"
    if done.exists():
        return json.loads(done.read_text())
    fr = json.loads((work / schedule.FRAMES_FILE).read_text())["schedule"]
    r = int(fr["refine"])
    if r == 1:
        return {"refine": 1, "skipped": "native steps are library steps"}
    keep = set(fr["keep"])
    moved = {"ascii": 0, "ascii_in": 0}
    for sub in moved:
        files = sorted(f for f in (work / sub).iterdir() if f.is_file() and NUM.match(f.name))
        extra = [f.name for f in files if int(NUM.match(f.name).group(2)) not in keep]
        if extra:
            raise RuntimeError(f"{sub}: frames outside the kept set {extra[:5]}")
        for f in files:                      # ascending native order; labels are smaller, so no collision with a pending name
            m = NUM.match(f.name)
            target = f.with_name(f"{m.group(1)}-{int(m.group(2)) // r:04d}")
            if target.exists():
                raise RuntimeError(f"relabel target exists: {target}")
            f.rename(target); moved[sub] += 1
    gc = work / "Global_conditions"
    for f in sorted(gc.glob("*-rfile.out")):
        raw = f.with_suffix(".out.raw"); shutil.move(f, raw)
        out = []
        for line in open(raw):
            parts = line.split()
            if parts and parts[0].isdigit():
                n = int(parts[0])
                if n % r:
                    continue
                line = line.replace(parts[0], str(n // r), 1)
            out.append(line)
        f.write_text("".join(out))
    for f in sorted(gc.glob(schedule.TRANSCRIPT_GLOB)):
        raw = f.with_suffix(".out.raw"); shutil.move(f, raw)
        out, block, started = [], [], False
        for line in open(raw, errors="replace"):
            m = re.search(r"time step = (\d+)", line)
            if not started and not m and "Updating solution at time level" not in line:
                out.append(line); continue      # set-up / case-reading header before the first step
            started = True
            if m:
                n = int(m.group(1))
                if n % r == 0:
                    out += block + [line.replace(f"time step = {n}", f"time step = {n // r}")]
                block = []
                continue
            block.append(line)
        out += block
        f.write_text("".join(out))
    rep = {"refine": r, "moved": moved, "labels": sorted(int(k) // r for k in keep)}
    done.write_text(json.dumps(rep))
    return rep


if __name__ == "__main__":
    print(relabel(Path(sys.argv[1])))
