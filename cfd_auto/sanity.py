"""Numerical checks of a finished cfd_auto run when there is no original solution to compare with (2026-09-29).

    python -m cfd_auto.sanity <work dir> [--library-ref <library unit> ...] [--json out.json]

Everything is read from the run's own outputs (transcript, UDF prints, exports) and from protocol_bc.json:
  exports      81 wall + 81 volume frames (steps 1120..1280 step 2), no NaN / Inf
  flow         last-cycle mean total outflow / protocol mean (library runs: 1.103, the truncated waveform), left share
               (protocol 50/50), within-side shares vs the Murray d^3 split the RCR was built from
  pressure     last-cycle mean outlet pressure vs the protocol mean 93.33 mmHg; outlets within a few % of each other
  periodicity  last two cycles: mean flow share and mean pressure per outlet
  solver       time steps of the last cycle that reached Fluent's convergence criterion before 20 iterations; final
               continuity residual per step
  fields       peak-frame (1162) WSS and volume velocity magnitude quantiles (compared with library units when given)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from cfd_auto import compare, udf

STEPS = list(range(1120, 1281, 2))
PROTOCOL_P_PA = udf.PROTOCOL_P
PROTOCOL_Q_M3S = 30.38e-6
RHO = 1060.0
CYCLE_STEPS = 160
# AAA/ILO gates, calibrated 2026-09-29 on the three cfd_auto runs that reproduced their library solutions (YU_TIAN_HAI,
# GUO_YU_SHU-0/before, YANG_BAO_KUI vs its January solve): flow ratio 1.1029-1.1030, left share 0.489-0.503, mean outlet
# pressure +5.5..+6.3 % over 93.33 mmHg (the waveform's mean inflow is 1.103 x the protocol), last-two-cycle pressure
# change 1.0-1.6 % (8 cycles do not fully settle the windkessels; the library labels carry the same drift).
# Converged-before-20-iterations is a property of the settings template (GUO: never), so it is reported, not gated.
GATES = {"flow_ratio": (1.098, 1.108), "left_share": (0.45, 0.55), "side_share_abs_dev_max": (0.0, 0.10),
         "outlet_p_rel_mean": (0.03, 0.09), "periodicity_share_abs_max": (0.0, 0.003), "periodicity_p_rel_max": (0.0, 0.025)}


def transcript(work: Path) -> Path:
    return compare.solver_log(work)


def step_prints(log: Path) -> dict[int, dict[str, float]]:
    """UDF prints per time step. Fluent prints 'time step = N' at the END of step N, so prints between the lines
    for N-1 and N belong to step N."""
    out, cur, last_done = {}, {}, 0
    for line in open(log, errors="replace"):
        m = re.search(r"time step = (\d+)", line)
        if m:
            n = int(m.group(1))
            if cur:
                out[n] = cur
            cur, last_done = {}, n
            continue
        if "P_ave_out" in line:
            for k, v in re.findall(r"(P_ave_out\w\w|Q_ave_out\w\w)=([-+0-9.eE]+)", line):
                cur[k] = float(v)
    return out


def solver_steps(log: Path) -> dict[int, dict]:
    """Per time step: iterations, converged flag, final continuity residual, reversed-flow warnings."""
    out, cur = {}, {"iters": 0, "converged": False, "cont": None, "reversed": 0}
    row = re.compile(r"^\s*!?\s*(\d+)\s+([0-9.]+e[-+]\d+)\s+([0-9.]+e[-+]\d+)\s+([0-9.]+e[-+]\d+)\s+([0-9.]+e[-+]\d+)")
    for line in open(log, errors="replace"):
        m = re.search(r"time step = (\d+)", line)
        if m:
            out[int(m.group(1))] = cur
            cur = {"iters": 0, "converged": False, "cont": None, "reversed": 0}
            continue
        r = row.match(line)
        if r:
            cur["iters"] += 1; cur["cont"] = float(r.group(2)); continue
        if "solution is converged" in line:
            cur["converged"] = True
        elif "Reversed flow" in line:
            cur["reversed"] += 1
    return out


def _q(rows: list[dict], k: str) -> np.ndarray:
    return np.array([r[f"Q_ave_{k}"] for r in rows])


def _p(rows: list[dict], k: str) -> np.ndarray:
    return np.array([r[f"P_ave_{k}"] for r in rows])


def frame_stats(work: Path, sub: str, step: int, col: str) -> dict:
    h, x = compare.read_export(compare.frame_file(work, sub, step))
    c = {k: i for i, k in enumerate(h)}
    v = x[:, c[col]]
    bad = int((~np.isfinite(x)).sum())
    return {"rows": int(len(x)), "nonfinite": bad, "p50": float(np.percentile(v, 50)), "p99": float(np.percentile(v, 99)), "max": float(v.max()), "min": float(v.min())}


def check(work: Path, library_refs: list[Path] | None = None, gates: dict | None = None, protocol_q_m3s: float | None = None) -> dict:
    """``gates`` (managed runs, protocol['sanity_gates']): {metric: {"warn": [lo, hi], "fail": [lo, hi]}} -> values outside
    'fail' are flags (``ok`` False), outside 'warn' only ``warnings``; default: the single-level ``GATES`` (all flags).
    ``protocol_q_m3s``: the protocol mean inflow for the flow ratio (default the AAA/ILO value)."""
    work = Path(work)
    rep: dict = {"work": str(work)}
    # exports
    n_w = [s for s in STEPS if len(list((work / "ascii").glob(f"*-{s:04d}"))) == 1]
    n_v = [s for s in STEPS if len(list((work / "ascii_in").glob(f"*-{s:04d}"))) == 1]
    rep["exports"] = {"wall_frames": len(n_w), "volume_frames": len(n_v), "ok": len(n_w) == 81 and len(n_v) == 81}
    log = transcript(work); rep["transcript"] = str(log)
    pr = step_prints(log)
    last = max(pr)
    rep["last_step"] = last
    cyc = [pr[s] for s in range(last - CYCLE_STEPS + 1, last + 1) if s in pr]
    prev = [pr[s] for s in range(last - 2 * CYCLE_STEPS + 1, last - CYCLE_STEPS + 1) if s in pr]
    keys = udf.OUTLETS
    q = {k: _q(cyc, k).mean() for k in keys}; tot = sum(q.values())
    share = {k: q[k] / tot for k in keys}
    rep["flow"] = {"flow_ratio": tot / RHO / (protocol_q_m3s or PROTOCOL_Q_M3S), "share": share, "left_share": share["outle"] + share["outli"]}
    prot = json.loads((work / "protocol_bc.json").read_text()) if (work / "protocol_bc.json").exists() else None
    if prot:
        d3 = {k: (2 * np.sqrt(prot["cut_area_mm2"][k] / np.pi)) ** 3 for k in keys}
        dev = {}
        for a, b in (("outle", "outli"), ("outre", "outri")):
            side = share[a] + share[b]
            for k in (a, b):
                dev[k] = share[k] / side - d3[k] / (d3[a] + d3[b])
        rep["flow"]["side_share_dev_vs_murray"] = dev
        rep["flow"]["side_share_abs_dev_max"] = max(abs(v) for v in dev.values())
        rep["rcr_R2_positive"] = all(prot["rcr"][k]["R2"] > 0 for k in keys)
    p = {k: _p(cyc, k).mean() for k in keys}
    rep["pressure"] = {"outlet_mean_pa": p, "outlet_p_rel_mean": float(np.mean(list(p.values())) / PROTOCOL_P_PA - 1),
                       "outlet_p_spread_rel": float((max(p.values()) - min(p.values())) / np.mean(list(p.values()))),
                       "outlet_p_min_max_pa": [float(min(_p(cyc, k).min() for k in keys)), float(max(_p(cyc, k).max() for k in keys))]}
    if len(prev) == CYCLE_STEPS:
        qp = {k: _q(prev, k).mean() for k in keys}; tp = sum(qp.values())
        pp = {k: _p(prev, k).mean() for k in keys}
        rep["periodicity"] = {"periodicity_share_abs_max": max(abs(qp[k] / tp - share[k]) for k in keys),
                              "periodicity_p_rel_max": max(abs(pp[k] / p[k] - 1) for k in keys),
                              "total_flow_rel": tp / tot - 1}
    ss = solver_steps(log)
    lc = [ss[s] for s in range(last - CYCLE_STEPS + 1, last + 1) if s in ss]
    cont = np.array([s["cont"] for s in lc if s["cont"] is not None])
    rep["solver"] = {"steps_last_cycle": len(lc), "converged_fraction": float(np.mean([s["converged"] for s in lc])) if lc else None,
                     "iters_mean": float(np.mean([s["iters"] for s in lc])) if lc else None,
                     "final_continuity_p50_p95_max": np.percentile(cont, [50, 95, 100]).tolist() if len(cont) else None,
                     "reversed_flow_warnings_last_cycle": int(sum(s["reversed"] for s in lc))}
    rep["fields"] = {"wall_wss_1162": frame_stats(work, "ascii", 1162, "wall-shear")}
    try:
        rep["fields"]["volume_speed_1162"] = frame_stats(work, "ascii_in", 1162, "velocity-magnitude")
    except (KeyError, FileNotFoundError) as exc:
        rep["fields"]["volume_speed_1162"] = {"error": str(exc)}
    if library_refs:
        rep["library_refs"] = {}
        for r in library_refs:
            try:
                rep["library_refs"][str(r)] = {"wall_wss_1162": frame_stats(r, "ascii", 1162, "wall-shear")}
            except Exception as exc:     # a library unit whose export layout differs
                rep["library_refs"][str(r)] = {"error": f"{type(exc).__name__}: {exc}"}
    flags = []
    vals = {"flow_ratio": rep["flow"]["flow_ratio"], "left_share": rep["flow"]["left_share"], "side_share_abs_dev_max": rep["flow"].get("side_share_abs_dev_max"),
            "outlet_p_rel_mean": rep["pressure"]["outlet_p_rel_mean"], **{k: rep.get("periodicity", {}).get(k) for k in ("periodicity_share_abs_max", "periodicity_p_rel_max")}}
    warnings = []
    for k, g in (gates or {k: {"fail": list(v)} for k, v in GATES.items()}).items():
        v = vals.get(k)
        lo, hi = g["fail"]
        if v is None or not (lo <= v <= hi):
            flags.append(f"{k}={v} outside [{lo}, {hi}]")
        elif "warn" in g and not (g["warn"][0] <= v <= g["warn"][1]):
            warnings.append(f"{k}={v} outside warn [{g['warn'][0]}, {g['warn'][1]}]")
    if not rep["exports"]["ok"]:
        flags.append(f"exports {rep['exports']}")
    for f in ("wall_wss_1162", "volume_speed_1162"):
        if rep["fields"][f].get("nonfinite"):
            flags.append(f"{f} has {rep['fields'][f]['nonfinite']} non-finite values")
    if last != 1280:
        flags.append(f"last step {last} != 1280")
    if prot and not rep["rcr_R2_positive"]:
        flags.append("protocol R2 <= 0")
    rep["gates"] = gates if gates is not None else {k: list(v) for k, v in GATES.items()}
    if gates is not None:
        rep["warnings"] = warnings
    rep["flags"] = flags
    rep["ok"] = not flags
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path); ap.add_argument("--library-ref", type=Path, nargs="*", default=None); ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    rep = check(a.work, a.library_ref)
    out = a.json or a.work / "sanity.json"
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"ok": rep["ok"], "flags": rep["flags"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
