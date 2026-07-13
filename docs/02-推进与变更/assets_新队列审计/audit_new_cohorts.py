#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读审计 AAA / ILO 队列的输入完整性 + 物理有效性。

- 不跑预处理、不做解剖配准、不写 data_wss_min。
- 复用 pipeline_wss_min.raw_io 按 pipeline 真实口径读取壁面 WSS 与 centerline。
- ILO 的 before/after 各算一个独立病例（用户口径）。
- 物理有效性判据镜像 qa_gate.LIMITS 中不依赖配准的项：
    peak_zero_frac<=0.01, all(sampled)_zero_frac<=0.01, peak_p90>0,
    n_wall<=50000, WSS/坐标全有限。
- 单位/尺度：用 centerline 包围盒对角线 / 壁面对角线估 mesh->mm 因子，
  |log10(factor)-3|>0.5 记 unit_anomaly（仅提示，不判死）。
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
sys.path.insert(0, str(ROOT))
from pipeline_wss_min import raw_io, config as C  # noqa: E402

DATA = ROOT / "data_new"
OUT_CSV = Path("/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/"
               "c63e71da-acd6-4ca8-b8c0-20257aea34aa/scratchpad/audit_new_cohorts.csv")

ZERO_FRAC_LIM = 0.01
NWALL_MAX = 50000
UNIT_LOG10_TOL = 0.5


def detect_case_prefix(ascii_dir: Path) -> str | None:
    """壁面文件名形如 <prefix>-<step>；目录名可能带 -0/-1 后缀而文件用裸姓名。"""
    for f in sorted(ascii_dir.iterdir()):
        if f.is_file():
            m = re.match(r"(.+)-(\d+)$", f.name)
            if m:
                return m.group(1)
    return None


def bbox_diag(pts: np.ndarray) -> float:
    lo = np.nanmin(pts, axis=0)
    hi = np.nanmax(pts, axis=0)
    return float(np.linalg.norm(hi - lo))


def audit_unit(unit_id: str, cohort: str, phase: str, case_dir: Path) -> dict:
    row = {
        "unit_id": unit_id, "cohort": cohort, "phase": phase,
        "status": "ok", "reasons": "",
        "n_wall": "", "n_steps": "", "peak_step": "",
        "peak_zero_frac": "", "sampled_zero_frac": "", "peak_p90": "",
        "peak_max": "", "peak_mean": "", "wss_nonfinite": "", "coords_nonfinite": "",
        "n_centerline": "", "cl_radius_min": "", "cl_radius_max": "",
        "unit_factor": "", "unit_anomaly": "",
    }
    reasons = []
    wall_dir = case_dir / C.RAW_LAYOUT["wall_ascii_dir"]
    cl_csv = case_dir / C.RAW_LAYOUT["centerline_csv"]

    # --- 结构完整性 ---
    if not wall_dir.is_dir() or not any(p.is_file() for p in wall_dir.iterdir()):
        row["status"] = "fail"; row["reasons"] = "no_wall_ascii"
        return row
    prefix = detect_case_prefix(wall_dir)
    if prefix is None:
        row["status"] = "fail"; row["reasons"] = "no_timestep_files"
        return row
    if not cl_csv.is_file():
        reasons.append("no_centerline")

    # --- 时间步 + 峰值收缩期 ---
    try:
        steps = raw_io.list_timesteps(case_dir)
    except Exception as e:
        row["status"] = "fail"; row["reasons"] = f"list_steps_err:{e}"
        return row
    if not steps:
        row["status"] = "fail"; row["reasons"] = "no_steps"
        return row
    row["n_steps"] = len(steps)
    try:
        peak = raw_io.peak_systole_step(case_dir, steps)
    except Exception:
        peak = steps[len(steps) // 2]
    row["peak_step"] = peak

    # --- 壁面 WSS 有效性（peak + 抽样多步） ---
    try:
        wf = raw_io.read_wall_fields(case_dir, prefix, peak)
    except Exception as e:
        row["status"] = "fail"; row["reasons"] = f"read_wall_err:{e}"
        return row
    wss = wf["wss"].astype(np.float64)
    coords = wf["coords"].astype(np.float64)
    n_wall = wss.shape[0]
    row["n_wall"] = n_wall
    finite = np.isfinite(wss)
    row["wss_nonfinite"] = int((~finite).sum())
    row["coords_nonfinite"] = int((~np.isfinite(coords)).sum())
    row["peak_zero_frac"] = round(float((wss[finite] <= 0).mean()) if finite.any() else float("nan"), 5)
    row["peak_p90"] = round(float(np.nanpercentile(wss, 90)), 4)
    row["peak_max"] = round(float(np.nanmax(wss)), 4)
    row["peak_mean"] = round(float(np.nanmean(wss)), 4)

    # 抽样若干步统计整体零值率 + 节点数一致性
    sample_steps = sorted(set([steps[len(steps)//2], peak]))
    zero_fracs = []
    ncounts = []
    for s in sample_steps:
        try:
            w = raw_io.read_wall_fields(case_dir, prefix, s)["wss"].astype(np.float64)
        except Exception:
            continue
        fw = np.isfinite(w)
        if fw.any():
            zero_fracs.append(float((w[fw] <= 0).mean()))
        ncounts.append(w.shape[0])
    row["sampled_zero_frac"] = round(max(zero_fracs), 5) if zero_fracs else float("nan")
    if len(set(ncounts)) > 1:
        reasons.append(f"wall_count_varies{sorted(set(ncounts))}")

    # --- centerline（直接读 CSV，跳过 vtk/vtp 以提速）---
    cl_diag = None
    if cl_csv.is_file():
        try:
            col = C.CENTERLINE_COLUMNS
            df = pd.read_csv(cl_csv)
            df.columns = [c.strip() for c in df.columns]
            clc = df[[col["x"], col["y"], col["z"]]].to_numpy(dtype=np.float64)
            rad = df[col["radius"]].to_numpy(dtype=np.float64)
            row["n_centerline"] = len(clc)
            row["cl_radius_min"] = round(float(np.nanmin(rad)), 4)
            row["cl_radius_max"] = round(float(np.nanmax(rad)), 4)
            cl_diag = bbox_diag(clc)
        except Exception as e:
            reasons.append(f"centerline_parse_err:{type(e).__name__}")

    # --- 单位/尺度提示（不判死）---
    wall_diag = bbox_diag(coords)
    if cl_diag and wall_diag > 0:
        factor = cl_diag / wall_diag
        row["unit_factor"] = round(factor, 3)
        anom = abs(np.log10(factor) - 3.0) > UNIT_LOG10_TOL if factor > 0 else True
        row["unit_anomaly"] = bool(anom)
        if anom:
            reasons.append(f"unit_anomaly(factor={factor:.3g})")

    # --- 物理有效性硬判据 ---
    if row["wss_nonfinite"] > 0 or row["coords_nonfinite"] > 0:
        reasons.append("nonfinite_values")
    if not np.isnan(row["peak_zero_frac"]) and row["peak_zero_frac"] > ZERO_FRAC_LIM:
        reasons.append(f"peak_zero_frac>{ZERO_FRAC_LIM}")
    if not np.isnan(row["sampled_zero_frac"]) and row["sampled_zero_frac"] > ZERO_FRAC_LIM:
        reasons.append(f"sampled_zero_frac>{ZERO_FRAC_LIM}")
    if row["peak_p90"] <= 0:
        reasons.append("peak_p90<=0")
    if n_wall > NWALL_MAX:
        reasons.append(f"n_wall>{NWALL_MAX}")

    hard = [r for r in reasons if not r.startswith("unit_anomaly")
            and not r.startswith("wall_count_varies")]
    soft = [r for r in reasons if r.startswith("unit_anomaly")
            or r.startswith("wall_count_varies")]
    # no_centerline 归为硬缺失（配准/几何特征必需）
    if "no_centerline" in reasons and "no_centerline" not in hard:
        hard.append("no_centerline")
    hard = [r for r in dict.fromkeys(hard)]
    if hard:
        row["status"] = "fail"; row["reasons"] = ";".join(hard + soft)
    elif soft:
        row["status"] = "warn"; row["reasons"] = ";".join(soft)
    return row


def enumerate_units():
    # AAA: 平铺 ruputer / unruputer
    for grp in ("ruputer", "unruputer"):
        base = DATA / "AAA" / grp
        if base.is_dir():
            for cd in sorted(p for p in base.iterdir() if p.is_dir()):
                yield f"AAA/{grp}/{cd.name}", f"AAA/{grp}", "-", cd
    # ILO: before/after 各算独立病例
    base = DATA / "ILO"
    if base.is_dir():
        for cd in sorted(p for p in base.iterdir() if p.is_dir()):
            for phase in ("before", "after"):
                pd_ = cd / phase
                if pd_.is_dir():
                    yield f"ILO/{cd.name}/{phase}", "ILO", phase, pd_


def main():
    rows = []
    units = list(enumerate_units())
    for i, (unit_id, cohort, phase, case_dir) in enumerate(units, 1):
        try:
            r = audit_unit(unit_id, cohort, phase, case_dir)
        except Exception as e:
            r = {"unit_id": unit_id, "cohort": cohort, "phase": phase,
                 "status": "fail", "reasons": f"crash:{type(e).__name__}:{e}"}
        rows.append(r)
        print(f"[{i:3}/{len(units)}] {r['status']:4} {unit_id}", flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    def tally(pred):
        sub = [r for r in rows if pred(r)]
        p = sum(r["status"] == "ok" for r in sub)
        wn = sum(r["status"] == "warn" for r in sub)
        fl = sum(r["status"] == "fail" for r in sub)
        return len(sub), p, wn, fl

    print(f"\n{'='*70}\nAUDIT SUMMARY (units; ILO before/after each counted)\n{'='*70}")
    print(f"{'group':<20}{'total':>7}{'PASS':>7}{'WARN':>7}{'FAIL':>7}")
    for name, pred in [
        ("AAA/ruputer",  lambda r: r["cohort"] == "AAA/ruputer"),
        ("AAA/unruputer", lambda r: r["cohort"] == "AAA/unruputer"),
        ("ILO/before",   lambda r: r["cohort"] == "ILO" and r.get("phase") == "before"),
        ("ILO/after",    lambda r: r["cohort"] == "ILO" and r.get("phase") == "after"),
        ("ALL",          lambda r: True),
    ]:
        t, p, wn, fl = tally(pred)
        print(f"{name:<20}{t:>7}{p:>7}{wn:>7}{fl:>7}")

    print(f"\n--- FAIL / WARN 明细 ---")
    for r in rows:
        if r["status"] != "ok":
            print(f"  [{r['status'].upper():4}] {r['unit_id']:<34} {r['reasons']}")
    print(f"\nCSV -> {OUT_CSV}")


if __name__ == "__main__":
    main()
