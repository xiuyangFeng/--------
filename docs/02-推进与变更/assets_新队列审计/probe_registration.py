#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 AAA/ILO 达标单元试跑 AG 解剖配准（只读几何，不生成 bundle、不写 data_wss_min）。

复现 preprocess 的配准前置链：
  read geometry(native) -> _resolve_unit_factor -> *factor -> compute_transform
采集配准诊断，判断 AG 的 flow_divider + wall_branches 框架在瘤体/髂动脉解剖上的成功率。
成功判据（与 AG 干净配准一致）：
  origin_kind=='flow_divider' & roll_source=='wall_branches' & roll_sign_reliable
  & 无 unit_anomaly/extent_mismatch & trunk_centering_offset_frac<=0.05
"""
from __future__ import annotations
import sys, re, csv
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
sys.path.insert(0, str(ROOT))
from pipeline_wss_min import raw_io, config as C            # noqa: E402
from pipeline_wss_min.preprocess import _resolve_unit_factor  # noqa: E402
from pipeline_wss_min.registration import compute_transform   # noqa: E402

SCR = Path("/tmp/claude-1006/-public-newhome-cy-Digital-twin-GNN/"
           "c63e71da-acd6-4ca8-b8c0-20257aea34aa/scratchpad")
CLS = pd.read_csv(SCR / "audit_classified.csv")
USABLE = CLS[CLS["cat"].isin(["F_clean_pass", "E_dense_clean"])]["unit_id"].tolist()
OUT = SCR / "probe_registration.csv"
WARN_OFFSET = 0.05


def case_dir_and_prefix(unit_id: str):
    cd = ROOT / "data_new" / unit_id
    ad = cd / "ascii"
    pfx = None
    for f in sorted(ad.iterdir()):
        m = re.match(r"(.+)-(\d+)$", f.name)
        if m:
            pfx = m.group(1); break
    return cd, pfx


def probe(unit_id: str) -> dict:
    row = {"unit_id": unit_id, "status": "ok", "note": ""}
    cd, pfx = case_dir_and_prefix(unit_id)
    if pfx is None:
        row["status"] = "err"; row["note"] = "no_prefix"; return row
    steps = raw_io.list_timesteps(cd)
    ref = steps[len(steps) // 2]
    try:
        wall_native = raw_io.read_wall_geometry(cd, pfx, ref)
        int_native = raw_io.read_interior_geometry(cd, pfx, ref)
        cl = raw_io.read_centerline(cd)
    except Exception as e:
        row["status"] = "err"; row["note"] = f"read:{type(e).__name__}:{e}"; return row

    ucfg = C.DEFAULT.unit
    factor, unit_anom, extent_mm = _resolve_unit_factor(wall_native, cl, ucfg)
    wall_pts = wall_native * factor
    int_pts = int_native * factor
    reg_cfg = C.DEFAULT.registration
    try:
        T = compute_transform(wall_pts, int_pts, cl, reg_cfg)
    except Exception as e:
        row.update(status="err", note=f"reg:{type(e).__name__}:{e}",
                   unit_factor=round(float(factor), 3), unit_anomaly=bool(unit_anom),
                   unit_extent_mismatch=bool(extent_mm))
        return row
    d = T.to_dict()
    row.update(
        unit_factor=round(float(factor), 3),
        unit_anomaly=bool(unit_anom),
        unit_extent_mismatch=bool(extent_mm),
        vtp_available=bool(np.asarray(cl.get("vtp_available", False))),
        has_dist_bif=("dist_to_bifurcation" in cl),
        origin_kind=d["origin_kind"],
        main_axis_source=d["main_axis_source"],
        roll_source=d["roll_source"],
        roll_sign_source=d["roll_sign_source"],
        roll_sign_cos=round(float(d["roll_sign_cos"]), 3),
        roll_sign_reliable=bool(d["roll_sign_reliable"]),
        trunk_offset_frac=round(float(d["trunk_centering_offset_frac"]), 4),
    )
    reasons = []
    if row["origin_kind"] != "flow_divider":
        reasons.append(f"origin={row['origin_kind']}")
    if row["roll_source"] != "wall_branches":
        reasons.append(f"roll={row['roll_source']}")
    if not row["roll_sign_reliable"]:
        reasons.append("roll_sign_unreliable")
    if row["unit_anomaly"]:
        reasons.append("unit_anomaly")
    if row["unit_extent_mismatch"]:
        reasons.append("unit_extent_mismatch")
    if row["main_axis_source"] != "centerline":
        reasons.append(f"axis={row['main_axis_source']}")
    if row["trunk_offset_frac"] > WARN_OFFSET:
        reasons.append(f"trunk_offset>{WARN_OFFSET}")
    row["reg_ok"] = len(reasons) == 0
    row["reg_reasons"] = ";".join(reasons)
    return row


def main():
    rows = []
    for i, uid in enumerate(USABLE, 1):
        try:
            r = probe(uid)
        except Exception as e:
            r = {"unit_id": uid, "status": "err", "note": f"crash:{type(e).__name__}:{e}"}
        rows.append(r)
        print(f"[{i:3}/{len(USABLE)}] {r.get('reg_ok','?')!s:5} {uid} "
              f"{r.get('reg_reasons','') or r.get('note','')}", flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    df = pd.DataFrame(rows)
    df["grp"] = df["unit_id"].apply(
        lambda u: "/".join(u.split("/")[:2]) if u.startswith("AAA")
        else f"ILO/{u.split('/')[-1]}")
    print("\n" + "=" * 72 + "\nREGISTRATION PROBE SUMMARY (AG flow_divider+wall_branches)\n" + "=" * 72)
    print(f"{'group':<16}{'n':>5}{'reg_ok':>8}{'ok%':>7}{'err':>6}")
    for g in ["AAA/ruputer", "AAA/unruputer", "ILO/before", "ILO/after"]:
        sub = df[df["grp"] == g]
        if not len(sub): continue
        ok = int((sub.get("reg_ok") == True).sum())
        er = int((sub["status"] == "err").sum())
        print(f"{g:<16}{len(sub):>5}{ok:>8}{100*ok/max(len(sub),1):>6.0f}%{er:>6}")
    ok = int((df.get("reg_ok") == True).sum())
    print(f"{'ALL':<16}{len(df):>5}{ok:>8}{100*ok/max(len(df),1):>6.0f}%{int((df['status']=='err').sum()):>6}")

    print("\n--- 失败原因分布 ---")
    from collections import Counter
    cnt = Counter()
    for r in rows:
        for x in (r.get("reg_reasons") or "").split(";"):
            if x: cnt[x.split(">")[0].split("=")[0]] += 1
    for k, v in cnt.most_common():
        print(f"  {k:<26} {v}")
    print(f"\nCSV -> {OUT}")


if __name__ == "__main__":
    main()
