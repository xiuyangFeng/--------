"""按 CFD 协议重算被标记新单元的 RCR（2026-09-22；配方与 09-15 的 rcr_corrected_values.py 完全相同）。
协议：r = sqrt(s/π)·1000 mm；d = (2r)³/((2r_nei)³+(2r_wai)³)；flow = A1·d·0.5；Rt = 93.33·133/flow；C = 1.79/Rt；
      R1 = 13.3/(2r)^0.3/s；R2 = Rt − R1。A1（总质量流量）逐例从 UDF 精确反推（R1 反解表内隐含半径 → 表内 d → A1 = 2P/(Rt·d)，
      四出口应一致；对录错的出口反推值会离群，取中位）。
入选：rcr_protocol_audit_new.csv 里 verdict != ok 的单元；同侧 |UDF 份额 − 网格 r³ 份额| > 0.03 的那一侧两口都重写。
用法：python rcr_correct_new.py [--apply] [--only CASE ...]   默认只出表（rcr_corrected_new.csv / .md），--apply 才改 UDF（备份 .orig_20260922_rcr）。
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import numpy as np

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
HERE = Path(__file__).resolve().parent
P = 93.33 * 133
SIDE_RERUN = 0.03
OUT = ("le", "li", "re", "ri")
SIDES = (("le", "li"), ("re", "ri"))


def rcr(s_nei, s_wai, A1):
    r_n, r_w = np.sqrt(s_nei / np.pi) * 1e3, np.sqrt(s_wai / np.pi) * 1e3
    out = {}
    for name, r, s in (("nei", r_n, s_nei), ("wai", r_w, s_wai)):
        d = (2 * r) ** 3 / ((2 * r_n) ** 3 + (2 * r_w) ** 3)
        flow = A1 * d * 0.5
        Rt = P / flow
        R1 = 13.3 / (2 * r) ** 0.3 / s
        out[name] = dict(d=d, Rt=Rt, C=1.79 / Rt, R1=R1, R2=Rt - R1)
    return out


def r_from_R1(R1):
    f = lambda r: 13.3 / ((2 * r) ** 0.3 * np.pi * (r * 1e-3) ** 2) - R1
    lo, hi = 0.1, 50.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return mid


BLOCK = lambda o: re.compile(r"(DEFINE_PROFILE\(pressure_out%s\s*,[^\n]*\n)(.*?)(^\})" % o, re.S | re.M)
LINE = lambda k: re.compile(r"^([ \t]*)%s[ \t]*=[ \t]*([0-9.Ee+-]+)[ \t]*;" % k, re.M)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    from wss_pinn.v4 import new_case_sources as ncs
    import importlib.util
    spec = importlib.util.spec_from_file_location("rpa", HERE / "rcr_protocol_audit_new.py"); rpa = importlib.util.module_from_spec(spec); spec.loader.exec_module(rpa)
    AREA_SCALE = {"AG/fast/PENG_JI_MING": 1e6}   # its .cas is 1000x too small in length (journal rescales at run time); areas from the cas are 1e6x too small
    FULL_RECOMPUTE = {"ILO/XUE_YOU_TANG-0/after"}
    HOLD = {"ILO/YANG_QING_REN-1/after": "left outlets 7.3 / 5.1 mm²: protocol R1 = 13.3/(2r)^0.3/s exceeds R_total → R2 < 0; needs a manual decision"}  # UDF on disk is not the one that ran (thread ids of another case): both sides rewritten, A1 = cohort median
    audit = {r["case"]: r for r in csv.DictReader(open(HERE / "rcr_protocol_audit_new.csv"))}
    flagged = [c for c, r in audit.items() if r["verdict"] != "ok" and not r["verdict"].startswith("ERROR")] + sorted(FULL_RECOMPUTE)
    if args.only is not None:
        flagged = [c for c in flagged if c in args.only]
    for c, why in HOLD.items():
        if c in flagged:
            flagged.remove(c); print(f"HOLD {c}: {why}")
    # cohort-median A1 from the ok units (protocol constant per cohort; used only for FULL_RECOMPUTE)
    A1_COHORT = {}
    for coh in ("AG", "AAA", "ILO"):
        vals = []
        for c, r in audit.items():
            if r["verdict"] != "ok" or c.split("/")[0] != coh: continue
            cur = {o: dict(R1=float(r[f"R1_{o}"]), R2=float(r[f"R2_{o}"])) for o in OUT}; ri = {o: r_from_R1(cur[o]["R1"]) for o in OUT}
            for ext, inte in SIDES:
                dd = (2 * ri[ext]) ** 3 / ((2 * ri[ext]) ** 3 + (2 * ri[inte]) ** 3)
                for o, dv in ((ext, dd), (inte, 1 - dd)): vals.append(2 * P / ((cur[o]["R1"] + cur[o]["R2"]) * dv))
        A1_COHORT[coh] = float(np.median(vals)) if vals else float("nan")
    print("A1 cohort medians (ok units):", {k: round(v, 6) for k, v in A1_COHORT.items()})
    rows = []; md = ["| 病例 | A1 kg/s（四口反推 cv） | 出口 | 网格面积 mm² | 原 R1 / R2 / C | 新 R1 / R2 / C | 该侧髂外份额 原 → 新 |", "|---|---|---|---|---|---|---|"]
    for c in flagged:
        s = {o: v * AREA_SCALE.get(c, 1.0) for o, v in rpa.outlet_areas(c).items()} if c not in FULL_RECOMPUTE else rpa.outlet_areas_by_name(c)
        cur = rpa.udf_params(ROOT / "data_new" / c)
        # A1 from the UDF-implied table of each side (robust: median over four outlets)
        ri = {o: r_from_R1(cur[o]["R1"]) for o in OUT}; vals = []
        for ext, inte in SIDES:
            dd = (2 * ri[ext]) ** 3 / ((2 * ri[ext]) ** 3 + (2 * ri[inte]) ** 3)
            for o, dv in ((ext, dd), (inte, 1 - dd)):
                vals.append(2 * P / ((cur[o]["R1"] + cur[o]["R2"]) * dv))
        A1 = float(np.median(vals)); cv = float(np.std(vals) / np.mean(vals))
        if c in FULL_RECOMPUTE:
            A1 = A1_COHORT[c.split("/")[0]]; cv = float("nan")
        for ext, inte in SIDES:
            G = {o: 1 / (cur[o]["R1"] + cur[o]["R2"]) for o in (ext, inte)}
            share_udf = G[ext] / (G[ext] + G[inte]); share_r3 = s[ext] ** 1.5 / (s[ext] ** 1.5 + s[inte] ** 1.5)
            if abs(share_udf - share_r3) <= SIDE_RERUN and c not in FULL_RECOMPUTE:
                continue
            new = rcr(s[inte], s[ext], A1)
            for o, key in ((ext, "wai"), (inte, "nei")):
                n = new[key]
                rows.append(dict(case=c, outlet=o, A1=f"{A1:.6f}", A1_cv=f"{cv:.2e}", area_m2=f"{s[o]:.6e}", R1_udf=f"{cur[o]['R1']:.4E}", R2_udf=f"{cur[o]['R2']:.4E}", C_udf=f"{cur[o]['C']:.4E}",
                                 R1=f"{n['R1']:.4E}", R2=f"{n['R2']:.4E}", C=f"{n['C']:.4E}", changed=True))
                md.append(f"| `{c}` | {A1:.5f}（cv {cv:.1e}） | {o} | {s[o]*1e6:.1f} | {cur[o]['R1']:.3e} / {cur[o]['R2']:.3e} / {cur[o]['C']:.2e} | {n['R1']:.3e} / {n['R2']:.3e} / {n['C']:.2e} | {share_udf:.2f} → {new['wai']['d']:.2f} |")
    with open(HERE / "rcr_corrected_new.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["case"]); w.writeheader(); w.writerows(rows)
    (HERE / "rcr_corrected_new.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    if not args.apply:
        print(f"\n[dry-run] {len(rows)} 个出口块待改（{len({r['case'] for r in rows})} 例）；--apply 写回 UDF")
        return
    for c in sorted({r["case"] for r in rows}):
        p = ncs.udf_path(ROOT / "data_new" / c); s = open(p, encoding="latin-1", newline="").read(); new = s
        for r in [x for x in rows if x["case"] == c]:
            o = r["outlet"]; m = BLOCK(o).search(new); assert m, (c, o)
            head, body, tail = m.groups(); body2 = body
            for k in ("R1", "R2", "C"):
                assert LINE(k).search(body2), (c, o, k)
                body2 = LINE(k).sub(lambda mm, k=k: f"{mm.group(1)}{k} = {float(r[k]):.4E};  /* 2026-09-22 outlet RCR protocol fix */" if k == "C" else f"{mm.group(1)}{k} = {float(r[k]):.4E};", body2, count=1)
            new = new[: m.start()] + head + body2 + tail + new[m.end():]
        bak = p.with_name(p.name + ".orig_20260922_rcr")
        if not bak.exists():
            shutil.copy2(p, bak)
        open(p, "w", encoding="latin-1", newline="").write(new)
        # re-parse and verify
        text = open(p, encoding="latin-1").read()
        for r in [x for x in rows if x["case"] == c]:
            m = BLOCK(r["outlet"]).search(text); body = m.group(2)
            for k in ("R1", "R2", "C"):
                v = float(LINE(k).search(body).group(2)); assert abs(v / float(r[k]) - 1) < 1e-4, (c, r["outlet"], k, v, r[k])
        print(f"applied: {c} ({p.name}, backup {bak.name})")


if __name__ == "__main__":
    main()
