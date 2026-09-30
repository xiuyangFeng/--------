"""运行态 RCR 审计（只读）：从每例最新一次（mtime）的 Fluent transcript 里读 UDF 每步打印的 P_ave_out*/Q_ave_out*，
核对 (1) 末周期左右分流（各出口净流量）是否 50/50、(2) 四个出口末步面压力是否都为正常量级（≈10–16 kPa）、(3) Σ|Q_out| 是否等于入口质量流。
动机：LV_GUO_YOU 重跑后发现 UDF 出口线程号与 cas 挂接的 pressure_out* 函数错位 → 两个出口压力恒为 0、分流 80/20，
而入口/导出/收敛看起来全部正常；这种错误只有从运行态才能抓到。适用于 A 层入库前筛查。

用法：python rcr_runtime_audit.py <病例目录 ...>   或   python rcr_runtime_audit.py --a-layer
输出：标准输出表格 + 本目录 rcr_runtime_audit.csv
"""
from __future__ import annotations

import csv
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/public/newhome/cy/Digital_twin/GNN/data_new")
HERE = Path(__file__).resolve().parent
PAT = re.compile(r"P_ave_out(le|li|ri|re)=(-?[0-9.]+(?:e[-+]?\d+)?) Q_ave_out\1=(-?[0-9.]+(?:e[-+]?\d+)?)")
RHO = 1060.0


def a_layer_dirs() -> list[Path]:
    dirs = sorted(ROOT.glob("ILO/*/after"))
    dirs += [ROOT / f"ILO/{u}/before" for u in (
        "DONG_JIA_JU-1", "LI_BO_JUN-1", "XIONG_DONG_SUO-0", "LI_FA_XIANG-1", "SUN_CHUN_PU-0", "ZHAO_JIAN_PING-0", "GUO_AI_JUN-0",
        "XUE_YOU_TANG-0", "DONG_KE_QIN-0", "WEI_QING_FENG-1", "ZHANG_WAN_ZENG-1", "LIU_CHUN_YANG-1", "LIU_YUN_ZHANG-0", "LI_YU_GANG-0",
        "WANG_LI_MIN-0", "YANG_QING_REN-1", "SHEN_CHUN_WANG-0", "ZHANG_MAO_JIN-0")]
    dirs += [ROOT / f"AG/slow/{u}" for u in ("ZHAO_XIU_XUAN", "LIN_SHU_TIAN", "NIE_QUAN_ZHONG", "WANG_DENG_FENG", "SUN_WEN_QING", "WANG_BAO_SHAN", "ZHANG_HUAN_LI")]
    dirs += [ROOT / "AAA/unruputer/CAO_DIAN_HE", ROOT / "AAA/ruputer/CHEN_FU", ROOT / "AAA/unruputer/GUO_YU_YING", ROOT / "AAA/unruputer/ZHANG_GUI_HUA"]
    return dirs


def transcript(d: Path) -> Path | None:
    cands = [p for p in list(d.glob("Fluent_*.out")) + list((d / "Global_conditions").glob("Fluent_*.out")) if "_bad_" not in p.name]
    # 取最新的一次运行（按 mtime）：在库病例 09-04/09-05 重跑过的目录里还留着旧的失败 transcript，按大小选会选到旧的
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def audit(d: Path) -> dict:
    row = dict(case=str(d.relative_to(ROOT)), transcript="", steps=0, verdict="no_transcript")
    t = transcript(d)
    if t is None:
        return row
    row["transcript"] = t.name
    P = {o: [] for o in ("le", "li", "ri", "re")}; Q = {o: [] for o in P}; last = {}
    with open(t, errors="replace") as fh:
        for line in fh:
            m = PAT.search(line)
            if not m:
                continue
            o = m.group(1); s = line.strip()
            if last.get(o) == s:      # 每步每个 MPI 进程各打一行
                continue
            last[o] = s; P[o].append(float(m.group(2))); Q[o].append(float(m.group(3)))
    n = min(len(v) for v in Q.values())
    row["steps"] = n
    if n < 320:
        row["verdict"] = "short_run"; return row
    cyc = slice(n - 160, n)
    q = {o: abs(float(np.array(Q[o][:n])[cyc].mean())) for o in Q}; tot = sum(q.values())   # 末周期净流量（带回流的出口按净值）
    p_last = {o: np.array(P[o][:n])[cyc].mean() for o in P}
    row.update(q_total_kgs=tot, L=(q["le"] + q["li"]) / tot, R=(q["re"] + q["ri"]) / tot,
               sh_le=q["le"] / tot, sh_li=q["li"] / tot, sh_re=q["re"] / tot, sh_ri=q["ri"] / tot,
               p_le=p_last["le"], p_li=p_last["li"], p_re=p_last["re"], p_ri=p_last["ri"])
    vf = list((d / "Global_conditions").glob("vf-in-rfile*.out")) + list(d.glob("vf-in-rfile*.out"))
    if vf:
        vf = max(vf, key=lambda p: p.stat().st_size)
        try:
            qin = np.loadtxt(vf, skiprows=3)[:, 1]
            row["inlet_mass_kgs"] = float(np.abs(qin[-160:]).mean() * RHO); row["mass_balance"] = tot / row["inlet_mass_kgs"]
        except Exception:
            pass
    flags = []
    if abs(row["L"] - 0.5) > 0.05:
        flags.append(f"LR={row['L']:.2f}/{row['R']:.2f}")
    low = [o for o in p_last if p_last[o] < 3000]
    if low:
        flags.append("P_low:" + ",".join(f"{o}={p_last[o]:.0f}" for o in low))
    if "mass_balance" in row and abs(row["mass_balance"] - 1) > 0.05:
        flags.append(f"mass={row['mass_balance']:.3f}")
    row["verdict"] = "ok" if not flags else ";".join(flags)
    return row


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    dirs = a_layer_dirs() if argv == ["--a-layer"] else [Path(a).resolve() for a in argv]
    rows = [audit(d) for d in dirs]
    keys = ["case", "verdict", "steps", "L", "R", "sh_le", "sh_li", "sh_re", "sh_ri", "p_le", "p_li", "p_re", "p_ri", "mass_balance", "q_total_kgs", "inlet_mass_kgs", "transcript"]
    with open(HERE / "rcr_runtime_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    bad = [r for r in rows if r["verdict"] != "ok"]
    for r in rows:
        if r["verdict"] == "ok":
            continue
        print(f"{r['case']:32s} {r['verdict']}  steps={r['steps']}")
    print(f"\n{len(rows)} cases: ok {len(rows) - len(bad)}, flagged {len(bad)} -> {HERE / 'rcr_runtime_audit.csv'}")


if __name__ == "__main__":
    main()
