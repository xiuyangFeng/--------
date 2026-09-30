"""RCR 出口协议审计（新单元版，2026-09-22；只读）。在库 170 例于 09-15 做过同一审计并重算了 26 例；95 个新单元此前没过这道门。

每个出口从 UDF 解析 R1/R2/C，从 .cas 网格算出口面积（按 UDF 线程号 = 压力出口区号），核四件事：
  (1) τ = (R1+R2)·C ≈ 1.79 s（协议常量）
  (2) 每侧髂外/髂内电导份额 G_ext/(G_ext+G_int) vs 网格半径³ 份额（协议 = Murray r³ 按出口面积）；|偏差| > 0.05 标记
  (3) R1 幂律隐含半径 vs 网格半径（用在库 170 例 R1–r 拟合），面积比 ∉ [1/1.5, 1.5] 标记（含"小数点错一位"）
  (4) R_total = 1/ΣG 与队列常量（AG 4.37e5 / AAA·ILO 3.76e5）偏差 > 5 % 标记
用法：python rcr_protocol_audit_new.py [--cases-file new_units_manifest.json 的 new_units] [--workers 16]
输出：rcr_protocol_audit_new.csv + 标记清单。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ("le", "li", "re", "ri")
SIDES = (("le", "li"), ("re", "ri"))
TAU_REF = 1.79
R_TOTAL_REF = {"AG": 4.37e5, "AAA": 3.76e5, "ILO": 3.76e5}
BLOCK = re.compile(r"DEFINE_PROFILE\(pressure_out(le|li|ri|re).*?R1\s*=\s*([0-9.E+-]+);\s*R2\s*=\s*([0-9.E+-]+);\s*C\s*=\s*([0-9.E+-]+);", re.S)


def udf_params(raw_dir: Path) -> dict[str, dict[str, float]]:
    from wss_pinn.v4 import new_case_sources as ncs
    text = ncs.udf_path(raw_dir).read_text(encoding="utf-8", errors="ignore")
    out = {}
    for o, r1, r2, c in BLOCK.findall(text):
        out[o] = dict(R1=float(r1), R2=float(r2), C=float(c))
    if set(out) != set(OUT):
        raise ValueError(f"{raw_dir}: RCR blocks found {sorted(out)}")
    return out


def outlet_areas(cid: str) -> dict[str, float]:
    """m² per outlet label: topology audit when it exists, else read the mesh (pressure-outlet zones via UDF thread ids)."""
    from wss_pinn.v4 import new_case_sources as ncs
    audit = ROOT / "outputs/wss_pinn/volume_uvwp_bc_rcr_v4_anatomy_prep_20260903/audits/topology/cases" / (cid.replace("/", "__") + ".json")
    if audit.is_file():
        rep = json.loads(audit.read_text(encoding="utf-8"))
        if "interfaces" in rep:
            return {row["semantic_label"][4:]: float(row["area_m2"]) for row in rep["interfaces"] if row["semantic_label"].startswith("out-")}
    from wss_pinn.v4.fluent_topology import read_fluent_mesh
    sem = ncs.outlet_semantics(cid)
    mesh = read_fluent_mesh(ncs.fluent_case(cid), keep_interior=False)
    areas = {}
    for s in mesh.face_sections:
        if s.zone_id in sem:
            _, av = mesh.face_geometry(s)
            areas[sem[s.zone_id][4:]] = float(np.linalg.norm(av, axis=1).sum())
    if set(areas) != set(OUT):
        raise ValueError(f"{cid}: outlet zones {sorted(areas)} from UDF ids {sem}")
    return areas


def outlet_areas_by_name(cid: str) -> dict[str, float]:
    """Fallback keyed by Fluent zone names (outlet-leftwai/leftnei/rightwai/rightnei or lw/ln/rw/rn) when the UDF ids do not exist in the case."""
    from wss_pinn.v4 import new_case_sources as ncs
    from wss_pinn.v4.fluent_topology import read_fluent_mesh
    mesh = read_fluent_mesh(ncs.fluent_case(cid), keep_interior=False)
    areas = {}
    for s in mesh.face_sections:
        nm = str(mesh.zone_names.get(s.zone_id, "")).lower()
        if "pressure-outlet" not in nm:
            continue
        side = "l" if "left" in nm or "-l" in nm else "r"
        kind = "i" if ("nei" in nm or nm.endswith("n')") or "-ln" in nm or "-rn" in nm) else "e"
        _, av = mesh.face_geometry(s); areas[side + kind] = float(np.linalg.norm(av, axis=1).sum())
    if set(areas) != set(OUT):
        raise ValueError(f"{cid}: could not name outlets by zone name: {areas}")
    return areas


def one(cid: str) -> dict:
    try:
        p = udf_params(ROOT / "data_new" / cid)
        a = outlet_areas(cid)
        return dict(case=cid, cohort=cid.split("/")[0], params=p, areas=a)
    except Exception as exc:  # noqa: BLE001
        return dict(case=cid, cohort=cid.split("/")[0], error=f"{type(exc).__name__}: {exc}")


def r1_powerlaw() -> dict[str, tuple[float, float]]:
    """log R1 = a + p log r_mm, robust fit on the 170 in-library cases (topology audits + UDFs)."""
    split = json.loads((ROOT / "data_wss_v5/views_v5_1/wss_min_view_v1/split_V5_train136_test34.json").read_text())
    lib = split["train_cases"] + split["test_cases"]
    pts = {"AG": [], "AAA": [], "ILO": []}
    for cid in lib:
        try:
            p = udf_params(ROOT / "data_new" / cid); a = outlet_areas(cid)
        except Exception:
            continue
        for o in OUT:
            pts[cid.split("/")[0]].append((np.log(np.sqrt(a[o] / np.pi) * 1e3), np.log(p[o]["R1"])))
    fit = {}
    for coh, rows in pts.items():
        lr = np.array([x for x, _ in rows]); lR = np.array([y for _, y in rows]); m = np.ones(len(rows), bool)
        for _ in range(5):
            pw, aa = np.polyfit(lr[m], lR[m], 1); res = lR - (aa + pw * lr); m = np.abs(res) < 0.2
        fit[coh] = (float(aa), float(pw)); print(f"R1 幂律 {coh}: log R1 = {aa:.3f} {pw:+.3f} log r_mm（内点 {m.sum()}/{len(rows)}）")
    return fit


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--cases-file", default=str(HERE / "new_units_manifest.json")); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(HERE / "rcr_protocol_audit_new.csv"))
    args = ap.parse_args()
    cases = json.loads(Path(args.cases_file).read_text())["new_units"] if args.cases_file.endswith(".json") else [l.strip() for l in open(args.cases_file) if l.strip()]
    fit = r1_powerlaw()
    with ProcessPoolExecutor(args.workers) as ex:
        recs = list(ex.map(one, cases))
    rows = []; flagged = []
    for r in recs:
        if "error" in r:
            rows.append(dict(case=r["case"], verdict="ERROR " + r["error"][:120])); flagged.append(rows[-1]); continue
        coh = r["cohort"]; p = r["params"]; a = r["areas"]; aa, pw = fit[coh]
        G = {o: 1.0 / (p[o]["R1"] + p[o]["R2"]) for o in OUT}; rt = 1.0 / sum(G.values())
        flags = []
        tau = {o: (p[o]["R1"] + p[o]["R2"]) * p[o]["C"] for o in OUT}
        if any(abs(t - TAU_REF) > 0.05 for t in tau.values()): flags.append("tau=" + "/".join(f"{tau[o]:.2f}" for o in OUT))
        if abs(rt / R_TOTAL_REF[coh] - 1) > 0.05: flags.append(f"R_total={rt:.3g}(ref {R_TOTAL_REF[coh]:.3g})")
        side = {}
        for ext, inte in SIDES:
            g = G[ext] / (G[ext] + G[inte]); r3 = a[ext] ** 1.5 / (a[ext] ** 1.5 + a[inte] ** 1.5)
            side[ext] = (g, r3)
            if abs(g - r3) > 0.05: flags.append(f"{ext}/{inte} share udf {g:.2f} vs mesh r³ {r3:.2f}")
        lr_side = G["le"] + G["li"]; lr = lr_side / sum(G.values())
        if abs(lr - 0.5) > 0.03: flags.append(f"L/R conductance {lr:.2f}/{1-lr:.2f}")
        ratio = {}
        for o in OUT:
            r_mm = np.sqrt(a[o] / np.pi) * 1e3; r_imp = np.exp((np.log(p[o]["R1"]) - aa) / pw); ratio[o] = (r_imp / r_mm) ** 2
            if not (1 / 1.5 <= ratio[o] <= 1.5): flags.append(f"{o} R1-implied area ×{ratio[o]:.2f}")
        row = dict(case=r["case"], cohort=coh, verdict="ok" if not flags else "; ".join(flags), R_total=f"{rt:.4g}",
                   **{f"area_{o}_mm2": f"{a[o]*1e6:.1f}" for o in OUT}, **{f"R1_{o}": f"{p[o]['R1']:.4g}" for o in OUT}, **{f"R2_{o}": f"{p[o]['R2']:.4g}" for o in OUT},
                   **{f"C_{o}": f"{p[o]['C']:.4g}" for o in OUT}, **{f"share_udf_{e}": f"{side[e][0]:.3f}" for e, _ in SIDES}, **{f"share_r3_{e}": f"{side[e][1]:.3f}" for e, _ in SIDES},
                   **{f"areaRatioR1_{o}": f"{ratio[o]:.2f}" for o in OUT})
        rows.append(row)
        if flags: flagged.append(row)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("case", "cohort", "verdict"), k))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} 单元：ok {len(rows)-len(flagged)}，标记 {len(flagged)} → {args.out}")
    for r in flagged:
        print(f"  {r['case']:30s} {r['verdict']}")


if __name__ == "__main__":
    main()
