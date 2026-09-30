"""母例同口径对照：把 numeric_sanity_scan.py 的周期收敛、开口质量守恒、流速极值、压力值域在 59 个母例（入库 bundle）上重算，
逐例与子例并列 → <包>/verification/numeric_child_vs_parent_59.csv。只读。"""
import csv, re
from multiprocessing import Pool
from pathlib import Path
import h5py, numpy as np
ROOT = Path("/public/newhome/cy/Digital_twin/GNN"); PKG = ROOT / "outputs/synth59_teacher_package_2026-09-28"
V52 = ROOT / "data_wss_v5/anatomy_pointcloud_v5_2_20260922/cases"; V51 = ROOT / "data_wss_v5/anatomy_pointcloud_v5_1_20260916/cases"
def bundle(cid):
    for r in (V52, V51):
        p = r / cid.replace("/", "__") / "case.h5"
        if p.is_file(): return p
def rel(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
def one(cid):
    p = re.sub(r"~m\d+", "", cid)
    with h5py.File(bundle(p), "r") as f:
        W = f["wall_temporal/wss_scalar_pa"]; U = f["volume_temporal/velocity_m_s"]; P = f["volume_temporal/pressure_pa"]; vol = f["volume_static/volume_m3"][...]
        n = W.shape[0]; vmax = max(float(np.linalg.norm(U[t], axis=1).max()) for t in range(n))
        pmin = min(float(P[t].min()) for t in range(n)) / 1000; pmax = max(float(P[t].max()) for t in range(n)) / 1000
        p0, pN = P[0], P[n - 1]
        q = {lab: f[f"interfaces/{lab}/flux_outward_m3s"][...] for lab in ("inlet", "out-le", "out-li", "out-re", "out-ri")}
        imb = sum(q.values())
        return {"case_id": cid, "parent_id": p,
                "parent_cycle_rel_diff_wss": rel(W[n - 1], W[0]), "parent_cycle_rel_diff_velocity": rel(U[n - 1], U[0]),
                "parent_cycle_abs_diff_pressure_mean_pa": float(abs(np.average(pN, weights=vol) - np.average(p0, weights=vol))),
                "parent_mass_imbalance_max_rel_to_peak": float(np.abs(imb).max() / np.abs(q["inlet"]).max()),
                "parent_velocity_cycle_max_m_s": vmax, "parent_p_volume_min_kpa": pmin, "parent_p_volume_max_kpa": pmax}
ids = [r["case_id"] for r in csv.DictReader(open(PKG / "manifest.csv"))]
child = {r["case_id"]: r for r in csv.DictReader(open(PKG / "verification/numeric_sanity_59.csv"))}
with Pool(8) as pool: res = pool.map(one, ids, chunksize=1)
keys = ["case_id", "parent_id"]; pairs = [("cycle_rel_diff_wss",) * 2, ("cycle_rel_diff_velocity",) * 2, ("cycle_abs_diff_pressure_mean_pa",) * 2,
                                          ("mass_imbalance_max_rel_to_peak",) * 2, ("velocity_cycle_max_m_s",) * 2, ("p_volume_min_kpa",) * 2, ("p_volume_max_kpa",) * 2]
rows = []
for r in res:
    row = {"case_id": r["case_id"], "parent_id": r["parent_id"]}
    for k, _ in pairs:
        row[f"child_{k}"] = float(child[r["case_id"]][k]); row[f"parent_{k}"] = r[f"parent_{k}"]
    rows.append(row)
with open(PKG / "verification/numeric_child_vs_parent_59.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for k, _ in pairs:
    c = np.array([r[f"child_{k}"] for r in rows]); pa = np.array([r[f"parent_{k}"] for r in rows])
    print(f"{k:34s} child med {np.median(c):.4g} p90 {np.percentile(c, 90):.4g} max {c.max():.4g} | parent med {np.median(pa):.4g} p90 {np.percentile(pa, 90):.4g} max {pa.max():.4g} | corr {np.corrcoef(c, pa)[0, 1]:.2f}")
big = sorted(rows, key=lambda r: -r["child_cycle_rel_diff_wss"])[:8]
for r in big: print(r["case_id"][:34].ljust(34), "WSS 周期差 子 %.3f 母 %.3f | 速度 子 %.3f 母 %.3f | 质量 子 %.3f 母 %.3f | vmax 子 %.2f 母 %.2f" % (
    r["child_cycle_rel_diff_wss"], r["parent_cycle_rel_diff_wss"], r["child_cycle_rel_diff_velocity"], r["parent_cycle_rel_diff_velocity"],
    r["child_mass_imbalance_max_rel_to_peak"], r["parent_mass_imbalance_max_rel_to_peak"], r["child_velocity_cycle_max_m_s"], r["parent_velocity_cycle_max_m_s"]))
fast = sorted(rows, key=lambda r: -r["child_velocity_cycle_max_m_s"])[:5]
for r in fast: print("vmax", r["case_id"][:34].ljust(34), "子 %.2f 母 %.2f m/s" % (r["child_velocity_cycle_max_m_s"], r["parent_velocity_cycle_max_m_s"]))
