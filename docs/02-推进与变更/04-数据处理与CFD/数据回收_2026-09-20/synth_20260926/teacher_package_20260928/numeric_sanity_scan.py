"""交付包数值合理性全扫描（59 例 × 81 帧，只读）→ <包>/verification/numeric_sanity_59.csv。

逐例：①全部帧全部量是否有限、WSS 标量是否 ≥ 0；②值域（压力、速度、WSS 的周期内极值与分位数）；
③逐帧质量守恒（入口流量 + 四出口流量 ≈ 0，按峰值入流归一）；④周期收敛：第 1 帧（t = 5.6 s）与第 81 帧（t = 6.4 s）相隔正好一个周期，
二者的相对差反映是否已进入周期稳态；⑤与母例对照：形变没有移动的壁面节点上，峰值帧 WSS 与 TAWSS 和母例的相对差（母例取入库 bundle）。
"""
import csv
import json
import re
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/public/newhome/cy/Digital_twin/GNN")
PKG = ROOT / "outputs/synth59_teacher_package_2026-09-28"
V52 = ROOT / "data_wss_v5/anatomy_pointcloud_v5_2_20260922/cases"
V51 = ROOT / "data_wss_v5/anatomy_pointcloud_v5_1_20260916/cases"


def bundle(cid):
    for r in (V52, V51):
        p = r / cid.replace("/", "__") / "case.h5"
        if p.is_file():
            return p
    raise FileNotFoundError(cid)


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def scan(cid):
    h5 = next((PKG / cid).glob("*.h5")); r = {"case_id": cid}
    with h5py.File(h5, "r") as f:
        k = int(f["time"].attrs["peak_index"]); nf = f["time/step"].shape[0]
        nonfinite = 0
        wmax = pmin_w = pmax_w = pmin_v = pmax_v = vmax = None
        wss_neg = 0
        tawss = np.zeros(f["wall/xyz_mm"].shape[0])
        for t in range(nf):
            w = f["wall/wss_scalar_pa"][t]; wv = f["wall/wss_vector_pa"][t]; pw = f["wall/pressure_pa"][t]
            pv = f["volume/pressure_pa"][t]; u = f["volume/velocity_m_s"][t]
            nonfinite += int((~np.isfinite(w)).sum() + (~np.isfinite(wv)).sum() + (~np.isfinite(pw)).sum() + (~np.isfinite(pv)).sum() + (~np.isfinite(u)).sum())
            wss_neg += int((w < 0).sum())
            um = np.linalg.norm(u, axis=1)
            wmax = max(wmax or 0, float(w.max())); vmax = max(vmax or 0, float(um.max()))
            pmin_w = min(pmin_w if pmin_w is not None else 1e30, float(pw.min())); pmax_w = max(pmax_w if pmax_w is not None else -1e30, float(pw.max()))
            pmin_v = min(pmin_v if pmin_v is not None else 1e30, float(pv.min())); pmax_v = max(pmax_v if pmax_v is not None else -1e30, float(pv.max()))
            if t < nf - 1:
                tawss += w / (nf - 1)
            if t == 0:
                w0, p0, u0 = w.copy(), pv.copy(), u.copy()
            if t == nf - 1:
                wN, pN, uN = w, pv, u
            if t == k:
                wk = w.copy(); vk99 = float(np.percentile(um, 99)); pvk_mean = float(pv.mean())
        vol = f["volume/volume_m3"][...]
        r.update({"nonfinite_values": nonfinite, "wss_negative_values": wss_neg,
                  "wss_peak_p50_pa": float(np.percentile(wk, 50)), "wss_peak_p99_pa": float(np.percentile(wk, 99)), "wss_cycle_max_pa": wmax,
                  "tawss_median_pa": float(np.median(tawss)), "tawss_p99_pa": float(np.percentile(tawss, 99)),
                  "velocity_cycle_max_m_s": vmax, "velocity_peak_p99_m_s": vk99,
                  "p_wall_min_kpa": pmin_w / 1000, "p_wall_max_kpa": pmax_w / 1000, "p_volume_min_kpa": pmin_v / 1000, "p_volume_max_kpa": pmax_v / 1000,
                  "p_volume_peak_mean_kpa": pvk_mean / 1000})
        # 周期收敛：t = 5.6 s 与 6.4 s（同相位、相隔一个周期）
        r.update({"cycle_rel_diff_wss": rel_l2(wN, w0), "cycle_rel_diff_velocity": rel_l2(uN, u0),
                  "cycle_rel_diff_pressure": rel_l2(pN - np.average(pN, weights=vol), p0 - np.average(p0, weights=vol)),
                  "cycle_abs_diff_pressure_mean_pa": float(abs(np.average(pN, weights=vol) - np.average(p0, weights=vol)))})
        # 逐帧质量守恒（解剖区 5 个开口）
        q = {lab: f[f"boundary/{lab}/flux_outward_m3s"][...] for lab in ("inlet", "out-le", "out-li", "out-re", "out-ri")}
        imb = sum(q.values()); qpk = float(np.abs(q["inlet"]).max())
        r["mass_imbalance_max_rel_to_peak"] = float(np.abs(imb).max() / qpk)
        r["mass_imbalance_cycle_rel"] = float(abs(imb[:-1].mean()) / abs(q["inlet"][:-1].mean()))
        xyz = f["wall/xyz_mm"][...]
    # 与母例：形变没移动的壁面节点
    with h5py.File(bundle(re.sub(r"~m\d+", "", cid)), "r") as p:
        xp = p["wall_static/xyz_mm"][...]; steps = list(p["wall_temporal/step"][...]); kp = steps.index(1162)
        wp_all = p["wall_temporal/wss_scalar_pa"]
        wpk = wp_all[kp]; tp = np.zeros(len(xp))
        for t in range(len(steps) - 1):
            tp += wp_all[t] / (len(steps) - 1)
    still = np.linalg.norm(xyz - xp, axis=1) < 1e-6
    r["wall_nodes_unmoved_frac"] = float(still.mean())
    m = still & (wpk > 0.1)
    r["unmoved_wss_peak_median_rel_diff_vs_parent"] = float(np.median(np.abs(wk[m] - wpk[m]) / wpk[m]))
    r["unmoved_wss_peak_p90_rel_diff_vs_parent"] = float(np.percentile(np.abs(wk[m] - wpk[m]) / wpk[m], 90))
    mt = still & (tp > 0.05)
    r["unmoved_tawss_median_rel_diff_vs_parent"] = float(np.median(np.abs(tawss[mt] - tp[mt]) / tp[mt]))
    return r


def main():
    ids = [r["case_id"] for r in csv.DictReader(open(PKG / "manifest.csv"))]
    with Pool(8) as pool:
        res = pool.map(scan, ids, chunksize=1)
    keys = list(res[0].keys())
    with open(PKG / "verification/numeric_sanity_59.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(res)
    arr = lambda k: np.array([r[k] for r in res], float)  # noqa: E731
    out = {}
    for k in keys[1:]:
        a = arr(k); out[k] = [round(float(a.min()), 5), round(float(np.median(a)), 5), round(float(a.max()), 5)]
    print(json.dumps(out, ensure_ascii=False, indent=0))
    for kk in ("cycle_rel_diff_wss", "cycle_rel_diff_velocity", "mass_imbalance_max_rel_to_peak", "velocity_cycle_max_m_s", "p_volume_min_kpa", "unmoved_wss_peak_median_rel_diff_vs_parent"):
        top = sorted(res, key=lambda r: -abs(r[kk]) if kk != "p_volume_min_kpa" else r[kk])[:4]
        print(kk, [(t["case_id"], round(t[kk], 4)) for t in top])


if __name__ == "__main__":
    main()
