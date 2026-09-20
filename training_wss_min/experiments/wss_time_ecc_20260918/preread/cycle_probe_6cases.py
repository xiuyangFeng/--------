"""只读：81 帧壁面 WSS 的时间结构核数（不训练、不写数据）。"""
import numpy as np, h5py, json, sys
from pathlib import Path

ROOT = Path("data_wss_v5/anatomy_pointcloud_v5_1_20260916/cases")
cases = sorted(p.name for p in ROOT.iterdir())
pick = []
for pre in ("AAA", "AG", "ILO"):
    pick += [c for c in cases if c.startswith(pre)][:2]

out = []
for cid in pick:
    with h5py.File(ROOT / cid / "case.h5", "r") as f:
        valid = f["wall_static/valid"][:]
        tau = f["wall_temporal/wss_scalar_pa"][:, valid].astype(np.float64)   # (81, N)
        vec = f["wall_temporal/wss_vector_pa"][:, valid, :].astype(np.float64)
        area = f["wall_static/area_m2"][valid].astype(np.float64)
        q = f["conditions/q_nom_m3s"][:]
        t = f["wall_temporal/time_s"][:]
    T, N = tau.shape
    w = area / area.sum()
    ipeak = int(np.argmax(q))
    # 1) 空间相关：各帧 vs 峰值帧（ln 空间，floor 0.05 Pa）
    L = np.log(np.maximum(tau, 0.05))
    Lc = L - L.mean(axis=1, keepdims=True)
    corr = (Lc @ Lc[ipeak]) / (np.linalg.norm(Lc, axis=1) * np.linalg.norm(Lc[ipeak]) + 1e-12)
    # 2) 秩-1「峰值场 × 共享时间系数」能解释多少（ln 空间，逐例最优 g(t) 与偏置）
    x = Lc[ipeak] / (np.linalg.norm(Lc[ipeak]) + 1e-12)
    g = Lc @ x                                    # 每帧最优系数
    resid = Lc - np.outer(g, x)
    r2_rank1 = 1 - (resid**2).sum() / (Lc**2).sum()
    # 3) 时间 PCA（跨帧，去帧均值后）能量谱
    U, S, Vt = np.linalg.svd(Lc, full_matrices=False)
    energy = np.cumsum(S**2) / (S**2).sum()
    # 4) 逐点峰值时刻的离散度
    tpk = np.argmax(tau, axis=0)
    lag_ms = (t[np.clip(tpk, 0, T-1)] - t[ipeak]) * 1000
    # 5) OSI / TAWSS / RRT（面积加权分布）
    dt = np.diff(t, prepend=t[0] - (t[1]-t[0]))
    dt = np.full(T, (t[-1]-t[0])/(T-1))
    tawss = (tau * dt[:, None]).sum(0) / dt.sum()
    netmag = np.linalg.norm((vec * dt[:, None, None]).sum(0), axis=1) / dt.sum()
    osi = 0.5 * (1 - netmag / np.maximum(tawss, 1e-12))
    out.append(dict(
        case=cid, n_wall=int(N), peak_step_idx=ipeak,
        corr_min=float(corr.min()), corr_min_frame=int(np.argmin(corr)),
        corr_p10=float(np.percentile(corr, 10)), corr_median=float(np.median(corr)),
        rank1_r2_ln=float(r2_rank1),
        pca_k1=float(energy[0]), pca_k2=float(energy[1]), pca_k3=float(energy[2]),
        pca_k5=float(energy[4]), pca_k8=float(energy[7]),
        lag_p05_ms=float(np.percentile(lag_ms, 5)), lag_p50_ms=float(np.percentile(lag_ms, 50)),
        lag_p95_ms=float(np.percentile(lag_ms, 95)),
        frac_peak_within_1frame=float(np.mean(np.abs(tpk - ipeak) <= 1)),
        tawss_over_peak=float((w*tawss).sum() / (w*tau[ipeak]).sum()),
        osi_area_mean=float((w*osi).sum()), osi_frac_gt01=float(w[osi > 0.1].sum()),
        osi_frac_gt02=float(w[osi > 0.2].sum()), osi_frac_gt03=float(w[osi > 0.3].sum()),
        lowtawss_frac_lt04pa=float(w[tawss < 0.4].sum()),
    ))
    print(json.dumps(out[-1], ensure_ascii=False))
print("\n=== 汇总（6 例均值）===")
keys = [k for k in out[0] if k not in ("case", "n_wall", "peak_step_idx", "corr_min_frame")]
for k in keys:
    print(f"{k:28s} {np.mean([o[k] for o in out]):.4f}   [{min(o[k] for o in out):.4f}, {max(o[k] for o in out):.4f}]")
