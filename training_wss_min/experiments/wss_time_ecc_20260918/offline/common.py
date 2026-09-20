"""阶段 0 公共读取/指标（只读）。数据合同见《WSS_V5_偏心与全周期时间实验矩阵_2026-09-18.md》§2。

- 81 帧标签：v5.1 视图 bundle.npz 的 wall_wss (81,N) / wall_wss_vec (81,N,3)
- 折外预测：train136 = wss_min_cascade_v1 的 wall_log_wss_base（cv3_v51 seed 1234 折外，ln Pa）
              test34 = X5D_v51 五 seed Pa 均值集成
- 目标空间：ln(max(τ, FLOOR) + EPS)，与 dataset.normalize_wss 一致
- R²_cb：复刻 metrics.casebalanced_field_metrics（病例等权，全局中心 = 病例均值的均值）
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
VIEW = ROOT / 'data_wss_v5/views_v5_1/wss_min_view_v1'
CAS = ROOT / 'data_wss_v5/views_v5_1/wss_min_cascade_v1'
H5 = ROOT / 'data_wss_v5/anatomy_pointcloud_v5_1_20260916/cases'
BASE = ROOT / 'training_wss_min/runs/wss_v51_wave1_20260916'
ECC = ROOT / 'training_wss_min/experiments/wss_x5d_longitudinal_20260917/frozen_longitudinal_features'
OUT = Path(__file__).resolve().parent
P = 'eval/ckpt_best/predictions/test'
SEEDS5 = (1234, 7, 2025, 11, 2026)
FLOOR, EPS = 0.05, 1e-6
BIN_MM, MIN_PTS = 4.0, 40
N_FRAMES, PEAK_INDEX, PEAK_STEP = 81, 21, 1162

split = json.loads((VIEW / 'split_V5_train136_test34.json').read_text())
TRAIN, TEST = list(split['train_cases']), list(split['test_cases'])
cv3 = json.loads((VIEW / 'cv3_v51/cv3_summary.json').read_text())
FOLD_OF = {k: int(v) for k, v in cv3['fold_of'].items()}
FOLD_TRAIN = {f: [c for c in TRAIN if FOLD_OF[c] != f] for f in range(3)}
FOLD_HELD = {f: [c for c in TRAIN if FOLD_OF[c] == f] for f in range(3)}


def ln_pa(tau):
    return np.log(np.clip(np.asarray(tau, dtype=np.float64), FLOOR, None) + EPS)


def load_bundle(cid, vec=False):
    with np.load(VIEW / cid / 'bundle.npz', allow_pickle=False) as z:
        steps = z['steps'].astype(np.int64)
        hit = np.flatnonzero(steps == int(z['peak_step']))
        if len(hit) != 1 or int(hit[0]) != PEAK_INDEX or len(steps) != N_FRAMES:
            raise ValueError(f'{cid}: unexpected steps/peak layout')
        d = dict(cid=cid, steps=steps, tau=z['wall_wss'].astype(np.float32),
                 theta=z['wall_theta_rad'].astype(np.float64), seg=z['wall_segment_id'].astype(int),
                 sem=z['wall_semantic_id'].astype(int), s_loc=z['wall_s_local_mm'].astype(np.float64),
                 rho=z['wall_rho'].astype(np.float64), rad=z['wall_local_radius'].astype(np.float64),
                 pos=z['wall_coords_aligned_mm'].astype(np.float64), ids=z['wall_node_id_cas'],
                 s_abs=z['wall_abscissa_mm'].astype(np.float64))
        if vec:
            d['vec'] = z['wall_wss_vec'].astype(np.float32)
    return d


def load_oof_peak_ln(cid, partition, ids=None):
    """折外峰值帧预测（ln Pa）。train → cascade OOF；test → 五 seed Pa 均值。"""
    if partition == 'train':
        with np.load(CAS / cid / 'features.npz', allow_pickle=False) as z:
            cid_ids, base = z['wall_node_id_cas'], z['wall_log_wss_base'].astype(np.float64)
        if ids is not None and not np.array_equal(ids, cid_ids):
            order = {v: i for i, v in enumerate(cid_ids)}
            base = base[np.array([order[v] for v in ids])]
        # cascade 的 ln 口径为 ln(max(τ,1e-6))：统一到 FLOOR 口径
        return ln_pa(np.exp(base))
    preds = []
    for s in SEEDS5:
        with np.load(BASE / f'X5D_v51_s{s}' / P / cid / 'predictions.npz') as z:
            preds.append(z['pred_pa'].astype(np.float64)); rows = z['row_index']
    if ids is not None and not np.array_equal(rows, np.arange(len(ids))):
        raise ValueError(f'{cid}: unexpected prediction row order')
    return ln_pa(np.mean(preds, axis=0))


def load_area(cid, ids):
    import h5py
    with h5py.File(H5 / cid.replace('/', '__') / 'case.h5', 'r') as f:
        valid = f['wall_static/valid'][:]
        hid = f['wall_static/node_id_cas'][:][valid]
        area = f['wall_static/area_m2'][:][valid].astype(np.float64)
    if not np.array_equal(hid, ids):
        order = {v: i for i, v in enumerate(hid)}
        area = area[np.array([order[v] for v in ids])]
    return area


def load_ecc(cid):
    with np.load(ECC / cid / 'features.npz', allow_pickle=False) as z:
        v = z['wall_geom_ref_eccentricity'].astype(np.float64)
        m = z['wall_geom_ref_eccentricity_valid'].astype(bool)
        r = z['wall_geom_ref_roundness'].astype(np.float64)
        rm = z['wall_geom_ref_roundness_valid'].astype(bool)
    return v, m, r, rm


def load_frame_stats(scope):
    """D0 产出：wss_frame_stats_<scope>.json（scope = train136 | fold0 | fold1 | fold2）。"""
    d = json.loads((OUT / f'wss_frame_stats_{scope}.json').read_text())
    return np.asarray(d['frame']['log_mean'], np.float64), np.asarray(d['frame']['log_std'], np.float64), d


def load_waveform():
    return json.loads((OUT / 'protocol_inlet_waveform_v51.json').read_text())


def bins_of(d):
    key = d['seg'].astype(np.int64) * 100000 + np.floor(d['s_loc'] / BIN_MM).astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uk))
    return inv, counts


def r2_score(yt, yp):
    yt = np.asarray(yt, np.float64); yp = np.asarray(yp, np.float64)
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return float('nan') if ss_tot < 1e-12 else 1.0 - float(np.sum((yt - yp) ** 2)) / ss_tot


class CaseBalanced:
    """按 metrics.casebalanced_field_metrics 的口径累计（可对 (81,) 帧向量或标量同时累计）。"""
    def __init__(self):
        self.mean, self.within, self.mse, self.mae = [], [], [], []

    def add(self, yt, yp):
        yt = np.asarray(yt, np.float64); yp = np.asarray(yp, np.float64)   # (..., N)
        m = yt.mean(axis=-1)
        self.mean.append(m); self.within.append(((yt - m[..., None]) ** 2).mean(axis=-1))
        self.mse.append(((yt - yp) ** 2).mean(axis=-1)); self.mae.append(np.abs(yt - yp).mean(axis=-1))

    def r2(self):
        mean = np.asarray(self.mean); within = np.asarray(self.within); mse = np.asarray(self.mse)
        g = mean.mean(axis=0)
        var = within + (mean - g) ** 2
        return 1.0 - mse.mean(axis=0) / var.mean(axis=0)

    def mae_mean(self):
        return np.asarray(self.mae).mean(axis=0)


def m1_fit(theta, y):
    """y ≈ c + a cos θ + b sin θ → (幅值 √(a²+b²), 相位 atan2(b,a), 解释份额)。y 为一维或 (F, n)。"""
    A = np.column_stack([np.ones(len(theta)), np.cos(theta), np.sin(theta)])
    Y = np.atleast_2d(y).T                                   # (n, F)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)            # (3, F)
    amp = np.hypot(coef[1], coef[2]); pha = np.arctan2(coef[2], coef[1])
    resid = Y - A @ coef
    var = Y.var(axis=0); share = np.where(var > 0, 1 - resid.var(axis=0) / np.maximum(var, 1e-300), np.nan)
    if np.ndim(y) == 1:
        return float(amp[0]), float(pha[0]), float(share[0])
    return amp, pha, share


def circ_diff(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


class TimeMetrics:
    """§8 时间口径：逐帧 R²_cb（Pa 与 ln）、峰值帧、谷底四分位（Q 最低 20 帧）、TAWSS、峰时误差、逐点时序相关、峰谷幅值误差。
    帧权 = 均匀（与 evaluate.py 的 tawss/frames 口径一致）。"""
    def __init__(self, q_norm):
        self.frame, self.frame_ln, self.tawss = CaseBalanced(), CaseBalanced(), CaseBalanced()
        self.peak_med, self.peak_p90, self.ts_corr, self.amp_err, self.n_cases = [], [], [], [], 0
        self.trough = np.sort(np.argsort(np.asarray(q_norm))[:20])

    def add(self, ln_true, ln_pred):
        ln_true = np.asarray(ln_true, np.float64); ln_pred = np.asarray(ln_pred, np.float64)
        tt = np.exp(ln_true) - EPS; tp = np.exp(ln_pred) - EPS
        self.frame.add(tt, tp); self.frame_ln.add(ln_true, ln_pred)
        self.tawss.add(tt.mean(axis=0), tp.mean(axis=0))
        d = np.abs(ln_true.argmax(axis=0) - ln_pred.argmax(axis=0)); d = np.minimum(d, N_FRAMES - 1 - d)
        self.peak_med.append(float(np.median(d))); self.peak_p90.append(float(np.quantile(d, .9)))
        a = ln_true - ln_true.mean(axis=0); b = ln_pred - ln_pred.mean(axis=0)
        den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
        ok = den > 1e-12
        self.ts_corr.append(float(np.mean((a * b).sum(axis=0)[ok] / den[ok])) if ok.any() else float('nan'))
        self.amp_err.append(float(np.median(np.abs((ln_pred.max(0) - ln_pred.min(0)) - (ln_true.max(0) - ln_true.min(0))))))
        self.n_cases += 1

    def summary(self):
        r2f, r2l = self.frame.r2(), self.frame_ln.r2()
        return dict(n_cases=self.n_cases,
                    cycle_r2cb_pa=float(r2f.mean()), cycle_r2cb_ln=float(r2l.mean()),
                    peak_r2cb_pa=float(r2f[PEAK_INDEX]), peak_r2cb_ln=float(r2l[PEAK_INDEX]),
                    trough_r2cb_pa=float(r2f[self.trough].mean()), trough_r2cb_ln=float(r2l[self.trough].mean()),
                    min_frame_r2cb_pa=float(r2f.min()), argmin_frame=int(r2f.argmin()),
                    tawss_r2cb=float(self.tawss.r2()), tawss_mae_pa=float(self.tawss.mae_mean()),
                    peak_time_err_med_frames=float(np.mean(self.peak_med)), peak_time_err_p90_frames=float(np.mean(self.peak_p90)),
                    ts_corr_ln=float(np.nanmean(self.ts_corr)), amp_err_ln_med=float(np.mean(self.amp_err)),
                    frame_r2cb_pa=[float(v) for v in r2f], frame_r2cb_ln=[float(v) for v in r2l])


def fmt_summary(s):
    return (f"cycle R²cb Pa {s['cycle_r2cb_pa']:.4f} | ln {s['cycle_r2cb_ln']:.4f} | peak {s['peak_r2cb_pa']:.4f} | "
            f"trough {s['trough_r2cb_pa']:.4f} | min {s['min_frame_r2cb_pa']:.4f}@{s['argmin_frame']} | TAWSS {s['tawss_r2cb']:.4f} "
            f"(MAE {s['tawss_mae_pa']:.3f} Pa) | peak-time med/p90 {s['peak_time_err_med_frames']:.1f}/{s['peak_time_err_p90_frames']:.1f} f | "
            f"ts-corr {s['ts_corr_ln']:.3f} | amp-err {s['amp_err_ln_med']:.3f}")
