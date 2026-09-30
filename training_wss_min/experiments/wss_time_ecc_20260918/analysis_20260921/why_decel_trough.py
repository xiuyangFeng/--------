"""为什么减速期 / 谷底难：真值本身的逐帧结构描述量（只读 bundle，不训练），再与 T0 逐帧 r² 对照。
描述量（每例每帧 → 136 例均值 / 中位）：
  c_peak      ln|τ|(t) 与 ln|τ|(峰值帧) 的空间 Pearson —— 场型态相对峰值场变了多少（几何+峰值信息能解释多少）
  c_adj       ln|τ|(t) 与 ln|τ|(t+1) 的空间 Pearson —— 型态随时间变化的速度（1 = 冻结）
  rev_frac    τ⃗(t)·τ⃗(峰值) < 0 的点份额 —— 相对峰值方向反向（回流）的壁面比例
  cos_peak    τ⃗(t) 与 τ⃗(峰值) 的余弦均值
  cos_adj     τ⃗(t) 与 τ⃗(t+1) 的余弦均值 —— 方向场变化速度
  sigma_ln    ln|τ|(t) 的空间标准差 —— 场的对比度（R² 分母）
  cv_lin      Pa 空间 std/mean
  within4mm   ln|τ|(t) 空间方差中 4 mm 截面段内（小尺度）份额 = 1 − Var(段均值)/Var(总)
  c_murray    ln|τ|(t) 与 Murray 先验 log τ0 的 Pearson —— 准稳态几何/分流先验解释力
  c_radius    ln|τ|(t) 与 −ln r 的 Pearson
  frac_lt005 / frac_lt04   |τ| < 0.05 Pa（地板）/ < 0.4 Pa（临床低 WSS）份额
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_ecc_20260918/offline')
from common import *  # noqa

OUTD = Path(__file__).resolve().parent
FLOWREF = ROOT / 'data_wss_v5/views_v5_1/wss_min_flowref_v1'
limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
cases = TRAIN[:limit]
wf = load_waveform(); q = np.asarray(wf['q_norm']); dq = np.asarray(wf['dq_norm'])
phase = {}
with open(OUTD.parent / 'analysis_20260919/frame_r2_source.csv') as f:
    for row in csv.DictReader(f):
        phase[int(row['frame'])] = ('trough' if row['in_trough'] == '1' else 'peak' if row['in_peak'] == '1'
                                    else 'accel' if row['in_systole'] == '1' else 'decel')
KEYS = ['c_peak', 'c_peak_lin', 'c_adj', 'rev_frac', 'cos_peak', 'cos_adj', 'sigma_ln', 'cv_lin', 'mean_pa', 'within4mm',
        'c_murray', 'c_radius', 'frac_lt005', 'frac_lt04']


def rowcorr(A, b):
    """A (F,N) 每行与 b (N,) 的 Pearson。"""
    A = A - A.mean(1, keepdims=True); b = b - b.mean()
    den = np.sqrt((A * A).sum(1) * (b * b).sum()); num = A @ b
    return np.where(den > 1e-12, num / np.maximum(den, 1e-300), np.nan)


def adjcorr(A):
    A = A - A.mean(1, keepdims=True)
    num = (A[:-1] * A[1:]).sum(1); den = np.sqrt((A[:-1] ** 2).sum(1) * (A[1:] ** 2).sum(1))
    out = np.full(A.shape[0], np.nan); out[:-1] = np.where(den > 1e-12, num / np.maximum(den, 1e-300), np.nan); return out


acc = {k: [] for k in KEYS}; t0 = time.time()
for i, cid in enumerate(cases):
    d = load_bundle(cid, vec=True)
    tau = d['tau'].astype(np.float64); vec = d['vec'].astype(np.float64); lg = ln_pa(tau)
    pk = lg[PEAK_INDEX]; vpk = vec[PEAK_INDEX]
    r = {}
    r['c_peak'] = rowcorr(lg, pk); r['c_peak_lin'] = rowcorr(tau, tau[PEAK_INDEX]); r['c_adj'] = adjcorr(lg)
    nrm = np.linalg.norm(vec, axis=2) + 1e-12
    dot = (vec * vpk[None]).sum(2)
    r['rev_frac'] = (dot < 0).mean(1); r['cos_peak'] = (dot / (nrm * nrm[PEAK_INDEX][None])).mean(1)
    ca = np.full(81, np.nan); ca[:-1] = ((vec[:-1] * vec[1:]).sum(2) / (nrm[:-1] * nrm[1:])).mean(1); r['cos_adj'] = ca
    r['sigma_ln'] = lg.std(1); r['cv_lin'] = tau.std(1) / tau.mean(1); r['mean_pa'] = tau.mean(1)
    inv, cnt = bins_of(d)
    bm = np.stack([np.bincount(inv, weights=lg[t], minlength=len(cnt)) / cnt for t in range(81)])  # (81, nbins)
    between = ((bm[:, inv] - lg.mean(1, keepdims=True)) ** 2).mean(1)
    r['within4mm'] = 1 - between / np.maximum(lg.var(1), 1e-12)
    with np.load(FLOWREF / cid / 'features.npz', allow_pickle=False) as z:
        fid = z['wall_node_id_cas']; mur = z['wall_log_tau0_murray'].astype(np.float64)
    if not np.array_equal(fid, d['ids']):
        order = {v: j for j, v in enumerate(fid)}; mur = mur[np.array([order[v] for v in d['ids']])]
    ok = np.isfinite(mur)
    r['c_murray'] = rowcorr(lg[:, ok], mur[ok]); r['c_radius'] = rowcorr(lg, -np.log(np.maximum(d['rad'], 1e-6)))
    r['frac_lt005'] = (tau < 0.05).mean(1); r['frac_lt04'] = (tau < 0.4).mean(1)
    for k in KEYS: acc[k].append(r[k])
    if (i + 1) % 20 == 0: print(f'{i+1}/{len(cases)} {time.time()-t0:.0f}s', flush=True)

A = {k: np.asarray(v) for k, v in acc.items()}   # (n_cases, 81)
mean = {k: np.nanmean(v, 0) for k, v in A.items()}; med = {k: np.nanmedian(v, 0) for k, v in A.items()}; sd = {k: np.nanstd(v, 0) for k, v in A.items()}

# 模型逐帧（三折均值）
model = {}
with open(OUTD / 'frame_metric_suite_best.csv') as f:
    for row in csv.DictReader(f):
        model.setdefault((row['arm'], int(row['step'])), []).append({k: float(row[k]) for k in ('r2cb', 'r2_fit', 'spearman', 'alpha')})
steps = sorted({s for _, s in model}); assert len(steps) == 81
M = {arm: {k: np.array([np.mean([x[k] for x in model[(arm, s)]]) for s in steps]) for k in ('r2cb', 'r2_fit', 'spearman', 'alpha')}
     for arm in ('T0', 'Tnull_Bscale', 'X5D_v51_frozen', 'TB8')}

with open(OUTD / 'why_decel_trough_frame.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['frame', 'step', 'phase', 'q_norm', 'dq_norm'] + [f'{k}_mean' for k in KEYS] + [f'{k}_med' for k in KEYS] + [f'{k}_sd' for k in KEYS]
               + ['T0_r2cb', 'T0_r2fit', 'T0_spearman', 'T0_alpha', 'Tnull_r2cb', 'Tnull_r2fit', 'Tnull_spearman', 'TB8_r2fit'])
    for t in range(81):
        w.writerow([t, steps[t], phase[t], f'{q[t]:.4f}', f'{dq[t]:.4f}'] + [f'{mean[k][t]:.4f}' for k in KEYS] + [f'{med[k][t]:.4f}' for k in KEYS] + [f'{sd[k][t]:.4f}' for k in KEYS]
                   + [f"{M['T0']['r2cb'][t]:.4f}", f"{M['T0']['r2_fit'][t]:.4f}", f"{M['T0']['spearman'][t]:.4f}", f"{M['T0']['alpha'][t]:.4f}",
                      f"{M['Tnull_Bscale']['r2cb'][t]:.4f}", f"{M['Tnull_Bscale']['r2_fit'][t]:.4f}", f"{M['Tnull_Bscale']['spearman'][t]:.4f}", f"{M['TB8']['r2_fit'][t]:.4f}"])

PH = ['accel', 'peak', 'decel', 'trough']
lines = [f'cases={len(cases)}  帧数 accel/peak/decel/trough = ' + '/'.join(str(sum(1 for t in range(81) if phase[t] == p)) for p in PH)]
lines.append('\n== 真值逐帧结构描述量（136 例均值）× 相位 ==')
lines.append(f"{'phase':7s} " + ' '.join(f'{k:>11s}' for k in KEYS) + '   T0_r2fit T0_spear Tnull_r2fit')
for p in PH + ['cycle']:
    sel = [t for t in range(81) if p == 'cycle' or phase[t] == p]
    lines.append(f'{p:7s} ' + ' '.join(f'{np.nanmean(mean[k][sel]):11.3f}' for k in KEYS)
                 + f"   {np.mean(M['T0']['r2_fit'][sel]):8.3f} {np.mean(M['T0']['spearman'][sel]):8.3f} {np.mean(M['Tnull_Bscale']['r2_fit'][sel]):11.3f}")
lines.append('\n== 描述量与模型逐帧 r² 的跨帧相关（81 帧；Pearson / Spearman）==')
def sp(a, b):
    a = np.asarray(a); b = np.asarray(b); ok = np.isfinite(a) & np.isfinite(b); ra = a[ok].argsort().argsort(); rb = b[ok].argsort().argsort()
    return np.corrcoef(ra, rb)[0, 1], np.corrcoef(a[ok], b[ok])[0, 1]
lines.append(f"{'descriptor':12s} {'vs T0 r²  ρ/r':>16s} {'vs T0 Spearman ρ/r':>20s} {'vs Tnull r² ρ/r':>18s} {'vs TB8 r² ρ/r':>16s}")
for k in KEYS + ['q', 'absdq']:
    x = q if k == 'q' else np.abs(dq) if k == 'absdq' else mean[k]
    a = sp(x, M['T0']['r2_fit']); b = sp(x, M['T0']['spearman']); c = sp(x, M['Tnull_Bscale']['r2_fit']); e = sp(x, M['TB8']['r2_fit'])
    lines.append(f'{k:12s} {a[0]:8.3f}/{a[1]:6.3f} {b[0]:11.3f}/{b[1]:6.3f} {c[0]:10.3f}/{c[1]:6.3f} {e[0]:8.3f}/{e[1]:6.3f}')
# 非峰值帧去掉后（避免峰值窗构造性 1.0 主导）
sel = [t for t in range(81) if phase[t] != 'peak']
lines.append('\n-- 同上，但剔除峰值窗 10 帧 --')
for k in ['c_peak', 'c_adj', 'rev_frac', 'sigma_ln', 'within4mm', 'c_murray', 'frac_lt04', 'q', 'absdq']:
    x = (q if k == 'q' else np.abs(dq) if k == 'absdq' else mean[k])[sel]
    a = sp(x, M['T0']['r2_fit'][sel]); b = sp(x, M['T0']['spearman'][sel])
    lines.append(f'{k:12s} vs T0 r² ρ={a[0]:.3f} r={a[1]:.3f} | vs T0 Spearman ρ={b[0]:.3f} r={b[1]:.3f}')

# 滞后（hysteresis）：加速 vs 减速 同流量帧配对
lines.append('\n== 同流量配对：加速帧 vs 减速帧（|Δq| < 0.04）==')
lines.append(f"{'q':>6s} {'acc_f':>5s} {'dec_f':>5s} | {'c_peak a/d':>13s} {'rev a/d':>13s} {'within a/d':>13s} {'c_mur a/d':>13s} | {'T0 r² a/d':>13s} {'T0 ρ a/d':>13s} {'Tnull r² a/d':>13s}")
pairs = []
for i in range(0, PEAK_INDEX):
    if phase[i] != 'accel' or q[i] < 0.15: continue
    js = [j for j in range(PEAK_INDEX + 1, 60) if phase[j] in ('decel', 'trough')]
    j = min(js, key=lambda j: abs(q[j] - q[i]))
    if abs(q[j] - q[i]) > 0.04: continue
    pairs.append((i, j))
    lines.append(f"{q[i]:6.2f} {i:5d} {j:5d} | {mean['c_peak'][i]:6.3f}/{mean['c_peak'][j]:6.3f} {mean['rev_frac'][i]:6.3f}/{mean['rev_frac'][j]:6.3f} "
                 f"{mean['within4mm'][i]:6.3f}/{mean['within4mm'][j]:6.3f} {mean['c_murray'][i]:6.3f}/{mean['c_murray'][j]:6.3f} | "
                 f"{M['T0']['r2_fit'][i]:6.3f}/{M['T0']['r2_fit'][j]:6.3f} {M['T0']['spearman'][i]:6.3f}/{M['T0']['spearman'][j]:6.3f} "
                 f"{M['Tnull_Bscale']['r2_fit'][i]:6.3f}/{M['Tnull_Bscale']['r2_fit'][j]:6.3f}")
if pairs:
    ai = [p[0] for p in pairs]; dj = [p[1] for p in pairs]
    lines.append(f"配对均值：c_peak {np.mean(mean['c_peak'][ai]):.3f}/{np.mean(mean['c_peak'][dj]):.3f}  rev {np.mean(mean['rev_frac'][ai]):.3f}/{np.mean(mean['rev_frac'][dj]):.3f}  "
                 f"T0 r² {np.mean(M['T0']['r2_fit'][ai]):.3f}/{np.mean(M['T0']['r2_fit'][dj]):.3f}  T0 ρ {np.mean(M['T0']['spearman'][ai]):.3f}/{np.mean(M['T0']['spearman'][dj]):.3f}")

# 逐例分散度：谷底 c_peak 的分布 + 与病例峰值 R² 的关系
tr = [t for t in range(81) if phase[t] == 'trough']
cp_case = np.nanmean(A['c_peak'][:, tr], 1); rv_case = np.nanmean(A['rev_frac'][:, tr], 1)
lines.append(f"\n谷底 c_peak 逐例：中位 {np.median(cp_case):.3f}，p10/p90 {np.quantile(cp_case,.1):.3f}/{np.quantile(cp_case,.9):.3f}；谷底反向份额逐例中位 {np.median(rv_case):.3f}，p10/p90 {np.quantile(rv_case,.1):.3f}/{np.quantile(rv_case,.9):.3f}")
lines.append(f"谷底 c_peak 与 反向份额 的逐例相关 r = {np.corrcoef(cp_case, rv_case)[0,1]:.3f}")
# 每帧最典型：c_peak 最低帧、rev 最高帧、c_adj 最低帧
lines.append(f"c_peak 最低帧 {int(np.nanargmin(mean['c_peak']))}（{np.nanmin(mean['c_peak']):.3f}），rev_frac 最高帧 {int(np.nanargmax(mean['rev_frac']))}（{np.nanmax(mean['rev_frac']):.3f}），c_adj 最低帧 {int(np.nanargmin(mean['c_adj']))}（{np.nanmin(mean['c_adj']):.3f}），sigma_ln 最低帧 {int(np.nanargmin(mean['sigma_ln']))}（{np.nanmin(mean['sigma_ln']):.3f}），T0 r² 最低帧 {int(np.argmin(M['T0']['r2_fit']))}")
txt = '\n'.join(lines); print(txt)
(OUTD / 'why_decel_trough.txt').write_text(txt + '\n')
json.dump({'n_cases': len(cases), 'keys': KEYS, 'frame_mean': {k: mean[k].tolist() for k in KEYS}, 'frame_med': {k: med[k].tolist() for k in KEYS},
           'frame_sd': {k: sd[k].tolist() for k in KEYS}, 'q_norm': q.tolist(), 'dq_norm': dq.tolist(), 'phase': [phase[t] for t in range(81)],
           'pairs': pairs, 'trough_case_c_peak': cp_case.tolist(), 'trough_case_rev_frac': rv_case.tolist(), 'cases': cases},
          open(OUTD / 'why_decel_trough.json', 'w'))
print('wrote why_decel_trough.{csv,json,txt}')
