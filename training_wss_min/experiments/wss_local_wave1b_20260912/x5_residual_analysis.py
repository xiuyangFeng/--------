"""X5 (three seeds) residual analysis on test34: ensemble gain, residual decomposition, oracle affine bound,
branch-mean residual vs Murray-vs-true split, and geometry->split predictability (deployment-legal features)."""
import glob, json, os, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import evaluate as E
M = E.M
ROOT = Path('/public/newhome/cy/Digital_twin/GNN')
RUNS = ROOT / 'training_wss_min/runs'
SEEDS = {'1234': RUNS / 'wss_local_wave1_20260912/X5_s1234', '7': RUNS / 'wss_local_wave1b_20260912/X5_s7', '2025': RUNS / 'wss_local_wave1b_20260912/X5_s2025'}
X0 = RUNS / 'wss_local_wave1_20260912/X0_s1234'
VIEW = ROOT / 'data_wss_v5/views/wss_min_view_v1'; FLOW = ROOT / 'data_wss_v5/views/wss_min_flowref_v1'
H5 = ROOT / 'data_wss_v5/anatomy_pointcloud_v5_20260906/cases'
units = sorted(str(p.parent.relative_to(SEEDS['1234'] / 'eval/ckpt_best/predictions/test')) for p in (SEEDS['1234'] / 'eval/ckpt_best/predictions/test').glob('*/*/*/predictions.npz'))
assert len(units) == 34
def load(run, unit):
    with np.load(run / 'eval/ckpt_best/predictions/test' / unit / 'predictions.npz') as z:
        return {k: z[k].astype(np.float64) for k in ('pred_norm', 'true_norm', 'pred_pa', 'true_pa')}
# ---- log_z -> Pa transform from one file (exact affine in log space)
d = load(SEEDS['1234'], units[0]); A = np.polyfit(d['pred_norm'], np.log(d['pred_pa']), 1); std, mean = A[0], A[1]
def to_pa(norm): return np.exp(norm * std + mean)
def cb(true_list, pred_list): return M.casebalanced_field_metrics(true_list, pred_list)['r2']
def tail(true_list, pred_list):
    t = np.concatenate(true_list); p = np.concatenate(pred_list); q = np.quantile(t, 0.9); m = t >= q
    return dict(top10_ratio=float(p[m].mean() / t[m].mean()), high_r2=float(1 - ((t[m] - p[m]) ** 2).sum() / ((t[m] - t[m].mean()) ** 2).sum()),
                p99_ratio=float(np.quantile(p, 0.99) / np.quantile(t, 0.99)))
data = {s: {u: load(r, u) for u in units} for s, r in SEEDS.items()}
truth_pa = [data['1234'][u]['true_pa'] for u in units]; truth_n = [data['1234'][u]['true_norm'] for u in units]
print('=== single seeds vs 3-seed ensemble (log-space mean), physical R2_cb / normalized R2_cb / tail')
for s in SEEDS:
    pp = [data[s][u]['pred_pa'] for u in units]; pn = [data[s][u]['pred_norm'] for u in units]
    print(f'  X5_s{s:>4}: Pa {cb(truth_pa, pp):.4f}  norm {cb(truth_n, pn):.4f}  {tail(truth_pa, pp)}')
ens_n = [np.mean([data[s][u]['pred_norm'] for s in SEEDS], axis=0) for u in units]; ens_pa = [to_pa(x) for x in ens_n]
print(f'  ensemble : Pa {cb(truth_pa, ens_pa):.4f}  norm {cb(truth_n, ens_n):.4f}  {tail(truth_pa, ens_pa)}')
ens_pa_mean = [np.mean([data[s][u]['pred_pa'] for s in SEEDS], axis=0) for u in units]
print(f'  ensemble (Pa-space mean): Pa {cb(truth_pa, ens_pa_mean):.4f}')
# ---- per-case R2 spread (ensemble, physical)
pc = [1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum() for t, p in zip(truth_pa, ens_pa)]
order = np.argsort(pc)
print('  per-case physical R2: median %.3f P10 %.3f min %.3f; worst 6:' % (np.median(pc), np.quantile(pc, 0.1), min(pc)), [(units[i].split('/')[0] + '/' + units[i].split('/')[-1], round(pc[i], 3)) for i in order[:6]])
# ---- oracle per-case affine / scale bound on the ensemble (log space fit -> physical R2)
aff_pa, scl_pa = [], []
for t_n, p_n in zip(truth_n, ens_n):
    a, b = np.polyfit(p_n, t_n, 1); aff_pa.append(to_pa(a * p_n + b)); scl_pa.append(to_pa(p_n + (t_n.mean() - p_n.mean())))
print(f'  oracle per-case affine (log) -> Pa R2_cb {cb(truth_pa, aff_pa):.4f}; per-case shift only -> {cb(truth_pa, scl_pa):.4f}  (ensemble {cb(truth_pa, ens_pa):.4f})')
# ---- residual decomposition on the ensemble (log_z residual)
BIN = 4.0
shares = {'section': [], 'within': [], 'branch': []}; branch_rows = []
peak_true, peak_murray = {}, {}
for u, t_n, p_n in zip(units, truth_n, ens_n):
    with np.load(VIEW / u / 'bundle.npz', allow_pickle=True) as b:
        seg = b['wall_segment_id'].astype(int); s_local = b['wall_s_local_mm'].astype(float)
    with np.load(FLOW / u / 'features.npz') as f:
        logq = f['wall_log_q_branch_murray'].astype(float)
    assert len(seg) == len(t_n)
    r = t_n - p_n; var = r.var()
    cell = seg * 100000 + np.floor(s_local / BIN).astype(int); _, inv, cnt = np.unique(cell, return_inverse=True, return_counts=True)
    between = (np.bincount(inv, weights=r) / cnt)[inv]
    shares['section'].append(between.var() / var); shares['within'].append((r - between).var() / var)
    bm = {int(s): r[seg == s].mean() for s in np.unique(seg)}
    shares['branch'].append(np.var([bm[int(s)] for s in seg]) / var)
    # true split at peak vs Murray share per leaf segment
    cid = u.replace('/', '__')
    import h5py
    with h5py.File(H5 / cid / 'case.h5', 'r') as h:
        step = h['wall_temporal/step'][:]; k = int(np.where(step == 1162)[0][0])
        q = {o: float(h[f'interfaces/{o}/flux_outward_m3s'][k]) for o in ('out-le', 'out-li', 'out-re', 'out-ri')}
        segs = json.loads(h['geometry'].attrs['atlas_segments'])
    tot = sum(q.values())
    for sg in segs:
        sid = sg['segment_id']
        if sg['ends_at_leaf'] and sg['outlet_name'] in q and sid in bm:
            true_share = q[sg['outlet_name']] / tot; murray_share = float(np.exp(logq[seg == sid][0]))
            branch_rows.append((u, sid, sg['outlet_name'], bm[sid], np.log(true_share) - np.log(murray_share), np.log(true_share), np.log(murray_share)))
        elif (not sg['ends_at_leaf']) and sg['parent_id'] == 0 and sid in bm:   # CIA: true share = sum of its leaves
            leaves = [x for x in segs if x['parent_id'] == sid]
            true_share = sum(q[x['outlet_name']] for x in leaves) / tot; murray_share = float(np.exp(logq[seg == sid][0]))
            branch_rows.append((u, sid, 'CIA', bm[sid], np.log(true_share) - np.log(murray_share), np.log(true_share), np.log(murray_share)))
print('=== ensemble residual decomposition (median [p25,p75] of variance shares): section-mean %.3f [%.3f,%.3f], within-section %.3f, branch-mean %.3f [%.3f,%.3f]' % (
    np.median(shares['section']), np.quantile(shares['section'], .25), np.quantile(shares['section'], .75), np.median(shares['within']), np.median(shares['branch']), np.quantile(shares['branch'], .25), np.quantile(shares['branch'], .75)))
br = np.array([(x[3], x[4]) for x in branch_rows]); leaf = np.array([x[2] != 'CIA' for x in branch_rows])
ok = np.isfinite(br).all(1); print('  branch rows dropped (non-finite):', int((~ok).sum())); br, leaf = br[ok], leaf[ok]
c_all = np.corrcoef(br[:, 0], br[:, 1])[0, 1]; c_leaf = np.corrcoef(br[leaf, 0], br[leaf, 1])[0, 1]
slope = np.polyfit(br[leaf, 1], br[leaf, 0], 1)[0]
print(f'=== branch-mean residual (log_z) vs log(true share / Murray share): corr all {c_all:+.3f} (n={len(br)}), leaves {c_leaf:+.3f} (n={leaf.sum()}), slope {slope:+.3f}; '
      f'sd of branch-mean residual {br[:,0].std():.3f}, sd of log split error {br[:,1].std():.3f}; explained share of branch-mean residual var (leaves) {c_leaf**2:.2f}')
# ---- geometry -> split predictability on all 172 cases (deployment-legal features), sides pooled
rows = []
for cdir in sorted(H5.glob('*')):
    with h5py.File(cdir / 'case.h5', 'r') as h:
        step = h['wall_temporal/step'][:]; k = int(np.where(step == 1162)[0][0]) if 1162 in step else int(np.argmax(np.abs(h['interfaces/inlet/flux_outward_m3s'][:])))
        q = {o: float(h[f'interfaces/{o}/flux_outward_m3s'][k]) for o in ('out-le', 'out-li', 'out-re', 'out-ri')}
        segs = json.loads(h['geometry'].attrs['atlas_segments']); caps = {c['label']: c for c in json.loads(h['geometry'].attrs['caps_detail'])}
        table = h['geometry/atlas_table'][:]; cols = json.loads(h['geometry'].attrs['atlas_columns']); role = h.attrs['role']; cohort = h.attrs['cohort']
        ai = {o: h[f'interfaces/{o}'].attrs['area_m2'] for o in q}
    r_col, sid_col, kap = cols.index('radius_mm'), cols.index('segment_id'), cols.index('curvature_per_mm')
    by_outlet = {s['outlet_name']: s for s in segs if s['ends_at_leaf']}
    def leaf_r(o): s = by_outlet[o]; rr = table[table[:, sid_col] == s['segment_id'], r_col]; return float(np.median(rr[-20:]))
    def leaf_len(o): return by_outlet[o]['length_mm']
    def leaf_kap(o): return by_outlet[o]['curvature_max_per_mm']
    for ext, inte in (('out-le', 'out-li'), ('out-re', 'out-ri')):
        if ext not in by_outlet or inte not in by_outlet or q[ext] <= 0 or q[inte] <= 0: continue
        rows.append(dict(role=str(role), cohort=str(cohort), y=np.log(q[ext] / q[inte]),
                         murray=3 * np.log(leaf_r(ext) / leaf_r(inte)), m233=2.33 * np.log(leaf_r(ext) / leaf_r(inte)),
                         cap=2 * np.log(caps[ext]['radius_mm'] / caps[inte]['radius_mm']), area=np.log(ai[ext] / ai[inte]),
                         f=[np.log(leaf_r(ext) / leaf_r(inte)), np.log(caps[ext]['radius_mm'] / caps[inte]['radius_mm']), np.log(ai[ext] / ai[inte]),
                            np.log(leaf_len(ext) / leaf_len(inte)), leaf_kap(ext) - leaf_kap(inte), np.log(leaf_r(ext)), np.log(leaf_r(inte))]))
rows = [r for r in rows if np.isfinite(r['y']) and np.isfinite(r['f']).all() and np.isfinite([r['murray'], r['m233'], r['cap'], r['area']]).all()]
Y = np.array([r['y'] for r in rows]); F = np.array([r['f'] for r in rows]); tr = np.array([r['role'] == 'train' for r in rows]); te = ~tr
def r2(y, yhat): return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f'=== geometry -> log(Q_ext/Q_int) at peak: n sides {len(rows)} (train {tr.sum()}, test {te.sum()}); sd of target {Y.std():.3f}; mean {Y.mean():+.3f}')
for name in ('murray', 'm233', 'cap', 'area'):
    v = np.array([r[name] for r in rows]); a, b = np.polyfit(v[tr], Y[tr], 1)
    print(f'  baseline {name:6s}: raw R2 all {r2(Y, v):+.3f} | corr {np.corrcoef(v, Y)[0,1]:+.3f} | linear recalibrated (fit train) test R2 {r2(Y[te], a * v[te] + b):+.3f}')
# ridge with LOO on train, then test
Xf = np.column_stack([F, np.ones(len(F))]); lam = 1.0
def ridge(Xa, ya): return np.linalg.solve(Xa.T @ Xa + lam * np.eye(Xa.shape[1]), Xa.T @ ya)
w = ridge(Xf[tr], Y[tr]); print(f'  ridge(7 geometry feats) train-fit R2 {r2(Y[tr], Xf[tr] @ w):+.3f}, test R2 {r2(Y[te], Xf[te] @ w):+.3f}')
loo = np.array([ (Xf[i] @ ridge(np.delete(Xf[tr], j, 0), np.delete(Y[tr], j))) for j, i in enumerate(np.where(tr)[0])])
print(f'  ridge LOO on train R2 {r2(Y[tr], loo):+.3f}; per cohort test R2: ' + ', '.join(f"{c} {r2(Y[te & np.array([r['cohort']==c for r in rows])], (Xf @ w)[te & np.array([r['cohort']==c for r in rows])]):+.3f}" for c in ('AG', 'AAA', 'ILO')))

# ---- per-cohort ensemble R2 and section/within decomposition of *explained* variance
coh = [u.split('/')[0] for u in units]
for c in ('AG', 'AAA', 'ILO'):
    idx = [i for i, x in enumerate(coh) if x == c]
    print(f'  ensemble per-cohort Pa R2_cb {c}: {cb([truth_pa[i] for i in idx], [ens_pa[i] for i in idx]):.4f} (n={len(idx)})  | X5_s1234: {cb([truth_pa[i] for i in idx], [data["1234"][units[i]]["pred_pa"] for i in idx]):.4f}')
sec_r2, within_r2, dec = [], [], np.zeros((34, 10))
for i, (u, t_n, p_n) in enumerate(zip(units, truth_n, ens_n)):
    with np.load(VIEW / u / 'bundle.npz', allow_pickle=True) as b:
        seg = b['wall_segment_id'].astype(int); s_local = b['wall_s_local_mm'].astype(float)
    cell = seg * 100000 + np.floor(s_local / BIN).astype(int); _, inv, cnt = np.unique(cell, return_inverse=True, return_counts=True)
    tm = (np.bincount(inv, weights=t_n) / cnt)[inv]; pm = (np.bincount(inv, weights=p_n) / cnt)[inv]
    sec_r2.append(1 - ((tm - pm) ** 2).sum() / ((tm - tm.mean()) ** 2).sum())
    tw, pw = t_n - tm, p_n - pm
    within_r2.append(1 - ((tw - pw) ** 2).sum() / ((tw - tw.mean()) ** 2).sum())
    q = np.quantile(t_n, np.linspace(0, 1, 11)); which = np.clip(np.searchsorted(q, t_n, side='right') - 1, 0, 9)
    dec[i] = [np.mean((p_n - t_n)[which == k]) if np.any(which == k) else np.nan for k in range(10)]
print('=== explained variance by scale (normalized, per-case median): section-mean track R2 %.3f, within-section pattern R2 %.3f' % (np.median(sec_r2), np.median(within_r2)))
print('=== mean residual (pred - true, log_z units) by true decile, median over cases:', np.round(np.nanmedian(dec, axis=0), 3).tolist())
# global (test-fitted) monotone/affine recalibration bound on the ensemble
allp = np.concatenate(ens_n); allt = np.concatenate(truth_n); a, b = np.polyfit(allp, allt, 1)
print(f'  global affine recal (oracle, fit on test): a={a:.3f} b={b:+.3f} -> Pa R2_cb {cb(truth_pa, [to_pa(a * x + b) for x in ens_n]):.4f} (ensemble {cb(truth_pa, ens_pa):.4f})')
# ---- per-case oracle slope/shift vs deployable proxies (ensemble spread, Murray-vs-area split disagreement, cohort)
import collections
split_err = collections.defaultdict(list)
for x in branch_rows:
    if np.isfinite(x[4]) and x[2] != 'CIA': split_err[x[0]].append(abs(x[4]))
slopes, shifts, spreads, splits, cohs, levels, pcR2 = [], [], [], [], [], [], []
for i, (u, t_n, p_n) in enumerate(zip(units, truth_n, ens_n)):
    a, b = np.polyfit(p_n, t_n, 1); slopes.append(a); shifts.append(t_n.mean() - p_n.mean())
    spreads.append(float(np.mean(np.std([data[s][u]['pred_norm'] for s in SEEDS], axis=0))))
    splits.append(float(np.mean(split_err[u])) if split_err[u] else np.nan); cohs.append(u.split('/')[0]); levels.append(float(p_n.mean())); pcR2.append(pc[i])
slopes, shifts, spreads, splits, levels, pcR2 = map(np.array, (slopes, shifts, spreads, splits, levels, pcR2))
def c(x, y): m = np.isfinite(x) & np.isfinite(y); return np.corrcoef(x[m], y[m])[0, 1]
print('=== per-case oracle calibration vs proxies: slope mean %.3f sd %.3f (range %.2f–%.2f); shift sd %.3f' % (slopes.mean(), slopes.std(), slopes.min(), slopes.max(), shifts.std()))
print('    corr(slope, ensemble spread) %+.2f | corr(|shift|, spread) %+.2f | corr(shift, mean |split err|) %+.2f | corr(case R2, spread) %+.2f | corr(case R2, |split err|) %+.2f' % (
    c(slopes, spreads), c(np.abs(shifts), spreads), c(shifts, splits), c(pcR2, spreads), c(pcR2, splits)))
for k in ('AG', 'AAA', 'ILO'):
    m = np.array([x == k for x in cohs]); print(f'    {k}: slope {slopes[m].mean():.3f}, shift {shifts[m].mean():+.3f}, spread {spreads[m].mean():.3f}, case R2 {pcR2[m].mean():.3f}, |split err| {np.nanmean(splits[m]):.3f}')
# what does a spread-driven slope (fit on test, oracle for now) buy?
A = np.column_stack([spreads, np.ones(34)]); coef = np.linalg.lstsq(A, slopes, rcond=None)[0]; pred_slope = A @ coef
cal = [to_pa(p_n.mean() + ps * (p_n - p_n.mean())) for p_n, ps in zip(ens_n, pred_slope)]
print(f'    spread-predicted slope (a = {coef[0]:+.2f}*spread + {coef[1]:.2f}) applied around case mean -> Pa R2_cb {cb(truth_pa, cal):.4f} (ensemble {cb(truth_pa, ens_pa):.4f}); constant slope {slopes.mean():.2f} -> {cb(truth_pa, [to_pa(p_n.mean() + slopes.mean() * (p_n - p_n.mean())) for p_n in ens_n]):.4f}')
print('=== within-cohort corr(case R2, |split err|):', {k: round(c(pcR2[np.array([x == k for x in cohs])], splits[np.array([x == k for x in cohs])]), 2) for k in ('AG', 'AAA', 'ILO')})
for name in ('cap', 'area', 'murray'):
    v = np.array([r[name] for r in rows]); a, b = np.polyfit(v[tr], Y[tr], 1)
    print(f'    fitted on train138: log(Qext/Qint) = {a:.2f} * {name}_term {b:+.2f}   (murray_term = 3 log r-ratio; cap_term = 2 log rcap-ratio; area_term = log A-ratio)')
res_m = Y - np.array([r['murray'] for r in rows]); v = np.array([r['area'] for r in rows]); a, b = np.polyfit(v[tr], Y[tr], 1); res_a = Y - (a * v + b)
print(f'    split error sd: Murray raw {res_m.std():.3f} -> area-fitted {res_a.std():.3f} (test only: {res_m[te].std():.3f} -> {res_a[te].std():.3f})')
