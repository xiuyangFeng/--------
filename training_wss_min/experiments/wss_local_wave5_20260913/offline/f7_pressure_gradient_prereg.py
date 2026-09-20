"""F7 pre-registration (no training): can the tangential gradient of the *predicted* wall pressure (PF6, deployment-legal)
explain what is left in the X5 ensemble residual on test34?

Poiseuille: tau = (R/2) |dp/ds|.  Features per wall point from the PF6 3-seed mean wall pressure (Pa, relative):
  g_ax   axial pressure gradient (Pa/mm) along the centreline tangent projected to the wall tangent plane
  g_tan  tangential gradient magnitude
  ln_tau_pois = ln((R/2) * |g_ax| + 1e-3)
  p_rel  case-standardised predicted wall pressure
at two neighbourhood scales (k=24 ~1.1 mm, k=96 ~2.3 mm on the CFD wall spacing).  Same features from the CFD
truth pressure give the ceiling (label-derived, residual analysis only).  Explained residual variance is estimated
leave-one-case-out (34 folds) with ridge (+ squares) and a small gradient-boosting model; the implied gain is
frac_explained * (1 - R2_current).
"""
import json, sys, time
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
import h5py
from training_wss_min import evaluate as E, next_geometry as NG
M = E.M
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); RUNS = ROOT / 'training_wss_min/runs'
X5 = [RUNS / 'wss_local_wave1_20260912/X5_s1234', RUNS / 'wss_local_wave1b_20260912/X5_s7', RUNS / 'wss_local_wave1b_20260912/X5_s2025', RUNS / 'wss_local_wave4_20260913/X5_s11', RUNS / 'wss_local_wave4_20260913/X5_s2026']
PF6 = [RUNS / 'wss_local_wave2_20260912/PF6_s1234', RUNS / 'wss_local_wave3_20260913/PF6_s7', RUNS / 'wss_local_wave3_20260913/PF6_s2025']
VIEW = ROOT / 'data_wss_v5/views/wss_min_view_v1'
PRED = 'eval/ckpt_best/predictions/test'
OUT = Path(__file__).resolve().parent
units = sorted(str(p.parent.relative_to(X5[0] / PRED)) for p in (X5[0] / PRED).glob('*/*/*/predictions.npz')); assert len(units) == 34
t0 = time.time()

def tangent_gradient(xyz, normals, tangent, values, k):
    """Least-squares gradient of `values` in the tangent plane; returns (g_axial, |g_tan|)."""
    n = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    ax = tangent - np.einsum('nd,nd->n', tangent, n)[:, None] * n
    ax /= np.maximum(np.linalg.norm(ax, axis=1, keepdims=True), 1e-12)
    ci = np.cross(n, ax)
    _, nb = cKDTree(xyz).query(xyz, k=k + 1, workers=-1); nb = nb[:, 1:]
    d = xyz[nb] - xyz[:, None, :]
    u = np.einsum('nkd,nd->nk', d, ax); v = np.einsum('nkd,nd->nk', d, ci)
    w = values[nb] - values[:, None]
    A = np.stack([u, v, np.ones_like(u)], -1)
    G = np.einsum('nki,nkj->nij', A, A) + 1e-9 * np.eye(3)
    rhs = np.einsum('nki,nk->ni', A, w)
    coef = np.linalg.solve(G, rhs)
    return coef[:, 0], np.hypot(coef[:, 0], coef[:, 1])

def wall_geometry(u):
    with np.load(VIEW / u / 'bundle.npz', allow_pickle=True) as z, h5py.File(NG.source_h5_for_bundle(VIEW / u / 'bundle.npz'), 'r') as h:
        ids = z['wall_node_id_cas'].astype(np.int64); src = h['wall_static/node_id_cas'][:].astype(np.int64)
        order = np.argsort(src); at = np.searchsorted(src[order], ids); rows = order[at]; assert np.array_equal(src[order][at], ids)
        atlas = h['geometry/atlas_table'][:]; cols = json.loads(h['geometry'].attrs['atlas_columns'])
        arow = h['wall_static/atlas_row'][:][rows]
        tangent = atlas[arow][:, [cols.index(f'tangent_{a}') for a in 'xyz']]; tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
        steps = z['steps'].tolist(); si = steps.index(int(z['peak_step']))
        return dict(xyz=z['wall_coords_raw'].astype(np.float64), normal=z['wall_normal_pca'].astype(np.float64), tangent=tangent,
                    radius=z['wall_local_radius'].astype(np.float64), p_true=z['wall_pressure'][si].astype(np.float64),
                    segment=z['wall_segment_id'].astype(int), s_local=z['wall_s_local_mm'].astype(np.float64))

def load_wss(run, u):
    with np.load(run / PRED / u / 'predictions.npz') as z:
        return z['pred_norm'].astype(np.float64), z['true_norm'].astype(np.float64), z['pred_pa'].astype(np.float64), z['true_pa'].astype(np.float64)

def load_pressure(run, u):
    with np.load(run / PRED / u / 'predictions.npz') as z:
        wall = z['point_kind'] == 0; qi = z['query_idx'][wall]; p = np.full(int(z['n_wall']), np.nan); p[qi] = z['pred_raw'][wall]; return p

def section_split(values, segment, s_local, bin_mm=4.0):
    """Split a field into per-(segment, 4 mm bin) mean and within-section remainder."""
    key = segment.astype(np.int64) * 100000 + np.floor(s_local / bin_mm).astype(np.int64)
    _, inv = np.unique(key, return_inverse=True); inv = inv.reshape(-1)
    sums = np.bincount(inv, weights=values); cnt = np.bincount(inv)
    mean = (sums / np.maximum(cnt, 1))[inv]
    return mean, values - mean

feats, resid, resid_within, case_id, lnwss_true, lnwss_pred, feats_true, meta = [], [], [], [], [], [], [], {}
ln_std = None
for i, u in enumerate(units):
    g = wall_geometry(u)
    preds = [load_wss(r, u) for r in X5]
    pn = np.mean([p[0] for p in preds], 0); tn = preds[0][1]; r = pn - tn
    p_pred = np.mean([load_pressure(rr, u) for rr in PF6], 0); assert np.isfinite(p_pred).all()
    if ln_std is None:
        A = np.polyfit(preds[0][0], np.log(preds[0][2]), 1); ln_std, ln_mean = A
    rows = {}
    for tag, p in (('pred', p_pred), ('true', g['p_true'] - g['p_true'].mean())):
        cols = []
        for k in (24, 96):
            gax, gtan = tangent_gradient(g['xyz'], g['normal'], g['tangent'], p, k)
            cols += [gax, gtan, np.log(0.5 * g['radius'] * np.abs(gax) + 1e-3)]
        cols += [(p - p.mean()) / (p.std() + 1e-9)]
        rows[tag] = np.stack(cols, 1)
    _, rw = section_split(r, g['segment'], g['s_local'])
    feats.append(rows['pred']); feats_true.append(rows['true']); resid.append(r); resid_within.append(rw); case_id.append(np.full(len(r), i))
    lnwss_true.append(tn * ln_std + ln_mean); lnwss_pred.append(pn * ln_std + ln_mean)
    print(f'[{i + 1}/34] {u} n={len(r)} {time.time() - t0:.0f}s', flush=True)
X = np.concatenate(feats); XT = np.concatenate(feats_true); R = np.concatenate(resid); RW = np.concatenate(resid_within); G = np.concatenate(case_id)
LT = np.concatenate(lnwss_true); LP = np.concatenate(lnwss_pred)
names = ['g_ax_k24', 'g_tan_k24', 'ln_tau_pois_k24', 'g_ax_k96', 'g_tan_k96', 'ln_tau_pois_k96', 'p_rel']
out = {'n_points': int(len(R)), 'features': names, 'current_norm_r2_cb_ensemble': None}

def cb_r2(true_by_case, pred_by_case):
    return float(M.casebalanced_field_metrics(true_by_case, pred_by_case)['r2'])

def loo_explained(Xm, target, groups, model='ridge', sub=None):
    """Leave-one-case-out out-of-fold fit; returns pooled R² of target and case-balanced R²."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingRegressor
    rng = np.random.default_rng(0)
    idx = np.arange(len(target)) if sub is None else np.sort(rng.choice(len(target), size=min(sub, len(target)), replace=False))
    Xs, ys, gs = Xm[idx], target[idx], groups[idx]
    oof = np.full(len(ys), np.nan)
    for c in np.unique(gs):
        tr, te = gs != c, gs == c
        if model == 'ridge':
            sc = StandardScaler().fit(Xs[tr]); Z = sc.transform(Xs[tr]); Z = np.hstack([Z, Z ** 2])
            m = Ridge(alpha=10.0).fit(Z, ys[tr]); Zt = sc.transform(Xs[te]); oof[te] = m.predict(np.hstack([Zt, Zt ** 2]))
        else:
            m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200, random_state=0).fit(Xs[tr], ys[tr])
            oof[te] = m.predict(Xs[te])
    pooled = float(1 - ((ys - oof) ** 2).sum() / ((ys - ys.mean()) ** 2).sum())
    per_case = [float(1 - ((ys[gs == c] - oof[gs == c]) ** 2).sum() / max(((ys[gs == c] - ys[gs == c].mean()) ** 2).sum(), 1e-12)) for c in np.unique(gs)]
    return dict(pooled_r2=pooled, case_mean_r2=float(np.mean(per_case)), case_median_r2=float(np.median(per_case)))

truth_n = [np.asarray(x) for x in resid]  # placeholder for shapes
res_var = float(np.var(R)); res_within_var = float(np.var(RW))
out['residual_var_total_logz'] = res_var; out['residual_var_within_section'] = res_within_var
# current ensemble R² in log_z (pooled and case-balanced)
tn_by = [preds_tn for preds_tn in np.split(LT, np.cumsum([len(x) for x in resid])[:-1])]
pn_by = [x for x in np.split(LP, np.cumsum([len(x) for x in resid])[:-1])]
out['current_ln_r2_pooled'] = float(1 - ((LP - LT) ** 2).sum() / ((LT - LT.mean()) ** 2).sum())
out['current_ln_r2_cb'] = cb_r2(tn_by, pn_by)
# 1) simple correlations of the residual with each feature (pooled, and within-section part)
out['corr_residual'] = {nm: float(np.corrcoef(X[:, j], R)[0, 1]) for j, nm in enumerate(names)}
out['corr_residual_within'] = {nm: float(np.corrcoef(X[:, j], RW)[0, 1]) for j, nm in enumerate(names)}
out['corr_residual_true_pressure'] = {nm: float(np.corrcoef(XT[:, j], R)[0, 1]) for j, nm in enumerate(names)}
# 2) Poiseuille estimate alone vs truth (how much of ln WSS does (R/2)|dp/ds| carry?)
for tag, Xm in (('pred', X), ('true', XT)):
    for j in (2, 5):
        out[f'poiseuille_{tag}_{names[j]}_vs_lnWSS_corr'] = float(np.corrcoef(Xm[:, j], LT)[0, 1])
# 3) LOO explained residual variance (pred-pressure features; true-pressure ceiling)
for tag, Xm in (('pred', X), ('true', XT)):
    for target_name, target in (('total', R), ('within_section', RW)):
        for model, sub in (('ridge', None), ('hgb', 400000)):
            key = f'loo_{tag}_{target_name}_{model}'
            out[key] = loo_explained(Xm, target, G, model, sub)
            frac = out[key]['pooled_r2']
            out[key]['implied_gain_norm_r2'] = float(max(frac, 0.0) * (1 - out['current_ln_r2_pooled']) * (res_var if target_name == 'total' else res_within_var) / res_var)
            print(key, out[key], f'{time.time() - t0:.0f}s', flush=True)
(OUT / 'f7_pressure_gradient_prereg.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if not k.startswith('loo_')}, indent=1))
print('wrote', OUT / 'f7_pressure_gradient_prereg.json')
