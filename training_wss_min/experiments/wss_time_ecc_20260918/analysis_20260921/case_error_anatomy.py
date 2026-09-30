"""中位病例 SHEN_FANG_JIN（fold2 留出）四相位同点误差解剖：误差落在哪（回流区 / 低 WSS / 小尺度），只读 postview 同点 CSV + bundle。"""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN/training_wss_min/experiments/wss_time_ecc_20260918/offline')
from common import *  # noqa
PV = ROOT / 'training_wss_min/experiments/wss_time_ecc_20260918/postview_median_phases_20260920/AAA__unruputer__SHEN_FANG_JIN'
OUTD = Path(__file__).resolve().parent
cid = 'AAA/unruputer/SHEN_FANG_JIN'
d = load_bundle(cid, vec=True); tau = d['tau'].astype(np.float64); vec = d['vec'].astype(np.float64)
PH = {'accel': 13, 'peak': 21, 'decel': 35, 'trough': 48}
inv, cnt = bins_of(d)
lines = [f'case {cid}  N_bundle={tau.shape[1]}']
tree = cKDTree(d['pos'])
out = {}
for arm in ('T0', 'T-null', 'TB8'):
    for ph, k in PH.items():
        p = PV / arm / ph / '_export/same_point_fields.csv'
        if not p.exists(): continue
        X = np.genfromtxt(p, delimiter=',', names=True)
        xyz = np.column_stack([X['x'], X['y'], X['z']]); dist, idx = tree.query(xyz)
        if arm == 'T0' and ph == 'peak':
            lines.append(f'坐标匹配：n={len(idx)} 唯一={len(np.unique(idx))} 距离中位 {np.median(dist):.2e} mm, p99 {np.quantile(dist,.99):.2e}')
        yt = X['wss_cfd'].astype(np.float64); yp = X['wss_pred'].astype(np.float64)
        assert np.allclose(yt, tau[k][idx], rtol=1e-4, atol=1e-4), 'cfd 列与 bundle 帧不一致'
        lt, lp = ln_pa(yt), ln_pa(yp); e = lp - lt; sse = (e ** 2).sum()
        r2 = r2_score(yt, yp); r2ln = r2_score(lt, lp)
        rr = np.corrcoef(lt, lp)[0, 1]
        # 回流：相对峰值方向反向
        dot = (vec[k][idx] * vec[PEAK_INDEX][idx]).sum(1); rev = dot < 0
        low = yt < 0.4
        # 尺度分解（ln 误差）：4 mm 截面段均值 vs 段内
        bi = inv[idx]; bm = np.bincount(bi, weights=e, minlength=len(cnt)) / np.maximum(np.bincount(bi, minlength=len(cnt)), 1)
        e_big = bm[bi]; e_small = e - e_big
        # 真值本身的尺度结构
        tm = np.bincount(bi, weights=lt, minlength=len(cnt)) / np.maximum(np.bincount(bi, minlength=len(cnt)), 1)
        t_within = 1 - ((tm[bi] - lt.mean()) ** 2).sum() / ((lt - lt.mean()) ** 2).sum()
        # 回流区内 / 外 的相关
        def corr(m):
            return np.corrcoef(lt[m], lp[m])[0, 1] if m.sum() > 50 else np.nan
        rec = dict(r2_pa=r2, r2_ln=r2ln, r_ln=rr, mean_pa=yt.mean(), pred_over_true=yp.mean() / yt.mean(),
                   rev_frac=rev.mean(), rev_sse_share=(e[rev] ** 2).sum() / sse, rev_rmse_ln=np.sqrt((e[rev] ** 2).mean()) if rev.any() else np.nan,
                   nonrev_rmse_ln=np.sqrt((e[~rev] ** 2).mean()), corr_rev=corr(rev), corr_nonrev=corr(~rev),
                   low_frac=low.mean(), low_sse_share=(e[low] ** 2).sum() / sse,
                   err_small_share=(e_small ** 2).sum() / sse, err_big_share=(e_big ** 2).sum() / sse, truth_within4mm=t_within,
                   bias_ln=e.mean(), rmse_ln=np.sqrt((e ** 2).mean()))
        out[f'{arm}|{ph}'] = rec
hdr = ['r2_pa', 'r2_ln', 'r_ln', 'pred_over_true', 'rev_frac', 'rev_sse_share', 'rev_rmse_ln', 'nonrev_rmse_ln', 'corr_rev', 'corr_nonrev', 'low_frac', 'low_sse_share', 'err_small_share', 'truth_within4mm', 'bias_ln']
lines.append(f"{'arm|phase':14s} " + ' '.join(f'{h:>14s}' for h in hdr))
for key, rec in out.items():
    lines.append(f'{key:14s} ' + ' '.join(f'{rec[h]:14.3f}' for h in hdr))
txt = '\n'.join(lines); print(txt)
(OUTD / 'case_error_anatomy.txt').write_text(txt + '\n'); json.dump(out, open(OUTD / 'case_error_anatomy.json', 'w'), indent=1)
