"""感受野阶梯读数 + 机制验证：周向模态份额有没有真的被压下去。"""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import evaluate as E
M = E.M
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); RUNS = ROOT / 'training_wss_min/runs'
VIEW = ROOT / 'data_wss_v5/views_v5_1/wss_min_view_v1'
OUT = Path(__file__).resolve().parent
P = 'eval/ckpt_best/predictions/test'
BIN_MM, MIN_PTS, N_PERM = 4.0, 40, 3
ARMS = [('X5D_v51 底座', RUNS / 'wss_v51_wave1_20260916/X5D_v51_s1234', 0.0, 16),
        ('X5Ddual(4mm+spacing)', RUNS / 'wss_v51_wave1_20260916/X5Ddual_s1234', 4.0, 16),
        ('W04', RUNS / 'wss_local_morph_radius_20260917/W04_s1234', 4.0, 16),
        ('W06', RUNS / 'wss_local_morph_radius_20260917/W06_s1234', 6.0, 16),
        ('W08', RUNS / 'wss_local_morph_radius_20260917/W08_s1234', 8.0, 16),
        ('W06K32', RUNS / 'wss_local_morph_radius_k_20260917/W06K32_s1234', 6.0, 32)]
units = sorted(str(p.parent.relative_to(ARMS[0][1] / P)) for p in (ARMS[0][1] / P).glob('*/*/*/predictions.npz'))
coh = [u.split('/')[0] for u in units]
avail = [(n, r, rm, k) for n, r, rm, k in ARMS if (r / P).is_dir()]
print('可读的臂:', [n for n, *_ in avail], flush=True)
data = {}
for name, run, _, _ in avail:
    d = {}
    for u in units:
        with np.load(run / P / u / 'predictions.npz') as z:
            d[u] = {kk: z[kk].astype(np.float64) for kk in ('pred_norm', 'true_norm', 'pred_pa', 'true_pa')}
    data[name] = d
truth = [data[avail[0][0]][u]['true_pa'] for u in units]
truth_n = [data[avail[0][0]][u]['true_norm'] for u in units]
pb = [data[avail[0][0]][u]['pred_pa'] for u in units]
pbn = [data[avail[0][0]][u]['pred_norm'] for u in units]


def r2(t, p): return 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()
def cb(t, p): return M.casebalanced_field_metrics(t, p)['r2']
def tail(tl, pl):
    t = np.concatenate(tl); p = np.concatenate(pl); q = np.quantile(t, .9); m = t >= q
    return float(1 - ((t[m] - p[m]) ** 2).sum() / ((t[m] - t[m].mean()) ** 2).sum())
def iou(tl, pl):
    return float(np.mean([((t >= np.quantile(t, .9)) & (p >= np.quantile(p, .9))).sum()
                          / ((t >= np.quantile(t, .9)) | (p >= np.quantile(p, .9))).sum() for t, p in zip(tl, pl)]))

res = {}
print(f'\n{"臂":22s} {"球":>5s} {"k":>3s} {"Pa R2_cb":>9s} {"Δ底座":>8s} {"归一化":>8s} {"Δ":>8s} {"胜":>7s} {"high":>6s} {"IoU":>6s}  AG/AAA/ILO')
for name, run, rm, k in avail:
    p = [data[name][u]['pred_pa'] for u in units]; pn = [data[name][u]['pred_norm'] for u in units]
    v, vn = cb(truth, p), cb(truth_n, pn)
    wins = sum(r2(truth[i], p[i]) > r2(truth[i], pb[i]) for i in range(len(units)))
    per = {c: cb([truth[i] for i in range(len(units)) if coh[i] == c],
                 [p[i] for i in range(len(units)) if coh[i] == c]) for c in ('AG', 'AAA', 'ILO')}
    res[name] = dict(radius_mm=rm, radius_k=k, pa_r2_cb=v, norm_r2_cb=vn,
                     d_pa=v - cb(truth, pb), d_norm=vn - cb(truth_n, pbn), wins=int(wins),
                     high=tail(truth, p), iou=iou(truth, p), per_cohort=per)
    print(f'{name:22s} {rm:5.1f} {k:3d} {v:9.4f} {v-cb(truth,pb):+8.4f} {vn:8.4f} {vn-cb(truth_n,pbn):+8.4f} '
          f'{wins:5d}/34 {res[name]["high"]:6.3f} {res[name]["iou"]:6.3f}  ' + '/'.join(f'{x:.3f}' for x in per.values()))

print('\n机制验证：区间内残差的周向模态净份额（越低 = 感受野越吃到了这块结构）')
rng = np.random.default_rng(0)
geo = {}
for u in units:
    with np.load(VIEW / u / 'bundle.npz', allow_pickle=False) as z:
        geo[u] = (z['wall_theta_rad'].astype(np.float64), z['wall_segment_id'].astype(int),
                  z['wall_s_local_mm'].astype(np.float64))
print(f'{"臂":22s} {"区间内份额":>10s} {"m=1..3 实际":>11s} {"零假设":>8s} {"净":>8s}')
for name, run, rm, k in avail:
    shares, nulls, within, wts = [], [], [], []
    for i, u in enumerate(units):
        th, seg, sl = geo[u]
        r = np.log(np.maximum(truth[i], 1e-6)) - np.log(np.maximum(data[name][u]['pred_pa'], 1e-6))
        key = seg.astype(np.int64) * 100000 + np.floor(sl / BIN_MM).astype(np.int64)
        uk, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(uk))
        bm = np.array([r[inv == b].mean() for b in range(len(uk))])
        within.append(1 - float(np.average((bm - r.mean()) ** 2, weights=cnt)) / r.var())
        for b in range(len(uk)):
            m_ = inv == b
            if cnt[b] < MIN_PTS:
                continue
            rl = r[m_] - r[m_].mean(); t = th[m_]
            if rl.var() <= 0:
                continue
            s_, n_ = 0.0, 0.0
            for mm in (1, 2, 3):
                A = np.column_stack([np.cos(mm * t), np.sin(mm * t)])
                c_, *_ = np.linalg.lstsq(A, rl, rcond=None)
                s_ += ((A @ c_) ** 2).mean() / rl.var()
                acc = []
                for _ in range(N_PERM):
                    tp = rng.permutation(t)
                    B = np.column_stack([np.cos(mm * tp), np.sin(mm * tp)])
                    cc, *_ = np.linalg.lstsq(B, rl, rcond=None)
                    acc.append(((B @ cc) ** 2).mean() / rl.var())
                n_ += float(np.mean(acc))
            shares.append(s_); nulls.append(n_); wts.append(int(cnt[b]))
    a = float(np.average(shares, weights=wts)); b_ = float(np.average(nulls, weights=wts))
    res[name].update(within_share=float(np.median(within)), modes=a, modes_null=b_, modes_net=a - b_)
    print(f'{name:22s} {np.median(within):10.4f} {a:11.4f} {b_:8.4f} {a-b_:+8.4f}')

(OUT / 'analyze_radius_ladder.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
print('\nwrote', OUT / 'analyze_radius_ladder.json')
