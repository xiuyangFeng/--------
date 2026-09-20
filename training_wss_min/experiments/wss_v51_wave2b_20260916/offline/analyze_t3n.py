"""v5.1 wave 2b: T3n residual stacking read two ways — paired on the exposed test34 (three seeds) and out-of-fold on
cv3_v51 (three folds, every train136 case once), each against the same-seed / same-fold X5D_v51 stage-1 model."""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import evaluate as E
M = E.M
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); RUNS = ROOT / 'training_wss_min/runs'
W1, W2A, W2B = RUNS / 'wss_v51_wave1_20260916', RUNS / 'wss_v51_wave2a_20260916', RUNS / 'wss_v51_wave2b_20260916'
P = 'eval/ckpt_best/predictions/test'; OUT = Path(__file__).resolve().parent
def load(run):
    units = sorted(str(p.parent.relative_to(run / P)) for p in (run / P).glob('*/*/*/predictions.npz'))
    return units, {u: {k: np.load(run / P / u / 'predictions.npz')[k].astype(np.float64) for k in ('pred_norm', 'true_norm', 'pred_pa', 'true_pa')} for u in units}
def cb(t, p): return M.casebalanced_field_metrics(t, p)['r2']
def r2(t, p): return 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()
def tail(tl, pl):
    t, p = np.concatenate(tl), np.concatenate(pl); q = np.quantile(t, .9); m = t >= q
    return dict(high_r2=float(1 - ((t[m] - p[m]) ** 2).sum() / ((t[m] - t[m].mean()) ** 2).sum()), top10_ratio=float(p[m].mean() / t[m].mean()),
                p99_ratio=float(np.quantile(p, .99) / np.quantile(t, .99)))
res = {}
print('=== test34 配对（T3n_s<seed> vs X5D_v51_s<seed>，best）')
rows = []
for s in (1234, 7, 2025):
    ua, A = load(W2B / f'T3n_s{s}'); ub, B = load(W1 / f'X5D_v51_s{s}'); assert ua == ub
    t = [A[u]['true_pa'] for u in ua]; tn = [A[u]['true_norm'] for u in ua]
    pa, pb = [A[u]['pred_pa'] for u in ua], [B[u]['pred_pa'] for u in ua]
    na, nb = [A[u]['pred_norm'] for u in ua], [B[u]['pred_norm'] for u in ua]
    coh = [u.split('/')[0] for u in ua]
    d_pa, d_n = cb(t, pa) - cb(t, pb), cb(tn, na) - cb(tn, nb)
    wins = int(sum(r2(x, y) > r2(x, z) for x, y, z in zip(t, pa, pb)))
    ta, tb = tail(t, pa), tail(t, pb)
    d_coh = {c: cb([t[i] for i in range(len(ua)) if coh[i] == c], [pa[i] for i in range(len(ua)) if coh[i] == c]) - cb([t[i] for i in range(len(ua)) if coh[i] == c], [pb[i] for i in range(len(ua)) if coh[i] == c]) for c in ('AG', 'AAA', 'ILO')}
    rows.append(dict(seed=s, t3n=cb(t, pa), base=cb(t, pb), d_pa=d_pa, d_norm=d_n, wins=wins, d_coh=d_coh,
                     d_high=ta['high_r2'] - tb['high_r2'], d_top10=ta['top10_ratio'] - tb['top10_ratio'], d_p99=ta['p99_ratio'] - tb['p99_ratio']))
    print(f'  seed {s}: T3n {cb(t,pa):.4f} vs 底座 {cb(t,pb):.4f} | Δ物理 {d_pa:+.4f} Δ归一化 {d_n:+.4f} 逐例胜 {wins}/34 | AG/AAA/ILO ' + '/'.join(f'{v:+.3f}' for v in d_coh.values()) + f' | Δhigh-WSS {ta["high_r2"]-tb["high_r2"]:+.3f} Δtop10 {ta["top10_ratio"]-tb["top10_ratio"]:+.3f} Δp99 {ta["p99_ratio"]-tb["p99_ratio"]:+.3f}')
print(f'  三 seed 均值: Δ物理 {np.mean([r["d_pa"] for r in rows]):+.4f} (sd {np.std([r["d_pa"] for r in rows]):.4f}) Δ归一化 {np.mean([r["d_norm"] for r in rows]):+.4f} (sd {np.std([r["d_norm"] for r in rows]):.4f}) Δhigh-WSS {np.mean([r["d_high"] for r in rows]):+.3f} Δtop10 {np.mean([r["d_top10"] for r in rows]):+.3f}')
res['test34_paired'] = rows
# 集成
ens_a = None
for tag, base, runs in (('T3n 3-seed', W2B, [f'T3n_s{s}' for s in (1234, 7, 2025)]), ('X5D_v51 3-seed', W1, [f'X5D_v51_s{s}' for s in (1234, 7, 2025)])):
    loaded = [load(base / r)[1] for r in runs]; units = load(base / runs[0])[0]
    t = [loaded[0][u]['true_pa'] for u in units]; e = [np.mean([d[u]['pred_pa'] for d in loaded], axis=0) for u in units]
    pc = np.array([r2(x, y) for x, y in zip(t, e)]); tl = tail(t, e)
    res[tag] = dict(pa_r2_cb=cb(t, e), case_r2_mean=float(pc.mean()), case_r2_p10=float(np.quantile(pc, .1)), **tl)
    print(f'  {tag} 集成(Pa均值): {cb(t,e):.4f} | 逐例均值/p10 {pc.mean():.3f}/{np.quantile(pc,.1):.3f} | high-WSS {tl["high_r2"]:.3f} top10 {tl["top10_ratio"]:.3f} p99 {tl["p99_ratio"]:.3f}')
print(f'  集成 Δ: {res["T3n 3-seed"]["pa_r2_cb"] - res["X5D_v51 3-seed"]["pa_r2_cb"]:+.4f}')
print('\n=== cv3_v51 折外配对（T3n_f<k> vs X5D_v51_f<k>，每例一次）')
pt, pa_all, pb_all, ptn, na_all, nb_all, cohs = [], [], [], [], [], [], []
frows = []
for k in range(3):
    ua, A = load(W2B / f'T3n_f{k}_s1234'); ub, B = load(W2A / f'X5D_v51_f{k}_s1234'); assert ua == ub
    t = [A[u]['true_pa'] for u in ua]; tn = [A[u]['true_norm'] for u in ua]
    pa, pb = [A[u]['pred_pa'] for u in ua], [B[u]['pred_pa'] for u in ua]
    na, nb = [A[u]['pred_norm'] for u in ua], [B[u]['pred_norm'] for u in ua]
    wins = int(sum(r2(x, y) > r2(x, z) for x, y, z in zip(t, pa, pb)))
    ta, tb = tail(t, pa), tail(t, pb)
    frows.append(dict(fold=k, n=len(ua), t3n=cb(t, pa), base=cb(t, pb), d_pa=cb(t, pa) - cb(t, pb), d_norm=cb(tn, na) - cb(tn, nb), wins=wins, n_cases=len(ua),
                      d_high=ta['high_r2'] - tb['high_r2'], d_top10=ta['top10_ratio'] - tb['top10_ratio']))
    print(f'  fold{k} ({len(ua)} 例): T3n {cb(t,pa):.4f} vs 底座 {cb(t,pb):.4f} | Δ物理 {cb(t,pa)-cb(t,pb):+.4f} Δ归一化 {cb(tn,na)-cb(tn,nb):+.4f} 逐例胜 {wins}/{len(ua)} | Δhigh-WSS {ta["high_r2"]-tb["high_r2"]:+.3f} Δtop10 {ta["top10_ratio"]-tb["top10_ratio"]:+.3f}')
    pt += t; ptn += tn; pa_all += pa; pb_all += pb; na_all += na; nb_all += nb; cohs += [u.split('/')[0] for u in ua]
wins = int(sum(r2(x, y) > r2(x, z) for x, y, z in zip(pt, pa_all, pb_all)))
ta, tb = tail(pt, pa_all), tail(pt, pb_all)
pooled = dict(n=len(pt), t3n=cb(pt, pa_all), base=cb(pt, pb_all), d_pa=cb(pt, pa_all) - cb(pt, pb_all), d_norm=cb(ptn, na_all) - cb(ptn, nb_all), wins=wins,
              d_high=ta['high_r2'] - tb['high_r2'], d_top10=ta['top10_ratio'] - tb['top10_ratio'],
              d_coh={c: cb([pt[i] for i in range(len(pt)) if cohs[i] == c], [pa_all[i] for i in range(len(pt)) if cohs[i] == c]) - cb([pt[i] for i in range(len(pt)) if cohs[i] == c], [pb_all[i] for i in range(len(pt)) if cohs[i] == c]) for c in ('AG', 'AAA', 'ILO')})
print(f'  合并 {len(pt)} 例折外: T3n {pooled["t3n"]:.4f} vs 底座 {pooled["base"]:.4f} | Δ物理 {pooled["d_pa"]:+.4f} Δ归一化 {pooled["d_norm"]:+.4f} 逐例胜 {wins}/{len(pt)} | AG/AAA/ILO ' + '/'.join(f'{v:+.3f}' for v in pooled['d_coh'].values()) + f' | Δhigh-WSS {pooled["d_high"]:+.3f} Δtop10 {pooled["d_top10"]:+.3f}')
res['cv3_folds'] = frows; res['cv3_pooled'] = pooled
(OUT / 'analyze_t3n.json').write_text(json.dumps(res, indent=1, default=float))
