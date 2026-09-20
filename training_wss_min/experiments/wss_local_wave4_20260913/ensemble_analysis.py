"""Wave-4 offline analysis: X5A vs X5 paired per seed, and seed ensembles from saved test34 predictions.

    python training_wss_min/experiments/wss_local_wave4_20260913/ensemble_analysis.py
"""
import itertools, statistics as st, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import evaluate as E
M = E.M
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); RUNS = ROOT / 'training_wss_min/runs'
W1, W1B, W4 = 'wss_local_wave1_20260912', 'wss_local_wave1b_20260912', 'wss_local_wave4_20260913'
X5 = {'1234': RUNS / W1 / 'X5_s1234', '7': RUNS / W1B / 'X5_s7', '2025': RUNS / W1B / 'X5_s2025', '11': RUNS / W4 / 'X5_s11', '2026': RUNS / W4 / 'X5_s2026'}
X5A = {s: RUNS / W4 / f'X5A_s{s}' for s in ('1234', '7', '2025', '11', '2026')}
X5B = RUNS / W4 / 'X5B_s1234'
PRED = 'eval/ckpt_best/predictions/test'
units = sorted(str(p.parent.relative_to(X5['1234'] / PRED)) for p in (X5['1234'] / PRED).glob('*/*/*/predictions.npz'))
assert len(units) == 34


def load(run):
    out = {}
    for u in units:
        with np.load(run / PRED / u / 'predictions.npz') as z:
            out[u] = {k: z[k].astype(np.float64) for k in ('pred_norm', 'true_norm', 'pred_pa', 'true_pa')}
    return out


def cb(t, p):
    return M.casebalanced_field_metrics(t, p)['r2']


def tail(t, p):
    tt, pp = np.concatenate(t), np.concatenate(p); q = np.quantile(tt, 0.9); m = tt >= q
    return f"high-WSS R² {1 - ((tt[m] - pp[m]) ** 2).sum() / ((tt[m] - tt[m].mean()) ** 2).sum():.3f}, top10 比 {pp[m].mean() / tt[m].mean():.3f}"


d0 = load(X5['1234']); A = np.polyfit(d0[units[0]]['pred_norm'], np.log(d0[units[0]]['pred_pa']), 1); std, mean = A
to_pa = lambda n: np.exp(n * std + mean)
truth_pa = [d0[u]['true_pa'] for u in units]; truth_n = [d0[u]['true_norm'] for u in units]
avail_x5 = {s: load(r) for s, r in X5.items() if (r / PRED).is_dir()}
avail_x5a = {s: load(r) for s, r in X5A.items() if (r / PRED).is_dir()}
print('=== paired X5A vs X5 (best, physical R²_cb / normalized / per-case wins):')
deltas = []
for s in avail_x5a:
    if s not in avail_x5:
        continue
    pa_a = [avail_x5a[s][u]['pred_pa'] for u in units]; pa_b = [avail_x5[s][u]['pred_pa'] for u in units]
    na = [avail_x5a[s][u]['pred_norm'] for u in units]; nb = [avail_x5[s][u]['pred_norm'] for u in units]
    pc = [(1 - ((t - a) ** 2).sum() / ((t - t.mean()) ** 2).sum()) - (1 - ((t - b) ** 2).sum() / ((t - t.mean()) ** 2).sum()) for t, a, b in zip(truth_pa, pa_a, pa_b)]
    d = cb(truth_pa, pa_a) - cb(truth_pa, pa_b); deltas.append(d)
    print(f"  seed {s:>4}: X5A {cb(truth_pa, pa_a):.4f} vs X5 {cb(truth_pa, pa_b):.4f}  Δ {d:+.4f} | norm Δ {cb(truth_n, na) - cb(truth_n, nb):+.4f} | wins/losses {sum(x > 0 for x in pc)}/{sum(x < 0 for x in pc)} | X5A {tail(truth_pa, pa_a)}")
if len(deltas) > 1:
    print(f"  mean Δ {st.mean(deltas):+.4f} sd {st.stdev(deltas):.4f} all same sign: {all(x > 0 for x in deltas) or all(x < 0 for x in deltas)}")
print('=== ensembles (log-space mean of pred_norm):')
for name, pool in (('X5', avail_x5), ('X5A', avail_x5a)):
    seeds = list(pool)
    for k in sorted({3, len(seeds)}):
        if k > len(seeds):
            continue
        vals = []
        for combo in itertools.combinations(seeds, k):
            ens = [np.mean([pool[s][u]['pred_norm'] for s in combo], axis=0) for u in units]
            vals.append(cb(truth_pa, [to_pa(x) for x in ens]))
        print(f"  {name} {k}-seed ensemble: mean over {len(vals)} combos {st.mean(vals):.4f} (min {min(vals):.4f}, max {max(vals):.4f}); singles mean {st.mean(cb(truth_pa, [pool[s][u]['pred_pa'] for u in units]) for s in seeds):.4f}")
    if len(seeds) >= 3:
        ens = [np.mean([pool[s][u]['pred_norm'] for s in seeds], axis=0) for u in units]
        print(f"  {name} all-{len(seeds)} ensemble: Pa {cb(truth_pa, [to_pa(x) for x in ens]):.4f}, norm {cb(truth_n, ens):.4f}, {tail(truth_pa, [to_pa(x) for x in ens])}")
if avail_x5 and avail_x5a:
    both = [np.mean([p[s][u]['pred_norm'] for p in (avail_x5, avail_x5a) for s in p], axis=0) for u in units]
    print(f"  X5+X5A mixed ensemble ({len(avail_x5) + len(avail_x5a)} models): Pa {cb(truth_pa, [to_pa(x) for x in both]):.4f}, norm {cb(truth_n, both):.4f}")
if (X5B / PRED).is_dir():
    db = load(X5B); print(f"=== X5B_s1234 (Murray + capfit): Pa {cb(truth_pa, [db[u]['pred_pa'] for u in units]):.4f}, norm {cb(truth_n, [db[u]['pred_norm'] for u in units]):.4f}")
