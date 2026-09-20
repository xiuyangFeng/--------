"""Wave-5 offline analysis: paired stenosis-index deltas (test34), cv3 paired X5+X11 vs X5 (held-out folds, pooled
over train138), fold-model quality, and the CV-based check of the ensemble/back-transform protocol.

    python training_wss_min/experiments/wss_local_wave5_20260913/offline/analyze_wave5.py
"""
import json, statistics as st, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/public/newhome/cy/Digital_twin/GNN')
from training_wss_min import evaluate as E
M = E.M
ROOT = Path('/public/newhome/cy/Digital_twin/GNN'); RUNS = ROOT / 'training_wss_min/runs'
W1, W1B, W4, W5, W5B = 'wss_local_wave1_20260912', 'wss_local_wave1b_20260912', 'wss_local_wave4_20260913', 'wss_local_wave5_20260913', 'wss_local_wave5b_20260913'
PRED = 'eval/ckpt_best/predictions/test'
OUT = Path(__file__).resolve().parent
report = {}


def units_of(run):
    return sorted(str(p.parent.relative_to(run / PRED)) for p in (run / PRED).glob('*/*/*/predictions.npz'))


def load(run, units):
    out = {}
    for u in units:
        with np.load(run / PRED / u / 'predictions.npz') as z:
            out[u] = {k: z[k].astype(np.float64) for k in ('pred_norm', 'true_norm', 'pred_pa', 'true_pa')}
    return out


def cb(t, p): return float(M.casebalanced_field_metrics(t, p)['r2'])


def tails(t, p):
    tt, pp = np.concatenate(t), np.concatenate(p); q = np.quantile(tt, 0.9); m = tt >= q
    ious = [float(((a >= np.percentile(a, 90)) & (b >= np.percentile(b, 90))).sum() / ((a >= np.percentile(a, 90)) | (b >= np.percentile(b, 90))).sum()) for a, b in zip(t, p)]
    return dict(high_r2=float(1 - ((tt[m] - pp[m]) ** 2).sum() / ((tt[m] - tt[m].mean()) ** 2).sum()), top10_ratio=float(pp[m].mean() / tt[m].mean()),
                p99_ratio=float(np.quantile(pp, 0.99) / np.quantile(tt, 0.99)), iou=float(np.mean(ious)), mae=float(np.mean(np.abs(tt - pp))))


def per_case(t, p):
    return [float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()) for a, b in zip(t, p)]


def domains(units, t, p):
    out = {}
    for d in ('AG', 'AAA', 'ILO'):
        idx = [i for i, u in enumerate(units) if u.startswith(d + '/')]
        out[d] = cb([t[i] for i in idx], [p[i] for i in idx]) if idx else None
    return out


def paired(name, arm_runs, ref_runs):
    """arm_runs / ref_runs: {seed: run}; both evaluated on the same units (test34)."""
    rows = []; d_pa = []; d_n = []
    for s in arm_runs:
        if s not in ref_runs or not (arm_runs[s] / PRED).is_dir() or not (ref_runs[s] / PRED).is_dir():
            continue
        units = units_of(arm_runs[s]); a = load(arm_runs[s], units); r = load(ref_runs[s], units)
        t_pa = [a[u]['true_pa'] for u in units]; t_n = [a[u]['true_norm'] for u in units]
        a_pa = [a[u]['pred_pa'] for u in units]; r_pa = [r[u]['pred_pa'] for u in units]
        a_n = [a[u]['pred_norm'] for u in units]; r_n = [r[u]['pred_norm'] for u in units]
        pa, pr = cb(t_pa, a_pa), cb(t_pa, r_pa); na, nr = cb(t_n, a_n), cb(t_n, r_n)
        pc = np.array(per_case(t_pa, a_pa)) - np.array(per_case(t_pa, r_pa))
        row = dict(seed=s, arm_pa=pa, ref_pa=pr, d_pa=pa - pr, arm_norm=na, ref_norm=nr, d_norm=na - nr,
                   wins=int((pc > 0).sum()), losses=int((pc < 0).sum()), worst8_delta=float(np.sort(pc)[:8].mean()) if False else None,
                   arm_tails=tails(t_pa, a_pa), ref_tails=tails(t_pa, r_pa), arm_domains=domains(units, t_pa, a_pa), ref_domains=domains(units, t_pa, r_pa))
        ref_pc = np.array(per_case(t_pa, r_pa)); worst = np.argsort(ref_pc)[:8]; row['worst8_delta'] = float(pc[worst].mean())
        rows.append(row); d_pa.append(pa - pr); d_n.append(na - nr)
    summary = dict(rows=rows)
    if len(d_pa) > 1:
        summary.update(mean_d_pa=st.mean(d_pa), sd_d_pa=st.stdev(d_pa), mean_d_norm=st.mean(d_n), sd_d_norm=st.stdev(d_n),
                       all_same_sign_pa=all(x > 0 for x in d_pa) or all(x < 0 for x in d_pa),
                       all_same_sign_norm=all(x > 0 for x in d_n) or all(x < 0 for x in d_n))
    print(f'=== {name}')
    for r in rows:
        print(f"  seed {r['seed']:>5}: arm {r['arm_pa']:.4f} vs ref {r['ref_pa']:.4f} Δ {r['d_pa']:+.4f} | norm Δ {r['d_norm']:+.4f} | wins/losses {r['wins']}/{r['losses']} | worst8 Δ {r['worst8_delta']:+.3f} | "
              f"high-WSS {r['arm_tails']['high_r2']:.3f} ({r['ref_tails']['high_r2']:.3f}) top10 {r['arm_tails']['top10_ratio']:.3f} ({r['ref_tails']['top10_ratio']:.3f}) IoU {r['arm_tails']['iou']:.3f} ({r['ref_tails']['iou']:.3f}) | "
              f"AG/AAA/ILO {r['arm_domains']['AG']:.3f}/{r['arm_domains']['AAA']:.3f}/{r['arm_domains']['ILO']:.3f} (ref {r['ref_domains']['AG']:.3f}/{r['ref_domains']['AAA']:.3f}/{r['ref_domains']['ILO']:.3f})")
    if 'mean_d_pa' in summary:
        print(f"  mean Δ物理 {summary['mean_d_pa']:+.4f} (sd {summary['sd_d_pa']:.4f}, same sign {summary['all_same_sign_pa']}) | Δ归一化 {summary['mean_d_norm']:+.4f} (sd {summary['sd_d_norm']:.4f}, same sign {summary['all_same_sign_norm']})")
    return summary


X5 = {'1234': RUNS / W1 / 'X5_s1234', '7': RUNS / W1B / 'X5_s7', '2025': RUNS / W1B / 'X5_s2025', '11': RUNS / W4 / 'X5_s11', '2026': RUNS / W4 / 'X5_s2026'}
X5A = {s: RUNS / W4 / f'X5A_s{s}' for s in ('1234', '7', '2025', '11', '2026')}
report['X5I_vs_X5'] = paired('X5I (Murray + stenosis index) vs X5', {s: RUNS / W5 / f'X5I_s{s}' for s in ('1234', '7', '2025')}, X5)
report['X5AI_vs_X5A'] = paired('X5AI (capfit + stenosis index) vs X5A', {s: RUNS / W5 / f'X5AI_s{s}' for s in ('1234', '7', '2025')}, X5A)
report['X5AI_vs_X5'] = paired('X5AI (capfit + stenosis index) vs X5', {s: RUNS / W5 / f'X5AI_s{s}' for s in ('1234', '7', '2025')}, X5)
for wave, label in ((W5B, 'wave-5b'),):
    t3 = {s: RUNS / wave / f'T3_s{s}' for s in ('1234', '7', '2025')}; t3n = {s: RUNS / wave / f'T3n_s{s}' for s in ('1234', '7', '2025')}
    if any((r / PRED).is_dir() for r in t3.values()):
        report['T3_vs_X5'] = paired('T3 cascade (focus + gate) vs X5', t3, X5)
        report['T3n_vs_X5'] = paired('T3n residual stacking (no focus) vs X5', t3n, X5)
        report['T3_vs_T3n'] = paired('T3 vs T3n (focus + gate effect)', t3, t3n)

# ---- cv3: X5X11 vs X5 per fold and pooled over the three held-out folds (= every train138 case once)
cv = {}
for seed, wave in (('1234', W5), ('7', W5B)):
    folds = {}
    pooled = {'units': [], 'true_pa': [], 'true_n': [], 'x5_pa': [], 'x5_n': [], 'x11_pa': [], 'x11_n': []}
    for k in range(3):
        a, b = RUNS / wave / f'X5_f{k}_s{seed}', RUNS / wave / f'X5X11_f{k}_s{seed}'
        if not (a / PRED).is_dir():
            continue
        units = units_of(a); da = load(a, units); db = load(b, units) if (b / PRED).is_dir() else None
        t_pa = [da[u]['true_pa'] for u in units]; t_n = [da[u]['true_norm'] for u in units]
        x5_pa = [da[u]['pred_pa'] for u in units]; x5_n = [da[u]['pred_norm'] for u in units]
        row = dict(n=len(units), x5_pa=cb(t_pa, x5_pa), x5_norm=cb(t_n, x5_n), x5_tails=tails(t_pa, x5_pa), x5_domains=domains(units, t_pa, x5_pa))
        pooled['units'] += units; pooled['true_pa'] += t_pa; pooled['true_n'] += t_n; pooled['x5_pa'] += x5_pa; pooled['x5_n'] += x5_n
        if db:
            x11_pa = [db[u]['pred_pa'] for u in units]; x11_n = [db[u]['pred_norm'] for u in units]
            pc = np.array(per_case(t_pa, x11_pa)) - np.array(per_case(t_pa, x5_pa))
            row.update(x11_pa=cb(t_pa, x11_pa), x11_norm=cb(t_n, x11_n), d_pa=cb(t_pa, x11_pa) - row['x5_pa'], d_norm=cb(t_n, x11_n) - row['x5_norm'],
                       wins=int((pc > 0).sum()), losses=int((pc < 0).sum()), x11_tails=tails(t_pa, x11_pa), x11_domains=domains(units, t_pa, x11_pa))
            pooled['x11_pa'] += x11_pa; pooled['x11_n'] += x11_n
        folds[k] = row
    if not folds:
        continue
    entry = {'folds': folds}
    t_pa, t_n = pooled['true_pa'], pooled['true_n']
    entry['pooled_X5'] = dict(n=len(pooled['units']), pa=cb(t_pa, pooled['x5_pa']), norm=cb(t_n, pooled['x5_n']), tails=tails(t_pa, pooled['x5_pa']), domains=domains(pooled['units'], t_pa, pooled['x5_pa']),
                             case_mean=float(np.mean(per_case(t_pa, pooled['x5_pa']))), case_p10=float(np.quantile(per_case(t_pa, pooled['x5_pa']), 0.1)))
    if len(pooled['x11_pa']) == len(pooled['x5_pa']):
        entry['pooled_X5X11'] = dict(pa=cb(t_pa, pooled['x11_pa']), norm=cb(t_n, pooled['x11_n']), tails=tails(t_pa, pooled['x11_pa']), domains=domains(pooled['units'], t_pa, pooled['x11_pa']),
                                    case_mean=float(np.mean(per_case(t_pa, pooled['x11_pa']))), case_p10=float(np.quantile(per_case(t_pa, pooled['x11_pa']), 0.1)))
        pc = np.array(per_case(t_pa, pooled['x11_pa'])) - np.array(per_case(t_pa, pooled['x5_pa']))
        entry['pooled_delta'] = dict(d_pa=entry['pooled_X5X11']['pa'] - entry['pooled_X5']['pa'], d_norm=entry['pooled_X5X11']['norm'] - entry['pooled_X5']['norm'],
                                     wins=int((pc > 0).sum()), losses=int((pc < 0).sum()), mean_case_delta=float(pc.mean()))
    cv[f'seed{seed}'] = entry
    print(f'=== cv3 seed {seed}')
    for k, r in folds.items():
        line = f"  fold{k} (n={r['n']}): X5 {r['x5_pa']:.4f}/{r['x5_norm']:.4f}"
        if 'x11_pa' in r:
            line += f" | X5X11 {r['x11_pa']:.4f}/{r['x11_norm']:.4f} | Δ {r['d_pa']:+.4f}/{r['d_norm']:+.4f} | wins/losses {r['wins']}/{r['losses']} | high-WSS {r['x11_tails']['high_r2']:.3f} ({r['x5_tails']['high_r2']:.3f}) top10 {r['x11_tails']['top10_ratio']:.3f} ({r['x5_tails']['top10_ratio']:.3f}) p99 {r['x11_tails']['p99_ratio']:.3f} ({r['x5_tails']['p99_ratio']:.3f}) IoU {r['x11_tails']['iou']:.3f} ({r['x5_tails']['iou']:.3f})"
        print(line)
    p = entry['pooled_X5']; print(f"  pooled X5 (n={p['n']}): Pa {p['pa']:.4f} norm {p['norm']:.4f} case mean {p['case_mean']:.3f} P10 {p['case_p10']:.3f} AG/AAA/ILO {p['domains']['AG']:.3f}/{p['domains']['AAA']:.3f}/{p['domains']['ILO']:.3f} | {p['tails']}")
    if 'pooled_X5X11' in entry:
        p = entry['pooled_X5X11']; d = entry['pooled_delta']
        print(f"  pooled X5X11: Pa {p['pa']:.4f} norm {p['norm']:.4f} case mean {p['case_mean']:.3f} P10 {p['case_p10']:.3f} AG/AAA/ILO {p['domains']['AG']:.3f}/{p['domains']['AAA']:.3f}/{p['domains']['ILO']:.3f} | {p['tails']} | Δ {d['d_pa']:+.4f}/{d['d_norm']:+.4f} wins/losses {d['wins']}/{d['losses']}")
report['cv3'] = cv

# ---- fold models on the frozen test34 (if make_cascade_sidecar produced eval_test34): 92-case models vs the 138-case X5
t34 = {}
for k in range(3):
    d = RUNS / W5 / f'X5_f{k}_s1234' / 'eval_test34/ckpt_best/predictions/test'
    if d.is_dir():
        units = sorted(str(p.parent.relative_to(d)) for p in d.glob('*/*/*/predictions.npz'))
        pr = {}
        for u in units:
            with np.load(d / u / 'predictions.npz') as z:
                pr[u] = {kk: z[kk].astype(np.float64) for kk in ('pred_norm', 'true_norm', 'pred_pa', 'true_pa')}
        t34[k] = dict(pa=cb([pr[u]['true_pa'] for u in units], [pr[u]['pred_pa'] for u in units]), norm=cb([pr[u]['true_norm'] for u in units], [pr[u]['pred_norm'] for u in units]))
if t34:
    report['fold_models_on_test34'] = t34; print('=== fold models (91-93 cases) on frozen test34:', t34)
(OUT / 'analyze_wave5.json').write_text(json.dumps(report, indent=1))
print('wrote', OUT / 'analyze_wave5.json')
