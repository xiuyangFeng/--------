"""Read saved v5.1 predictions; no inference or training.

Keep Pa-mean ensemble accuracy separate from log_z-mean residual structure.
The 4 mm bins contain axial as well as circumferential variation.
"""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from training_wss_min import metrics as M

RUNS = ROOT / 'training_wss_min/runs'
VIEW = ROOT / 'data_wss_v5/views_v5_1/wss_min_view_v1'
PRED = 'eval/ckpt_best/predictions/test'


def load(run, unit):
    with np.load(run / PRED / unit / 'predictions.npz') as z:
        return {k: z[k].copy() for k in
                ('row_index', 'true_pa', 'pred_pa', 'true_norm', 'pred_norm')}


def summarize(items):
    t = [d['true_pa'] for d in items]
    p = [d['pred_pa'] for d in items]
    high_t, high_p = [], []
    for a, b in zip(t, p):
        mask = a >= np.quantile(a, .9)
        high_t.append(a[mask]); high_p.append(b[mask])
    details = [d['residual'] for d in items]
    pc = [M.r2_score(a, b) for a, b in zip(t, p)]
    return dict(
        n=len(items), pa=M.casebalanced_field_metrics(t, p),
        case_r2_mean=float(np.mean(pc)), case_r2_p10=float(np.quantile(pc, .1)),
        high_wss_case_p90_then_pool_r2=M.r2_score(np.concatenate(high_t), np.concatenate(high_p)),
        calibration_pooled=M.calibration_metrics(np.concatenate(t), np.concatenate(p)),
        residual_medians={k: float(np.median([d[k] for d in details])) for k in
                          ('within_bin_share', 'bin_mean_share', 'branch_mean_share',
                           'bin_track_r2', 'within_bin_pattern_r2')})


def diagnose(unit, d):
    with np.load(VIEW / unit / 'bundle.npz', allow_pickle=True) as z:
        idx = d['row_index']
        seg = z['wall_segment_id'][idx].astype(int)
        s = z['wall_s_local_mm'][idx].astype(float)
    _, inv, cnt = np.unique(np.column_stack([seg, np.floor(s / 4).astype(int)]),
                            axis=0, return_inverse=True, return_counts=True)
    t, p = d['true_norm'].astype(float), d['pred_norm'].astype(float)
    tm = (np.bincount(inv, weights=t) / cnt)[inv]
    pm = (np.bincount(inv, weights=p) / cnt)[inv]
    r, rm = t - p, tm - pm
    _, bi, bc = np.unique(seg, return_inverse=True, return_counts=True)
    br = (np.bincount(bi, weights=r) / bc)[bi]
    within = float(np.var(r-rm) / np.var(r))
    between = float(np.var(rm) / np.var(r))
    np.testing.assert_allclose(within + between, 1, atol=1e-10)
    return dict(within_bin_share=within, bin_mean_share=between,
                branch_mean_share=float(np.var(br) / np.var(r)),
                bin_track_r2=M.r2_score(tm, pm),
                within_bin_pattern_r2=M.r2_score(t-tm, p-pm))


def case_summary(d):
    t, p = d['true_pa'].astype(float), d['pred_pa'].astype(float)
    high = t >= np.quantile(t, .9)
    pred_high = p >= np.quantile(p, .9)
    return dict(unit=d['unit'], mse_pa=float(np.mean((t-p)**2)),
                r2_pa=M.r2_score(t, p), true_p99_pa=float(np.quantile(t, .99)),
                high_sse_fraction=float(np.sum((t[high]-p[high])**2)/np.sum((t-p)**2)),
                top10_ratio=float(p[high].mean()/t[high].mean()),
                top10_iou=float(np.sum(high & pred_high)/np.sum(high | pred_high)),
                **d['residual'])


def collect(name, run_groups):
    items, seen, verified = [], set(), []
    for runs in run_groups:
        units = sorted(str(p.parent.relative_to(runs[0] / PRED))
                       for p in (runs[0] / PRED).glob('*/*/*/predictions.npz'))
        single_truth = [[] for _ in runs]
        single_preds = [[] for _ in runs]
        single_high_t = [[] for _ in runs]
        single_high_p = [[] for _ in runs]
        for unit in units:
            assert unit not in seen, unit
            seen.add(unit)
            data = [load(run, unit) for run in runs]
            for j, d in enumerate(data):
                for key in ('row_index', 'true_pa', 'true_norm'):
                    np.testing.assert_array_equal(d[key], data[0][key])
                assert np.isfinite(d['pred_pa']).all()
                assert np.isfinite(d['pred_norm']).all()
                single_truth[j].append(d['true_pa'])
                single_preds[j].append(d['pred_pa'])
                mask = d['true_pa'] >= np.quantile(d['true_pa'], .9)
                single_high_t[j].append(d['true_pa'][mask])
                single_high_p[j].append(d['pred_pa'][mask])
            d = dict(data[0])
            d['pred_pa'] = np.mean([x['pred_pa'] for x in data], axis=0)
            d['pred_norm'] = np.mean([x['pred_norm'] for x in data], axis=0)
            d.update(unit=unit, residual=diagnose(unit, d))
            items.append(d)
        for j, run in enumerate(runs):
            expected = json.loads((run / 'eval/ckpt_best/metrics.json').read_text())['test']
            actual = M.casebalanced_field_metrics(single_truth[j], single_preds[j])['r2']
            high = M.r2_score(np.concatenate(single_high_t[j]), np.concatenate(single_high_p[j]))
            np.testing.assert_allclose(actual, expected['field_casebalanced']['r2'], atol=1e-9, rtol=0)
            np.testing.assert_allclose(high, expected['regional_field']['high_wss']['r2'], atol=1e-9, rtol=0)
            verified.append(dict(run=str(run.relative_to(ROOT)), pa_r2_cb=actual, high_wss_r2=high))
    result = dict(summary=summarize(items),
                  per_cohort={c: summarize([d for d in items if d['unit'].split('/')[0] == c])
                              for c in ('AG', 'AAA', 'ILO')},
                  verified_single_runs=verified,
                  per_case=[case_summary(d) for d in items])
    print(name, json.dumps({k: v for k, v in result.items() if k in ('summary', 'per_cohort')}), flush=True)
    return result


output = dict(protocol=dict(
    data='v5.1, train136/test34', checkpoint='ckpt_best, train_loss/top3',
    scope='saved caches only; test34 development-exposed; cv3 is one seed per fold',
    accuracy='Pa-mean ensemble; legacy_vertex; R2_cb uses a shared case-balanced mean',
    structure='log_z-mean ensemble; branch x 4 mm bins; median of within-case statistics',
    note='Within-bin residual variance is neither Pa SSE nor purely circumferential error.'))
base = RUNS / 'wss_v51_wave1_20260916'
output['test34_base5'] = collect('test34_base5', [[base / ('X5D_v51_s%d' % s)
                                                 for s in (1234, 7, 2025, 11, 2026)]])
output['cv3_oof'] = collect('cv3_oof', [[RUNS / 'wss_v51_wave2a_20260916' /
                                        ('X5D_v51_f%d_s1234' % k)] for k in range(3)])
assert output['test34_base5']['summary']['n'] == 34
assert output['cv3_oof']['summary']['n'] == 136
Path(__file__).with_name('current_residuals.json').write_text(
    json.dumps(output, ensure_ascii=False, indent=2) + '\n')
